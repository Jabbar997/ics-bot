"""Calibrating the DQS threshold from decisions the system declined.

The weekly loop already tunes *component weights* from the handful of trades it
took. This tunes the *cut-off* from the thousands it refused — a far larger
sample, and the only place the threshold's own correctness is observable.

The question it answers is narrow and decision-relevant: **look at the setups
that scored just below the current cut-off and were rejected. Did they go on to
beat the market or lag it?**

* They beat it -> the cut-off is turning away good setups -> lower it.
* They lagged it -> the cut-off is earning its place -> raise it or hold.
* Neither -> the score is not discriminating there -> change nothing.

Everything is judged against the measured base rate, never against zero: in a
rising market every band has a positive forward return, and comparing to zero
would recommend lowering the threshold forever.

Bounded exactly like the weight loop: a minimum sample before anything moves, a
small cap per cycle, hard floor and ceiling, and a LearningEvent every time.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import List, Optional

from sqlalchemy import select

from app.db.models import RejectedOutcome
from app.db.repositories import SystemConfigRepository
from app.learning.counterfactuals import LEARNABLE
from app.logging_config import get_logger

log = get_logger(__name__)

THRESHOLD_KEY = "dqs_minimum"
# How many scored rejections existed when the cut-off last moved. Without this
# the weekly cycle re-reads the SAME evidence every Friday and walks the
# threshold 2 points a week until it hits a bound — learning from repetition
# rather than from anything new.
EVIDENCE_KEY = "dqs_threshold_evidence_at_last_change"

# --- bounds ---------------------------------------------------------------- #
MIN_BAND_SAMPLE = 200   # near-miss rejections required before moving at all
MIN_NEW_EVIDENCE = 150  # newly scored rejections required before moving *again*
MAX_SHIFT_POINTS = 2.0  # per weekly cycle
MIN_THRESHOLD = 60.0
MAX_THRESHOLD = 85.0
BAND_WIDTH = 10.0       # how far below the cut-off counts as a near miss
EDGE_TOLERANCE = 0.0025  # 0.25% — below this the band is indistinguishable from the market


@dataclass
class ThresholdProposal:
    current: float
    proposed: float
    band_low: float
    band_high: float
    band_n: int
    band_return: Optional[float] = None
    baseline_return: Optional[float] = None
    edge: Optional[float] = None
    applied: bool = False
    reason: str = ""
    total_evidence: int = 0
    new_evidence: int = 0

    @property
    def changed(self) -> bool:
        return abs(self.proposed - self.current) > 1e-9


def load_threshold(session, default: float) -> float:
    """Learned cut-off, or the configured default when never tuned."""
    raw = SystemConfigRepository(session).get(THRESHOLD_KEY)
    if not raw:
        return float(default)
    try:
        value = float(json.loads(raw))
    except (ValueError, TypeError):
        log.warning("Stored DQS threshold is unreadable; using the configured default.")
        return float(default)
    return max(MIN_THRESHOLD, min(MAX_THRESHOLD, value))


def save_threshold(session, value: float) -> float:
    bounded = max(MIN_THRESHOLD, min(MAX_THRESHOLD, float(value)))
    SystemConfigRepository(session).set(THRESHOLD_KEY, json.dumps(round(bounded, 2)))
    return bounded


def reset_threshold(session, default: float) -> float:
    return save_threshold(session, default)


def _total_scored(session) -> int:
    """Every rejection scored so far — the yardstick for 'new evidence'."""
    from sqlalchemy import func

    return int(session.scalar(func.count(RejectedOutcome.id)) or 0)


def _evidence_at_last_change(session) -> int:
    raw = SystemConfigRepository(session).get(EVIDENCE_KEY)
    try:
        return int(raw) if raw else 0
    except (TypeError, ValueError):
        return 0


def _near_miss_band(session, current: float) -> List[RejectedOutcome]:
    """Learnable rejections scoring within BAND_WIDTH just below the cut-off."""
    low = current - BAND_WIDTH
    return [
        r for r in session.scalars(select(RejectedOutcome))
        if r.forward_return is not None
        and r.dqs_score is not None
        and r.category in LEARNABLE
        and low <= float(r.dqs_score) < current
    ]


def propose_threshold(
    session,
    current: float,
    baseline_return: Optional[float],
    *,
    min_sample: int = MIN_BAND_SAMPLE,
    max_shift: float = MAX_SHIFT_POINTS,
    min_new_evidence: int = MIN_NEW_EVIDENCE,
) -> ThresholdProposal:
    """Bounded proposal for the DQS cut-off. Never applies anything itself."""
    band = _near_miss_band(session, current)
    total = _total_scored(session)
    seen = _evidence_at_last_change(session)
    proposal = ThresholdProposal(
        current=current, proposed=current,
        band_low=current - BAND_WIDTH, band_high=current,
        band_n=len(band), baseline_return=baseline_return,
        total_evidence=total, new_evidence=max(0, total - seen),
    )

    if baseline_return is None:
        proposal.reason = "لا معدّل أساس للمقارنة؛ لا تعديل."
        return proposal
    if len(band) < min_sample:
        proposal.reason = (
            f"عيّنة النطاق {len(band)} < الحد الأدنى {min_sample}؛ لا تعديل."
        )
        return proposal
    # Only move on evidence that did not exist last time. Re-reading the same
    # rows every week is repetition, not learning.
    if seen and proposal.new_evidence < min_new_evidence:
        proposal.reason = (
            f"أدلة جديدة {proposal.new_evidence} < الحد {min_new_evidence} "
            "منذ آخر تعديل؛ لا تعديل."
        )
        return proposal

    returns = [float(r.forward_return) for r in band]
    band_return = sum(returns) / len(returns)
    edge = band_return - baseline_return
    proposal.band_return = band_return
    proposal.edge = edge

    if abs(edge) <= EDGE_TOLERANCE:
        proposal.reason = (
            f"النطاق {proposal.band_low:.0f}–{current:.0f} يطابق السوق "
            f"({edge*100:+.2f}%)؛ العتبة معايَرة."
        )
        return proposal

    # Beat the market -> we are refusing good setups -> lower the bar, and vice versa.
    direction = -1.0 if edge > 0 else 1.0
    proposed = max(MIN_THRESHOLD, min(MAX_THRESHOLD, current + direction * max_shift))
    proposal.proposed = proposed
    if not proposal.changed:
        proposal.reason = "العتبة عند حدّها؛ لا تعديل ممكن."
        return proposal

    what = "خفض" if direction < 0 else "رفع"
    proposal.reason = (
        f"{what} العتبة {current:.0f} ← {proposed:.0f}: النطاق "
        f"{proposal.band_low:.0f}–{current:.0f} ({len(band)} فرصة) "
        f"{'تفوّق على' if edge > 0 else 'تخلّف عن'} السوق بـ {abs(edge)*100:.2f}%."
    )
    return proposal


def apply_threshold(session, proposal: ThresholdProposal) -> ThresholdProposal:
    if proposal.changed:
        save_threshold(session, proposal.proposed)
        # Remember how much evidence justified this move, so the next cycle needs
        # genuinely new data before moving again.
        SystemConfigRepository(session).set(
            EVIDENCE_KEY, str(proposal.total_evidence or _total_scored(session))
        )
        proposal.applied = True
        log.info(
            "DQS threshold: %.0f -> %.0f (on %d new rejections)",
            proposal.current, proposal.proposed, proposal.new_evidence,
        )
    return proposal
