"""Weekly DQS feedback loop (ICS-DOC-004 Phase 0).

What it does, once a week:

1. Record a :class:`DecisionOutcome` for every newly closed paper position.
2. Gate: do nothing unless at least ``MIN_CLOSED_TRADES`` outcomes exist.
3. For each DQS component, measure the Spearman rank correlation between the
   points that component contributed at entry and the realised return.
4. Nudge each weight toward the components that actually predicted returns,
   capped at ``MAX_SHIFT_PCT`` **relative** change per component per cycle.
5. Re-normalise so the weights still sum to exactly 100, and persist.
6. Write a :class:`LearningEvent` — always, including on a skipped cycle.

Safety: this loop only re-balances how candidates are *scored*. It cannot place,
size, or exit a trade, and it never touches the risk limits or the kill switch.
Weights are additionally floored/ceilinged so no component can be starved or
allowed to dominate.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from app.db.models import LearningEvent
from app.decision.dqs import COMPONENT_NAMES, TOTAL_WEIGHT, normalize_weights
from app.learning.outcomes import load_outcomes, record_outcomes
from app.learning.weights import load_weights, save_weights
from app.logging_config import get_logger

log = get_logger(__name__)

# --- Phase 0 bounds --------------------------------------------------------- #
MIN_CLOSED_TRADES = 30      # gate: fewer closed trades than this changes nothing
MAX_SHIFT_PCT = 5.0         # cap: max RELATIVE change per component per cycle
MIN_COMPONENT_WEIGHT = 5.0  # no component may be starved below this
MAX_COMPONENT_WEIGHT = 40.0 # ...nor dominate above this


@dataclass
class FeedbackResult:
    applied: bool
    reason: str
    trades_considered: int = 0
    correlations: Dict[str, Optional[float]] = field(default_factory=dict)
    weights_before: Dict[str, float] = field(default_factory=dict)
    weights_after: Dict[str, float] = field(default_factory=dict)

    def summary(self) -> str:
        head = "تم تحديث الأوزان" if self.applied else "لم تُحدَّث الأوزان"
        return f"{head} — {self.reason} (صفقات: {self.trades_considered})"


def spearman(xs: List[float], ys: List[float]) -> Optional[float]:
    """Spearman rank correlation, or None when it is not defined."""
    if len(xs) != len(ys) or len(xs) < 3:
        return None
    if len(set(xs)) < 2 or len(set(ys)) < 2:
        return None  # a constant series has no rank correlation
    from scipy.stats import spearmanr

    rho, _p = spearmanr(xs, ys)
    if rho is None:
        return None
    try:
        rho = float(rho)
    except (TypeError, ValueError):
        return None
    return None if rho != rho else rho  # drop NaN


def compute_correlations(outcomes) -> Dict[str, Optional[float]]:
    """Spearman(component points at entry, realised return) per component."""
    returns = [float(o.realized_return) for o in outcomes]
    out: Dict[str, Optional[float]] = {}
    for name in COMPONENT_NAMES:
        xs, ys = [], []
        for o, r in zip(outcomes, returns):
            comps = o.dqs_components_json or {}
            val = comps.get(name)
            if val is None:
                continue
            xs.append(float(val))
            ys.append(r)
        out[name] = spearman(xs, ys)
    return out


def propose_weights(
    current: Dict[str, float],
    correlations: Dict[str, Optional[float]],
    max_shift_pct: float = MAX_SHIFT_PCT,
) -> Dict[str, float]:
    """Bounded, zero-sum re-balance of the weights.

    A component whose contribution correlated positively with realised return
    gains weight; a negatively-correlated one loses weight.

    The move is constructed to be **zero-sum**: signals are centred on the
    weight-weighted mean correlation, so the deltas cancel out and the total
    stays at exactly 100 with no re-normalisation. That matters because
    re-normalising *after* capping would silently push a component past its cap.
    Each delta is then at most ``max_shift_pct`` percent of that component's own
    current weight.

    An undefined correlation (constant or missing data) is treated as neutral
    evidence (0.0) rather than being frozen, so a single informative component
    can still be rewarded relative to the uninformative ones.
    """
    weights = {name: float(current.get(name, 0.0)) for name in COMPONENT_NAMES}
    total_w = sum(weights.values())
    if total_w <= 0:
        return normalize_weights(current)

    rhos = {name: (0.0 if correlations.get(name) is None else float(correlations[name]))
            for name in COMPONENT_NAMES}

    # Weight-weighted mean => sum(w_i * (rho_i - mean)) == 0.
    mean_rho = sum(weights[n] * rhos[n] for n in COMPONENT_NAMES) / total_w
    signals = {n: rhos[n] - mean_rho for n in COMPONENT_NAMES}

    # Scale into [-1, 1] (a constant factor preserves the zero-sum property).
    peak = max(abs(v) for v in signals.values())
    if peak <= 1e-12:
        return normalize_weights(weights)  # no differential signal at all
    signals = {n: v / peak for n, v in signals.items()}

    proposed = {
        n: weights[n] * (1.0 + (max_shift_pct / 100.0) * signals[n]) for n in COMPONENT_NAMES
    }

    # Safety floor/ceiling wins over gradualism; it only binds in extremes, and
    # normalize_weights keeps the sum at 100 if clamping ever does bind.
    clamped = {
        n: max(MIN_COMPONENT_WEIGHT, min(MAX_COMPONENT_WEIGHT, v)) for n, v in proposed.items()
    }
    if any(abs(clamped[n] - proposed[n]) > 1e-12 for n in COMPONENT_NAMES):
        return normalize_weights(clamped)
    return clamped


def _record_event(
    session,
    *,
    event_type: str,
    applied: bool,
    trades: int,
    reason: str,
    correlations: Dict[str, Optional[float]],
    before: Dict[str, float],
    after: Dict[str, float],
    max_shift_pct: float,
) -> LearningEvent:
    """Immutable record of the cycle — written even when nothing changed."""
    event = LearningEvent(
        event_type=event_type,
        applied=applied,
        trades_considered=trades,
        min_trades_required=MIN_CLOSED_TRADES,
        reason=reason,
        correlations_json={k: (None if v is None else round(v, 6)) for k, v in correlations.items()},
        weights_before_json={k: round(v, 6) for k, v in before.items()},
        weights_after_json={k: round(v, 6) for k, v in after.items()},
        max_shift_pct=max_shift_pct,
        raw_context_json={
            "min_component_weight": MIN_COMPONENT_WEIGHT,
            "max_component_weight": MAX_COMPONENT_WEIGHT,
            "total_weight": TOTAL_WEIGHT,
            "mode": "paper_only",
        },
    )
    session.add(event)
    session.flush()
    return event


def run_feedback_cycle(
    session,
    *,
    min_trades: int = MIN_CLOSED_TRADES,
    max_shift_pct: float = MAX_SHIFT_PCT,
    price_provider=None,
) -> FeedbackResult:
    """Run one weekly cycle. Always writes exactly one LearningEvent."""
    record_outcomes(session, price_provider=price_provider)
    outcomes = load_outcomes(session)
    before = load_weights(session)
    n = len(outcomes)

    if n < min_trades:
        reason = f"عدد الصفقات المغلقة {n} < الحد الأدنى {min_trades}؛ لا تعديل."
        _record_event(
            session, event_type="skipped", applied=False, trades=n, reason=reason,
            correlations={}, before=before, after=before, max_shift_pct=max_shift_pct,
        )
        log.info("Feedback cycle skipped: %s", reason)
        return FeedbackResult(False, reason, n, {}, before, before)

    correlations = compute_correlations(outcomes)
    if all(v is None for v in correlations.values()):
        reason = "تعذّر حساب أي ارتباط (بيانات ثابتة أو ناقصة)؛ لا تعديل."
        _record_event(
            session, event_type="skipped", applied=False, trades=n, reason=reason,
            correlations=correlations, before=before, after=before, max_shift_pct=max_shift_pct,
        )
        return FeedbackResult(False, reason, n, correlations, before, before)

    after = propose_weights(before, correlations, max_shift_pct=max_shift_pct)
    save_weights(session, after)
    reason = f"أُعيد توازن الأوزان من {n} صفقة مغلقة (سقف {max_shift_pct}% لكل مكوّن)."
    _record_event(
        session, event_type="weight_update", applied=True, trades=n, reason=reason,
        correlations=correlations, before=before, after=after, max_shift_pct=max_shift_pct,
    )
    log.info("Feedback cycle applied: %s", reason)
    return FeedbackResult(True, reason, n, correlations, before, after)
