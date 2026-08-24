"""Calibrating the DQS cut-off from rejected decisions.

The threshold decides what the system is allowed to trade at all, so these tests
lean on the guardrails: nothing moves without a real sample, nothing moves more
than the per-cycle cap, and — the failure mode that already bit once — a rising
market must not be mistaken for evidence that the cut-off is too high.
"""
from datetime import datetime, timedelta

import pytest

from app.config import load_config
from app.db.database import session_scope
from app.db.models import RejectedOutcome
from app.learning.counterfactuals import CAT_SLOTS_FULL, CAT_STRATEGY_FILTER
from app.learning.threshold_tuner import (
    BAND_WIDTH,
    MAX_SHIFT_POINTS,
    MAX_THRESHOLD,
    MIN_BAND_SAMPLE,
    MIN_THRESHOLD,
    apply_threshold,
    load_threshold,
    propose_threshold,
    reset_threshold,
    save_threshold,
)

BASE = 0.0106  # the measured market base rate: +1.06% over 10 days


def _seed(session, n, dqs, forward_return, category=CAT_STRATEGY_FILTER):
    when = datetime.utcnow() - timedelta(days=90)
    for i in range(n):
        session.add(
            RejectedOutcome(
                ticker=f"S{i%7}", strategy="trend", rejected_at=when,
                category=category, rejection_reason="فلتر الحجم.",
                dqs_score=dqs, horizon_days=10, forward_return=forward_return,
                rejection_helped=forward_return <= 0,
            )
        )
    session.flush()


# --------------------------------------------------------------------------- #
# storage
# --------------------------------------------------------------------------- #
def test_defaults_to_the_configured_threshold(db_url):
    with session_scope() as s:
        assert load_threshold(s, 70) == 70.0


def test_persists_and_clamps_to_bounds(db_url):
    with session_scope() as s:
        assert save_threshold(s, 999) == MAX_THRESHOLD
        assert load_threshold(s, 70) == MAX_THRESHOLD
        assert save_threshold(s, 0) == MIN_THRESHOLD
        assert reset_threshold(s, 70) == 70.0


# --------------------------------------------------------------------------- #
# the gate
# --------------------------------------------------------------------------- #
def test_nothing_moves_without_a_baseline(db_url):
    with session_scope() as s:
        _seed(s, MIN_BAND_SAMPLE + 10, dqs=65, forward_return=0.05)
        p = propose_threshold(s, 70.0, baseline_return=None)
    assert not p.changed
    assert "معدّل أساس" in p.reason


def test_nothing_moves_below_the_minimum_sample(db_url):
    with session_scope() as s:
        _seed(s, MIN_BAND_SAMPLE - 1, dqs=65, forward_return=0.05)
        p = propose_threshold(s, 70.0, baseline_return=BASE)
    assert not p.changed
    assert "الحد الأدنى" in p.reason


def test_capacity_rejections_never_reach_the_band(db_url):
    """Slot exhaustion says nothing about whether the cut-off is right."""
    with session_scope() as s:
        _seed(s, MIN_BAND_SAMPLE + 50, dqs=65, forward_return=0.05,
              category=CAT_SLOTS_FULL)
        p = propose_threshold(s, 70.0, baseline_return=BASE)
    assert p.band_n == 0
    assert not p.changed


# --------------------------------------------------------------------------- #
# direction
# --------------------------------------------------------------------------- #
def test_lowers_the_bar_when_near_misses_beat_the_market(db_url):
    with session_scope() as s:
        _seed(s, MIN_BAND_SAMPLE + 20, dqs=65, forward_return=BASE + 0.03)
        p = propose_threshold(s, 70.0, baseline_return=BASE)
    assert p.proposed == pytest.approx(70.0 - MAX_SHIFT_POINTS)
    assert p.edge > 0
    assert "خفض" in p.reason


def test_raises_the_bar_when_near_misses_lag_the_market(db_url):
    with session_scope() as s:
        _seed(s, MIN_BAND_SAMPLE + 20, dqs=65, forward_return=BASE - 0.03)
        p = propose_threshold(s, 70.0, baseline_return=BASE)
    assert p.proposed == pytest.approx(70.0 + MAX_SHIFT_POINTS)
    assert p.edge < 0
    assert "رفع" in p.reason


def test_a_rising_market_alone_does_not_lower_the_bar(db_url):
    """The false alarm that already happened once, pinned shut.

    Every band has a positive forward return in a bull market. Only an edge
    *over the base rate* may move the threshold.
    """
    with session_scope() as s:
        _seed(s, MIN_BAND_SAMPLE + 20, dqs=65, forward_return=BASE)
        p = propose_threshold(s, 70.0, baseline_return=BASE)
    assert p.band_return > 0          # the market rose after these rejections...
    assert p.edge == pytest.approx(0.0, abs=1e-9)
    assert not p.changed              # ...but that is not evidence of anything
    assert "معايَرة" in p.reason


# --------------------------------------------------------------------------- #
# bounds
# --------------------------------------------------------------------------- #
def test_never_moves_more_than_the_cap_in_one_cycle(db_url):
    with session_scope() as s:
        _seed(s, MIN_BAND_SAMPLE + 20, dqs=65, forward_return=BASE + 0.50)
        p = propose_threshold(s, 70.0, baseline_return=BASE)
    assert abs(p.proposed - p.current) <= MAX_SHIFT_POINTS + 1e-9


def test_respects_the_floor(db_url):
    with session_scope() as s:
        _seed(s, MIN_BAND_SAMPLE + 20, dqs=MIN_THRESHOLD - 5, forward_return=BASE + 0.05)
        p = propose_threshold(s, MIN_THRESHOLD, baseline_return=BASE)
    assert p.proposed >= MIN_THRESHOLD


def test_only_the_near_miss_band_counts(db_url):
    """Setups far below the cut-off say nothing about where it should sit."""
    with session_scope() as s:
        _seed(s, MIN_BAND_SAMPLE + 20, dqs=int(70 - BAND_WIDTH - 5),
              forward_return=BASE + 0.05)
        p = propose_threshold(s, 70.0, baseline_return=BASE)
    assert p.band_n == 0
    assert not p.changed


# --------------------------------------------------------------------------- #
# applying
# --------------------------------------------------------------------------- #
def test_apply_persists_and_is_picked_up_by_the_engine(db_url):
    with session_scope() as s:
        _seed(s, MIN_BAND_SAMPLE + 20, dqs=65, forward_return=BASE - 0.03)
        p = apply_threshold(s, propose_threshold(s, 70.0, baseline_return=BASE))
        assert p.applied is True
    with session_scope() as s:
        assert load_threshold(s, 70) == pytest.approx(72.0)

    # The decision engine must read the learned value, not the config default.
    from app.paper.broker import PaperBroker
    from app.decision.decision_engine import DecisionEngine

    cfg = load_config()
    with session_scope() as s:
        engine = DecisionEngine(s, cfg, PaperBroker(s, cfg))
        assert engine.minimum_dqs == 72


def test_unchanged_proposal_is_not_applied(db_url):
    with session_scope() as s:
        _seed(s, 10, dqs=65, forward_return=BASE)
        p = apply_threshold(s, propose_threshold(s, 70.0, baseline_return=BASE))
        assert p.applied is False
        assert load_threshold(s, 70) == 70.0
