"""Learning from rejected decisions.

The system records ~97% of its decisions as rejections and learns from none of
them. These tests cover turning that into calibration data — and the guardrails
that keep it honest: capacity rejections must never be mistaken for signal
quality, and nothing may be scored before its forward window has actually
elapsed.
"""
from datetime import datetime, timedelta

import pandas as pd
import pytest

from app.db.database import session_scope
from app.db.models import Decision, RejectedOutcome
from app.db.repositories import DecisionRepository
from app.learning.counterfactuals import (
    CAT_ALREADY_HOLDING,
    CAT_DQS_BELOW,
    CAT_RISK_LIMIT,
    CAT_SLOTS_FULL,
    CAT_STRATEGY_FILTER,
    LEARNABLE,
    analyze_calibration,
    categorize_rejection,
    record_rejected_outcomes,
    threshold_calibration,
)


# --------------------------------------------------------------------------- #
# categorisation — the axis everything else groups on
# --------------------------------------------------------------------------- #
def test_capacity_rejections_are_not_signal_quality():
    """Slot exhaustion says nothing about whether the setup was good."""
    assert categorize_rejection("بلغ الحد الأقصى للمراكز المفتوحة (3).") == CAT_SLOTS_FULL
    assert categorize_rejection("blocked", "MAX_OPEN_POSITIONS") == CAT_SLOTS_FULL
    assert categorize_rejection("يوجد مركز مفتوح في VTI؛ لا تكديس.") == CAT_ALREADY_HOLDING
    assert CAT_SLOTS_FULL not in LEARNABLE
    assert CAT_ALREADY_HOLDING not in LEARNABLE


def test_filter_and_dqs_rejections_are_learnable():
    assert categorize_rejection("اتجاه صاعد لكن الحجم دون متوسط 20 يومًا.") == CAT_STRATEGY_FILTER
    assert categorize_rejection("DQS 63 < الحد 70؛ فرصة مرفوضة.") == CAT_DQS_BELOW
    assert CAT_STRATEGY_FILTER in LEARNABLE
    assert CAT_DQS_BELOW in LEARNABLE


def test_risk_limit_rejections_are_their_own_category():
    assert categorize_rejection("بلغ حد الخسارة الأسبوعي.") == CAT_RISK_LIMIT
    assert categorize_rejection("x", "WEEKLY_LOSS_LIMIT") == CAT_RISK_LIMIT
    assert CAT_RISK_LIMIT not in LEARNABLE


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _prices(start="2026-01-05", n=60, drift=0.0):
    """Deterministic frame: close moves by `drift` per bar from 100."""
    idx = pd.bdate_range(start, periods=n)
    close = [100.0 * (1 + drift) ** i for i in range(n)]
    return pd.DataFrame(
        {"open": close, "high": [c * 1.01 for c in close],
         "low": [c * 0.99 for c in close], "close": close,
         "adjusted_close": close, "volume": [1e6] * n},
        index=idx,
    )


def _reject(session, ticker, when, reason, dqs=68):
    return DecisionRepository(session).record(
        ticker=ticker, action="reject", strategy="trend", price=100.0,
        dqs_score=dqs, confidence=0.3, reason=reason,
        market_regime="bull", rejected_opportunity=True, rejection_reason=reason,
        raw_context={"dqs": {"components": {"timing_quality": 12}}},
        created_at=when,
    )


# --------------------------------------------------------------------------- #
# recording
# --------------------------------------------------------------------------- #
def test_scores_a_rejection_against_what_the_market_did(db_url):
    old = datetime.utcnow() - timedelta(days=120)
    with session_scope() as s:
        _reject(s, "AAPL", old, "اتجاه صاعد لكن الحجم دون متوسط 20 يومًا.")

    rising = _prices(start=old.strftime("%Y-%m-%d"), drift=0.01)  # +1%/bar
    with session_scope() as s:
        created = record_rejected_outcomes(s, price_provider=lambda t: rising)
        assert len(created) == 1
        row = created[0]
        assert row.category == CAT_STRATEGY_FILTER
        assert row.forward_return > 0            # market rose after we said no
        assert row.rejection_helped is False     # ...so the rejection cost us
        assert row.forward_mfe >= row.forward_return
        assert row.forward_mae <= row.forward_mfe


def test_rejection_that_avoided_a_loss_is_marked_helpful(db_url):
    old = datetime.utcnow() - timedelta(days=120)
    with session_scope() as s:
        _reject(s, "XOM", old, "اتجاه صاعد لكن RSI مرتفع (71).")

    falling = _prices(start=old.strftime("%Y-%m-%d"), drift=-0.01)
    with session_scope() as s:
        row = record_rejected_outcomes(s, price_provider=lambda t: falling)[0]
        assert row.forward_return < 0
        assert row.rejection_helped is True


def test_recent_rejections_are_not_scored_early(db_url):
    """Nothing may be judged before its forward window has actually elapsed."""
    with session_scope() as s:
        _reject(s, "AAPL", datetime.utcnow() - timedelta(days=1), "فلتر.")
    with session_scope() as s:
        assert record_rejected_outcomes(s, price_provider=lambda t: _prices()) == []


def test_recording_is_idempotent(db_url):
    old = datetime.utcnow() - timedelta(days=120)
    with session_scope() as s:
        _reject(s, "AAPL", old, "فلتر الحجم.")
    frame = _prices(start=old.strftime("%Y-%m-%d"), drift=0.005)
    with session_scope() as s:
        assert len(record_rejected_outcomes(s, price_provider=lambda t: frame)) == 1
    with session_scope() as s:
        assert record_rejected_outcomes(s, price_provider=lambda t: frame) == []
        assert len(list(s.scalars(__import__("sqlalchemy").select(RejectedOutcome)))) == 1


def test_missing_price_history_is_skipped_not_fatal(db_url):
    old = datetime.utcnow() - timedelta(days=120)
    with session_scope() as s:
        _reject(s, "GONE", old, "فلتر.")
    with session_scope() as s:
        assert record_rejected_outcomes(s, price_provider=lambda t: None) == []


# --------------------------------------------------------------------------- #
# calibration
# --------------------------------------------------------------------------- #
def _seed_mixed(session, n_filter=40, n_slots=20, drift_days=120):
    old = datetime.utcnow() - timedelta(days=drift_days)
    for i in range(n_filter):
        _reject(session, "AAA", old + timedelta(days=i % 5),
                "اتجاه صاعد لكن الحجم دون متوسط 20 يومًا.", dqs=60 + i % 25)
    for i in range(n_slots):
        _reject(session, "BBB", old + timedelta(days=i % 5),
                "بلغ الحد الأقصى للمراكز المفتوحة (3).", dqs=85)


def test_calibration_separates_capacity_from_signal_quality(db_url):
    with session_scope() as s:
        _seed_mixed(s)
    old = (datetime.utcnow() - timedelta(days=120)).strftime("%Y-%m-%d")
    frames = {"AAA": _prices(start=old, drift=-0.01), "BBB": _prices(start=old, drift=0.01)}
    with session_scope() as s:
        record_rejected_outcomes(s, price_provider=lambda t: frames[t])
        rep = analyze_calibration(s)

    assert CAT_STRATEGY_FILTER in rep.by_category
    assert CAT_SLOTS_FULL in rep.by_category
    # Capacity rejections are excluded from the learnable sample...
    assert rep.learnable_n() == rep.by_category[CAT_STRATEGY_FILTER].n
    # ...but their opportunity cost is still surfaced separately.
    assert rep.slots_full_n > 0
    assert rep.slots_full_mean_return > 0


def _stat_with_baseline(drift, baseline):
    """Score 40 filter rejections against a market moving at `drift`."""
    old = (datetime.utcnow() - timedelta(days=120)).strftime("%Y-%m-%d")
    with session_scope() as s:
        record_rejected_outcomes(s, price_provider=lambda t: _prices(start=old, drift=drift))
        rep = analyze_calibration(s, baseline_return=baseline, baseline_hit_rate=0.58)
    return rep.by_category[CAT_STRATEGY_FILTER]


def test_a_protective_filter_reads_as_protective(db_url):
    """Rejected setups did clearly worse than the market -> the filter earned it."""
    with session_scope() as s:
        _seed_mixed(s, n_filter=40, n_slots=0)
    stat = _stat_with_baseline(drift=-0.01, baseline=0.01)
    assert stat.hit_rate == pytest.approx(1.0)
    assert stat.edge < 0
    assert "أسوأ من السوق" in stat.verdict


def test_a_costly_filter_is_flagged(db_url):
    """Rejected setups clearly beat the market -> the filter is costing us."""
    with session_scope() as s:
        _seed_mixed(s, n_filter=40, n_slots=0)
    stat = _stat_with_baseline(drift=0.01, baseline=0.0)
    assert stat.edge > 0
    assert "أفضل من السوق" in stat.verdict


def test_a_filter_matching_the_market_is_not_flagged_as_costly(db_url):
    """The false alarm this logic was built to prevent.

    In a rising market everything has a positive forward return, so comparing a
    filter to ZERO reports "it rejects winners" for a filter that is merely
    average. Judged against the base rate, that same filter must read as having
    no predictive power — which is what the real data showed: rejected setups
    returned +1.01% against a +1.06% market.
    """
    with session_scope() as s:
        _seed_mixed(s, n_filter=40, n_slots=0)
    old = (datetime.utcnow() - timedelta(days=120)).strftime("%Y-%m-%d")
    with session_scope() as s:
        record_rejected_outcomes(s, price_provider=lambda t: _prices(start=old, drift=0.001))
        stat = analyze_calibration(s).by_category[CAT_STRATEGY_FILTER]
        # Without a baseline the verdict must refuse to judge, not cry wolf.
        assert "لا معدّل أساس" in stat.verdict

        matched = analyze_calibration(
            s, baseline_return=stat.mean_forward_return, baseline_hit_rate=0.58,
        ).by_category[CAT_STRATEGY_FILTER]
    assert matched.edge == pytest.approx(0.0, abs=1e-9)
    assert "بلا قوة تنبؤية" in matched.verdict


def test_compute_baseline_measures_the_universe(db_url):
    from app.learning.counterfactuals import compute_baseline

    frames = {"UP": _prices(n=260, drift=0.002), "FLAT": _prices(n=260, drift=0.0)}
    mean, hit = compute_baseline(frames, horizon=10, warmup=20)
    assert mean is not None and 0.0 < mean < 0.05
    assert 0.0 <= hit <= 1.0


def test_small_samples_refuse_to_render_a_verdict(db_url):
    with session_scope() as s:
        _seed_mixed(s, n_filter=5, n_slots=0)
    old = (datetime.utcnow() - timedelta(days=120)).strftime("%Y-%m-%d")
    with session_scope() as s:
        record_rejected_outcomes(s, price_provider=lambda t: _prices(start=old, drift=0.01))
        stat = analyze_calibration(s).by_category[CAT_STRATEGY_FILTER]
    assert "غير كافية" in stat.verdict


def test_threshold_calibration_only_uses_learnable_rejections(db_url):
    with session_scope() as s:
        _seed_mixed(s, n_filter=40, n_slots=20)
    old = (datetime.utcnow() - timedelta(days=120)).strftime("%Y-%m-%d")
    frames = {"AAA": _prices(start=old, drift=0.01), "BBB": _prices(start=old, drift=0.01)}
    with session_scope() as s:
        record_rejected_outcomes(s, price_provider=lambda t: frames[t])
        rows = threshold_calibration(s, thresholds=(60, 70, 80))

    assert [r["threshold"] for r in rows] == [60, 70, 80]
    assert rows[0]["n"] >= rows[1]["n"] >= rows[2]["n"]  # higher bar, fewer pass
    # The 85-DQS capacity rejections must not leak into this analysis.
    assert rows[-1]["n"] < 20


# --------------------------------------------------------------------------- #
# baseline persistence
# --------------------------------------------------------------------------- #
def test_baseline_survives_so_verdicts_can_be_rendered(db_url):
    """Measuring the base rate needs the whole price history.

    /learning cannot refetch that on every invocation, so the value is stored
    when it is measured. Without this the calibration showed real numbers and
    then refused to judge them: "no baseline to compare against".
    """
    from app.learning.counterfactuals import load_baseline, save_baseline

    with session_scope() as s:
        assert load_baseline(s) == (None, None)
        save_baseline(s, 0.0106, 0.58)
    with session_scope() as s:
        mean, hit = load_baseline(s)
    assert mean == pytest.approx(0.0106)
    assert hit == pytest.approx(0.58)


def test_calibration_uses_the_stored_baseline_without_being_told(db_url):
    from app.learning.counterfactuals import save_baseline

    with session_scope() as s:
        _seed_mixed(s, n_filter=40, n_slots=0)
    old = (datetime.utcnow() - timedelta(days=120)).strftime("%Y-%m-%d")
    with session_scope() as s:
        record_rejected_outcomes(s, price_provider=lambda t: _prices(start=old, drift=-0.01))
        # No baseline stored yet -> refuses to judge.
        assert "لا معدّل أساس" in analyze_calibration(s).by_category[CAT_STRATEGY_FILTER].verdict
        save_baseline(s, 0.0106, 0.58)
        # Stored -> judges without the caller passing anything.
        stat = analyze_calibration(s).by_category[CAT_STRATEGY_FILTER]
    assert stat.edge is not None
    assert "لا معدّل أساس" not in stat.verdict


def test_an_explicit_baseline_overrides_the_stored_one(db_url):
    from app.learning.counterfactuals import save_baseline

    with session_scope() as s:
        _seed_mixed(s, n_filter=40, n_slots=0)
    old = (datetime.utcnow() - timedelta(days=120)).strftime("%Y-%m-%d")
    with session_scope() as s:
        record_rejected_outcomes(s, price_provider=lambda t: _prices(start=old, drift=0.0))
        save_baseline(s, 0.99, 0.58)  # deliberately absurd
        stat = analyze_calibration(s, baseline_return=0.0).by_category[CAT_STRATEGY_FILTER]
    assert stat.baseline_return == pytest.approx(0.0)
