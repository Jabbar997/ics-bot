"""ICS-DOC-004 Phase 0 — learning feedback loop.

Acceptance test: simulate >= 30 synthetic closed trades with a *known* relationship
between one DQS component and the realised return, then assert
  (a) the adjustment moves in the correct direction,
  (b) no component moves more than the +/-5% per-cycle cap,
  (c) the weights still sum to exactly 100,
  (d) the gate blocks any change below 30 trades,
  (e) every cycle is recorded as a LearningEvent (AuditLog-grade rigour).
"""
from datetime import datetime, timedelta

import pytest

from app.db.database import session_scope
from app.db.models import DecisionOutcome, LearningEvent
from app.decision.dqs import COMPONENT_NAMES, DEFAULT_WEIGHTS, normalize_weights
from app.learning.feedback_loop import (
    MAX_SHIFT_PCT,
    MIN_CLOSED_TRADES,
    compute_correlations,
    propose_weights,
    run_feedback_cycle,
    spearman,
)
from app.learning.weights import load_weights, reset_weights, save_weights


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _make_outcomes(session, n=35, *, driver="timing_quality"):
    """Synthetic closed trades where `driver` predicts the return perfectly.

    The driving component rises with i and so does the realised return; the other
    components are held constant so only the driver can correlate.
    """
    base = datetime(2026, 1, 1)
    for i in range(n):
        components = {name: 12 for name in COMPONENT_NAMES}
        components[driver] = i  # monotonically increasing
        session.add(
            DecisionOutcome(
                ticker=f"T{i:02d}",
                strategy="trend",
                entry_at=base + timedelta(days=i),
                exit_at=base + timedelta(days=i + 3),
                realized_return=0.001 * i,  # monotonically increasing with driver
                holding_period_days=3,
                mfe=0.02,
                mae=-0.01,
                dqs_at_entry=sum(components.values()),
                dqs_components_json=components,
            )
        )
    session.flush()


# --------------------------------------------------------------------------- #
# spearman + correlations
# --------------------------------------------------------------------------- #
def test_spearman_detects_monotonic_relationships():
    assert spearman([1, 2, 3, 4, 5], [2, 4, 6, 8, 10]) == pytest.approx(1.0)
    assert spearman([1, 2, 3, 4, 5], [10, 8, 6, 4, 2]) == pytest.approx(-1.0)


def test_spearman_undefined_cases_return_none():
    assert spearman([1, 2], [1, 2]) is None            # too few points
    assert spearman([1, 1, 1, 1], [1, 2, 3, 4]) is None  # constant series


def test_correlation_isolates_the_driving_component(db_url):
    with session_scope() as s:
        _make_outcomes(s, 35, driver="timing_quality")
        from app.learning.outcomes import load_outcomes

        corrs = compute_correlations(load_outcomes(s))
    assert corrs["timing_quality"] == pytest.approx(1.0)
    # Constant components have no defined rank correlation.
    assert corrs["risk_management"] is None


# --------------------------------------------------------------------------- #
# weight proposal: direction, cap, sum
# --------------------------------------------------------------------------- #
def test_positive_correlation_increases_that_weight():
    before = dict(DEFAULT_WEIGHTS)
    after = propose_weights(before, {"timing_quality": 1.0, "risk_management": -1.0})
    assert after["timing_quality"] > before["timing_quality"]
    assert after["risk_management"] < before["risk_management"]


def test_shift_never_exceeds_the_five_percent_cap():
    before = dict(DEFAULT_WEIGHTS)
    # Extreme, contradictory correlations must still respect the cap.
    corrs = {
        "strategy_alignment": 1.0,
        "risk_management": -1.0,
        "timing_quality": 1.0,
        "market_regime_strength": -1.0,
        "reason_clarity": 1.0,
    }
    after = propose_weights(before, corrs, max_shift_pct=MAX_SHIFT_PCT)
    for name in COMPONENT_NAMES:
        rel = abs(after[name] - before[name]) / before[name] * 100.0
        assert rel <= MAX_SHIFT_PCT + 1e-6, f"{name} moved {rel:.3f}% > {MAX_SHIFT_PCT}%"


def test_weights_always_sum_to_one_hundred():
    after = propose_weights(dict(DEFAULT_WEIGHTS), {"timing_quality": 0.9})
    assert sum(after.values()) == pytest.approx(100.0)
    assert sum(normalize_weights({"timing_quality": 3, "risk_management": 1}).values()) == (
        pytest.approx(100.0)
    )


def test_unknown_correlation_leaves_weight_untouched():
    before = dict(DEFAULT_WEIGHTS)
    after = propose_weights(before, {name: None for name in COMPONENT_NAMES})
    for name in COMPONENT_NAMES:
        assert after[name] == pytest.approx(before[name])


# --------------------------------------------------------------------------- #
# full cycle
# --------------------------------------------------------------------------- #
def test_cycle_applies_and_moves_driver_up(db_url):
    with session_scope() as s:
        _make_outcomes(s, 35, driver="timing_quality")
        before = load_weights(s)
        result = run_feedback_cycle(s, price_provider=lambda *a: None)

    assert result.applied is True
    assert result.trades_considered == 35
    assert result.weights_after["timing_quality"] > before["timing_quality"]
    assert sum(result.weights_after.values()) == pytest.approx(100.0)

    with session_scope() as s:
        assert load_weights(s)["timing_quality"] > before["timing_quality"]


def test_gate_blocks_below_thirty_trades(db_url):
    with session_scope() as s:
        _make_outcomes(s, MIN_CLOSED_TRADES - 1, driver="timing_quality")
        before = load_weights(s)
        result = run_feedback_cycle(s, price_provider=lambda *a: None)

    assert result.applied is False
    assert result.trades_considered == MIN_CLOSED_TRADES - 1
    assert result.weights_after == before  # untouched
    with session_scope() as s:
        assert load_weights(s) == before


def test_every_cycle_records_a_learning_event(db_url):
    """Applied or skipped, each cycle leaves exactly one immutable record."""
    from sqlalchemy import func, select

    with session_scope() as s:
        _make_outcomes(s, 10)          # below the gate -> skipped
        run_feedback_cycle(s, price_provider=lambda *a: None)
    with session_scope() as s:
        _make_outcomes(s, 35, driver="reason_clarity")  # above the gate -> applied
        run_feedback_cycle(s, price_provider=lambda *a: None)

    with session_scope() as s:
        events = list(s.scalars(select(LearningEvent).order_by(LearningEvent.timestamp)))
        assert s.scalar(select(func.count()).select_from(LearningEvent)) == 2
        assert [e.applied for e in events] == [False, True]
        for e in events:
            # AuditLog-grade: full context reconstructable from the record.
            assert e.reason and e.weights_before_json and e.weights_after_json
            assert e.min_trades_required == MIN_CLOSED_TRADES
            assert e.raw_context_json["mode"] == "paper_only"


def test_repeated_cycles_stay_bounded_and_normalised(db_url):
    """Ten consecutive cycles must never break the sum or the floor/ceiling."""
    with session_scope() as s:
        _make_outcomes(s, 40, driver="timing_quality")
    for _ in range(10):
        with session_scope() as s:
            run_feedback_cycle(s, price_provider=lambda *a: None)
    with session_scope() as s:
        w = load_weights(s)
    assert sum(w.values()) == pytest.approx(100.0)
    for name, v in w.items():
        assert 5.0 <= v <= 40.0, f"{name} escaped its bounds: {v}"


def test_reset_restores_defaults(db_url):
    with session_scope() as s:
        save_weights(s, {**DEFAULT_WEIGHTS, "timing_quality": 30.0})
        assert load_weights(s)["timing_quality"] != pytest.approx(DEFAULT_WEIGHTS["timing_quality"])
        reset_weights(s)
        assert load_weights(s) == pytest.approx(DEFAULT_WEIGHTS)


# --------------------------------------------------------------------------- #
# outcomes recording
# --------------------------------------------------------------------------- #
def test_record_outcomes_links_closed_position_to_its_entry_decision(db_url):
    from app.config import load_config
    from app.db.models import Decision
    from app.decision.decision_engine import DecisionEngine
    from app.domain import Action, FeatureSnapshot, Regime, RegimeResult, Signal, StrategyName
    from app.learning.outcomes import record_outcomes
    from app.paper.broker import PaperBroker
    from app.risk.risk_manager import PortfolioState
    from sqlalchemy import select

    cfg = load_config()
    feat = FeatureSnapshot(
        ticker="AAPL", as_of=datetime(2026, 1, 2), close=190.0, ma20=185, ma50=180,
        ma200=160, rsi14=55.0, macd=1.2, macd_signal=0.8, atr14=3.5,
        volume=2_000_000, volume_ma20=1_500_000, drawdown_20d=-0.01,
    )
    regime = RegimeResult(Regime.BULL, 500, 492, 450, -0.01, "bull")
    buy = Signal("AAPL", StrategyName.TREND, Action.BUY,
                 "trend up with healthy RSI and confirming volume quality entry",
                 0.85, {"x": 1})

    with session_scope() as s:
        broker = PaperBroker(s, cfg)
        de = DecisionEngine(s, cfg, broker)
        de.process_signal(buy, feat, regime, PortfolioState(266.0, 266.0), [])
        pos = broker.positions.get_open_by_ticker("AAPL")
        broker.close_position(pos, 210.0, reason="target")

    with session_scope() as s:
        created = record_outcomes(s, price_provider=lambda *a: None)
        assert len(created) == 1
        o = created[0]
        assert o.ticker == "AAPL"
        assert o.realized_return > 0            # closed at a profit
        assert o.dqs_components_json            # DQS captured from the audit log
        entry = s.get(Decision, o.decision_id)
        assert entry is not None and entry.action == "buy"

    # Running again must not duplicate.
    with session_scope() as s:
        assert record_outcomes(s, price_provider=lambda *a: None) == []
        assert len(list(s.scalars(select(DecisionOutcome)))) == 1
