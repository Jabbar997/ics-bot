"""Risk manager tests — limits, sizing, stops."""
from datetime import datetime

import pytest

from app.config import load_config
from app.decision.dqs import calculate_dqs
from app.domain import Action, DQSResult, FeatureSnapshot, Regime, RegimeResult, Signal, StrategyName
from app.risk.risk_manager import (
    OpenPositionView,
    PortfolioState,
    compute_stop_and_target,
    validate_order,
)


@pytest.fixture
def cfg():
    return load_config()


def _signal(ticker="AAPL"):
    return Signal(
        ticker=ticker, strategy=StrategyName.TREND, action=Action.BUY,
        reason="trend up, rsi healthy, volume confirms, structure positive overall",
        confidence=0.85,
        raw_conditions={"close": 190, "ma50": 180, "ma200": 160, "rsi14": 55, "volume": 2e6, "volume_ma20": 1.5e6},
    )


def _features(close=190.0, atr=3.5):
    return FeatureSnapshot(
        ticker="AAPL", as_of=datetime(2024, 1, 2), close=close, ma20=185, ma50=180,
        ma200=160, rsi14=55, macd=1, macd_signal=0.5, atr14=atr, volume=2e6,
        volume_ma20=1.5e6, drawdown_20d=-0.01,
    )


def _dqs(score=85):
    return DQSResult(score=score, components={}, allowed=score >= 70, reason="ok")


def _bull():
    return RegimeResult(Regime.BULL, 500, 492, 450, -0.01, "bull")


# -- stop loss --------------------------------------------------------------- #
def test_stop_uses_atr_when_atr_is_tighter():
    # 2*ATR (2.0) < 7% of 100 (7.0) -> ATR governs.
    st = compute_stop_and_target(entry=100.0, atr=1.0)
    assert abs(st.stop_distance - 2.0) < 1e-9
    assert abs(st.stop_loss - 98.0) < 1e-9
    assert abs(st.target_price - 104.0) < 1e-9  # RR 2:1


def test_stop_uses_absolute_pct_when_atr_is_wider():
    # 2*ATR (20) > 7% of 100 (7) -> 7% absolute governs.
    st = compute_stop_and_target(entry=100.0, atr=10.0, absolute_stop_pct=7.0)
    assert abs(st.stop_distance - 7.0) < 1e-9
    assert abs(st.stop_loss - 93.0) < 1e-9


# -- sizing ------------------------------------------------------------------ #
def test_position_size_is_capped_at_max_pct(cfg):
    pf = PortfolioState(total_value=266.0, cash=266.0)
    rd = validate_order(_signal(), _dqs(), pf, [], cfg, features=_features())
    assert rd.allowed
    assert rd.position_size_pct == pytest.approx(cfg.risk.max_position_size_pct, abs=1e-6)
    assert rd.notional == pytest.approx(26.6, abs=1e-6)


# -- limits ------------------------------------------------------------------ #
def test_max_open_positions(cfg):
    pf = PortfolioState(total_value=266.0, cash=266.0)
    held = [OpenPositionView(t) for t in ("MSFT", "NVDA", "AMZN")]
    rd = validate_order(_signal("AAPL"), _dqs(), pf, held, cfg, features=_features())
    assert not rd.allowed
    assert "MAX_OPEN_POSITIONS" in rd.violations


def test_already_holding_blocks_pyramiding(cfg):
    pf = PortfolioState(total_value=266.0, cash=266.0)
    rd = validate_order(_signal("AAPL"), _dqs(), pf, [OpenPositionView("AAPL")], cfg, features=_features())
    assert not rd.allowed
    assert "ALREADY_HOLDING" in rd.violations


def test_not_in_watchlist(cfg):
    pf = PortfolioState(total_value=266.0, cash=266.0)
    rd = validate_order(_signal("ZZZZ"), _dqs(), pf, [], cfg, features=_features())
    assert not rd.allowed
    assert "NOT_IN_WATCHLIST" in rd.violations


def test_dqs_below_minimum(cfg):
    pf = PortfolioState(total_value=266.0, cash=266.0)
    rd = validate_order(_signal(), _dqs(score=60), pf, [], cfg, features=_features())
    assert not rd.allowed
    assert "DQS_BELOW_MIN" in rd.violations


def test_missing_dqs_rejected(cfg):
    pf = PortfolioState(total_value=266.0, cash=266.0)
    rd = validate_order(_signal(), None, pf, [], cfg, features=_features())
    assert not rd.allowed
    assert "DQS_MISSING" in rd.violations


def test_kill_switch_blocks(cfg):
    pf = PortfolioState(total_value=266.0, cash=266.0)
    rd = validate_order(_signal(), _dqs(), pf, [], cfg, features=_features(), kill_switch_active=True)
    assert not rd.allowed
    assert "KILL_SWITCH_ACTIVE" in rd.violations


def test_weekly_loss_limit_blocks(cfg):
    pf = PortfolioState(total_value=266.0, cash=266.0, weekly_return_pct=-0.06)
    rd = validate_order(_signal(), _dqs(), pf, [], cfg, features=_features())
    assert not rd.allowed
    assert "WEEKLY_LOSS_LIMIT" in rd.violations


def test_drawdown_limit_blocks(cfg):
    pf = PortfolioState(total_value=226.0, cash=226.0, current_drawdown_pct=-0.16)
    rd = validate_order(_signal(), _dqs(), pf, [], cfg, features=_features())
    assert not rd.allowed
    assert "MAX_DRAWDOWN_LIMIT" in rd.violations
