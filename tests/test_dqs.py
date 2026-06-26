"""Decision Quality Score tests."""
from datetime import datetime

from app.decision.dqs import calculate_dqs
from app.domain import (
    Action,
    FeatureSnapshot,
    Regime,
    RegimeResult,
    Signal,
    StrategyName,
)


def _bull():
    return RegimeResult(Regime.BULL, 500, 492, 450, -0.01, "bull")


def _good_features():
    return FeatureSnapshot(
        ticker="AAPL", as_of=datetime(2024, 1, 2), close=190.0,
        ma20=185, ma50=180, ma200=160, rsi14=55.0, macd=1.2, macd_signal=0.8,
        atr14=3.5, volume=2_000_000, volume_ma20=1_500_000, high_20d=192, drawdown_20d=-0.01,
    )


def _strong_buy():
    return Signal(
        ticker="AAPL", strategy=StrategyName.TREND, action=Action.BUY,
        reason="Up-trend: close above MA50 and MA200, RSI healthy, volume confirms entry quality.",
        confidence=0.85,
        raw_conditions={"close": 190, "ma50": 180, "ma200": 160, "rsi14": 55, "volume": 2e6, "volume_ma20": 1.5e6, "regime": "bull"},
    )


def _good_risk_ctx():
    return {"has_stop": True, "risk_reward": 2.0, "within_position_limit": True, "atr_available": True}


def test_components_within_weights_and_sum():
    res = calculate_dqs(_strong_buy(), _good_features(), _bull(), _good_risk_ctx())
    caps = {
        "strategy_alignment": 25, "risk_management": 25, "timing_quality": 20,
        "market_regime_strength": 15, "reason_clarity": 15,
    }
    for k, cap in caps.items():
        assert 0 <= res.components[k] <= cap
    assert res.score == sum(res.components.values())


def test_strong_signal_allowed():
    res = calculate_dqs(_strong_buy(), _good_features(), _bull(), _good_risk_ctx())
    assert res.score >= 70
    assert res.allowed is True


def test_weak_signal_rejected():
    weak = Signal(
        ticker="XYZ", strategy=StrategyName.TREND, action=Action.REJECT,
        reason="weak", confidence=0.2, raw_conditions={"close": 1},
    )
    bad_feat = FeatureSnapshot(
        ticker="XYZ", as_of=datetime(2024, 1, 2), close=10.0, ma50=9, ma200=8,
        rsi14=78.0, macd=-1, macd_signal=0.5, atr14=1.0, volume=100, volume_ma20=500,
        drawdown_20d=-0.04,
    )
    res = calculate_dqs(weak, bad_feat, RegimeResult(Regime.BEAR, 1, 1, 1, -0.04, "bear"), {"has_stop": False})
    assert res.score < 70
    assert res.allowed is False


def test_missing_risk_context_does_not_crash():
    res = calculate_dqs(_strong_buy(), _good_features(), _bull(), None)
    assert 0 <= res.score <= 100


def test_threshold_is_configurable():
    res = calculate_dqs(_strong_buy(), _good_features(), _bull(), _good_risk_ctx(), minimum_dqs=99)
    assert res.allowed is False
    assert "99" in res.reason
