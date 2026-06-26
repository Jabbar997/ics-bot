"""Market regime classification tests."""
from app.domain import Regime
from app.market.regime import classify_regime


def test_bull():
    r = classify_regime(spy_close=500, spy_ma50=492, spy_ma200=450, drawdown_20d=-0.01)
    assert r.regime == Regime.BULL


def test_bear():
    r = classify_regime(spy_close=440, spy_ma50=460, spy_ma200=450, drawdown_20d=-0.02)
    assert r.regime == Regime.BEAR


def test_weak_bull():
    # Above MA200, below MA50, MA50 still above MA200 -> weak bull.
    r = classify_regime(spy_close=455, spy_ma50=460, spy_ma200=450, drawdown_20d=-0.01)
    assert r.regime == Regime.WEAK_BULL


def test_sideways_when_ma50_below_ma200():
    r = classify_regime(spy_close=455, spy_ma50=452, spy_ma200=460, drawdown_20d=-0.01)
    assert r.regime == Regime.SIDEWAYS


def test_panic_overrides_bull():
    # Levels look bullish but a >5% 20d drawdown forces panic.
    r = classify_regime(spy_close=500, spy_ma50=492, spy_ma200=450, drawdown_20d=-0.06)
    assert r.regime == Regime.PANIC


def test_panic_threshold_boundary():
    assert classify_regime(500, 492, 450, -0.05).regime == Regime.PANIC
    assert classify_regime(500, 492, 450, -0.049).regime == Regime.BULL
