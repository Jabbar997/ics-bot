"""Indicator calculation tests (MA, RSI, ATR, MACD)."""
import numpy as np
import pandas as pd

from app.features import indicators as ind


def test_moving_average_simple():
    s = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
    ma3 = ind.calculate_moving_average(s, 3)
    assert np.isnan(ma3.iloc[0]) and np.isnan(ma3.iloc[1])
    assert ma3.iloc[2] == 2.0  # mean(1,2,3)
    assert ma3.iloc[-1] == 9.0  # mean(8,9,10)


def test_rsi_pure_uptrend_is_100():
    s = pd.Series(np.arange(1, 60, dtype=float))
    rsi = ind.calculate_rsi(s, 14)
    assert rsi.iloc[-1] == 100.0


def test_rsi_pure_downtrend_is_zero():
    s = pd.Series(np.arange(60, 1, -1, dtype=float))
    rsi = ind.calculate_rsi(s, 14)
    assert rsi.iloc[-1] == 0.0


def test_rsi_bounds():
    rng = np.random.default_rng(0)
    s = pd.Series(100 + np.cumsum(rng.normal(0, 1, 200)))
    rsi = ind.calculate_rsi(s, 14).dropna()
    assert (rsi >= 0).all() and (rsi <= 100).all()


def test_atr_constant_range():
    # Constant daily range of 1.0, no overnight gaps -> ATR converges to 1.0.
    n = 100
    low = pd.Series(np.full(n, 10.0))
    high = low + 1.0
    close = low + 0.5
    atr = ind.calculate_atr(high, low, close, 14)
    assert abs(atr.iloc[-1] - 1.0) < 1e-6


def test_macd_shape_and_crossover_sign():
    s = pd.Series(np.linspace(100, 200, 100))
    macd = ind.calculate_macd(s)
    assert set(["macd", "macd_signal", "macd_hist"]).issubset(macd.columns)
    # In a steady uptrend, MACD line should be above its signal line at the end.
    assert macd["macd"].iloc[-1] >= macd["macd_signal"].iloc[-1]


def test_drawdown_is_non_positive():
    s = pd.Series([10, 11, 12, 9, 8, 13], dtype=float)
    dd = ind.calculate_drawdown(s, window=20)
    assert (dd <= 1e-9).all()
    # At the trough (8) vs rolling high (12): 8/12 - 1
    assert abs(dd.iloc[4] - (8 / 12 - 1)) < 1e-9
