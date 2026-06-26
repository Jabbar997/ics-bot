"""Technical indicators, implemented manually (no pandas-ta dependency).

Each function takes/returns pandas Series so they compose cleanly and stay
transparent — there is no hidden state. Wilder's smoothing (RMA) is used for RSI
and ATR, matching the conventional definitions used by most charting platforms.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _rma(series: pd.Series, period: int) -> pd.Series:
    """Wilder's moving average (a.k.a. RMA / SMMA)."""
    return series.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def calculate_moving_average(close: pd.Series, window: int) -> pd.Series:
    """Simple moving average."""
    return close.rolling(window=window, min_periods=window).mean()


def calculate_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index (Wilder). Bounded 0..100."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    avg_gain = _rma(gain, period)
    avg_loss = _rma(loss, period)

    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))
    # When average loss is zero (pure uptrend) RSI is 100; when both zero, neutral.
    rsi = rsi.where(avg_loss != 0, 100.0)
    rsi = rsi.where(~((avg_gain == 0) & (avg_loss == 0)), 50.0)
    return rsi


def calculate_macd(
    close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> pd.DataFrame:
    """MACD line, signal line and histogram as a DataFrame."""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return pd.DataFrame(
        {"macd": macd_line, "macd_signal": signal_line, "macd_hist": hist}
    )


def calculate_atr(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> pd.Series:
    """Average True Range (Wilder)."""
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return _rma(tr, period)


def calculate_volume_ma(volume: pd.Series, window: int = 20) -> pd.Series:
    return volume.rolling(window=window, min_periods=window).mean()


def calculate_drawdown(close: pd.Series, window: int = 20) -> pd.Series:
    """Percent drawdown vs the rolling ``window``-period high.

    Returns a fraction (e.g. -0.021 means -2.1% below the 20-day high).
    """
    rolling_high = close.rolling(window=window, min_periods=1).max()
    return close / rolling_high - 1.0


def rolling_high(close: pd.Series, window: int = 20) -> pd.Series:
    return close.rolling(window=window, min_periods=1).max()


def calculate_volatility(close: pd.Series, window: int = 20, annualize: bool = True) -> pd.Series:
    """Rolling standard deviation of daily returns (optionally annualised)."""
    returns = close.pct_change()
    vol = returns.rolling(window=window, min_periods=window).std()
    if annualize:
        vol = vol * np.sqrt(252.0)
    return vol


def calculate_beta(
    asset_close: pd.Series, market_close: pd.Series, window: int = 60
) -> pd.Series:
    """Rolling beta of the asset vs the market (cov / var of returns)."""
    asset_ret = asset_close.pct_change()
    market_ret = market_close.pct_change()
    aligned = pd.concat([asset_ret, market_ret], axis=1, join="inner").dropna()
    if aligned.empty:
        return pd.Series(dtype=float, index=asset_close.index)
    a = aligned.iloc[:, 0]
    m = aligned.iloc[:, 1]
    cov = a.rolling(window=window, min_periods=window).cov(m)
    var = m.rolling(window=window, min_periods=window).var()
    beta = cov / var
    return beta.reindex(asset_close.index)
