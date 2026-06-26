"""Feature engine — turns a clean OHLCV frame into indicator columns and
point-in-time :class:`FeatureSnapshot` value objects.

Expected input: a DataFrame indexed by date with columns
``open, high, low, close, volume`` (``adjusted_close`` optional).
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

import pandas as pd

from app.domain import FeatureSnapshot
from app.features import indicators as ind


def compute_features(df: pd.DataFrame, spy_close: Optional[pd.Series] = None) -> pd.DataFrame:
    """Append all indicator columns to ``df`` and return a new frame.

    ``spy_close`` (the benchmark close series) is optional; when supplied, a
    rolling beta column is added.
    """
    if df.empty:
        return df.copy()

    out = df.copy()
    close = out["close"]

    out["ma20"] = ind.calculate_moving_average(close, 20)
    out["ma50"] = ind.calculate_moving_average(close, 50)
    out["ma200"] = ind.calculate_moving_average(close, 200)
    out["rsi14"] = ind.calculate_rsi(close, 14)

    macd = ind.calculate_macd(close)
    out["macd"] = macd["macd"]
    out["macd_signal"] = macd["macd_signal"]

    out["atr14"] = ind.calculate_atr(out["high"], out["low"], close, 14)
    out["volume_ma20"] = ind.calculate_volume_ma(out["volume"], 20)
    out["high_20d"] = ind.rolling_high(close, 20)
    out["drawdown_20d"] = ind.calculate_drawdown(close, 20)
    out["volatility_20d"] = ind.calculate_volatility(close, 20)

    if spy_close is not None and not spy_close.empty:
        out["beta_vs_spy"] = ind.calculate_beta(close, spy_close, window=60)
    else:
        out["beta_vs_spy"] = pd.NA

    return out


def _as_of(idx_value) -> datetime:
    if isinstance(idx_value, pd.Timestamp):
        dt = idx_value.to_pydatetime()
    elif isinstance(idx_value, datetime):
        dt = idx_value
    else:
        dt = datetime.now(timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _f(row, key) -> Optional[float]:
    val = row.get(key)
    if val is None or pd.isna(val):
        return None
    return float(val)


def build_feature_snapshot(
    symbol: str,
    features_df: pd.DataFrame,
    as_of_index: int = -1,
) -> FeatureSnapshot:
    """Build a :class:`FeatureSnapshot` for one row of a features frame.

    ``as_of_index`` is a positional index into the frame (default last row).
    """
    if features_df.empty:
        raise ValueError(f"No data to build feature snapshot for {symbol}")

    row = features_df.iloc[as_of_index]
    idx_value = features_df.index[as_of_index]

    return FeatureSnapshot(
        ticker=symbol,
        as_of=_as_of(idx_value),
        close=_f(row, "close") or 0.0,
        ma20=_f(row, "ma20"),
        ma50=_f(row, "ma50"),
        ma200=_f(row, "ma200"),
        rsi14=_f(row, "rsi14"),
        macd=_f(row, "macd"),
        macd_signal=_f(row, "macd_signal"),
        atr14=_f(row, "atr14"),
        volume=_f(row, "volume"),
        volume_ma20=_f(row, "volume_ma20"),
        high_20d=_f(row, "high_20d"),
        drawdown_20d=_f(row, "drawdown_20d"),
        volatility_20d=_f(row, "volatility_20d"),
        beta_vs_spy=_f(row, "beta_vs_spy"),
    )
