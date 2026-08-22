"""Training data for the Phase 1 model — features in, forward return out.

**No new indicators.** ICS-DOC-004 Phase 1 says to use only what
`feature_engine.py` already computes, and that is respected: every column below
is a ratio of existing indicators. Ratios rather than raw levels are necessary
because the model is trained across symbols — a raw `close` of 700 (SPY) and 60
(KO) is not comparable, while `close / ma50` is. This is normalisation of
existing features, not new feature engineering (see D-11).

**No look-ahead.** Two separate guarantees:

1. Features at row *i* are backward-looking (`compute_features` uses rolling and
   ewm windows only).
2. The label at row *i* is the return of a trade that a signal on day *i* would
   actually have got: filled at the **next open** and exited at the open
   ``HORIZON_DAYS`` sessions later — matching the backtester's documented
   execution convention. A row is therefore only usable once bar ``i+1+N``
   exists, and :func:`build_training_frame` refuses to emit any row whose label
   would need data after the requested training cut-off.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from app.logging_config import get_logger

log = get_logger(__name__)

# Approved via the Phase 1 parameter decision: the median real holding period is
# 18 calendar days (~13 sessions) over 164 closed trades, so 10 sessions is the
# nearest round horizon that stays inside how long the system actually holds.
HORIZON_DAYS = 10

FEATURE_COLUMNS: List[str] = [
    "close_over_ma20",
    "close_over_ma50",
    "close_over_ma200",
    "ma20_over_ma50",
    "ma50_over_ma200",
    "rsi14",
    "macd_hist_over_close",
    "atr_over_close",
    "volume_over_ma20",
    "drawdown_20d",
    "volatility_20d",
    "beta_vs_spy",
    "close_over_high20d",
]

LABEL_COLUMN = "forward_return"


def _safe_ratio(num: pd.Series, den: pd.Series) -> pd.Series:
    return num / den.replace(0.0, np.nan)


def make_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Derive the model's feature columns from an existing feature frame."""
    out = pd.DataFrame(index=frame.index)
    out["close_over_ma20"] = _safe_ratio(frame["close"], frame["ma20"])
    out["close_over_ma50"] = _safe_ratio(frame["close"], frame["ma50"])
    out["close_over_ma200"] = _safe_ratio(frame["close"], frame["ma200"])
    out["ma20_over_ma50"] = _safe_ratio(frame["ma20"], frame["ma50"])
    out["ma50_over_ma200"] = _safe_ratio(frame["ma50"], frame["ma200"])
    out["rsi14"] = frame["rsi14"]
    out["macd_hist_over_close"] = _safe_ratio(frame["macd"] - frame["macd_signal"], frame["close"])
    out["atr_over_close"] = _safe_ratio(frame["atr14"], frame["close"])
    out["volume_over_ma20"] = _safe_ratio(frame["volume"], frame["volume_ma20"])
    out["drawdown_20d"] = frame["drawdown_20d"]
    out["volatility_20d"] = frame["volatility_20d"]
    out["beta_vs_spy"] = frame["beta_vs_spy"]
    out["close_over_high20d"] = _safe_ratio(frame["close"], frame["high_20d"])
    return out


def make_labels(frame: pd.DataFrame, horizon: int = HORIZON_DAYS) -> pd.Series:
    """Forward return of a trade signalled on day *i*.

    Entered at ``open[i+1]`` and exited at ``open[i+1+horizon]`` — the same
    next-open convention the backtester uses, so the label matches what the
    system could actually have captured. The last ``horizon+1`` rows are NaN by
    construction because their exit bar does not exist yet.
    """
    entry = frame["open"].shift(-1)
    exit_ = frame["open"].shift(-(1 + horizon))
    return (exit_ / entry) - 1.0


@dataclass
class TrainingFrame:
    X: pd.DataFrame
    y: pd.Series
    tickers: pd.Series
    dates: pd.DatetimeIndex

    def __len__(self) -> int:
        return len(self.y)

    @property
    def span(self) -> tuple[Optional[datetime], Optional[datetime]]:
        if len(self.dates) == 0:
            return None, None
        return self.dates.min().to_pydatetime(), self.dates.max().to_pydatetime()


def build_training_frame(
    features_by_symbol: Dict[str, pd.DataFrame],
    *,
    train_end: Optional[datetime] = None,
    train_start: Optional[datetime] = None,
    horizon: int = HORIZON_DAYS,
    exclude: Optional[List[str]] = None,
) -> TrainingFrame:
    """Stack every symbol into one supervised dataset.

    ``train_end`` is a hard information barrier: a row survives only if its
    **exit bar** (``i+1+horizon``) is at or before ``train_end``. Filtering on
    the signal date alone would leak up to ``horizon+1`` sessions of future
    prices into training — the exact mistake the no-look-ahead rule exists to
    prevent.
    """
    exclude = set(exclude or [])
    frames: List[pd.DataFrame] = []

    for ticker, frame in features_by_symbol.items():
        if ticker in exclude or frame is None or frame.empty:
            continue
        feats = make_features(frame)
        labels = make_labels(frame, horizon)
        # Timestamp of the bar whose price closes the trade.
        exit_dates = pd.Series(frame.index, index=frame.index).shift(-(1 + horizon))

        part = feats.copy()
        part[LABEL_COLUMN] = labels
        part["_ticker"] = ticker
        part["_exit_date"] = exit_dates
        frames.append(part)

    if not frames:
        empty = pd.DataFrame(columns=FEATURE_COLUMNS)
        return TrainingFrame(empty, pd.Series(dtype=float), pd.Series(dtype=object),
                             pd.DatetimeIndex([]))

    data = pd.concat(frames)
    data = data.dropna(subset=FEATURE_COLUMNS + [LABEL_COLUMN, "_exit_date"])

    if train_start is not None:
        data = data[data.index >= pd.Timestamp(train_start)]
    if train_end is not None:
        # The barrier is on the EXIT bar, not the signal bar.
        data = data[data["_exit_date"] <= pd.Timestamp(train_end)]

    data = data.sort_index()
    return TrainingFrame(
        X=data[FEATURE_COLUMNS].astype(float),
        y=data[LABEL_COLUMN].astype(float),
        tickers=data["_ticker"],
        dates=pd.DatetimeIndex(data.index),
    )
