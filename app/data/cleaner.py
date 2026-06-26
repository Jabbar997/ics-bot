"""Data cleaning and validation.

Decisions must never be made from incomplete data, so cleaning is conservative:
duplicate bars are dropped, the frame is sorted, missing closes are forward
filled (and rows still missing a close are dropped), and symbols without enough
history are rejected.
"""
from __future__ import annotations

import pandas as pd

from app.logging_config import get_logger

log = get_logger(__name__)

REQUIRED_COLUMNS = ["open", "high", "low", "close", "volume"]
# Need ~200 bars for MA200 plus a buffer to produce a usable snapshot.
MIN_BARS_FOR_DECISION = 210


class InsufficientDataError(ValueError):
    """Raised when a symbol does not have enough clean history to decide on."""


def clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Clean a raw OHLCV frame. Returns a new, sorted, de-duplicated frame."""
    if df is None or df.empty:
        return pd.DataFrame(columns=REQUIRED_COLUMNS)

    out = df.copy()

    # 1. Normalise column names to lowercase.
    out.columns = [str(c).lower().replace(" ", "_") for c in out.columns]

    # 2. Sort by date (index).
    out = out.sort_index()

    # 3. Remove duplicate rows / duplicate timestamps (keep last).
    out = out[~out.index.duplicated(keep="last")]
    out = out.drop_duplicates()

    # 4. Handle missing close/volume.
    if "close" in out.columns:
        out["close"] = out["close"].ffill()
    if "adjusted_close" in out.columns:
        out["adjusted_close"] = out["adjusted_close"].ffill()
    if "volume" in out.columns:
        out["volume"] = out["volume"].fillna(0.0)

    # Fill OHLC gaps from close so indicators do not break on NaN.
    for col in ("open", "high", "low"):
        if col in out.columns:
            out[col] = out[col].fillna(out["close"])

    # 5. Drop any rows that still lack a usable close.
    if "close" in out.columns:
        out = out[out["close"].notna()]

    return out


def is_sufficient(df: pd.DataFrame, min_bars: int = MIN_BARS_FOR_DECISION) -> bool:
    if df is None or df.empty:
        return False
    if not all(c in df.columns for c in ("close",)):
        return False
    return len(df) >= min_bars


def validate_for_decision(symbol: str, df: pd.DataFrame, min_bars: int = MIN_BARS_FOR_DECISION) -> pd.DataFrame:
    """Clean and assert there is enough data; raise otherwise."""
    cleaned = clean_ohlcv(df)
    if not is_sufficient(cleaned, min_bars):
        raise InsufficientDataError(
            f"{symbol}: only {len(cleaned)} clean bars, need >= {min_bars}."
        )
    return cleaned
