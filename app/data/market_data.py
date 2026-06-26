"""Market data layer (yfinance).

yfinance is imported lazily so the deterministic core and the test suite run
without network access or the yfinance dependency installed.

SAFETY: this module is read-only market data. It never places orders. There is
no broker, account, or order-routing code anywhere in this layer.
"""
from __future__ import annotations

from typing import Dict, List

import pandas as pd

from app.data.cleaner import clean_ohlcv
from app.logging_config import get_logger

log = get_logger(__name__)

# Standard output column order for every frame this module returns.
STANDARD_COLUMNS = ["open", "high", "low", "close", "adjusted_close", "volume"]


def _import_yfinance():
    try:
        import yfinance as yf  # noqa: WPS433 (lazy import is intentional)
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "yfinance is not installed. Install requirements.txt to fetch data."
        ) from exc
    return yf


def _standardise(df: pd.DataFrame) -> pd.DataFrame:
    """Map a yfinance frame to ICS standard columns."""
    if df is None or df.empty:
        return pd.DataFrame(columns=STANDARD_COLUMNS)

    # yfinance can return a MultiIndex for multiple tickers; flatten if needed.
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = [c[0] for c in df.columns]

    rename = {
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Adj Close": "adjusted_close",
        "Volume": "volume",
    }
    df = df.rename(columns=rename)
    df.index = pd.to_datetime(df.index)
    df.index.name = "date"

    if "adjusted_close" not in df.columns and "close" in df.columns:
        df["adjusted_close"] = df["close"]

    keep = [c for c in STANDARD_COLUMNS if c in df.columns]
    return clean_ohlcv(df[keep])


def fetch_history(symbol: str, period: str = "5y", interval: str = "1d") -> pd.DataFrame:
    """Fetch and clean historical OHLCV for one symbol."""
    yf = _import_yfinance()
    log.info("Fetching history: %s period=%s interval=%s", symbol, period, interval)
    raw = yf.download(
        symbol,
        period=period,
        interval=interval,
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    return _standardise(raw)


def fetch_latest(symbol: str) -> dict:
    """Fetch the most recent clean bar as a dict."""
    df = fetch_history(symbol, period="1mo", interval="1d")
    if df.empty:
        raise RuntimeError(f"No recent data for {symbol}")
    last = df.iloc[-1]
    return {
        "symbol": symbol,
        "timestamp": df.index[-1].to_pydatetime(),
        "open": float(last["open"]),
        "high": float(last["high"]),
        "low": float(last["low"]),
        "close": float(last["close"]),
        "adjusted_close": float(last.get("adjusted_close", last["close"])),
        "volume": float(last.get("volume", 0.0)),
    }


def fetch_watchlist_history(
    symbols: List[str], period: str = "5y", interval: str = "1d"
) -> Dict[str, pd.DataFrame]:
    """Fetch history for every symbol in the watchlist."""
    result: Dict[str, pd.DataFrame] = {}
    for sym in symbols:
        try:
            result[sym] = fetch_history(sym, period=period, interval=interval)
        except Exception as exc:  # pragma: no cover - network dependent
            log.error("Failed to fetch %s: %s", sym, exc)
            result[sym] = pd.DataFrame(columns=STANDARD_COLUMNS)
    return result
