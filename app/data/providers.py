"""Market-data providers behind one interface.

Why this exists: `yfinance` returns an **empty frame** instead of raising when a
bulk fetch is rate-limited. The old code caught exceptions, so nothing ever
fired, and the symbol was silently stored as empty — it simply vanished from
that day's decisions. That was measured twice (JPM, then NVDA) and it silently
changed backtest results.

So: a provider either returns a usable frame or raises :class:`ProviderError`.
Emptiness is a failure, not a result. Retries and fallback live one layer up in
``market_data.py``.

SAFETY: read-only price data. No provider here can place an order.
"""
from __future__ import annotations

from typing import Optional, Protocol

import pandas as pd

from app.data.cleaner import clean_ohlcv
from app.logging_config import get_logger

log = get_logger(__name__)

STANDARD_COLUMNS = ["open", "high", "low", "close", "adjusted_close", "volume"]

# Roughly how many calendar days a period string covers, for providers that
# return full history and need slicing.
_PERIOD_DAYS = {
    "1mo": 31, "3mo": 92, "6mo": 183,
    "1y": 366, "2y": 731, "3y": 1096, "5y": 1827, "10y": 3653,
}


class ProviderError(RuntimeError):
    """A provider could not return usable data for a symbol."""


class Provider(Protocol):
    name: str

    def fetch(self, symbol: str, period: str, interval: str) -> pd.DataFrame: ...


def _standardise(df: pd.DataFrame, symbol: str, provider: str) -> pd.DataFrame:
    """Map a raw frame to ICS standard columns, or raise if unusable."""
    if df is None or len(df) == 0:
        raise ProviderError(f"{provider}: no data for {symbol}")

    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = [c[0] for c in df.columns]

    df = df.rename(columns={
        "Open": "open", "High": "high", "Low": "low", "Close": "close",
        "Adj Close": "adjusted_close", "Volume": "volume",
    })
    df.index = pd.to_datetime(df.index)
    df.index.name = "date"

    if "adjusted_close" not in df.columns and "close" in df.columns:
        df["adjusted_close"] = df["close"]

    missing = [c for c in ("open", "high", "low", "close") if c not in df.columns]
    if missing:
        raise ProviderError(f"{provider}: {symbol} missing columns {missing}")

    keep = [c for c in STANDARD_COLUMNS if c in df.columns]
    cleaned = clean_ohlcv(df[keep])
    if cleaned.empty:
        raise ProviderError(f"{provider}: {symbol} produced no rows after cleaning")
    return cleaned


class YFinanceProvider:
    """Primary source. Imported lazily so the core runs without it installed."""

    name = "yfinance"

    def fetch(self, symbol: str, period: str = "5y", interval: str = "1d") -> pd.DataFrame:
        try:
            import yfinance as yf
        except ImportError as exc:  # pragma: no cover
            raise ProviderError("yfinance is not installed") from exc

        try:
            raw = yf.download(
                symbol, period=period, interval=interval,
                auto_adjust=False, progress=False, threads=False,
            )
        except Exception as exc:  # network/parse failures
            raise ProviderError(f"yfinance: {symbol} download failed: {exc}") from exc
        return _standardise(raw, symbol, self.name)


class StooqProvider:
    """Free fallback (no API key). Returns full daily history as CSV.

    Only daily data is supported; anything else is rejected rather than silently
    returning the wrong interval.
    """

    name = "stooq"
    URL = "https://stooq.com/q/d/l/?s={sym}&i=d"

    @staticmethod
    def map_symbol(symbol: str) -> str:
        """US tickers on Stooq are lowercase with a `.us` suffix."""
        s = symbol.strip().lower()
        return s if "." in s else f"{s}.us"

    def fetch(self, symbol: str, period: str = "5y", interval: str = "1d") -> pd.DataFrame:
        if interval != "1d":
            raise ProviderError(f"stooq: interval {interval} not supported")
        url = self.URL.format(sym=self.map_symbol(symbol))
        try:
            raw = pd.read_csv(url, parse_dates=["Date"], index_col="Date")
        except Exception as exc:
            raise ProviderError(f"stooq: {symbol} fetch failed: {exc}") from exc

        frame = _standardise(raw, symbol, self.name)
        days = _PERIOD_DAYS.get(period)
        if days:
            cutoff = frame.index.max() - pd.Timedelta(days=days)
            frame = frame[frame.index >= cutoff]
            if frame.empty:
                raise ProviderError(f"stooq: {symbol} empty after applying period {period}")
        return frame


def default_providers(include_fallback: bool = True) -> list:
    providers: list = [YFinanceProvider()]
    if include_fallback:
        providers.append(StooqProvider())
    return providers
