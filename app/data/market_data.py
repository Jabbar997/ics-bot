"""Market data layer.

Fetching is layered: a primary provider, a bounded number of retries (the
failures observed in practice were transient rate-limits, not delistings), then
a fallback provider. Anything that still fails is reported **by name** — never
silently turned into an empty frame, which is what used to make a symbol
disappear from a day's decisions without a trace.

SAFETY: this module is read-only market data. It never places orders. There is
no broker, account, or order-routing code anywhere in this layer.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import pandas as pd

from app.data.providers import (
    STANDARD_COLUMNS,
    Provider,
    ProviderError,
    default_providers,
)
from app.logging_config import get_logger

log = get_logger(__name__)

DEFAULT_RETRIES = 2      # attempts per provider, beyond the first try
DEFAULT_BACKOFF = 1.5    # seconds, doubled each retry


@dataclass
class FetchReport:
    """What actually came back — including what did not."""

    frames: Dict[str, pd.DataFrame] = field(default_factory=dict)
    failed: List[str] = field(default_factory=list)
    recovered: Dict[str, str] = field(default_factory=dict)  # symbol -> provider

    @property
    def ok(self) -> bool:
        return not self.failed

    def summary(self) -> str:
        parts = [f"{len(self.frames)} رمزًا"]
        if self.recovered:
            parts.append("استُعيد: " + ", ".join(f"{k}({v})" for k, v in self.recovered.items()))
        if self.failed:
            parts.append("فشل: " + ", ".join(self.failed))
        return " | ".join(parts)


def fetch_history(
    symbol: str,
    period: str = "5y",
    interval: str = "1d",
    *,
    providers: Optional[Sequence[Provider]] = None,
    retries: int = DEFAULT_RETRIES,
    backoff: float = DEFAULT_BACKOFF,
    sleep: Callable[[float], None] = time.sleep,
) -> pd.DataFrame:
    """Fetch and clean historical OHLCV for one symbol.

    Tries each provider in turn, retrying transient failures with exponential
    backoff. Raises :class:`ProviderError` if every provider is exhausted —
    callers must not receive a silent empty frame.
    """
    provs = list(providers) if providers is not None else default_providers()
    last_error: Optional[Exception] = None

    for provider in provs:
        for attempt in range(retries + 1):
            try:
                frame = provider.fetch(symbol, period, interval)
                if attempt or provider is not provs[0]:
                    log.info(
                        "Recovered %s via %s (attempt %d).", symbol, provider.name, attempt + 1
                    )
                return frame
            except ProviderError as exc:
                last_error = exc
                if attempt < retries:
                    wait = backoff * (2 ** attempt)
                    log.warning(
                        "%s failed for %s (%s); retrying in %.1fs.",
                        provider.name, symbol, exc, wait,
                    )
                    sleep(wait)
                else:
                    log.warning("%s exhausted for %s: %s", provider.name, symbol, exc)

    raise ProviderError(f"All providers failed for {symbol}: {last_error}")


def fetch_latest(symbol: str, **kwargs) -> dict:
    """Fetch the most recent clean bar as a dict."""
    df = fetch_history(symbol, period="1mo", interval="1d", **kwargs)
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


def fetch_watchlist(
    symbols: Sequence[str],
    period: str = "5y",
    interval: str = "1d",
    *,
    required: Sequence[str] = (),
    **kwargs,
) -> FetchReport:
    """Fetch every symbol, reporting failures instead of hiding them.

    ``required`` names symbols the cycle cannot proceed without (the benchmark):
    if one of those fails, the error propagates rather than producing a report
    that looks merely incomplete.
    """
    report = FetchReport()
    primary_name = kwargs.get("providers")
    primary_name = (list(primary_name)[0].name if primary_name else default_providers()[0].name)

    for sym in symbols:
        try:
            frame = fetch_history(sym, period=period, interval=interval, **kwargs)
            report.frames[sym] = frame
        except ProviderError as exc:
            if sym in required:
                raise
            log.error("Dropping %s from this cycle: %s", sym, exc)
            report.failed.append(sym)

    if report.failed:
        log.error(
            "Market data incomplete — %d/%d symbols unavailable: %s",
            len(report.failed), len(symbols), ", ".join(report.failed),
        )
    return report


def fetch_watchlist_history(
    symbols: List[str], period: str = "5y", interval: str = "1d", **kwargs
) -> Dict[str, pd.DataFrame]:
    """Backwards-compatible wrapper returning only the frames.

    Prefer :func:`fetch_watchlist`, which also tells you what was missing.
    """
    return fetch_watchlist(symbols, period=period, interval=interval, **kwargs).frames


__all__ = [
    "STANDARD_COLUMNS",
    "FetchReport",
    "ProviderError",
    "fetch_history",
    "fetch_latest",
    "fetch_watchlist",
    "fetch_watchlist_history",
]
