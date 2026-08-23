"""Market-data reliability: retries, fallback, and no silent drops.

The bug these lock down: yfinance returns an EMPTY frame rather than raising
when a bulk fetch is throttled. The old layer caught exceptions only, so the
symbol was stored as an empty frame and quietly vanished from that day's
decisions — observed twice in practice (JPM, then NVDA), and it changed backtest
results without any warning.
"""
import pandas as pd
import pytest

from app.data.market_data import (
    FetchReport,
    ProviderError,
    fetch_history,
    fetch_watchlist,
    fetch_watchlist_history,
)
from app.data.providers import StooqProvider, _standardise

NO_SLEEP = lambda _s: None  # noqa: E731 — keep the tests instant


def _frame(n=5):
    idx = pd.bdate_range("2026-01-05", periods=n)
    return pd.DataFrame(
        {"Open": 1.0, "High": 1.2, "Low": 0.9, "Close": 1.1,
         "Adj Close": 1.1, "Volume": 1000.0},
        index=idx,
    )


class FakeProvider:
    """Fails a configurable number of times, then succeeds."""

    def __init__(self, name, fail_times=0, always_fail=False, empty=False):
        self.name = name
        self.fail_times = fail_times
        self.always_fail = always_fail
        self.empty = empty
        self.calls = 0

    def fetch(self, symbol, period, interval):
        self.calls += 1
        if self.always_fail:
            raise ProviderError(f"{self.name}: down")
        if self.empty:
            return _standardise(pd.DataFrame(), symbol, self.name)  # raises
        if self.calls <= self.fail_times:
            raise ProviderError(f"{self.name}: transient")
        return _standardise(_frame(), symbol, self.name)


# --------------------------------------------------------------------------- #
# emptiness is a failure, not a result
# --------------------------------------------------------------------------- #
def test_empty_frame_raises_instead_of_being_returned():
    """The exact shape of the original bug."""
    with pytest.raises(ProviderError):
        _standardise(pd.DataFrame(), "AAPL", "fake")


def test_frame_missing_price_columns_is_rejected():
    bad = pd.DataFrame({"Volume": [1, 2, 3]}, index=pd.bdate_range("2026-01-05", periods=3))
    with pytest.raises(ProviderError, match="missing columns"):
        _standardise(bad, "AAPL", "fake")


# --------------------------------------------------------------------------- #
# retries
# --------------------------------------------------------------------------- #
def test_transient_failure_is_retried_and_recovers():
    p = FakeProvider("primary", fail_times=2)
    df = fetch_history("AAPL", providers=[p], retries=2, sleep=NO_SLEEP)
    assert not df.empty
    assert p.calls == 3  # two failures then success


def test_retries_are_bounded():
    p = FakeProvider("primary", always_fail=True)
    with pytest.raises(ProviderError):
        fetch_history("AAPL", providers=[p], retries=2, sleep=NO_SLEEP)
    assert p.calls == 3  # first attempt + 2 retries, then give up


def test_backoff_grows_between_attempts():
    waits = []
    p = FakeProvider("primary", always_fail=True)
    with pytest.raises(ProviderError):
        fetch_history("AAPL", providers=[p], retries=2, backoff=1.0, sleep=waits.append)
    assert waits == [1.0, 2.0]  # exponential, and no sleep after the last try


# --------------------------------------------------------------------------- #
# fallback provider
# --------------------------------------------------------------------------- #
def test_falls_back_to_the_secondary_provider():
    primary = FakeProvider("primary", always_fail=True)
    secondary = FakeProvider("secondary")
    df = fetch_history("AAPL", providers=[primary, secondary], retries=1, sleep=NO_SLEEP)
    assert not df.empty
    assert primary.calls == 2 and secondary.calls == 1


def test_error_names_the_symbol_when_everything_fails():
    a = FakeProvider("a", always_fail=True)
    b = FakeProvider("b", always_fail=True)
    with pytest.raises(ProviderError, match="NVDA"):
        fetch_history("NVDA", providers=[a, b], retries=0, sleep=NO_SLEEP)


# --------------------------------------------------------------------------- #
# watchlist reporting
# --------------------------------------------------------------------------- #
def test_failed_symbols_are_reported_not_silently_empty():
    class Selective:
        name = "selective"

        def fetch(self, symbol, period, interval):
            if symbol == "NVDA":
                raise ProviderError("throttled")
            return _standardise(_frame(), symbol, self.name)

    report = fetch_watchlist(
        ["AAPL", "NVDA", "MSFT"], providers=[Selective()], retries=0, sleep=NO_SLEEP
    )
    assert isinstance(report, FetchReport)
    assert report.failed == ["NVDA"]
    assert report.ok is False
    assert "NVDA" not in report.frames          # not present as an empty frame
    assert set(report.frames) == {"AAPL", "MSFT"}
    assert "NVDA" in report.summary()


def test_a_required_symbol_failing_raises():
    """Losing the benchmark invalidates the whole cycle — it must not pass quietly."""
    p = FakeProvider("primary", always_fail=True)
    with pytest.raises(ProviderError):
        fetch_watchlist(
            ["SPY"], required=("SPY",), providers=[p], retries=0, sleep=NO_SLEEP
        )


def test_all_good_reports_ok():
    report = fetch_watchlist(
        ["AAPL", "MSFT"], providers=[FakeProvider("p")], retries=0, sleep=NO_SLEEP
    )
    assert report.ok is True and report.failed == []


def test_legacy_wrapper_still_returns_plain_frames():
    frames = fetch_watchlist_history(
        ["AAPL"], providers=[FakeProvider("p")], retries=0, sleep=NO_SLEEP
    )
    assert set(frames) == {"AAPL"}
    assert isinstance(frames["AAPL"], pd.DataFrame)


# --------------------------------------------------------------------------- #
# stooq symbol mapping (pure, no network)
# --------------------------------------------------------------------------- #
def test_stooq_maps_us_tickers():
    assert StooqProvider.map_symbol("SPY") == "spy.us"
    assert StooqProvider.map_symbol("aapl") == "aapl.us"
    assert StooqProvider.map_symbol("BRK.B") == "brk.b"  # already qualified


def test_stooq_rejects_non_daily_intervals():
    with pytest.raises(ProviderError, match="interval"):
        StooqProvider().fetch("SPY", "5y", "1h")
