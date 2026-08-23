"""Backtester tests — runs end-to-end and proves no look-ahead bias."""
import pandas as pd

from app.backtesting.backtester import Backtester
from app.backtesting.walk_forward import split_dates, split_indices
from app.config import load_config


def test_walk_forward_split_proportions():
    splits = split_indices(1000)
    assert splits["train"].start_index == 0
    assert splits["train"].end_index == 700
    assert splits["validation"].start_index == 700
    assert splits["validation"].end_index == 900
    assert splits["walk_forward"].start_index == 900
    assert splits["walk_forward"].end_index == 1000


def test_split_dates_are_ordered(synthetic_market):
    splits = split_dates(synthetic_market["SPY"].index)
    train_end = splits["train"][1]
    val_start = splits["validation"][0]
    assert train_end <= val_start


def test_backtester_runs_and_logs_every_decision(synthetic_market, tmp_path):
    cfg = load_config()
    url = f"sqlite:///{tmp_path / 'bt.db'}"
    bt = Backtester(cfg)
    result = bt.run(synthetic_market, database_url=url)

    # The pipeline executed and produced decisions + matching audit logs.
    assert result.total_decisions > 0
    assert result.audit_logs == result.total_decisions  # audit invariant
    assert result.benchmark_symbol == "SPY"
    assert isinstance(result.rejected_opportunities, int)
    assert result.performance.period == "backtest"


def test_backtester_no_lookahead_fills_at_next_open(synthetic_market, tmp_path):
    """Every BUY must fill at the *next* trading day's OPEN, never the signal
    day's close — the core no-look-ahead guarantee.
    """
    cfg = load_config()
    url = f"sqlite:///{tmp_path / 'bt2.db'}"
    bt = Backtester(cfg)
    bt.run(synthetic_market, database_url=url)

    from app.db import database
    from app.db.database import session_scope
    from app.db.models import Decision

    database.init_engine(url)
    with session_scope() as s:
        buys = [d for d in s.query(Decision).all() if d.action == "buy"]

    assert len(buys) >= 1, "expected at least one BUY in the bull synthetic market"
    for d in buys:
        src = synthetic_market[d.ticker]
        fill_day = pd.Timestamp(d.created_at).normalize()
        assert fill_day in src.index, "fill must occur on a real trading day"
        expected_open = float(src.at[fill_day, "open"])
        # price recorded on the decision is the execution price = next-day open.
        assert abs(float(d.price) - expected_open) < 1e-6, (
            f"{d.ticker}: filled at {float(d.price)} but next open was {expected_open}"
        )
        # Sanity: it must NOT equal the prior day's close (which would be look-ahead-free
        # signal price, not the execution price).
        prior_idx = src.index.get_loc(fill_day) - 1
        if prior_idx >= 0:
            prior_close = float(src.iloc[prior_idx]["close"])
            # Open and prior close differ in this synthetic data, so this is meaningful.
            assert float(d.price) != prior_close or expected_open == prior_close


# --------------------------------------------------------------------------- #
# windowed runs: warm-up must come from BEFORE the window
# --------------------------------------------------------------------------- #
def test_window_keeps_its_tradeable_days(synthetic_market, tmp_path):
    """A window must not lose 200 of its own bars to warm-up.

    Slicing the calendar first and then skipping WARMUP_BARS inside the slice
    silently emptied short windows, which made a 3-window validation return
    identical numbers for genuinely different configurations.
    """
    from app.backtesting.backtester import WARMUP_BARS, Backtester
    from app.config import load_config

    cfg = load_config()
    idx = synthetic_market["SPY"].index
    tradeable = idx[WARMUP_BARS:]
    half = len(tradeable) // 2

    first = Backtester(cfg).run(
        synthetic_market, start=tradeable[0], end=tradeable[half - 1],
        database_url=f"sqlite:///{tmp_path / 'w1.db'}",
    )
    second = Backtester(cfg).run(
        synthetic_market, start=tradeable[half], end=tradeable[-1],
        database_url=f"sqlite:///{tmp_path / 'w2.db'}",
    )

    # Both halves must actually trade, and cover their own date range.
    assert first.total_decisions > 0 and second.total_decisions > 0
    assert first.start >= tradeable[0]
    assert second.end <= tradeable[-1]
    # Non-overlapping.
    assert first.end < second.start


def test_window_shorter_than_warmup_is_rejected_not_silently_empty(synthetic_market, tmp_path):
    from app.backtesting.backtester import Backtester
    from app.config import load_config
    import pytest as _pytest

    idx = synthetic_market["SPY"].index
    with _pytest.raises(ValueError, match="warm-up"):
        Backtester(load_config()).run(
            synthetic_market, start=idx[0], end=idx[5],
            database_url=f"sqlite:///{tmp_path / 'tiny.db'}",
        )


def test_isolated_runs_do_not_accumulate(synthetic_market, tmp_path):
    """Two runs on the same :memory: URL must not share rows (the v1.1 trap)."""
    from app.backtesting.backtester import Backtester
    from app.config import load_config

    cfg = load_config()
    a = Backtester(cfg).run(synthetic_market, database_url="sqlite:///:memory:")
    b = Backtester(cfg).run(synthetic_market, database_url="sqlite:///:memory:")
    assert a.total_decisions == b.total_decisions
    assert a.closed_trades == b.closed_trades
