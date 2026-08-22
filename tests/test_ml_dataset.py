"""ICS-DOC-004 Phase 1 — dataset, labels, and the no-look-ahead guarantee.

The central test here is `test_training_frame_excludes_rows_whose_label_needs_future_data`:
filtering on the signal date alone would leak `horizon + 1` sessions of future
prices into training. That is the exact failure the roadmap's no-look-ahead rule
exists to prevent, so it is asserted directly.
"""
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from app.ml.dataset import (
    FEATURE_COLUMNS,
    HORIZON_DAYS,
    LABEL_COLUMN,
    build_training_frame,
    make_features,
    make_labels,
)


def _frame(n=400, seed=1, start="2022-01-03"):
    """Deterministic OHLCV + indicator frame shaped like feature_engine output."""
    rng = np.random.default_rng(seed)
    close = 100 * np.cumprod(1 + rng.normal(0.0006, 0.012, n))
    idx = pd.bdate_range(start, periods=n)
    df = pd.DataFrame(
        {
            "open": np.concatenate([[close[0]], close[:-1]]),
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": rng.integers(1e6, 3e6, n).astype(float),
        },
        index=idx,
    )
    df["ma20"] = df["close"].rolling(20).mean()
    df["ma50"] = df["close"].rolling(50).mean()
    df["ma200"] = df["close"].rolling(200).mean()
    df["rsi14"] = 50 + rng.normal(0, 10, n)
    df["macd"] = rng.normal(0, 1, n)
    df["macd_signal"] = rng.normal(0, 1, n)
    df["atr14"] = df["close"] * 0.02
    df["volume_ma20"] = df["volume"].rolling(20).mean()
    df["high_20d"] = df["close"].rolling(20).max()
    df["drawdown_20d"] = df["close"] / df["high_20d"] - 1
    df["volatility_20d"] = 0.2
    df["beta_vs_spy"] = 1.0
    return df


# --------------------------------------------------------------------------- #
# features
# --------------------------------------------------------------------------- #
def test_features_are_ratios_not_raw_levels():
    """Cross-symbol training needs scale-free inputs, so no raw price columns."""
    f = make_features(_frame())
    assert list(f.columns) == FEATURE_COLUMNS
    assert "close" not in f.columns and "ma50" not in f.columns
    tail = f.dropna()
    # Ratios of a price to its own moving average sit near 1, not near 100.
    assert 0.5 < tail["close_over_ma50"].median() < 1.5


def test_features_use_only_existing_indicators():
    """No new indicator may sneak in: every column must be derivable from these."""
    allowed = {"close", "open", "high", "low", "volume", "ma20", "ma50", "ma200",
               "rsi14", "macd", "macd_signal", "atr14", "volume_ma20", "high_20d",
               "drawdown_20d", "volatility_20d", "beta_vs_spy"}
    df = _frame()
    assert set(df.columns) <= allowed | {"adjusted_close"}
    make_features(df)  # must not require anything outside `allowed`


def test_zero_denominator_becomes_nan_not_inf():
    df = _frame()
    df.loc[df.index[250], "ma50"] = 0.0
    f = make_features(df)
    assert pd.isna(f.loc[df.index[250], "close_over_ma50"])
    assert not np.isinf(f["close_over_ma50"].dropna()).any()


# --------------------------------------------------------------------------- #
# labels
# --------------------------------------------------------------------------- #
def test_label_is_next_open_to_next_open_matching_execution_convention():
    df = _frame()
    y = make_labels(df, horizon=HORIZON_DAYS)
    i = 100
    expected = df["open"].iloc[i + 1 + HORIZON_DAYS] / df["open"].iloc[i + 1] - 1
    assert y.iloc[i] == pytest.approx(expected)


def test_last_rows_have_no_label():
    df = _frame(n=300)
    y = make_labels(df, horizon=HORIZON_DAYS)
    assert y.iloc[-(HORIZON_DAYS + 1):].isna().all()
    assert not pd.isna(y.iloc[-(HORIZON_DAYS + 2)])


# --------------------------------------------------------------------------- #
# the no-look-ahead barrier
# --------------------------------------------------------------------------- #
def test_training_frame_excludes_rows_whose_label_needs_future_data():
    """A row is only legal once its EXIT bar is at or before the cut-off."""
    data = {"AAA": _frame(seed=2), "BBB": _frame(seed=3)}
    cutoff = pd.Timestamp("2023-01-02")
    tf = build_training_frame(data, train_end=cutoff, horizon=HORIZON_DAYS)

    assert len(tf) > 0
    # Every signal date must be far enough before the cut-off for its trade to
    # have closed by then — never merely "before the cut-off".
    latest_signal = tf.dates.max()
    frame = data["AAA"]
    pos = frame.index.get_loc(latest_signal)
    exit_bar = frame.index[pos + 1 + HORIZON_DAYS]
    assert exit_bar <= cutoff, "a label used prices from after the training cut-off"


def test_naive_signal_date_filter_would_have_leaked():
    """Guards the guard: prove the barrier is stricter than a signal-date filter."""
    data = {"AAA": _frame(seed=2)}
    cutoff = pd.Timestamp("2023-01-02")
    strict = build_training_frame(data, train_end=cutoff, horizon=HORIZON_DAYS)
    naive = build_training_frame(data, horizon=HORIZON_DAYS)
    naive_rows = naive.dates[naive.dates <= cutoff]
    assert len(naive_rows) > len(strict.dates), (
        "the exit-bar barrier must drop rows a naive signal-date filter would keep"
    )


def test_train_start_and_end_bracket_the_data():
    data = {"AAA": _frame(seed=5)}
    tf = build_training_frame(
        data, train_start=datetime(2022, 6, 1), train_end=datetime(2023, 1, 1),
        horizon=HORIZON_DAYS,
    )
    assert tf.dates.min() >= pd.Timestamp("2022-06-01")
    lo, hi = tf.span
    assert lo is not None and hi is not None and lo <= hi


def test_empty_input_returns_empty_frame():
    tf = build_training_frame({}, horizon=HORIZON_DAYS)
    assert len(tf) == 0
    assert list(tf.X.columns) == FEATURE_COLUMNS


def test_frame_has_no_nan_and_stacks_symbols():
    data = {"AAA": _frame(seed=6), "BBB": _frame(seed=7)}
    tf = build_training_frame(data, horizon=HORIZON_DAYS)
    assert not tf.X.isna().any().any()
    assert not tf.y.isna().any()
    assert set(tf.tickers.unique()) == {"AAA", "BBB"}
    assert LABEL_COLUMN not in tf.X.columns
