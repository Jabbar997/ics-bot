"""ICS-DOC-004 Phase 1 — model, bounded tuning, and walk-forward retraining."""
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from app.ml.dataset import FEATURE_COLUMNS, HORIZON_DAYS
from app.ml.model import (
    DEFAULT_PARAMS,
    TrainedModel,
    make_model_version,
    spearman_ic,
    train_model,
)
from app.ml.tuning import MAX_TRIALS, SEARCH_SPACE, chronological_split, tune
from app.ml.walk_forward import MIN_TRAIN_DAYS, run_walk_forward, split_windows


def _xy(n=600, seed=0, signal=True):
    """Synthetic set where feature 0 genuinely predicts the label (if signal)."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(0, 1, (n, len(FEATURE_COLUMNS))), columns=FEATURE_COLUMNS)
    noise = rng.normal(0, 0.01, n)
    y = pd.Series(0.02 * X["rsi14"] + noise if signal else noise)
    return X, y


# --------------------------------------------------------------------------- #
# metric
# --------------------------------------------------------------------------- #
def test_spearman_ic_is_rank_based():
    assert spearman_ic([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert spearman_ic([1, 2, 3, 4], [40, 30, 20, 10]) == pytest.approx(-1.0)


def test_spearman_ic_never_returns_nan():
    assert spearman_ic([1, 1, 1, 1], [1, 2, 3, 4]) == 0.0
    assert spearman_ic([1, 2], [1, 2]) == 0.0


# --------------------------------------------------------------------------- #
# training
# --------------------------------------------------------------------------- #
def test_model_learns_a_real_relationship():
    X, y = _xy(signal=True)
    m = train_model(X, y)
    assert spearman_ic(y, m.predict(X)) > 0.5


def test_model_version_is_stable_and_traceable():
    a = make_model_version(datetime(2022, 1, 1), datetime(2024, 1, 1), DEFAULT_PARAMS)
    b = make_model_version(datetime(2022, 1, 1), datetime(2024, 1, 1), DEFAULT_PARAMS)
    c = make_model_version(datetime(2022, 1, 1), datetime(2024, 6, 1), DEFAULT_PARAMS)
    assert a == b            # deterministic
    assert a != c            # window is part of the identity
    assert a.startswith(f"lgbm-h{HORIZON_DAYS}-20220101-20240101-")


def test_trained_model_records_its_training_window():
    X, y = _xy(n=300)
    m = train_model(X, y, train_start=datetime(2022, 1, 1), train_end=datetime(2023, 1, 1))
    assert m.train_start and m.train_end and m.n_samples == 300
    assert m.horizon_days == HORIZON_DAYS
    assert m.feature_columns == FEATURE_COLUMNS


def test_model_round_trips_through_joblib(tmp_path):
    X, y = _xy(n=300)
    m = train_model(X, y)
    path = m.save(tmp_path)
    loaded = TrainedModel.load(path)
    assert loaded.model_version == m.model_version
    assert np.allclose(loaded.predict(X), m.predict(X))


# --------------------------------------------------------------------------- #
# tuning
# --------------------------------------------------------------------------- #
def test_validation_split_is_chronological_not_random():
    X, y = _xy(n=100)
    X_tr, y_tr, X_val, y_val = chronological_split(X, y, 0.2)
    assert len(X_tr) == 80 and len(X_val) == 20
    # Validation must be strictly the LATER rows.
    assert X_val.index.min() > X_tr.index.max()


def test_tuning_logs_every_trial_and_respects_the_cap():
    X, y = _xy(n=400, signal=True)
    res = tune(X, y, n_trials=6)
    assert res.n_trials == 6
    assert [t.number for t in res.trials] == list(range(6))
    for t in res.trials:
        assert set(t.params) == set(SEARCH_SPACE)  # only the declared space
        assert isinstance(t.score, float)
    assert len(res.scores) == 6
    assert res.to_dict()["n_trials"] == 6


def test_trial_ceiling_is_enforced_even_if_more_are_requested():
    X, y = _xy(n=200)
    res = tune(X, y, n_trials=MAX_TRIALS + 500)
    assert res.n_trials <= MAX_TRIALS


def test_tuned_params_stay_inside_the_declared_space():
    X, y = _xy(n=400)
    res = tune(X, y, n_trials=5)
    for name, (lo, hi) in SEARCH_SPACE.items():
        assert lo <= res.best_params[name] <= hi


# --------------------------------------------------------------------------- #
# walk-forward
# --------------------------------------------------------------------------- #
def _panel(n=800, seed=1):
    rng = np.random.default_rng(seed)
    close = 100 * np.cumprod(1 + rng.normal(0.0005, 0.012, n))
    idx = pd.bdate_range("2021-01-04", periods=n)
    df = pd.DataFrame({
        "open": np.concatenate([[close[0]], close[:-1]]),
        "high": close * 1.01, "low": close * 0.99, "close": close,
        "volume": rng.integers(1e6, 3e6, n).astype(float),
    }, index=idx)
    df["ma20"] = df["close"].rolling(20).mean()
    df["ma50"] = df["close"].rolling(50).mean()
    df["ma200"] = df["close"].rolling(200).mean()
    df["rsi14"] = 50 + rng.normal(0, 10, n)
    df["macd"] = rng.normal(0, 1, n); df["macd_signal"] = rng.normal(0, 1, n)
    df["atr14"] = df["close"] * 0.02
    df["volume_ma20"] = df["volume"].rolling(20).mean()
    df["high_20d"] = df["close"].rolling(20).max()
    df["drawdown_20d"] = df["close"] / df["high_20d"] - 1
    df["volatility_20d"] = 0.2; df["beta_vs_spy"] = 1.0
    return df


def test_walk_forward_retrains_monthly_and_never_predicts_from_the_future():
    data = {"AAA": _panel(seed=1), "BBB": _panel(seed=2)}
    res = run_walk_forward(data, tune_first=False, min_train_days=MIN_TRAIN_DAYS)
    assert res.n_retrains >= 2, "expected at least a couple of monthly retrains"
    assert not res.predictions.empty

    # THE invariant: the model used on day t was trained only on trades that had
    # already closed before t.
    for _, row in res.predictions.iterrows():
        assert row["train_end"] < row["date"], (
            f"{row['ticker']} on {row['date']} used a model trained to {row['train_end']}"
        )


def test_walk_forward_uses_one_model_per_month():
    data = {"AAA": _panel(seed=4)}
    res = run_walk_forward(data, tune_first=False)
    df = res.predictions.copy()
    df["month"] = pd.to_datetime(df["date"]).dt.to_period("M")
    assert (df.groupby("month")["model_version"].nunique() == 1).all()


def test_walk_forward_needs_enough_history():
    data = {"AAA": _panel(n=300, seed=5)}
    res = run_walk_forward(data, tune_first=False, min_train_days=MIN_TRAIN_DAYS)
    assert res.n_retrains == 0 and res.predictions.empty


def test_split_windows_are_contiguous_and_disjoint():
    dates = pd.DatetimeIndex(pd.bdate_range("2021-01-04", periods=1250))
    windows = split_windows(dates, 3)
    assert len(windows) == 3
    for i in range(len(windows) - 1):
        assert windows[i][1] < windows[i + 1][0], "windows must not overlap"
    assert windows[0][0] == dates[0].to_pydatetime()
    assert windows[-1][1] == dates[-1].to_pydatetime()
