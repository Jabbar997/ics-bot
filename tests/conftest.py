"""Shared pytest fixtures and synthetic-data helpers (no network required)."""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

# Ensure the package is importable when running pytest from anywhere.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.config import load_config  # noqa: E402
from app.db import database  # noqa: E402


@pytest.fixture
def config():
    return load_config()


@pytest.fixture
def db_url(tmp_path):
    """A fresh file-backed SQLite DB per test (shared across sessions)."""
    url = f"sqlite:///{tmp_path / 'test.db'}"
    database.init_engine(url)
    database.create_all()
    yield url
    database.drop_all()


def make_ohlcv(n=320, start=100.0, drift=0.0009, vol=0.012, seed=1, start_date="2023-01-02"):
    """Generate a deterministic synthetic OHLCV frame."""
    rng = np.random.default_rng(seed)
    close = start * np.cumprod(1 + rng.normal(drift, vol, n))
    idx = pd.bdate_range(start_date, periods=n)
    high = close * (1 + np.abs(rng.normal(0, 0.004, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.004, n)))
    openp = np.concatenate([[close[0]], close[:-1]])
    volume = rng.integers(1_000_000, 3_000_000, n).astype(float)
    return pd.DataFrame(
        {"open": openp, "high": high, "low": low, "close": close, "adjusted_close": close, "volume": volume},
        index=idx,
    )


@pytest.fixture
def synthetic_market():
    """A small multi-symbol market with a rising SPY (bull regime)."""
    data = {"SPY": make_ohlcv(seed=7, drift=0.0010)}
    for i, sym in enumerate(["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META"]):
        data[sym] = make_ohlcv(seed=20 + i, drift=0.0009 + 0.0001 * i)
    return data
