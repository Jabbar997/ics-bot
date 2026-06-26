"""Benchmark (SPY) return helpers and the two ICS return measures.

Per the benchmark policy the system tracks two returns and compares both to SPY:
* Total Portfolio Return = total_value / initial_capital - 1
* Invested Capital Return = realised+unrealised P/L / invested capital
"""
from __future__ import annotations

from typing import Optional

import pandas as pd


def period_return(close: pd.Series) -> float:
    """Simple return over a close-price series (first to last)."""
    s = close.dropna()
    if len(s) < 2 or s.iloc[0] == 0:
        return 0.0
    return float(s.iloc[-1] / s.iloc[0] - 1.0)


def spy_return_between(spy_df: pd.DataFrame, start=None, end=None) -> float:
    """SPY return between two dates (inclusive), using the 'close' column."""
    if spy_df is None or spy_df.empty or "close" not in spy_df.columns:
        return 0.0
    df = spy_df
    if start is not None:
        df = df[df.index >= pd.Timestamp(start)]
    if end is not None:
        df = df[df.index <= pd.Timestamp(end)]
    return period_return(df["close"])


def total_portfolio_return(total_value: float, initial_capital: float) -> float:
    if not initial_capital:
        return 0.0
    return total_value / initial_capital - 1.0


def invested_capital_return(realized_unrealized_pnl: float, invested_capital: float) -> float:
    if not invested_capital:
        return 0.0
    return realized_unrealized_pnl / invested_capital
