"""Walk-forward split helpers.

Splits the historical window into Training (first 70%), Validation (next 20%)
and Walk-forward / out-of-sample (last 10%). Splits are purely positional so
they never leak future data into an earlier segment.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import pandas as pd


@dataclass
class Split:
    name: str
    start_index: int
    end_index: int  # exclusive

    def slice(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.iloc[self.start_index : self.end_index]


def split_indices(n: int, train: float = 0.70, val: float = 0.20) -> Dict[str, Split]:
    """Return positional Train/Validation/WalkForward splits for ``n`` rows.

    Computed additively (not via ``train + val``) to avoid float rounding, e.g.
    ``0.7 + 0.2`` evaluating to ``0.8999...`` and truncating a boundary.
    """
    train_end = int(n * train)
    val_end = train_end + int(n * val)
    return {
        "train": Split("train", 0, train_end),
        "validation": Split("validation", train_end, val_end),
        "walk_forward": Split("walk_forward", val_end, n),
    }


def split_dates(index: pd.DatetimeIndex, train: float = 0.70, val: float = 0.20) -> Dict[str, Tuple]:
    """Return (start_date, end_date) tuples for each split given a date index."""
    n = len(index)
    splits = split_indices(n, train, val)
    out = {}
    for name, sp in splits.items():
        if sp.end_index <= sp.start_index:
            out[name] = (None, None)
            continue
        out[name] = (index[sp.start_index], index[min(sp.end_index, n) - 1])
    return out
