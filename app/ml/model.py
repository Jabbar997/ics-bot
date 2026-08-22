"""LightGBM wrapper for the Phase 1 shadow model.

CPU-only and deliberately small — it has to train inside the existing Render
worker without a resource upgrade. Every trained model carries a
``model_version`` and the exact training window, so any shadow prediction can be
traced back to the model that produced it (ICS-DOC-004 Phase 1 §2).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from app.logging_config import get_logger
from app.ml.dataset import FEATURE_COLUMNS, HORIZON_DAYS

log = get_logger(__name__)

MODEL_DIR = Path("models")

# Conservative defaults for a small, noisy financial dataset.
DEFAULT_PARAMS: Dict[str, Any] = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.05,
    "num_leaves": 15,
    "max_depth": 4,
    "min_child_samples": 100,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "n_estimators": 300,
    "verbose": -1,
    "n_jobs": 2,
}


def spearman_ic(y_true, y_pred) -> float:
    """Rank correlation between prediction and outcome (the tuning objective).

    The model is used to *rank* candidates, not to estimate a return precisely,
    so rank correlation is the metric that matches the use. Returns 0.0 when it
    is undefined (constant series), never NaN.
    """
    from scipy.stats import spearmanr

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) < 3 or len(set(y_pred.tolist())) < 2 or len(set(y_true.tolist())) < 2:
        return 0.0
    rho, _ = spearmanr(y_pred, y_true)
    rho = float(rho)
    return 0.0 if rho != rho else rho


def make_model_version(train_start: Optional[datetime], train_end: Optional[datetime],
                       params: Dict[str, Any], horizon: int = HORIZON_DAYS) -> str:
    """Stable, human-readable id: date range + horizon + a params fingerprint."""
    digest = hashlib.sha256(
        json.dumps(params, sort_keys=True, default=str).encode()
    ).hexdigest()[:8]
    s = train_start.strftime("%Y%m%d") if train_start else "na"
    e = train_end.strftime("%Y%m%d") if train_end else "na"
    return f"lgbm-h{horizon}-{s}-{e}-{digest}"


@dataclass
class TrainedModel:
    booster: Any
    model_version: str
    params: Dict[str, Any] = field(default_factory=dict)
    train_start: Optional[datetime] = None
    train_end: Optional[datetime] = None
    horizon_days: int = HORIZON_DAYS
    n_samples: int = 0
    feature_columns: List[str] = field(default_factory=lambda: list(FEATURE_COLUMNS))

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.booster.predict(X[self.feature_columns]), dtype=float)

    def save(self, directory: Path = MODEL_DIR) -> Path:
        import joblib

        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{self.model_version}.joblib"
        joblib.dump(self, path)
        return path

    @staticmethod
    def load(path: Path) -> "TrainedModel":
        import joblib

        return joblib.load(path)


def train_model(
    X: pd.DataFrame,
    y: pd.Series,
    params: Optional[Dict[str, Any]] = None,
    *,
    train_start: Optional[datetime] = None,
    train_end: Optional[datetime] = None,
    horizon: int = HORIZON_DAYS,
) -> TrainedModel:
    """Fit a LightGBM regressor on an already-prepared, leak-free dataset."""
    from lightgbm import LGBMRegressor

    merged = {**DEFAULT_PARAMS, **(params or {})}
    model = LGBMRegressor(**merged)
    model.fit(X[FEATURE_COLUMNS], y)

    return TrainedModel(
        booster=model,
        model_version=make_model_version(train_start, train_end, merged, horizon),
        params=merged,
        train_start=train_start,
        train_end=train_end,
        horizon_days=horizon,
        n_samples=len(y),
    )
