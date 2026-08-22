"""Monthly walk-forward retraining and out-of-sample shadow prediction.

The single rule this module exists to enforce: a prediction for day *t* may only
ever come from a model trained on data whose **trades had already closed** before
*t*. Concretely, for each month:

    train on everything with an exit bar <= the last day of the previous month
    predict every day of this month with that model, and only that model

Nothing here influences a decision. It produces shadow predictions and the
out-of-sample series that the DSR / 3-window gate is computed from.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from app.logging_config import get_logger
from app.ml.dataset import (
    FEATURE_COLUMNS,
    HORIZON_DAYS,
    LABEL_COLUMN,
    build_training_frame,
    make_features,
    make_labels,
)
from app.ml.model import TrainedModel, spearman_ic, train_model
from app.ml.tuning import MAX_TRIALS, TuningResult, tune

log = get_logger(__name__)

# Approved via the Phase 1 parameter decision: expanding window, minimum 2 years.
MIN_TRAIN_DAYS = 504  # ~2 years of sessions


@dataclass
class MonthlyPrediction:
    date: pd.Timestamp
    ticker: str
    predicted: float
    realized: Optional[float]
    model_version: str
    train_start: Optional[datetime]
    train_end: Optional[datetime]


@dataclass
class WalkForwardResult:
    predictions: pd.DataFrame = field(default_factory=pd.DataFrame)
    models: List[TrainedModel] = field(default_factory=list)
    tuning: Optional[TuningResult] = None
    n_retrains: int = 0

    @property
    def ic(self) -> float:
        if self.predictions.empty:
            return 0.0
        df = self.predictions.dropna(subset=["predicted", "realized"])
        return spearman_ic(df["realized"], df["predicted"]) if len(df) >= 3 else 0.0


def _month_key(ts: pd.Timestamp) -> tuple[int, int]:
    return (ts.year, ts.month)


def run_walk_forward(
    features_by_symbol: Dict[str, pd.DataFrame],
    *,
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
    horizon: int = HORIZON_DAYS,
    min_train_days: int = MIN_TRAIN_DAYS,
    tune_first: bool = True,
    n_trials: int = MAX_TRIALS,
    exclude: Optional[List[str]] = None,
) -> WalkForwardResult:
    """Retrain monthly on an expanding window; predict the following month.

    ``tune_first`` runs one bounded optuna search on the earliest legal training
    window and reuses those hyper-parameters for the whole run. Re-tuning every
    month would multiply the trial count (and therefore the DSR deflation) for no
    modelling benefit on a dataset this size.
    """
    # One tidy panel of (date, ticker, features, label).
    panels = []
    for ticker, frame in features_by_symbol.items():
        if exclude and ticker in exclude:
            continue
        if frame is None or frame.empty:
            continue
        feats = make_features(frame)
        feats[LABEL_COLUMN] = make_labels(frame, horizon)
        feats["_ticker"] = ticker
        exit_dates = pd.Series(frame.index, index=frame.index).shift(-(1 + horizon))
        feats["_exit_date"] = exit_dates
        panels.append(feats)

    if not panels:
        return WalkForwardResult()

    panel = pd.concat(panels).dropna(subset=FEATURE_COLUMNS).sort_index()
    if start is not None:
        panel = panel[panel.index >= pd.Timestamp(start)]
    if end is not None:
        panel = panel[panel.index <= pd.Timestamp(end)]
    if panel.empty:
        return WalkForwardResult()

    all_dates = pd.DatetimeIndex(sorted(panel.index.unique()))
    if len(all_dates) <= min_train_days:
        log.warning("Not enough history for walk-forward: %d sessions.", len(all_dates))
        return WalkForwardResult()

    first_predict_date = all_dates[min_train_days]
    tuning: Optional[TuningResult] = None
    params = None

    if tune_first:
        seed_frame = build_training_frame(
            features_by_symbol, train_end=first_predict_date, horizon=horizon, exclude=exclude
        )
        if len(seed_frame) >= 200:
            tuning = tune(seed_frame.X, seed_frame.y, n_trials=n_trials, horizon=horizon)
            params = tuning.best_params
            log.info("Tuned on %d rows; best IC %.4f", len(seed_frame), tuning.best_score)

    rows: List[MonthlyPrediction] = []
    models: List[TrainedModel] = []
    current_month: Optional[tuple[int, int]] = None
    model: Optional[TrainedModel] = None

    predict_dates = all_dates[all_dates >= first_predict_date]
    for day in predict_dates:
        key = _month_key(day)
        if key != current_month:
            # New month -> retrain on everything that had already closed.
            cutoff = day - pd.Timedelta(days=1)
            frame = build_training_frame(
                features_by_symbol, train_end=cutoff, horizon=horizon, exclude=exclude
            )
            if len(frame) < 200:
                current_month = key
                continue
            ts_start, ts_end = frame.span
            model = train_model(
                frame.X, frame.y, params,
                train_start=ts_start, train_end=ts_end, horizon=horizon,
            )
            models.append(model)
            current_month = key

        if model is None:
            continue

        today = panel[panel.index == day]
        if today.empty:
            continue
        preds = model.predict(today[FEATURE_COLUMNS])
        for (_, row), pred in zip(today.iterrows(), preds):
            rows.append(
                MonthlyPrediction(
                    date=day,
                    ticker=row["_ticker"],
                    predicted=float(pred),
                    realized=(None if pd.isna(row[LABEL_COLUMN]) else float(row[LABEL_COLUMN])),
                    model_version=model.model_version,
                    train_start=model.train_start,
                    train_end=model.train_end,
                )
            )

    predictions = pd.DataFrame([r.__dict__ for r in rows])
    log.info("Walk-forward: %d retrains, %d predictions.", len(models), len(predictions))
    return WalkForwardResult(
        predictions=predictions, models=models, tuning=tuning, n_retrains=len(models)
    )


def split_windows(dates: pd.DatetimeIndex, n_windows: int = 3) -> List[tuple[datetime, datetime]]:
    """Split a date range into ``n_windows`` contiguous, non-overlapping windows.

    Approved via the Phase 1 parameter decision: 3 windows of roughly 20 months
    each over the 5-year dataset.
    """
    if len(dates) == 0:
        return []
    ordered = pd.DatetimeIndex(sorted(dates))
    edges = np.linspace(0, len(ordered) - 1, n_windows + 1).astype(int)
    out = []
    for i in range(n_windows):
        lo, hi = edges[i], edges[i + 1]
        if i > 0:
            lo = min(lo + 1, hi)  # keep the windows disjoint
        out.append((ordered[lo].to_pydatetime(), ordered[hi].to_pydatetime()))
    return out
