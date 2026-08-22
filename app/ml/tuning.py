"""Bounded optuna tuning for the Phase 1 model.

Per the Phase 1 amendment:

* the search space is **fixed and declared here in code** — no open search;
* at most :data:`MAX_TRIALS` (100) trials per monthly training cycle;
* **every trial is logged** — number, parameters, validation score. That log is
  not bookkeeping: :func:`app.ml.evaluation.deflated_sharpe_ratio` needs the
  spread of the trial scores to deflate the winner honestly.

Validation is a **time-ordered holdout**, never a random split: shuffling rows
of a time series lets the model learn from its own future.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd

from app.logging_config import get_logger
from app.ml.model import DEFAULT_PARAMS, spearman_ic, train_model

log = get_logger(__name__)

MAX_TRIALS = 100          # hard ceiling per monthly cycle (amendment §1)
VALIDATION_FRACTION = 0.2  # last 20% of the window, chronologically

# The predeclared, bounded search space. Ranges are deliberately narrow: this is
# a small, noisy dataset, and a wide space is exactly how multiplicity produces
# a lucky winner.
SEARCH_SPACE: Dict[str, Any] = {
    "learning_rate": (0.01, 0.10),      # log-uniform
    "num_leaves": (7, 63),
    "max_depth": (3, 8),
    "min_child_samples": (20, 300),
    "feature_fraction": (0.6, 1.0),
    "bagging_fraction": (0.6, 1.0),
    "lambda_l2": (0.0, 10.0),
}


@dataclass
class TrialRecord:
    number: int
    params: Dict[str, Any]
    score: float

    def to_dict(self) -> Dict[str, Any]:
        return {"number": self.number, "params": self.params, "score": self.score}


@dataclass
class TuningResult:
    best_params: Dict[str, Any]
    best_score: float
    trials: List[TrialRecord] = field(default_factory=list)

    @property
    def n_trials(self) -> int:
        return len(self.trials)

    @property
    def scores(self) -> List[float]:
        return [t.score for t in self.trials]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "best_params": self.best_params,
            "best_score": self.best_score,
            "n_trials": self.n_trials,
            "trials": [t.to_dict() for t in self.trials],
        }


def chronological_split(X: pd.DataFrame, y: pd.Series, fraction: float = VALIDATION_FRACTION):
    """Split by time, not at random: validation is always the later rows."""
    n = len(y)
    if n < 10:
        return X, y, X, y
    cut = int(n * (1.0 - fraction))
    return X.iloc[:cut], y.iloc[:cut], X.iloc[cut:], y.iloc[cut:]


def tune(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    n_trials: int = MAX_TRIALS,
    seed: int = 42,
    horizon: Optional[int] = None,
) -> TuningResult:
    """Bayesian search over :data:`SEARCH_SPACE`, maximising validation Spearman IC."""
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    n_trials = max(1, min(int(n_trials), MAX_TRIALS))  # the ceiling is not advisory

    X_tr, y_tr, X_val, y_val = chronological_split(X, y)
    records: List[TrialRecord] = []

    def objective(trial: "optuna.Trial") -> float:
        params = {
            "learning_rate": trial.suggest_float("learning_rate", *SEARCH_SPACE["learning_rate"], log=True),
            "num_leaves": trial.suggest_int("num_leaves", *SEARCH_SPACE["num_leaves"]),
            "max_depth": trial.suggest_int("max_depth", *SEARCH_SPACE["max_depth"]),
            "min_child_samples": trial.suggest_int("min_child_samples", *SEARCH_SPACE["min_child_samples"]),
            "feature_fraction": trial.suggest_float("feature_fraction", *SEARCH_SPACE["feature_fraction"]),
            "bagging_fraction": trial.suggest_float("bagging_fraction", *SEARCH_SPACE["bagging_fraction"]),
            "lambda_l2": trial.suggest_float("lambda_l2", *SEARCH_SPACE["lambda_l2"]),
        }
        model = train_model(X_tr, y_tr, params, horizon=horizon or 10)
        score = spearman_ic(y_val, model.predict(X_val))
        records.append(TrialRecord(trial.number, dict(params), float(score)))
        return score

    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed)
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = {**DEFAULT_PARAMS, **study.best_params}
    log.info(
        "Tuning finished: %d trials, best validation IC %.4f", len(records), study.best_value
    )
    return TuningResult(best_params=best, best_score=float(study.best_value), trials=records)
