"""Shadow-mode evaluation: does the Phase 1 model deserve to influence anything?

Runs the whole out-of-sample pipeline and applies the activation gate:

    walk-forward predictions -> split into >= 3 non-overlapping windows
    -> per window: information coefficient, portfolio return, drawdown,
       Sharpe, and a Deflated Sharpe Ratio that counts every optuna trial
    -> activate only on a majority of windows beating the documented baseline
       AND a minimum DSR

`ml_confidence` is never written into a live DQS here. This module's only output
is evidence and a verdict.

**Reading the numbers honestly:** the per-window portfolio is a screening
construction — the top-N predicted symbols each day, equally weighted, with the
holding-period return spread evenly across its days. It measures the model's
standalone ranking edge; it is not the full "DQS + ML" counterfactual backtest.
It is deliberately the *first* gate: a model that cannot pass this has no case
for the more expensive one.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from app.logging_config import get_logger
from app.ml.evaluation import GateResult, WindowResult, deflated_sharpe_ratio, evaluate_gate
from app.ml.evaluation import trial_sharpe_variance
from app.ml.model import spearman_ic
from app.ml.walk_forward import WalkForwardResult, run_walk_forward, split_windows

log = get_logger(__name__)

TOP_N = 3            # mirrors the live cap of 3 open positions
TRADING_DAYS = 252


@dataclass
class ShadowReport:
    gate: GateResult
    windows: List[WindowResult] = field(default_factory=list)
    overall_ic: float = 0.0
    validation_ic: Optional[float] = None
    n_trials: int = 0
    n_retrains: int = 0
    n_predictions: int = 0
    best_params: Dict = field(default_factory=dict)

    @property
    def activate(self) -> bool:
        return self.gate.passed

    def summary(self) -> str:
        lines = [
            "🔬 تقرير وضع الظل — المرحلة 1",
            "",
            f"إعادات تدريب: {self.n_retrains} | تنبؤات: {self.n_predictions} | "
            f"محاولات optuna: {self.n_trials}",
        ]
        if self.validation_ic is not None:
            lines.append(
                f"IC داخل التحقق: {self.validation_ic:+.4f} → خارج العينة: {self.overall_ic:+.4f}"
            )
        lines.append("")
        for w in self.windows:
            mark = "✅" if w.beats_baseline else "❌"
            lines.append(
                f"{mark} {w.name} ({w.start} → {w.end}): عائد {w.total_return*100:+.2f}% | "
                f"تراجع {w.max_drawdown*100:.2f}% | Sharpe {w.sharpe:.2f} | DSR {w.dsr:.3f}"
            )
        lines += ["", self.gate.summary()]
        lines.append(
            "ml_confidence: يُفعَّل" if self.activate
            else "ml_confidence: يبقى في وضع الظل — التفعيل ممنوع"
        )
        return "\n".join(lines)


def portfolio_daily_returns(preds: pd.DataFrame, top_n: int = TOP_N, horizon: int = 10) -> pd.Series:
    """Daily return series of a top-N, equally weighted, model-ranked portfolio.

    The realised label is a ``horizon``-day return, so it is spread evenly across
    the holding period to put it on a daily footing. That is an approximation —
    it smooths the path and therefore flatters Sharpe — which is why this is a
    screening metric, not the final word.
    """
    out: Dict[pd.Timestamp, float] = {}
    usable = preds.dropna(subset=["realized", "predicted"])
    for day, group in usable.groupby("date"):
        top = group.nlargest(top_n, "predicted")
        if top.empty:
            continue
        out[pd.Timestamp(day)] = float(top["realized"].mean()) / max(1, horizon)
    return pd.Series(out).sort_index()


def evaluate_windows(
    preds: pd.DataFrame,
    *,
    n_trials: int,
    trial_variance: float,
    n_windows: int = 3,
    horizon: int = 10,
    min_days: int = 20,
) -> List[WindowResult]:
    """Per-window return, drawdown, Sharpe, IC and DSR."""
    if preds.empty:
        return []
    preds = preds.copy()
    preds["date"] = pd.to_datetime(preds["date"])
    dates = pd.DatetimeIndex(sorted(preds["date"].unique()))
    results: List[WindowResult] = []

    for i, (start, end) in enumerate(split_windows(dates, n_windows), 1):
        window = preds[(preds["date"] >= start) & (preds["date"] <= end)]
        clean = window.dropna(subset=["realized", "predicted"])
        rets = portfolio_daily_returns(window, horizon=horizon)
        if len(rets) < min_days:
            continue

        equity = (1.0 + rets).cumprod()
        total = float(equity.iloc[-1] - 1.0)
        drawdown = float((equity / equity.cummax() - 1.0).min())
        sharpe = (
            float(rets.mean() / rets.std() * np.sqrt(TRADING_DAYS)) if rets.std() > 0 else 0.0
        )
        dsr = deflated_sharpe_ratio(
            rets.tolist(), n_trials=n_trials, variance_of_trial_sharpes=trial_variance
        )["dsr"]
        results.append(
            WindowResult(
                name=f"نافذة{i}",
                start=str(pd.Timestamp(start).date()),
                end=str(pd.Timestamp(end).date()),
                total_return=total,
                max_drawdown=drawdown,
                sharpe=sharpe,
                win_rate=float((rets > 0).mean()),
                average_dqs=0.0,
                dsr=dsr,
                # Informational: rank correlation inside this window.
                beats_baseline=False,
            )
        )
        log.info(
            "%s IC %.4f return %.2f%% dd %.2f%% sharpe %.2f dsr %.3f",
            results[-1].name,
            spearman_ic(clean["realized"], clean["predicted"]) if len(clean) >= 3 else 0.0,
            total * 100, drawdown * 100, sharpe, dsr,
        )
    return results


def run_shadow_evaluation(
    features_by_symbol: Dict[str, pd.DataFrame],
    *,
    n_windows: int = 3,
    n_trials: int = 100,
    horizon: int = 10,
    walk_forward: Optional[WalkForwardResult] = None,
) -> ShadowReport:
    """Full Phase 1 shadow evaluation. Returns evidence plus a go/no-go verdict."""
    wf = walk_forward or run_walk_forward(
        features_by_symbol, tune_first=True, n_trials=n_trials, horizon=horizon
    )
    if wf.predictions.empty:
        return ShadowReport(gate=evaluate_gate([]), overall_ic=0.0)

    scores = wf.tuning.scores if wf.tuning else []
    variance = trial_sharpe_variance(scores) if scores else 0.0
    trials = wf.tuning.n_trials if wf.tuning else 0

    windows = evaluate_windows(
        wf.predictions,
        n_trials=max(trials, 1),
        trial_variance=variance,
        n_windows=n_windows,
        horizon=horizon,
    )
    gate = evaluate_gate(windows)

    return ShadowReport(
        gate=gate,
        windows=gate.windows,
        overall_ic=wf.ic,
        validation_ic=(wf.tuning.best_score if wf.tuning else None),
        n_trials=trials,
        n_retrains=wf.n_retrains,
        n_predictions=len(wf.predictions),
        best_params=(wf.tuning.best_params if wf.tuning else {}),
    )
