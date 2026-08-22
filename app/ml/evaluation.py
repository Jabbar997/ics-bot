"""Deflated Sharpe Ratio and the Phase 1 multi-window gate.

Why DSR (ICS-DOC-004 Phase 1 amendment): if you try N configurations and keep the
best one, its Sharpe ratio is biased upward simply because you looked N times.
DSR (Bailey & López de Prado, 2014) discounts the observed Sharpe by the number
of trials actually run — every optuna trial counts — plus the skew and kurtosis
of the returns, and reports the probability that the true Sharpe exceeds a
benchmark.

Nothing here can affect a live decision; it only decides whether the Phase 1
model is allowed to *stop* being shadow-only.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

# Euler-Mascheroni constant, used in the expected-maximum-Sharpe approximation.
_EULER_GAMMA = 0.5772156649015329


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_ppf(p: float) -> float:
    """Inverse standard normal CDF (Acklam's rational approximation)."""
    if not 0.0 < p < 1.0:
        raise ValueError("p must be in (0, 1)")
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


def _moments(returns: Sequence[float]) -> tuple[float, float, float, float]:
    """(mean, stdev, skew, kurtosis) — kurtosis is non-excess (normal = 3)."""
    n = len(returns)
    mean = sum(returns) / n
    var = sum((r - mean) ** 2 for r in returns) / n
    sd = math.sqrt(var)
    if sd == 0:
        return mean, 0.0, 0.0, 3.0
    skew = sum((r - mean) ** 3 for r in returns) / (n * sd ** 3)
    kurt = sum((r - mean) ** 4 for r in returns) / (n * sd ** 4)
    return mean, sd, skew, kurt


_SD_EPS = 1e-12  # a series this flat has no meaningful Sharpe


def sharpe_ratio(returns: Sequence[float]) -> float:
    """Non-annualised Sharpe of the given return series."""
    if len(returns) < 2:
        return 0.0
    mean, sd, _, _ = _moments(returns)
    # Guard against float noise: [0.01]*50 has sd ~3e-18, not exactly 0, which
    # would otherwise report an astronomically large Sharpe.
    return 0.0 if sd <= _SD_EPS else mean / sd


def trial_sharpe_variance(trial_scores: Sequence[float]) -> float:
    """Variance of the Sharpe ratios observed across tuning trials.

    This is what makes the deflation honest, and it is why the Phase 1 amendment
    makes logging every optuna trial mandatory: the correction needs the spread
    of the scores that were actually tried, not a guess.
    """
    n = len(trial_scores)
    if n < 2:
        return 0.0
    mean = sum(trial_scores) / n
    return sum((s - mean) ** 2 for s in trial_scores) / (n - 1)


def expected_max_sharpe(n_trials: int, variance_of_trial_sharpes: float = 1.0) -> float:
    """Expected maximum Sharpe from ``n_trials`` independent, useless trials.

    This is the bar a genuinely-skilled model must clear: it is what the *best of
    N random tries* would score by luck alone.
    """
    if n_trials <= 1:
        return 0.0
    sd = math.sqrt(max(variance_of_trial_sharpes, 0.0))
    if sd == 0:
        return 0.0
    n = float(n_trials)
    term = (1 - _EULER_GAMMA) * _norm_ppf(1 - 1.0 / n) + _EULER_GAMMA * _norm_ppf(
        1 - 1.0 / (n * math.e)
    )
    return sd * term


def deflated_sharpe_ratio(
    returns: Sequence[float],
    n_trials: int,
    variance_of_trial_sharpes: float,
    benchmark_sharpe: Optional[float] = None,
) -> Dict[str, float]:
    """Probability that the true Sharpe beats the multiplicity-adjusted benchmark.

    ``variance_of_trial_sharpes`` is **required** and must come from the observed
    optuna trial log (see :func:`trial_sharpe_variance`). There is deliberately no
    default: assuming unit variance would set the bar at a Sharpe no real daily
    strategy reaches, and would reject everything for the wrong reason — which
    looks conservative but is simply broken.

    ``benchmark_sharpe`` defaults to :func:`expected_max_sharpe`, i.e. what the
    best of ``n_trials`` worthless attempts would have scored by chance.
    Returns the observed Sharpe, the benchmark used, and DSR in [0, 1].
    A conventional pass mark is DSR >= 0.95.

    Note: ``returns`` and the trial Sharpes must be on the **same scale**
    (both per-period, e.g. daily) or the deflation is meaningless.
    """
    n = len(returns)
    if n < 3:
        return {"sharpe": 0.0, "benchmark_sharpe": 0.0, "dsr": 0.0, "n_obs": float(n)}

    sr = sharpe_ratio(returns)
    _, _, skew, kurt = _moments(returns)
    sr_star = (
        expected_max_sharpe(n_trials, variance_of_trial_sharpes)
        if benchmark_sharpe is None
        else benchmark_sharpe
    )

    # Standard error of the Sharpe estimator under non-normal returns.
    denom = 1.0 - skew * sr + ((kurt - 1.0) / 4.0) * sr ** 2
    if denom <= 0 or n <= 1:
        return {"sharpe": sr, "benchmark_sharpe": sr_star, "dsr": 0.0, "n_obs": float(n)}
    se = math.sqrt(denom / (n - 1))
    if se == 0:
        return {"sharpe": sr, "benchmark_sharpe": sr_star, "dsr": 0.0, "n_obs": float(n)}

    dsr = _norm_cdf((sr - sr_star) / se)
    return {"sharpe": sr, "benchmark_sharpe": sr_star, "dsr": dsr, "n_obs": float(n)}


# --------------------------------------------------------------------------- #
# Multi-window gate
# --------------------------------------------------------------------------- #
@dataclass
class WindowResult:
    name: str
    start: Optional[str] = None
    end: Optional[str] = None
    total_return: float = 0.0
    max_drawdown: float = 0.0
    sharpe: float = 0.0
    win_rate: float = 0.0
    average_dqs: float = 0.0
    dsr: float = 0.0
    beats_baseline: bool = False


@dataclass
class GateResult:
    passed: bool
    reason: str
    windows: List[WindowResult] = field(default_factory=list)
    windows_won: int = 0
    windows_total: int = 0
    dsr_min: float = 0.0

    def summary(self) -> str:
        head = "✅ اجتاز" if self.passed else "❌ لم يجتز"
        return f"{head} — {self.reason} ({self.windows_won}/{self.windows_total} نافذة)"


# The documented baseline (ICS-DOC-004): 5 years, walk-forward.
BASELINE = {
    "total_return": 0.2188,
    "max_drawdown": -0.0542,
    "sharpe": 1.01,
    "win_rate": 0.387,
    "average_dqs": 85.15,
}

MIN_WINDOWS = 3
DSR_PASS = 0.95


def beats_baseline(w: WindowResult, baseline: Optional[Dict[str, float]] = None) -> bool:
    """A window is won only if return, drawdown AND Sharpe all beat the baseline."""
    b = baseline or BASELINE
    return (
        w.total_return > b["total_return"]
        and w.max_drawdown >= b["max_drawdown"]  # less negative is better
        and w.sharpe > b["sharpe"]
    )


def evaluate_gate(
    windows: Sequence[WindowResult],
    baseline: Optional[Dict[str, float]] = None,
    dsr_pass: float = DSR_PASS,
    min_windows: int = MIN_WINDOWS,
) -> GateResult:
    """Phase 1 activation gate: >= 3 non-overlapping windows, majority beaten, DSR ok."""
    ws = list(windows)
    if len(ws) < min_windows:
        return GateResult(
            False,
            f"عدد النوافذ {len(ws)} < الحد الأدنى {min_windows}.",
            ws, 0, len(ws), 0.0,
        )

    for w in ws:
        w.beats_baseline = beats_baseline(w, baseline)

    won = sum(1 for w in ws if w.beats_baseline)
    dsr_min = min(w.dsr for w in ws)
    majority = won > len(ws) / 2.0

    if not majority:
        return GateResult(False, f"لم يتفوق في أغلبية النوافذ ({won}/{len(ws)}).", ws, won, len(ws), dsr_min)
    if dsr_min < dsr_pass:
        return GateResult(
            False,
            f"أدنى DSR {dsr_min:.3f} < الحد {dsr_pass} (تحسّن قد يكون صدفة تعدد المحاولات).",
            ws, won, len(ws), dsr_min,
        )
    return GateResult(
        True,
        f"تفوّق في {won}/{len(ws)} نافذة وأدنى DSR {dsr_min:.3f} ≥ {dsr_pass}.",
        ws, won, len(ws), dsr_min,
    )
