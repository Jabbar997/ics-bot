"""Decision Quality Score (DQS) engine.

Every candidate trade is scored 0-100 across five weighted components before it
is allowed anywhere near the risk manager:

    strategy_alignment      25
    risk_management         25
    timing_quality          20
    market_regime_strength  15
    reason_clarity          15
    ----------------------------
    total                  100

Scores >= the minimum threshold (default 70) are eligible for execution if the
risk manager also approves; scores below are rejected and logged as rejected
opportunities.
"""
from __future__ import annotations

from typing import Any, Mapping, Optional

from app.domain import (
    Action,
    DQSResult,
    FeatureSnapshot,
    Regime,
    RegimeResult,
    Signal,
)

DEFAULT_MIN_DQS = 70

# Max points per component (sum = 100).
W_STRATEGY = 25
W_RISK = 25
W_TIMING = 20
W_REGIME = 15
W_REASON = 15

_REGIME_STRENGTH = {
    Regime.BULL: 15,
    Regime.WEAK_BULL: 11,
    Regime.SIDEWAYS: 7,
    Regime.BEAR: 3,
    Regime.PANIC: 0,
}


def _strategy_alignment(signal: Signal) -> int:
    """Confidence in the setup maps directly to alignment points."""
    return int(round(max(0.0, min(1.0, signal.confidence)) * W_STRATEGY))


def _market_regime_strength(regime: RegimeResult) -> int:
    return _REGIME_STRENGTH.get(regime.regime, 7)


def _timing_quality(features: FeatureSnapshot) -> int:
    score = 0.0

    rsi = features.rsi14
    if rsi is None:
        rsi_q = 0.3
    elif 45 <= rsi <= 60:
        rsi_q = 1.0
    elif 40 <= rsi < 45 or 60 < rsi <= 68:
        rsi_q = 0.7
    elif 35 <= rsi < 40 or 68 < rsi <= 72:
        rsi_q = 0.4
    else:
        rsi_q = 0.15
    score += 12.0 * rsi_q

    if features.macd is not None and features.macd_signal is not None:
        score += 5.0 if features.macd >= features.macd_signal else 1.0
    else:
        score += 2.0

    if features.volume is not None and features.volume_ma20:
        score += 3.0 if features.volume >= features.volume_ma20 else 1.0
    else:
        score += 1.0

    return int(round(min(W_TIMING, score)))


def _risk_management(risk_context: Optional[Mapping[str, Any]]) -> int:
    ctx = risk_context or {}
    score = 0.0

    if ctx.get("has_stop"):
        score += 10.0

    rr = ctx.get("risk_reward")
    if rr is None:
        score += 2.0
    elif rr >= 2.0:
        score += 9.0
    elif rr >= 1.5:
        score += 7.0
    elif rr >= 1.0:
        score += 4.0
    else:
        score += 1.0

    if ctx.get("within_position_limit", True):
        score += 4.0

    if ctx.get("atr_available"):
        score += 2.0

    return int(round(min(W_RISK, score)))


def _reason_clarity(signal: Signal) -> int:
    reason = signal.reason or ""
    n_words = len(reason.split())
    n_conds = len(signal.raw_conditions or {})
    score = min(7.0, n_words / 4.0) + min(8.0, n_conds * 1.2)
    return int(round(min(W_REASON, score)))


def calculate_dqs(
    signal: Signal,
    features: FeatureSnapshot,
    market_regime: RegimeResult,
    risk_context: Optional[Mapping[str, Any]] = None,
    minimum_dqs: int = DEFAULT_MIN_DQS,
) -> DQSResult:
    """Compute the Decision Quality Score for a candidate signal."""
    components = {
        "strategy_alignment": _strategy_alignment(signal),
        "risk_management": _risk_management(risk_context),
        "timing_quality": _timing_quality(features),
        "market_regime_strength": _market_regime_strength(market_regime),
        "reason_clarity": _reason_clarity(signal),
    }
    score = int(sum(components.values()))
    allowed = score >= minimum_dqs

    if allowed:
        reason = f"DQS {score} ≥ الحد {minimum_dqs}."
    else:
        reason = f"DQS {score} < الحد {minimum_dqs}؛ فرصة مرفوضة."

    return DQSResult(score=score, components=components, allowed=allowed, reason=reason)
