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

ICS-DOC-004 Phase 0
-------------------
The weights are now *injectable* so the learning feedback loop can re-balance
them (bounded, and always re-normalised to 100). Each component is computed as a
fraction of its own maximum and then scaled by its weight, so passing
``DEFAULT_WEIGHTS`` reproduces the pre-Phase-0 scores exactly. The scoring rules
themselves are unchanged.
"""
from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

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

COMPONENT_NAMES = (
    "strategy_alignment",
    "risk_management",
    "timing_quality",
    "market_regime_strength",
    "reason_clarity",
)

DEFAULT_WEIGHTS: Dict[str, float] = {
    "strategy_alignment": float(W_STRATEGY),
    "risk_management": float(W_RISK),
    "timing_quality": float(W_TIMING),
    "market_regime_strength": float(W_REGIME),
    "reason_clarity": float(W_REASON),
}

TOTAL_WEIGHT = 100.0

# Natural maximum of each raw sub-score, used to convert it to a 0..1 fraction.
_RAW_MAX = {
    "strategy_alignment": 25.0,   # confidence (0..1) * 25
    "risk_management": 25.0,      # 10 + 9 + 4 + 2
    "timing_quality": 20.0,       # 12 + 5 + 3
    "market_regime_strength": 15.0,
    "reason_clarity": 15.0,       # 7 + 8
}

_REGIME_STRENGTH = {
    Regime.BULL: 15,
    Regime.WEAK_BULL: 11,
    Regime.SIDEWAYS: 7,
    Regime.BEAR: 3,
    Regime.PANIC: 0,
}


def _strategy_alignment(signal: Signal) -> float:
    """Confidence in the setup maps directly to alignment points."""
    return max(0.0, min(1.0, signal.confidence)) * 25.0


def _market_regime_strength(regime: RegimeResult) -> float:
    return float(_REGIME_STRENGTH.get(regime.regime, 7))


def _timing_quality(features: FeatureSnapshot) -> float:
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

    return min(20.0, score)


def _risk_management(risk_context: Optional[Mapping[str, Any]]) -> float:
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

    return min(25.0, score)


def _reason_clarity(signal: Signal) -> float:
    reason = signal.reason or ""
    n_words = len(reason.split())
    n_conds = len(signal.raw_conditions or {})
    score = min(7.0, n_words / 4.0) + min(8.0, n_conds * 1.2)
    return min(15.0, score)


def normalize_weights(weights: Mapping[str, float]) -> Dict[str, float]:
    """Return weights restricted to the known components and summing to 100."""
    vals = {name: max(0.0, float(weights.get(name, 0.0))) for name in COMPONENT_NAMES}
    total = sum(vals.values())
    if total <= 0:
        return dict(DEFAULT_WEIGHTS)
    return {name: v * TOTAL_WEIGHT / total for name, v in vals.items()}


def calculate_dqs(
    signal: Signal,
    features: FeatureSnapshot,
    market_regime: RegimeResult,
    risk_context: Optional[Mapping[str, Any]] = None,
    minimum_dqs: int = DEFAULT_MIN_DQS,
    weights: Optional[Mapping[str, float]] = None,
) -> DQSResult:
    """Compute the Decision Quality Score for a candidate signal.

    ``weights`` defaults to :data:`DEFAULT_WEIGHTS`; any supplied mapping is
    normalised so the components can still only add up to 100.
    """
    w = DEFAULT_WEIGHTS if weights is None else normalize_weights(weights)

    raw = {
        "strategy_alignment": _strategy_alignment(signal),
        "risk_management": _risk_management(risk_context),
        "timing_quality": _timing_quality(features),
        "market_regime_strength": _market_regime_strength(market_regime),
        "reason_clarity": _reason_clarity(signal),
    }
    # fraction-of-own-max * weight — identical to the old fixed caps when the
    # default weights are used.
    components = {
        name: int(round((raw[name] / _RAW_MAX[name]) * w[name])) for name in COMPONENT_NAMES
    }
    score = int(sum(components.values()))
    allowed = score >= minimum_dqs

    if allowed:
        reason = f"DQS {score} ≥ الحد {minimum_dqs}."
    else:
        reason = f"DQS {score} < الحد {minimum_dqs}؛ فرصة مرفوضة."

    return DQSResult(score=score, components=components, allowed=allowed, reason=reason)
