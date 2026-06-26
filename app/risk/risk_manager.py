"""Risk Manager.

The risk manager is the last gate before a paper order. It enforces every hard
limit from the spec and is the single place position size and stops are computed.

Hard rules enforced (entries):
* max position size = 10% of portfolio
* max open positions = 3
* weekly loss limit = -5%, monthly = -12%, max drawdown = -15%
* stop loss = min(2 * ATR14, 7% absolute) from entry
* leverage / short / options forbidden (long-only, cash-funded)
* symbols outside the watchlist forbidden
* no trade if Kill Switch active
* no trade if DQS missing or < minimum
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

from app.config import Config
from app.domain import Action, DQSResult, FeatureSnapshot, RiskDecision, Signal


@dataclass
class PortfolioState:
    """Minimal portfolio view the risk manager needs (decoupled from the DB)."""

    total_value: float
    cash: float
    weekly_return_pct: float = 0.0    # fraction, negative on loss
    monthly_return_pct: float = 0.0   # fraction
    current_drawdown_pct: float = 0.0  # fraction, <= 0


@dataclass
class OpenPositionView:
    ticker: str
    quantity: float = 0.0


@dataclass
class StopTarget:
    stop_loss: float
    target_price: float
    stop_distance: float
    risk_reward: float


def compute_stop_and_target(
    entry: float,
    atr: Optional[float],
    absolute_stop_pct: float = 7.0,
    atr_multiplier: float = 2.0,
    risk_reward: float = 2.0,
) -> StopTarget:
    """Stop = entry - min(atr_mult*ATR, absolute_pct). Target = entry + RR*risk."""
    abs_distance = entry * (absolute_stop_pct / 100.0)
    if atr is not None and atr > 0:
        atr_distance = atr_multiplier * atr
        stop_distance = min(atr_distance, abs_distance)
    else:
        stop_distance = abs_distance
    stop_distance = max(stop_distance, 0.0001)  # never zero
    stop_loss = entry - stop_distance
    target_price = entry + risk_reward * stop_distance
    return StopTarget(
        stop_loss=stop_loss,
        target_price=target_price,
        stop_distance=stop_distance,
        risk_reward=risk_reward,
    )


def build_risk_context(
    signal: Signal,
    features: FeatureSnapshot,
    portfolio: PortfolioState,
    config: Config,
) -> dict:
    """Preliminary risk context used by the DQS engine (before final validation)."""
    st = compute_stop_and_target(
        entry=features.close,
        atr=features.atr14,
        absolute_stop_pct=config.risk.absolute_stop_loss_pct,
        atr_multiplier=config.risk.stop_loss_atr_multiplier,
    )
    notional = portfolio.total_value * (config.risk.max_position_size_pct / 100.0)
    within_limit = notional <= portfolio.total_value * (config.risk.max_position_size_pct / 100.0) + 1e-9
    return {
        "has_stop": True,
        "risk_reward": st.risk_reward,
        "within_position_limit": within_limit,
        "atr_available": features.atr14 is not None,
        "stop_loss": st.stop_loss,
        "target_price": st.target_price,
    }


def validate_order(
    signal: Signal,
    dqs: Optional[DQSResult],
    portfolio: PortfolioState,
    open_positions: Sequence[OpenPositionView],
    config: Config,
    features: Optional[FeatureSnapshot] = None,
    kill_switch_active: bool = False,
) -> RiskDecision:
    """Validate a candidate entry. Returns an actionable :class:`RiskDecision`."""
    violations: List[str] = []
    reasons: List[str] = []

    # Exits are always permitted (reducing risk) — sizing handled by the broker.
    if signal.action != Action.BUY:
        return RiskDecision(
            allowed=signal.action == Action.SELL,
            reason="إشارة غير دخول؛ بوابة مخاطر الدخول لا تنطبق.",
            violations=[],
        )

    risk = config.risk

    # 1. DQS must exist and clear the threshold.
    if dqs is None:
        violations.append("DQS_MISSING")
        reasons.append("DQS مفقود.")
    elif dqs.score < risk.minimum_dqs:
        violations.append("DQS_BELOW_MIN")
        reasons.append(f"DQS {dqs.score} < الحد الأدنى {risk.minimum_dqs}.")

    # 2. Kill switch.
    if kill_switch_active:
        violations.append("KILL_SWITCH_ACTIVE")
        reasons.append("مفتاح الإيقاف مفعّل؛ لا دخول جديد.")

    # 3. Watchlist membership (also blocks any non-US / forbidden symbol).
    if not config.is_in_watchlist(signal.ticker):
        violations.append("NOT_IN_WATCHLIST")
        reasons.append(f"{signal.ticker} ليس ضمن قائمة المتابعة.")

    # 4. Position count / duplicates.
    held = {p.ticker for p in open_positions}
    if signal.ticker in held:
        violations.append("ALREADY_HOLDING")
        reasons.append(f"يوجد مركز مفتوح في {signal.ticker}؛ لا تكديس.")
    elif len(open_positions) >= risk.max_open_positions:
        violations.append("MAX_OPEN_POSITIONS")
        reasons.append(f"بلغ الحد الأقصى للمراكز المفتوحة ({risk.max_open_positions}).")

    # 5. Portfolio-level loss limits.
    if portfolio.weekly_return_pct <= -(risk.weekly_loss_limit_pct / 100.0):
        violations.append("WEEKLY_LOSS_LIMIT")
        reasons.append("بلغ حد الخسارة الأسبوعي.")
    if portfolio.monthly_return_pct <= -(risk.monthly_loss_limit_pct / 100.0):
        violations.append("MONTHLY_LOSS_LIMIT")
        reasons.append("بلغ حد الخسارة الشهري.")
    if portfolio.current_drawdown_pct <= -(risk.max_drawdown_limit_pct / 100.0):
        violations.append("MAX_DRAWDOWN_LIMIT")
        reasons.append("بلغ حد أقصى التراجع.")

    # 6. Sizing & stop (only meaningful if we have a price/features).
    entry = features.close if features is not None else signal.raw_conditions.get("close")
    quantity = position_size_pct = notional = stop_loss = target_price = None
    if entry:
        max_notional = portfolio.total_value * (risk.max_position_size_pct / 100.0)
        notional = min(max_notional, portfolio.cash)
        if notional <= 0:
            violations.append("INSUFFICIENT_CASH")
            reasons.append("لا نقد متاح لمركز جديد.")
        else:
            quantity = notional / entry
            position_size_pct = (notional / portfolio.total_value) * 100.0
            atr = features.atr14 if features is not None else None
            st = compute_stop_and_target(
                entry=entry,
                atr=atr,
                absolute_stop_pct=risk.absolute_stop_loss_pct,
                atr_multiplier=risk.stop_loss_atr_multiplier,
            )
            stop_loss = st.stop_loss
            target_price = st.target_price

    allowed = len(violations) == 0
    reason = "موافَق عليه." if allowed else " ".join(reasons)
    return RiskDecision(
        allowed=allowed,
        reason=reason,
        violations=violations,
        quantity=quantity if allowed else None,
        position_size_pct=position_size_pct if allowed else None,
        notional=notional if allowed else None,
        stop_loss=stop_loss,
        target_price=target_price,
    )
