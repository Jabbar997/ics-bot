"""Defensive Cash 'strategy' — a market-state gate rather than a per-symbol setup.

Activates (no new entries, hold cash, review open positions) when any of:
* SPY below MA50, or
* SPY in Panic, or
* Kill Switch Level 1+ active, or
* most opportunities have DQS < 70.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from app.domain import KillSwitchLevel, Regime, RegimeResult, StrategyName


@dataclass
class DefensiveDecision:
    active: bool
    reason: str

    def to_dict(self) -> dict:
        return {"active": self.active, "reason": self.reason}


class DefensiveCashStrategy:
    name = StrategyName.DEFENSIVE_CASH

    def should_activate(
        self,
        regime: RegimeResult,
        kill_switch_level: KillSwitchLevel = KillSwitchLevel.NONE,
        low_quality_ratio: Optional[float] = None,
    ) -> DefensiveDecision:
        """``low_quality_ratio`` = fraction of candidate signals scoring DQS < 70."""
        reasons = []

        if regime.regime == Regime.PANIC:
            reasons.append("SPY في حالة ذعر")
        if regime.spy_ma50 is not None and regime.spy_close < regime.spy_ma50:
            reasons.append("SPY دون MA50")
        if kill_switch_level.value >= KillSwitchLevel.LEVEL_1.value:
            reasons.append(f"مفتاح الإيقاف المستوى {kill_switch_level.value} مفعّل")
        if low_quality_ratio is not None and low_quality_ratio >= 0.7:
            reasons.append("معظم الفرص سجّلت DQS < 70")

        if reasons:
            return DefensiveDecision(
                active=True,
                reason="نقد دفاعي: " + "؛ ".join(reasons) + ". لا دخول جديد.",
            )
        return DefensiveDecision(active=False, reason="وضع المخاطرة: الدخول مسموح.")
