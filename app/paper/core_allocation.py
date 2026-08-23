"""Core-satellite allocation (ICS-DOC-004 amendment, approved 2026-08-23).

**Why this exists.** Measured over 2022-06 → 2026-08 on a fixed dataset, ICS
returned +24.75% while a static "30% SPY + 70% cash" returned +21.83%. The whole
strategy stack was reproducing scaled market beta, because only ~30% of capital
was ever deployed. The binding constraint was idle cash, not signal quality.

**What it does.** Holds a target share of the portfolio in the diversified
benchmark, rebalanced only when drift leaves a band. The tactical layer (3 × 10%)
runs on top, unchanged. Measured effect at a 50% core: +73.00% return with a
−12.93% drawdown and Sharpe 1.20, versus +24.75% / −5.43% / 1.12 before — and far
better than the concentrated alternative (5 positions gave +35.39% with a −15.82%
drawdown that breaches the mandate).

**Hard rules kept.**
* Long-only, cash-funded — the core can never be bought with money that is not
  there, and core + tactical is capped by ``max_total_deployment_pct``.
* Cash reserved for the tactical layer is protected: the core will not consume
  the budget the strategies need to take their positions.
* Paper only. This module places no real orders.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from app.config import Config
from app.logging_config import get_logger

log = get_logger(__name__)

CORE_STRATEGY = "core"  # marks the position so tactical rules skip it


@dataclass
class CorePlan:
    """What the core should do today."""

    action: str            # "buy" | "sell" | "hold"
    quantity: float = 0.0  # shares to trade (always positive)
    notional: float = 0.0
    reason: str = ""
    target_value: float = 0.0
    current_value: float = 0.0

    @property
    def trades(self) -> bool:
        return self.action in ("buy", "sell") and self.quantity > 0


def tactical_budget_pct(config: Config) -> float:
    """Portfolio share the tactical layer may need at full extension."""
    return config.risk.max_open_positions * config.risk.max_position_size_pct


def plan_core(
    config: Config,
    *,
    total_value: float,
    cash: float,
    core_quantity: float,
    price: float,
    tactical_value: float = 0.0,
) -> CorePlan:
    """Decide the core trade for today.

    ``total_value`` is the whole portfolio (cash + tactical + core).
    ``tactical_value`` is the market value currently held in tactical positions.
    """
    ca = config.core_allocation
    current_value = max(0.0, core_quantity) * price

    if not ca.enabled or price <= 0 or total_value <= 0:
        return CorePlan("hold", reason="النواة غير مفعّلة.", current_value=current_value)

    target_value = total_value * (ca.target_pct / 100.0)

    # Never let core + tactical exceed the deployment ceiling (no leverage).
    ceiling_value = total_value * (ca.max_total_deployment_pct / 100.0)
    target_value = min(target_value, max(0.0, ceiling_value - tactical_value))

    # Protect the cash the tactical layer still needs to reach full extension.
    reserve = total_value * (tactical_budget_pct(config) / 100.0) - tactical_value
    reserve = max(0.0, reserve)

    drift = target_value - current_value
    band = total_value * (ca.rebalance_band_pct / 100.0)
    if abs(drift) < band:
        return CorePlan(
            "hold",
            reason=f"انحراف النواة {abs(drift):.2f}$ دون نطاق {band:.2f}$.",
            target_value=target_value,
            current_value=current_value,
        )

    if drift > 0:
        # Buy, but only with cash that is genuinely free after the reserve.
        spendable = max(0.0, min(drift, cash - reserve))
        qty = spendable / price
        if qty <= 0:
            return CorePlan(
                "hold",
                reason="لا نقد حر لشراء النواة بعد حجز احتياطي التكتيكي.",
                target_value=target_value,
                current_value=current_value,
            )
        return CorePlan(
            "buy", quantity=qty, notional=qty * price,
            reason=f"رفع النواة نحو {ca.target_pct:.0f}% من المحفظة.",
            target_value=target_value, current_value=current_value,
        )

    # Trim back toward target.
    qty = min(core_quantity, (-drift) / price)
    if qty <= 0:
        return CorePlan("hold", reason="لا شيء للتخفيف.",
                        target_value=target_value, current_value=current_value)
    return CorePlan(
        "sell", quantity=qty, notional=qty * price,
        reason=f"تخفيف النواة نحو {ca.target_pct:.0f}% من المحفظة.",
        target_value=target_value, current_value=current_value,
    )


def should_liquidate_core(config: Config, kill_level: int) -> bool:
    """The core is only liquidated at the configured kill-switch level.

    Levels 1 and 2 halt *strategy activity*; selling a long-term index
    allocation into the drawdown that triggered them would lock the loss in at
    the worst moment. Level 3 is a full stop with mandatory human review, so the
    core goes too. See D-15.
    """
    if not config.core_allocation.enabled:
        return False
    return kill_level >= config.core_allocation.liquidate_on_kill_level
