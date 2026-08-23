"""Core-satellite allocation — the no-leverage and reserve guarantees.

The structural change is only safe because of three invariants, each asserted
here: the core never spends cash the tactical layer needs, core + tactical never
exceeds the deployment ceiling, and the core is not dumped on a Level 1/2 kill
switch.
"""
import pytest

from app.config import load_config
from app.paper.core_allocation import (
    CORE_STRATEGY,
    plan_core,
    should_liquidate_core,
    tactical_budget_pct,
)


def _cfg(enabled=True, target=50.0, band=5.0, ceiling=85.0):
    c = load_config()
    c.core_allocation.enabled = enabled
    c.core_allocation.target_pct = target
    c.core_allocation.rebalance_band_pct = band
    c.core_allocation.max_total_deployment_pct = ceiling
    return c


# --------------------------------------------------------------------------- #
# basics
# --------------------------------------------------------------------------- #
def test_disabled_core_never_trades():
    plan = plan_core(_cfg(enabled=False), total_value=1000, cash=1000,
                     core_quantity=0, price=100)
    assert plan.action == "hold" and not plan.trades


def test_tactical_budget_is_positions_times_size():
    c = _cfg()
    assert tactical_budget_pct(c) == pytest.approx(
        c.risk.max_open_positions * c.risk.max_position_size_pct
    )


def test_buys_toward_target_when_underweight():
    # 1000 total, nothing in core, target 50% -> wants 500.
    plan = plan_core(_cfg(), total_value=1000, cash=1000, core_quantity=0, price=100)
    assert plan.action == "buy"
    assert plan.notional == pytest.approx(500.0)
    assert plan.quantity == pytest.approx(5.0)


def test_trims_when_overweight():
    # 700 of 1000 in core, target 500 -> sell 200.
    plan = plan_core(_cfg(), total_value=1000, cash=300, core_quantity=7, price=100)
    assert plan.action == "sell"
    assert plan.notional == pytest.approx(200.0)


def test_no_trade_inside_the_rebalance_band():
    # Core 480 vs target 500: drift 20 < band 50 -> hold, no churn.
    plan = plan_core(_cfg(band=5.0), total_value=1000, cash=520,
                     core_quantity=4.8, price=100)
    assert plan.action == "hold"
    assert "نطاق" in plan.reason


# --------------------------------------------------------------------------- #
# the safety invariants
# --------------------------------------------------------------------------- #
def test_core_never_spends_the_tactical_reserve():
    """3 x 10% = 30% must stay available for the strategies."""
    c = _cfg(target=50.0)
    # 1000 total, all cash, no tactical positions yet -> reserve is 300.
    plan = plan_core(c, total_value=1000, cash=1000, core_quantity=0,
                     price=100, tactical_value=0.0)
    assert plan.notional <= 1000 - 300 + 1e-9, "core ate into the tactical reserve"


def test_core_is_capped_by_the_total_deployment_ceiling():
    """core + tactical may never exceed max_total_deployment_pct."""
    c = _cfg(target=80.0, ceiling=85.0)
    # 300 already deployed tactically; ceiling 850 -> core may reach 550 at most.
    plan = plan_core(c, total_value=1000, cash=700, core_quantity=0,
                     price=100, tactical_value=300.0)
    assert plan.target_value <= 550.0 + 1e-9
    assert plan.current_value + plan.notional + 300.0 <= 850.0 + 1e-9


def test_core_never_buys_more_than_available_cash():
    """No leverage: a rich target with a thin wallet buys only what exists."""
    c = _cfg(target=90.0, ceiling=100.0)
    plan = plan_core(c, total_value=1000, cash=350, core_quantity=0,
                     price=100, tactical_value=0.0)
    assert plan.notional <= 350.0 + 1e-9


def test_core_holds_when_no_free_cash_after_reserve():
    c = _cfg()
    # Only 200 cash and the tactical reserve is 300 -> nothing free.
    plan = plan_core(c, total_value=1000, cash=200, core_quantity=0,
                     price=100, tactical_value=0.0)
    assert plan.action == "hold"


def test_sell_never_exceeds_the_position_held():
    c = _cfg(target=0.0, band=0.1)
    plan = plan_core(c, total_value=1000, cash=0, core_quantity=2, price=100)
    assert plan.action == "sell"
    assert plan.quantity <= 2.0 + 1e-9


# --------------------------------------------------------------------------- #
# kill switch interaction (D-15)
# --------------------------------------------------------------------------- #
def test_core_survives_level_1_and_2_but_not_level_3():
    """Selling a long-term index holding into an L1/L2 drawdown locks in the loss."""
    c = _cfg()
    assert should_liquidate_core(c, 0) is False
    assert should_liquidate_core(c, 1) is False
    assert should_liquidate_core(c, 2) is False
    assert should_liquidate_core(c, 3) is True
    assert should_liquidate_core(c, 4) is True  # manual STOP


def test_disabled_core_is_never_liquidated():
    assert should_liquidate_core(_cfg(enabled=False), 3) is False


def test_core_strategy_marker_is_stable():
    """Tactical rules identify the core by this marker; it must not drift."""
    assert CORE_STRATEGY == "core"
