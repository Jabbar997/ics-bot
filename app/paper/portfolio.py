"""Virtual portfolio accounting (cash + open positions valuation + snapshots).

Cash is persisted in the SystemConfig key/value store; positions live in the
``positions`` table. Everything is virtual — there is no real money or account.
"""
from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Dict, Optional

from app.db.models import PortfolioSnapshot
from app.db.repositories import (
    PortfolioSnapshotRepository,
    PositionRepository,
    SystemConfigRepository,
)
from app.utils.money import round_money, to_decimal

CASH_KEY = "cash_usd"


class Portfolio:
    def __init__(self, session, initial_capital: float):
        self.session = session
        self.initial_capital = Decimal(str(initial_capital))
        self.cfg = SystemConfigRepository(session)
        self.positions = PositionRepository(session)
        self.snapshots = PortfolioSnapshotRepository(session)
        self._ensure_initialized()

    def _ensure_initialized(self) -> None:
        if self.cfg.get(CASH_KEY) is None:
            self.cfg.set(CASH_KEY, str(self.initial_capital))

    # -- cash ------------------------------------------------------------- #
    def calculate_cash(self) -> Decimal:
        return to_decimal(self.cfg.get(CASH_KEY, "0"))

    def set_cash(self, value: Decimal) -> None:
        self.cfg.set(CASH_KEY, str(round_money(value)))

    def adjust_cash(self, delta: Decimal) -> Decimal:
        new_cash = self.calculate_cash() + to_decimal(delta)
        self.set_cash(new_cash)
        return new_cash

    # -- valuation -------------------------------------------------------- #
    def calculate_invested_value(self, prices: Optional[Dict[str, float]] = None) -> Decimal:
        prices = prices or {}
        total = Decimal("0")
        for pos in self.positions.open_positions():
            mark = prices.get(pos.ticker, pos.current_price or pos.entry_price)
            total += to_decimal(pos.quantity) * to_decimal(mark)
        return round_money(total)

    def calculate_total_value(self, prices: Optional[Dict[str, float]] = None) -> Decimal:
        return round_money(self.calculate_cash() + self.calculate_invested_value(prices))

    # -- mark-to-market & snapshots -------------------------------------- #
    def mark_to_market(self, prices: Dict[str, float]) -> None:
        for pos in self.positions.open_positions():
            if pos.ticker in prices:
                pos.current_price = prices[pos.ticker]

    def update_portfolio_snapshot(
        self, prices: Optional[Dict[str, float]] = None, ts: Optional[datetime] = None
    ) -> PortfolioSnapshot:
        prices = prices or {}
        self.mark_to_market(prices)
        cash = self.calculate_cash()
        invested = self.calculate_invested_value(prices)
        total = round_money(cash + invested)

        prev = self.snapshots.latest()
        daily_pnl = None
        daily_pnl_pct = None
        if prev is not None and prev.total_value:
            daily_pnl = float(total - to_decimal(prev.total_value))
            daily_pnl_pct = daily_pnl / float(prev.total_value)

        snapshot = PortfolioSnapshot(
            cash=float(cash),
            invested_value=float(invested),
            total_value=float(total),
            open_positions=len(self.positions.open_positions()),
            daily_pnl=daily_pnl,
            daily_pnl_pct=daily_pnl_pct,
        )
        if ts is not None:
            snapshot.ts = ts
        return self.snapshots.add(snapshot)

    # -- derived metrics for the risk manager ----------------------------- #
    def total_return_pct(self, prices: Optional[Dict[str, float]] = None) -> float:
        total = self.calculate_total_value(prices)
        return float(total / self.initial_capital - 1) if self.initial_capital else 0.0
