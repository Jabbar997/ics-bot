"""Paper broker — executes virtual orders only.

SAFETY: This is the *only* execution path in ICS and it is entirely simulated.
There is deliberately no real-broker integration, no API keys, no account, and
no order routing. Commission and slippage are configurable but cosmetic.
"""
from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Dict, Optional

from app.config import Config
from app.db.models import PaperOrder, Position
from app.db.repositories import OrderRepository, PositionRepository
from app.domain import OrderSide, utcnow
from app.paper.orders import OrderRequest, make_paper_order
from app.paper.portfolio import Portfolio
from app.utils.money import round_money, round_shares, to_decimal


class PaperBroker:
    def __init__(self, session, config: Config):
        self.session = session
        self.config = config
        self.portfolio = Portfolio(session, config.initial_capital)
        self.orders = OrderRepository(session)
        self.positions = PositionRepository(session)
        self.commission = Decimal(str(config.paper.commission_per_trade_usd))
        self.slippage = Decimal(str(config.paper.slippage_pct)) / Decimal("100")

    # -- fills ------------------------------------------------------------ #
    def _buy_fill(self, price: float) -> Decimal:
        return round_money(to_decimal(price) * (Decimal("1") + self.slippage))

    def _sell_fill(self, price: float) -> Decimal:
        return round_money(to_decimal(price) * (Decimal("1") - self.slippage))

    # -- buys ------------------------------------------------------------- #
    def place_buy_order(
        self,
        ticker: str,
        quantity: float,
        price: float,
        *,
        strategy: Optional[str] = None,
        reason: str = "",
        decision_id: Optional[str] = None,
        stop_loss: Optional[float] = None,
        target_price: Optional[float] = None,
        dqs: Optional[int] = None,
        ts: Optional[datetime] = None,
    ) -> Position:
        """Open a virtual long position, funded from cash only (no leverage)."""
        fill = self._buy_fill(price)
        qty = round_shares(quantity)
        cost = round_money(fill * qty + self.commission)

        cash = self.portfolio.calculate_cash()
        if cost > cash:
            # Never spend cash we do not have — long-only, cash-funded.
            qty = round_shares((cash - self.commission) / fill) if fill > 0 else Decimal("0")
            cost = round_money(fill * qty + self.commission)
        if qty <= 0:
            raise ValueError(f"Cannot buy {ticker}: insufficient cash.")

        order = make_paper_order(
            OrderRequest(ticker, OrderSide.BUY, float(qty), float(fill), reason, decision_id),
            fill_price=float(fill),
            commission=float(self.commission),
            filled_at=ts,
        )
        self.orders.add(order)
        self.portfolio.adjust_cash(-cost)

        position = Position(
            ticker=ticker,
            strategy=strategy,
            quantity=float(qty),
            entry_price=float(fill),
            current_price=float(fill),
            stop_loss=stop_loss,
            target_price=target_price,
            dqs_at_entry=dqs,
            is_open=True,
        )
        if ts is not None:
            position.entry_at = ts
        return self.positions.add(position)

    # -- sells / closes --------------------------------------------------- #
    def close_position(
        self,
        position: Position,
        price: float,
        *,
        reason: str = "",
        ts: Optional[datetime] = None,
    ) -> PaperOrder:
        """Close (fully) an open paper position and realise P/L."""
        fill = self._sell_fill(price)
        qty = to_decimal(position.quantity)
        proceeds = round_money(fill * qty - self.commission)

        order = make_paper_order(
            OrderRequest(position.ticker, OrderSide.SELL, float(qty), float(fill), reason),
            fill_price=float(fill),
            commission=float(self.commission),
            filled_at=ts,
        )
        self.orders.add(order)
        self.portfolio.adjust_cash(proceeds)

        entry = to_decimal(position.entry_price)
        realized = round_money(fill * qty - entry * qty - self.commission)
        realized_pct = float(realized / (entry * qty)) if entry * qty != 0 else 0.0

        position.is_open = False
        position.exit_price = float(fill)
        position.exit_at = ts or utcnow().replace(tzinfo=None)
        position.current_price = float(fill)
        position.realized_pnl = float(realized)
        position.realized_pnl_pct = realized_pct
        position.exit_reason = reason
        return order

    def place_sell_order(self, ticker: str, price: float, *, reason: str = "", ts=None) -> Optional[PaperOrder]:
        pos = self.positions.get_open_by_ticker(ticker)
        if pos is None:
            return None
        return self.close_position(pos, price, reason=reason, ts=ts)

    # -- valuation passthroughs ------------------------------------------ #
    def mark_to_market(self, prices: Dict[str, float]) -> None:
        self.portfolio.mark_to_market(prices)

    def update_portfolio_snapshot(self, prices: Optional[Dict[str, float]] = None, ts=None):
        return self.portfolio.update_portfolio_snapshot(prices, ts)

    def calculate_cash(self) -> Decimal:
        return self.portfolio.calculate_cash()

    def calculate_invested_value(self, prices: Optional[Dict[str, float]] = None) -> Decimal:
        return self.portfolio.calculate_invested_value(prices)

    def calculate_total_value(self, prices: Optional[Dict[str, float]] = None) -> Decimal:
        return self.portfolio.calculate_total_value(prices)
