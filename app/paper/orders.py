"""Paper order helpers.

SAFETY: there is no live-order path here. ``make_paper_order`` only ever creates
a local :class:`PaperOrder` row — it never contacts a broker or routes an order.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from app.db.models import PaperOrder
from app.domain import OrderSide, OrderStatus, utcnow


@dataclass
class OrderRequest:
    ticker: str
    side: OrderSide
    quantity: float
    price: float
    reason: str = ""
    decision_id: Optional[str] = None


def make_paper_order(
    req: OrderRequest,
    fill_price: float,
    commission: float = 0.0,
    filled_at: Optional[datetime] = None,
) -> PaperOrder:
    """Create a FILLED paper order record (simulation only)."""
    quantity = float(req.quantity)
    notional = quantity * fill_price
    return PaperOrder(
        ticker=req.ticker,
        side=req.side.value,
        quantity=quantity,
        price=fill_price,
        notional=notional,
        commission=commission,
        status=OrderStatus.FILLED.value,
        created_at=filled_at or utcnow().replace(tzinfo=None),
        filled_at=filled_at or utcnow().replace(tzinfo=None),
        reason=req.reason,
        decision_id=req.decision_id,
    )
