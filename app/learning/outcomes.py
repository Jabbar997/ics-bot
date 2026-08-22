"""Recording of realised outcomes for closed paper positions.

For every closed :class:`Position` that has no :class:`DecisionOutcome` yet, this
module pairs it with the BUY decision that opened it and stores what actually
happened: realised return, holding period, and the best/worst excursion reached
while the position was held (MFE / MAE).

MFE/MAE need the price path between entry and exit. A ``price_provider`` may be
injected (tests, backtests); otherwise daily bars are fetched lazily and any
failure simply leaves those two fields ``None`` — never blocking the rest.
"""
from __future__ import annotations

from datetime import datetime
from typing import Callable, Dict, List, Optional

from sqlalchemy import select

from app.db.models import Decision, DecisionOutcome, Position
from app.logging_config import get_logger

log = get_logger(__name__)

# (ticker, start, end) -> DataFrame with high/low/close columns
PriceProvider = Callable[[str, datetime, datetime], Optional[object]]


def _find_entry_decision(session, position: Position) -> Optional[Decision]:
    """The BUY decision that opened this position.

    Positions carry no decision_id (adding one would need a migration on the
    live database), so match on ticker + the entry timestamp: the engine records
    the decision and the fill with the same ``ts``.
    """
    stmt = (
        select(Decision)
        .where(Decision.ticker == position.ticker, Decision.action == "buy")
        .order_by(Decision.created_at.desc())
    )
    candidates = list(session.scalars(stmt))
    if not candidates:
        return None
    if position.entry_at is None:
        return candidates[0]
    # Closest decision at or before the fill; fall back to the nearest overall.
    at_or_before = [d for d in candidates if d.created_at and d.created_at <= position.entry_at]
    pool = at_or_before or candidates
    return min(pool, key=lambda d: abs((d.created_at - position.entry_at).total_seconds()))


def _default_price_provider(ticker: str, start: datetime, end: datetime):
    from app.data.market_data import fetch_history

    df = fetch_history(ticker, period="1y")
    if df is None or df.empty:
        return None
    return df.loc[(df.index >= start) & (df.index <= end)]


def _excursions(
    position: Position, provider: PriceProvider
) -> tuple[Optional[float], Optional[float]]:
    """(MFE, MAE) as fractions of the entry price, or (None, None)."""
    if position.entry_at is None or position.exit_at is None:
        return None, None
    entry = float(position.entry_price or 0.0)
    if entry <= 0:
        return None, None
    try:
        window = provider(position.ticker, position.entry_at, position.exit_at)
        if window is None or len(window) == 0:
            return None, None
        high = float(window["high"].max())
        low = float(window["low"].min())
    except Exception:
        log.warning("Could not compute MFE/MAE for %s", position.ticker, exc_info=False)
        return None, None
    return (high / entry - 1.0), (low / entry - 1.0)


def record_outcomes(
    session,
    price_provider: Optional[PriceProvider] = None,
) -> List[DecisionOutcome]:
    """Create a DecisionOutcome for every closed position that lacks one."""
    provider = price_provider or _default_price_provider

    known = set(session.scalars(select(DecisionOutcome.position_id)))
    closed = list(
        session.scalars(select(Position).where(Position.is_open.is_(False)))
    )

    created: List[DecisionOutcome] = []
    for pos in closed:
        if pos.id in known:
            continue

        decision = _find_entry_decision(session, pos)
        components: Optional[Dict] = None
        if decision is not None and decision.audit_log is not None:
            ctx = decision.audit_log.raw_context_json or {}
            dqs_ctx = ctx.get("dqs") if isinstance(ctx, dict) else None
            if isinstance(dqs_ctx, dict):
                components = dqs_ctx.get("components")

        holding_days = None
        if pos.entry_at and pos.exit_at:
            holding_days = max(0, (pos.exit_at - pos.entry_at).days)

        mfe, mae = _excursions(pos, provider)

        outcome = DecisionOutcome(
            decision_id=decision.id if decision is not None else None,
            position_id=pos.id,
            ticker=pos.ticker,
            strategy=pos.strategy,
            entry_at=pos.entry_at,
            exit_at=pos.exit_at,
            realized_return=(
                float(pos.realized_pnl_pct) if pos.realized_pnl_pct is not None else None
            ),
            holding_period_days=holding_days,
            mfe=mfe,
            mae=mae,
            dqs_at_entry=pos.dqs_at_entry,
            dqs_components_json=components,
        )
        session.add(outcome)
        created.append(outcome)

    if created:
        session.flush()
        log.info("Recorded %d new decision outcome(s).", len(created))
    return created


def load_outcomes(session) -> List[DecisionOutcome]:
    """All outcomes that carry both a realised return and DQS components."""
    rows = list(session.scalars(select(DecisionOutcome)))
    return [
        r for r in rows if r.realized_return is not None and r.dqs_components_json
    ]
