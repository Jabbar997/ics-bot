"""Repository layer — the only abstraction the app uses to touch the database.

Keeping all queries here means a future migration to PostgreSQL (or swapping the
ORM) touches one layer, not the whole codebase.

The most important guarantee lives here: :meth:`DecisionRepository.record` writes
a Decision *and* its AuditLog in the same transaction. Per the spec, a decision
without an audit log is invalid, so the two are never created separately.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import (
    AuditLog,
    Decision,
    KillSwitchEvent,
    MarketPrice,
    MarketRegime,
    PaperOrder,
    PerformanceReport,
    Position,
    PortfolioSnapshot,
    SystemConfig,
    User,
    WatchlistSymbol,
)


class DecisionRepository:
    def __init__(self, session: Session):
        self.session = session

    def record(
        self,
        *,
        ticker: str,
        action: str,
        strategy: str,
        price: float,
        dqs_score: int,
        confidence: Optional[float] = None,
        reason: Optional[str] = None,
        market_regime: Optional[str] = None,
        quantity: Optional[float] = None,
        position_size_pct: Optional[float] = None,
        stop_loss: Optional[float] = None,
        target_price: Optional[float] = None,
        rule_violated: bool = False,
        violation_details: Optional[str] = None,
        rejected_opportunity: bool = False,
        rejection_reason: Optional[str] = None,
        result_usd: Optional[float] = None,
        result_pct: Optional[float] = None,
        learning_note: Optional[str] = None,
        raw_context: Optional[Dict[str, Any]] = None,
        created_at: Optional[datetime] = None,
    ) -> Decision:
        """Create a Decision and its mandatory AuditLog atomically."""
        decision = Decision(
            ticker=ticker,
            action=action,
            strategy=strategy,
            price=price,
            quantity=quantity,
            position_size_pct=position_size_pct,
            dqs_score=dqs_score,
            confidence=confidence,
            reason=reason,
            market_regime=market_regime,
            stop_loss=stop_loss,
            target_price=target_price,
            rule_violated=rule_violated,
            violation_details=violation_details,
            rejected_opportunity=rejected_opportunity,
            rejection_reason=rejection_reason,
        )
        if created_at is not None:
            decision.created_at = created_at
        self.session.add(decision)
        self.session.flush()  # assign decision.id

        audit = AuditLog(
            decision_id=decision.id,
            timestamp=created_at or decision.created_at,
            ticker=ticker,
            action=action,
            strategy=strategy,
            price=price,
            quantity=quantity,
            position_size_pct=position_size_pct,
            dqs_score=dqs_score,
            confidence=confidence,
            reason=reason,
            market_regime=market_regime,
            stop_loss=stop_loss,
            target_price=target_price,
            rule_violated=rule_violated,
            violation_details=violation_details,
            rejected_opportunity=rejected_opportunity,
            rejection_reason=rejection_reason,
            result_usd=result_usd,
            result_pct=result_pct,
            learning_note=learning_note,
            raw_context_json=raw_context or {},
        )
        self.session.add(audit)
        self.session.flush()
        return decision

    def for_day(self, day: datetime) -> List[Decision]:
        start = day.replace(hour=0, minute=0, second=0, microsecond=0)
        end = start.replace(hour=23, minute=59, second=59)
        stmt = (
            select(Decision)
            .where(Decision.created_at >= start, Decision.created_at <= end)
            .order_by(Decision.created_at)
        )
        return list(self.session.scalars(stmt))

    def rejected_opportunities(self, limit: int = 50) -> List[Decision]:
        stmt = (
            select(Decision)
            .where(Decision.rejected_opportunity.is_(True))
            .order_by(Decision.created_at.desc())
            .limit(limit)
        )
        return list(self.session.scalars(stmt))


class AuditRepository:
    def __init__(self, session: Session):
        self.session = session

    def recent(self, limit: int = 20) -> List[AuditLog]:
        stmt = select(AuditLog).order_by(AuditLog.timestamp.desc()).limit(limit)
        return list(self.session.scalars(stmt))

    def count(self) -> int:
        return self.session.query(AuditLog).count()


class PositionRepository:
    def __init__(self, session: Session):
        self.session = session

    def open_positions(self) -> List[Position]:
        stmt = select(Position).where(Position.is_open.is_(True))
        return list(self.session.scalars(stmt))

    def get_open_by_ticker(self, ticker: str) -> Optional[Position]:
        stmt = select(Position).where(
            Position.ticker == ticker, Position.is_open.is_(True)
        )
        return self.session.scalars(stmt).first()

    def closed_positions(self) -> List[Position]:
        stmt = select(Position).where(Position.is_open.is_(False))
        return list(self.session.scalars(stmt))

    def add(self, position: Position) -> Position:
        self.session.add(position)
        self.session.flush()
        return position


class OrderRepository:
    def __init__(self, session: Session):
        self.session = session

    def add(self, order: PaperOrder) -> PaperOrder:
        self.session.add(order)
        self.session.flush()
        return order

    def all(self) -> List[PaperOrder]:
        return list(self.session.scalars(select(PaperOrder).order_by(PaperOrder.created_at)))


class PortfolioSnapshotRepository:
    def __init__(self, session: Session):
        self.session = session

    def add(self, snapshot: PortfolioSnapshot) -> PortfolioSnapshot:
        self.session.add(snapshot)
        self.session.flush()
        return snapshot

    def latest(self) -> Optional[PortfolioSnapshot]:
        stmt = select(PortfolioSnapshot).order_by(PortfolioSnapshot.ts.desc()).limit(1)
        return self.session.scalars(stmt).first()

    def since(self, start: datetime) -> List[PortfolioSnapshot]:
        stmt = (
            select(PortfolioSnapshot)
            .where(PortfolioSnapshot.ts >= start)
            .order_by(PortfolioSnapshot.ts)
        )
        return list(self.session.scalars(stmt))

    def all(self) -> List[PortfolioSnapshot]:
        return list(self.session.scalars(select(PortfolioSnapshot).order_by(PortfolioSnapshot.ts)))


class MarketPriceRepository:
    def __init__(self, session: Session):
        self.session = session

    def upsert_bar(self, **kwargs) -> None:
        symbol = kwargs["symbol"]
        ts = kwargs["ts"]
        existing = self.session.scalars(
            select(MarketPrice).where(MarketPrice.symbol == symbol, MarketPrice.ts == ts)
        ).first()
        if existing:
            for k, v in kwargs.items():
                setattr(existing, k, v)
        else:
            self.session.add(MarketPrice(**kwargs))


class RegimeRepository:
    def __init__(self, session: Session):
        self.session = session

    def add(self, regime: MarketRegime) -> MarketRegime:
        self.session.add(regime)
        self.session.flush()
        return regime

    def latest(self) -> Optional[MarketRegime]:
        stmt = select(MarketRegime).order_by(MarketRegime.ts.desc()).limit(1)
        return self.session.scalars(stmt).first()


class KillSwitchRepository:
    def __init__(self, session: Session):
        self.session = session

    def add_event(self, event: KillSwitchEvent) -> KillSwitchEvent:
        self.session.add(event)
        self.session.flush()
        return event

    def active_event(self) -> Optional[KillSwitchEvent]:
        stmt = (
            select(KillSwitchEvent)
            .where(KillSwitchEvent.active.is_(True))
            .order_by(KillSwitchEvent.level.desc(), KillSwitchEvent.ts.desc())
            .limit(1)
        )
        return self.session.scalars(stmt).first()

    def deactivate_all(self) -> int:
        events = list(
            self.session.scalars(
                select(KillSwitchEvent).where(KillSwitchEvent.active.is_(True))
            )
        )
        for e in events:
            e.active = False
        return len(events)


class SystemConfigRepository:
    def __init__(self, session: Session):
        self.session = session

    def get(self, key: str, default: Optional[str] = None) -> Optional[str]:
        row = self.session.get(SystemConfig, key)
        return row.value if row else default

    def set(self, key: str, value: str) -> None:
        row = self.session.get(SystemConfig, key)
        if row:
            row.value = value
        else:
            self.session.add(SystemConfig(key=key, value=value))

    def get_bool(self, key: str, default: bool = False) -> bool:
        val = self.get(key)
        if val is None:
            return default
        return val.lower() in ("1", "true", "yes", "on")


class PerformanceReportRepository:
    def __init__(self, session: Session):
        self.session = session

    def add(self, report: PerformanceReport) -> PerformanceReport:
        self.session.add(report)
        self.session.flush()
        return report

    def latest(self, period: Optional[str] = None) -> Optional[PerformanceReport]:
        stmt = select(PerformanceReport)
        if period:
            stmt = stmt.where(PerformanceReport.period == period)
        stmt = stmt.order_by(PerformanceReport.created_at.desc()).limit(1)
        return self.session.scalars(stmt).first()


class WatchlistRepository:
    def __init__(self, session: Session):
        self.session = session

    def sync(self, symbols: List[str]) -> None:
        existing = {w.symbol for w in self.session.scalars(select(WatchlistSymbol))}
        for sym in symbols:
            if sym not in existing:
                self.session.add(WatchlistSymbol(symbol=sym))

    def all_symbols(self) -> List[str]:
        return [w.symbol for w in self.session.scalars(select(WatchlistSymbol))]


class UserRepository:
    def __init__(self, session: Session):
        self.session = session

    def authorize(self, telegram_user_id: int, name: Optional[str] = None) -> User:
        user = self.session.scalars(
            select(User).where(User.telegram_user_id == telegram_user_id)
        ).first()
        if not user:
            user = User(telegram_user_id=telegram_user_id, name=name, is_authorized=True)
            self.session.add(user)
        else:
            user.is_authorized = True
        self.session.flush()
        return user
