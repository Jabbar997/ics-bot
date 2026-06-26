"""SQLAlchemy 2.x ORM models for ICS.

All money/quantity columns use ``Numeric`` for exactness. UUID primary keys are
stored as 36-char strings so the schema is portable to PostgreSQL unchanged.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import (
    Boolean,
    DateTime,
    ForeignKey,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.types import JSON


def _uuid() -> str:
    return str(uuid.uuid4())


def _utcnow() -> datetime:
    # Naive UTC: stored consistently so the same models work on both SQLite and
    # Postgres (TIMESTAMP WITHOUT TIME ZONE). Values remain UTC, just tz-naive.
    return datetime.now(timezone.utc).replace(tzinfo=None)


class Base(DeclarativeBase):
    pass


# Reusable column types
Money = Numeric(18, 6)
Pct = Numeric(10, 4)


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    telegram_user_id: Mapped[Optional[int]] = mapped_column(Integer, unique=True, nullable=True)
    name: Mapped[Optional[str]] = mapped_column(String(120), nullable=True)
    role: Mapped[str] = mapped_column(String(40), default="analyst")  # analyst / trainee
    is_authorized: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)


class WatchlistSymbol(Base):
    __tablename__ = "watchlist_symbols"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    symbol: Mapped[str] = mapped_column(String(20), unique=True, index=True)
    asset_type: Mapped[str] = mapped_column(String(20), default="stock")  # stock / etf
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)


class MarketPrice(Base):
    __tablename__ = "market_prices"
    __table_args__ = (UniqueConstraint("symbol", "ts", name="uq_symbol_ts"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    symbol: Mapped[str] = mapped_column(String(20), index=True)
    ts: Mapped[datetime] = mapped_column(DateTime, index=True)
    open: Mapped[float] = mapped_column(Money)
    high: Mapped[float] = mapped_column(Money)
    low: Mapped[float] = mapped_column(Money)
    close: Mapped[float] = mapped_column(Money)
    adjusted_close: Mapped[Optional[float]] = mapped_column(Money, nullable=True)
    volume: Mapped[Optional[float]] = mapped_column(Money, nullable=True)


class FeatureSnapshot(Base):
    __tablename__ = "feature_snapshots"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    symbol: Mapped[str] = mapped_column(String(20), index=True)
    ts: Mapped[datetime] = mapped_column(DateTime, index=True)
    close: Mapped[float] = mapped_column(Money)
    ma20: Mapped[Optional[float]] = mapped_column(Money, nullable=True)
    ma50: Mapped[Optional[float]] = mapped_column(Money, nullable=True)
    ma200: Mapped[Optional[float]] = mapped_column(Money, nullable=True)
    rsi14: Mapped[Optional[float]] = mapped_column(Pct, nullable=True)
    macd: Mapped[Optional[float]] = mapped_column(Money, nullable=True)
    macd_signal: Mapped[Optional[float]] = mapped_column(Money, nullable=True)
    atr14: Mapped[Optional[float]] = mapped_column(Money, nullable=True)
    volume_ma20: Mapped[Optional[float]] = mapped_column(Money, nullable=True)
    high_20d: Mapped[Optional[float]] = mapped_column(Money, nullable=True)
    drawdown_20d: Mapped[Optional[float]] = mapped_column(Pct, nullable=True)
    volatility_20d: Mapped[Optional[float]] = mapped_column(Pct, nullable=True)
    beta_vs_spy: Mapped[Optional[float]] = mapped_column(Pct, nullable=True)


class MarketRegime(Base):
    __tablename__ = "market_regimes"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    ts: Mapped[datetime] = mapped_column(DateTime, index=True, default=_utcnow)
    regime: Mapped[str] = mapped_column(String(20))  # bull/weak_bull/sideways/bear/panic
    spy_close: Mapped[Optional[float]] = mapped_column(Money, nullable=True)
    spy_ma50: Mapped[Optional[float]] = mapped_column(Money, nullable=True)
    spy_ma200: Mapped[Optional[float]] = mapped_column(Money, nullable=True)
    drawdown_20d: Mapped[Optional[float]] = mapped_column(Pct, nullable=True)
    reason: Mapped[Optional[str]] = mapped_column(Text, nullable=True)


class Decision(Base):
    __tablename__ = "decisions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, index=True)
    ticker: Mapped[str] = mapped_column(String(20), index=True)
    action: Mapped[str] = mapped_column(String(10))  # buy/sell/hold/cash/reject
    strategy: Mapped[str] = mapped_column(String(20))
    price: Mapped[float] = mapped_column(Money)
    quantity: Mapped[Optional[float]] = mapped_column(Money, nullable=True)
    position_size_pct: Mapped[Optional[float]] = mapped_column(Pct, nullable=True)
    dqs_score: Mapped[int] = mapped_column(Integer, default=0)
    confidence: Mapped[Optional[float]] = mapped_column(Pct, nullable=True)
    reason: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    market_regime: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    stop_loss: Mapped[Optional[float]] = mapped_column(Money, nullable=True)
    target_price: Mapped[Optional[float]] = mapped_column(Money, nullable=True)
    rule_violated: Mapped[bool] = mapped_column(Boolean, default=False)
    violation_details: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    rejected_opportunity: Mapped[bool] = mapped_column(Boolean, default=False)
    rejection_reason: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    audit_log: Mapped[Optional["AuditLog"]] = relationship(
        back_populates="decision", uselist=False
    )


class AuditLog(Base):
    """Immutable record. Every Decision MUST have exactly one AuditLog."""

    __tablename__ = "audit_logs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, index=True)
    decision_id: Mapped[str] = mapped_column(
        ForeignKey("decisions.id"), index=True, nullable=False
    )
    ticker: Mapped[str] = mapped_column(String(20), index=True)
    action: Mapped[str] = mapped_column(String(10))
    strategy: Mapped[str] = mapped_column(String(20))
    price: Mapped[float] = mapped_column(Money)
    quantity: Mapped[Optional[float]] = mapped_column(Money, nullable=True)
    position_size_pct: Mapped[Optional[float]] = mapped_column(Pct, nullable=True)
    dqs_score: Mapped[int] = mapped_column(Integer, default=0)
    confidence: Mapped[Optional[float]] = mapped_column(Pct, nullable=True)
    reason: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    market_regime: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    stop_loss: Mapped[Optional[float]] = mapped_column(Money, nullable=True)
    target_price: Mapped[Optional[float]] = mapped_column(Money, nullable=True)
    rule_violated: Mapped[bool] = mapped_column(Boolean, default=False)
    violation_details: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    rejected_opportunity: Mapped[bool] = mapped_column(Boolean, default=False)
    rejection_reason: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    result_usd: Mapped[Optional[float]] = mapped_column(Money, nullable=True)
    result_pct: Mapped[Optional[float]] = mapped_column(Pct, nullable=True)
    learning_note: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    raw_context_json: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

    decision: Mapped["Decision"] = relationship(back_populates="audit_log")


class PaperOrder(Base):
    __tablename__ = "paper_orders"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    ticker: Mapped[str] = mapped_column(String(20), index=True)
    side: Mapped[str] = mapped_column(String(4))  # buy / sell
    quantity: Mapped[float] = mapped_column(Money)
    price: Mapped[float] = mapped_column(Money)
    notional: Mapped[float] = mapped_column(Money)
    commission: Mapped[float] = mapped_column(Money, default=0)
    status: Mapped[str] = mapped_column(String(12), default="pending")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
    filled_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    reason: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    decision_id: Mapped[Optional[str]] = mapped_column(
        ForeignKey("decisions.id"), nullable=True
    )


class Position(Base):
    __tablename__ = "positions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    ticker: Mapped[str] = mapped_column(String(20), index=True)
    strategy: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    quantity: Mapped[float] = mapped_column(Money)
    entry_price: Mapped[float] = mapped_column(Money)
    entry_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
    current_price: Mapped[Optional[float]] = mapped_column(Money, nullable=True)
    stop_loss: Mapped[Optional[float]] = mapped_column(Money, nullable=True)
    target_price: Mapped[Optional[float]] = mapped_column(Money, nullable=True)
    dqs_at_entry: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    is_open: Mapped[bool] = mapped_column(Boolean, default=True, index=True)
    exit_price: Mapped[Optional[float]] = mapped_column(Money, nullable=True)
    exit_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    realized_pnl: Mapped[Optional[float]] = mapped_column(Money, nullable=True)
    realized_pnl_pct: Mapped[Optional[float]] = mapped_column(Pct, nullable=True)
    exit_reason: Mapped[Optional[str]] = mapped_column(Text, nullable=True)


class PortfolioSnapshot(Base):
    __tablename__ = "portfolio_snapshots"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    ts: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, index=True)
    cash: Mapped[float] = mapped_column(Money)
    invested_value: Mapped[float] = mapped_column(Money)
    total_value: Mapped[float] = mapped_column(Money)
    open_positions: Mapped[int] = mapped_column(Integer, default=0)
    daily_pnl: Mapped[Optional[float]] = mapped_column(Money, nullable=True)
    daily_pnl_pct: Mapped[Optional[float]] = mapped_column(Pct, nullable=True)


class PerformanceReport(Base):
    __tablename__ = "performance_reports"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, index=True)
    period: Mapped[str] = mapped_column(String(20))  # daily / weekly / backtest
    period_start: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    period_end: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    total_return: Mapped[Optional[float]] = mapped_column(Pct, nullable=True)
    invested_return: Mapped[Optional[float]] = mapped_column(Pct, nullable=True)
    spy_return: Mapped[Optional[float]] = mapped_column(Pct, nullable=True)
    sharpe_ratio: Mapped[Optional[float]] = mapped_column(Pct, nullable=True)
    max_drawdown: Mapped[Optional[float]] = mapped_column(Pct, nullable=True)
    win_rate: Mapped[Optional[float]] = mapped_column(Pct, nullable=True)
    average_win: Mapped[Optional[float]] = mapped_column(Pct, nullable=True)
    average_loss: Mapped[Optional[float]] = mapped_column(Pct, nullable=True)
    expectancy: Mapped[Optional[float]] = mapped_column(Pct, nullable=True)
    rule_compliance: Mapped[Optional[float]] = mapped_column(Pct, nullable=True)
    average_dqs: Mapped[Optional[float]] = mapped_column(Pct, nullable=True)
    rejected_opportunities: Mapped[int] = mapped_column(Integer, default=0)
    best_decision: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    worst_decision: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    recommendation: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    raw_json: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)


class KillSwitchEvent(Base):
    __tablename__ = "kill_switch_events"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    ts: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, index=True)
    level: Mapped[int] = mapped_column(Integer)  # 0..4 (see domain.KillSwitchLevel)
    active: Mapped[bool] = mapped_column(Boolean, default=True, index=True)
    trigger: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    action_taken: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    cooldown_until: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    resolved_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)


class SystemConfig(Base):
    """Key/value store for runtime flags (e.g. system frozen, kill-switch state)."""

    __tablename__ = "system_config"

    key: Mapped[str] = mapped_column(String(60), primary_key=True)
    value: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, onupdate=_utcnow)
