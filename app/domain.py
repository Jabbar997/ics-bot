"""Core domain types shared across the ICS pipeline.

Centralising the enums and dataclasses here keeps the pipeline modules
(strategies → DQS → risk → paper → audit) decoupled and avoids circular
imports. These are pure value objects with no I/O.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional


# --------------------------------------------------------------------------- #
# Enums
# --------------------------------------------------------------------------- #
class Action(str, Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    CASH = "cash"
    REJECT = "reject"


class StrategyName(str, Enum):
    TREND = "trend"
    MOMENTUM = "momentum"
    PULLBACK = "pullback"
    DEFENSIVE_CASH = "defensive_cash"


class Regime(str, Enum):
    BULL = "bull"
    WEAK_BULL = "weak_bull"
    SIDEWAYS = "sideways"
    BEAR = "bear"
    PANIC = "panic"


class KillSwitchLevel(int, Enum):
    NONE = 0
    LEVEL_1 = 1  # Warning  — stop new entries
    LEVEL_2 = 2  # Freeze   — close 50% of positions, stop new entries
    LEVEL_3 = 3  # Full stop — close all positions, freeze system
    MANUAL = 4   # Manual STOP — immediate freeze, no real positions to close


class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderStatus(str, Enum):
    PENDING = "pending"
    FILLED = "filled"
    REJECTED = "rejected"
    CANCELLED = "cancelled"


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


# --------------------------------------------------------------------------- #
# Value objects
# --------------------------------------------------------------------------- #
@dataclass
class FeatureSnapshot:
    """All engineered features for one symbol at one point in time."""

    ticker: str
    as_of: datetime
    close: float
    ma20: Optional[float] = None
    ma50: Optional[float] = None
    ma200: Optional[float] = None
    rsi14: Optional[float] = None
    macd: Optional[float] = None
    macd_signal: Optional[float] = None
    atr14: Optional[float] = None
    volume: Optional[float] = None
    volume_ma20: Optional[float] = None
    high_20d: Optional[float] = None
    drawdown_20d: Optional[float] = None  # percent, negative below the high
    volatility_20d: Optional[float] = None  # annualised stdev of daily returns
    beta_vs_spy: Optional[float] = None

    def is_complete(self) -> bool:
        """True only if the indicators required for decisions are present."""
        required = [self.ma50, self.ma200, self.rsi14, self.atr14, self.volume_ma20]
        return self.close is not None and all(v is not None for v in required)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ticker": self.ticker,
            "as_of": self.as_of.isoformat() if self.as_of else None,
            "close": self.close,
            "ma20": self.ma20,
            "ma50": self.ma50,
            "ma200": self.ma200,
            "rsi14": self.rsi14,
            "macd": self.macd,
            "macd_signal": self.macd_signal,
            "atr14": self.atr14,
            "volume": self.volume,
            "volume_ma20": self.volume_ma20,
            "high_20d": self.high_20d,
            "drawdown_20d": self.drawdown_20d,
            "volatility_20d": self.volatility_20d,
            "beta_vs_spy": self.beta_vs_spy,
        }


@dataclass
class RegimeResult:
    regime: Regime
    spy_close: float
    spy_ma50: Optional[float]
    spy_ma200: Optional[float]
    drawdown_20d: float
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "regime": self.regime.value,
            "spy_close": self.spy_close,
            "spy_ma50": self.spy_ma50,
            "spy_ma200": self.spy_ma200,
            "drawdown_20d": self.drawdown_20d,
            "reason": self.reason,
        }


@dataclass
class Signal:
    """A raw strategy signal, before scoring and risk validation."""

    ticker: str
    strategy: StrategyName
    action: Action
    reason: str
    confidence: float  # 0.0 - 1.0
    raw_conditions: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ticker": self.ticker,
            "strategy": self.strategy.value,
            "action": self.action.value,
            "reason": self.reason,
            "confidence": self.confidence,
            "raw_conditions": self.raw_conditions,
        }


@dataclass
class DQSResult:
    """Decision Quality Score result."""

    score: int
    components: Dict[str, int]
    allowed: bool
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "score": self.score,
            "components": self.components,
            "allowed": self.allowed,
            "reason": self.reason,
        }


@dataclass
class RiskDecision:
    """Result of risk validation for a candidate order."""

    allowed: bool
    reason: str
    violations: List[str] = field(default_factory=list)
    quantity: Optional[float] = None
    position_size_pct: Optional[float] = None
    notional: Optional[float] = None
    stop_loss: Optional[float] = None
    target_price: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "allowed": self.allowed,
            "reason": self.reason,
            "violations": self.violations,
            "quantity": self.quantity,
            "position_size_pct": self.position_size_pct,
            "notional": self.notional,
            "stop_loss": self.stop_loss,
            "target_price": self.target_price,
        }
