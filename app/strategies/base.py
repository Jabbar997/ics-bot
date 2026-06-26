"""Strategy base class and shared helpers.

Each strategy turns a :class:`FeatureSnapshot` + market :class:`RegimeResult`
into a :class:`Signal` with one of: BUY / SELL / HOLD / CASH / REJECT.

Convention used across all strategies:

* **BUY**    — entry conditions fully met.
* **REJECT** — the setup is *close* (core trend present) but a quality gate
  failed (RSI band, volume, etc.). These are surfaced as rejected opportunities.
* **HOLD**   — no setup; stand aside silently (not logged as an opportunity).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from app.domain import (
    Action,
    FeatureSnapshot,
    Regime,
    RegimeResult,
    Signal,
    StrategyName,
)


def clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def rsi_centeredness(rsi: float, low: float, high: float) -> float:
    """1.0 when RSI sits at the centre of [low, high], 0.0 at/over the edges."""
    center = (low + high) / 2.0
    half = (high - low) / 2.0
    if half <= 0:
        return 0.0
    return clamp(1.0 - abs(rsi - center) / half)


class Strategy(ABC):
    name: StrategyName

    @abstractmethod
    def evaluate(self, snapshot: FeatureSnapshot, regime: RegimeResult) -> Signal:
        """Return an entry signal for a single symbol."""

    def should_exit(
        self,
        snapshot: FeatureSnapshot,
        regime: RegimeResult,
        entry_price: float,
        stop_loss: Optional[float],
        target_price: Optional[float],
    ) -> Optional[Signal]:
        """Default exit logic shared by all strategies (stop / target / panic).

        Sub-classes call ``super().should_exit(...)`` first, then add their own
        structure-break rules.
        """
        conds: Dict[str, Any] = {
            "close": snapshot.close,
            "stop_loss": stop_loss,
            "target_price": target_price,
            "regime": regime.regime.value,
        }
        if stop_loss is not None and snapshot.close <= stop_loss:
            return self._exit("تم بلوغ وقف الخسارة.", conds, confidence=0.95)
        if target_price is not None and snapshot.close >= target_price:
            return self._exit("تم بلوغ السعر المستهدف.", conds, confidence=0.9)
        if regime.regime == Regime.PANIC:
            return self._exit("دخل السوق حالة الذعر.", conds, confidence=0.85)
        return None

    # -- helpers ---------------------------------------------------------- #
    def _buy(self, ticker: str, reason: str, conds: Dict[str, Any], confidence: float) -> Signal:
        return Signal(
            ticker=ticker,
            strategy=self.name,
            action=Action.BUY,
            reason=reason,
            confidence=clamp(confidence),
            raw_conditions=conds,
        )

    def _reject(self, ticker: str, reason: str, conds: Dict[str, Any], confidence: float = 0.3) -> Signal:
        return Signal(
            ticker=ticker,
            strategy=self.name,
            action=Action.REJECT,
            reason=reason,
            confidence=clamp(confidence),
            raw_conditions=conds,
        )

    def _hold(self, ticker: str, reason: str, conds: Dict[str, Any]) -> Signal:
        return Signal(
            ticker=ticker,
            strategy=self.name,
            action=Action.HOLD,
            reason=reason,
            confidence=0.0,
            raw_conditions=conds,
        )

    def _exit(self, reason: str, conds: Dict[str, Any], confidence: float) -> Signal:
        return Signal(
            ticker=conds.get("ticker", ""),
            strategy=self.name,
            action=Action.SELL,
            reason=reason,
            confidence=clamp(confidence),
            raw_conditions=conds,
        )
