"""Strategy engine — runs all strategies over the watchlist and selects the
best candidate signal per symbol.

Selection rule per symbol:
* If any strategy issues a BUY, keep the highest-confidence BUY.
* Else if a strategy issues a REJECT (a near-miss worth logging), keep the
  highest-confidence REJECT.
* Else stand aside (no signal emitted for that symbol).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from app.domain import Action, FeatureSnapshot, RegimeResult, Signal
from app.strategies.base import Strategy
from app.strategies.momentum import MomentumStrategy
from app.strategies.pullback import PullbackStrategy
from app.strategies.trend import TrendStrategy


class StrategyEngine:
    def __init__(self, strategies: Optional[Sequence[Strategy]] = None):
        self.strategies: List[Strategy] = list(strategies) if strategies else [
            TrendStrategy(),
            MomentumStrategy(),
            PullbackStrategy(),
        ]

    def evaluate_symbol(self, snapshot: FeatureSnapshot, regime: RegimeResult) -> Optional[Signal]:
        signals = [strat.evaluate(snapshot, regime) for strat in self.strategies]
        buys = [s for s in signals if s.action == Action.BUY]
        if buys:
            return max(buys, key=lambda s: s.confidence)
        rejects = [s for s in signals if s.action == Action.REJECT]
        if rejects:
            return max(rejects, key=lambda s: s.confidence)
        return None

    def generate_signals(
        self, snapshots: Dict[str, FeatureSnapshot], regime: RegimeResult
    ) -> List[Signal]:
        out: List[Signal] = []
        for symbol, snap in snapshots.items():
            sig = self.evaluate_symbol(snap, regime)
            if sig is not None:
                out.append(sig)
        return out

    def evaluate_exit(
        self,
        strategy_name: str,
        snapshot: FeatureSnapshot,
        regime: RegimeResult,
        entry_price: float,
        stop_loss: Optional[float],
        target_price: Optional[float],
    ) -> Optional[Signal]:
        """Run the owning strategy's exit logic for an open position."""
        strat = next(
            (s for s in self.strategies if s.name.value == strategy_name),
            self.strategies[0],
        )
        return strat.should_exit(snapshot, regime, entry_price, stop_loss, target_price)
