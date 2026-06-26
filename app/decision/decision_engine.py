"""Decision Engine — the orchestration spine.

For each candidate signal it runs the canonical sequence:

    signal -> DQS -> (reject if < min) -> Risk Manager -> (reject if blocked)
           -> paper execution -> Decision + AuditLog

Every path records a Decision with its mandatory AuditLog (including rejected
opportunities), so nothing the system considers is ever lost.
"""
from __future__ import annotations

from datetime import datetime
from typing import List, Optional, Sequence

from app.config import Config
from app.db.repositories import DecisionRepository
from app.decision.dqs import calculate_dqs
from app.domain import (
    Action,
    DQSResult,
    FeatureSnapshot,
    RegimeResult,
    Signal,
)
from app.logging_config import get_logger
from app.paper.broker import PaperBroker
from app.risk.risk_manager import (
    OpenPositionView,
    PortfolioState,
    build_risk_context,
    validate_order,
)

log = get_logger(__name__)


class DecisionEngine:
    def __init__(self, session, config: Config, broker: PaperBroker, kill_switch_active: bool = False):
        self.session = session
        self.config = config
        self.broker = broker
        self.kill_switch_active = kill_switch_active
        self.decisions = DecisionRepository(session)

    def _raw_context(self, signal, features, regime, dqs, risk) -> dict:
        return {
            "signal": signal.to_dict() if signal else None,
            "features": features.to_dict() if features else None,
            "regime": regime.to_dict() if regime else None,
            "dqs": dqs.to_dict() if dqs else None,
            "risk": risk.to_dict() if risk else None,
            "kill_switch_active": self.kill_switch_active,
        }

    def process_signal(
        self,
        signal: Signal,
        features: FeatureSnapshot,
        regime: RegimeResult,
        portfolio_state: PortfolioState,
        open_positions: Sequence[OpenPositionView],
        ts: Optional[datetime] = None,
    ):
        """Run the full pipeline for one BUY/REJECT candidate signal."""
        # Compute DQS for every candidate (BUY and near-miss REJECT alike).
        risk_ctx = build_risk_context(signal, features, portfolio_state, self.config)
        dqs = calculate_dqs(
            signal, features, regime, risk_ctx, minimum_dqs=self.config.risk.minimum_dqs
        )

        # Strategy explicitly stood aside but flagged a near-miss: log as rejected.
        if signal.action == Action.REJECT:
            return self.decisions.record(
                ticker=signal.ticker,
                action=Action.REJECT.value,
                strategy=signal.strategy.value,
                price=features.close,
                dqs_score=dqs.score,
                confidence=signal.confidence,
                reason=signal.reason,
                market_regime=regime.regime.value,
                rejected_opportunity=True,
                rejection_reason=signal.reason,
                raw_context=self._raw_context(signal, features, regime, dqs, None),
                created_at=ts,
            )

        # DQS gate.
        if not dqs.allowed:
            return self.decisions.record(
                ticker=signal.ticker,
                action=Action.REJECT.value,
                strategy=signal.strategy.value,
                price=features.close,
                dqs_score=dqs.score,
                confidence=signal.confidence,
                reason=dqs.reason,
                market_regime=regime.regime.value,
                rejected_opportunity=True,
                rejection_reason=f"DQS {dqs.score} < {self.config.risk.minimum_dqs}.",
                raw_context=self._raw_context(signal, features, regime, dqs, None),
                created_at=ts,
            )

        # Risk gate.
        risk = validate_order(
            signal,
            dqs,
            portfolio_state,
            open_positions,
            self.config,
            features=features,
            kill_switch_active=self.kill_switch_active,
        )
        if not risk.allowed:
            return self.decisions.record(
                ticker=signal.ticker,
                action=Action.REJECT.value,
                strategy=signal.strategy.value,
                price=features.close,
                dqs_score=dqs.score,
                confidence=signal.confidence,
                reason=risk.reason,
                market_regime=regime.regime.value,
                rule_violated=True,
                violation_details=", ".join(risk.violations),
                rejected_opportunity=True,
                rejection_reason=risk.reason,
                raw_context=self._raw_context(signal, features, regime, dqs, risk),
                created_at=ts,
            )

        # Approved — execute the paper buy, then record the decision.
        decision = self.decisions.record(
            ticker=signal.ticker,
            action=Action.BUY.value,
            strategy=signal.strategy.value,
            price=features.close,
            quantity=risk.quantity,
            position_size_pct=risk.position_size_pct,
            dqs_score=dqs.score,
            confidence=signal.confidence,
            reason=signal.reason,
            market_regime=regime.regime.value,
            stop_loss=risk.stop_loss,
            target_price=risk.target_price,
            raw_context=self._raw_context(signal, features, regime, dqs, risk),
            created_at=ts,
        )
        self.broker.place_buy_order(
            ticker=signal.ticker,
            quantity=risk.quantity,
            price=features.close,
            strategy=signal.strategy.value,
            reason=signal.reason,
            decision_id=decision.id,
            stop_loss=risk.stop_loss,
            target_price=risk.target_price,
            dqs=dqs.score,
            ts=ts,
        )
        return decision

    def record_exit(
        self,
        position,
        exit_signal: Signal,
        features: FeatureSnapshot,
        regime: RegimeResult,
        exit_price: float,
        ts: Optional[datetime] = None,
    ):
        """Close a paper position and record the SELL decision + audit."""
        order = self.broker.close_position(position, exit_price, reason=exit_signal.reason, ts=ts)
        result_usd = position.realized_pnl
        result_pct = position.realized_pnl_pct
        return self.decisions.record(
            ticker=position.ticker,
            action=Action.SELL.value,
            strategy=position.strategy or exit_signal.strategy.value,
            price=exit_price,
            quantity=float(position.quantity),
            dqs_score=position.dqs_at_entry or 0,
            confidence=exit_signal.confidence,
            reason=exit_signal.reason,
            market_regime=regime.regime.value,
            stop_loss=position.stop_loss,
            target_price=position.target_price,
            result_usd=result_usd,
            result_pct=result_pct,
            learning_note=exit_signal.reason,
            raw_context=self._raw_context(exit_signal, features, regime, None, None),
            created_at=ts,
        )
