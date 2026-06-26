"""Backtester — replays history through the *exact* live pipeline.

No look-ahead:
* Indicators are backward-looking only (MA/RSI/ATR use data up to day *t*).
* Signals are generated from day *t* (after the close).
* Orders fill at the **next trading day's open** (``i+1``) — and only if that
  bar exists. This is the single, documented execution convention.

The backtest runs against an in-memory SQLite database using the same
PaperBroker + DecisionEngine as live paper trading, so it produces real
decisions, orders, positions, portfolio snapshots and audit logs.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd

from app.config import Config
from app.db import database
from app.db.repositories import (
    AuditRepository,
    PortfolioSnapshotRepository,
    PositionRepository,
)
from app.decision.decision_engine import DecisionEngine
from app.domain import RegimeResult
from app.features.feature_engine import build_feature_snapshot, compute_features
from app.logging_config import get_logger
from app.market.regime import analyze_from_features_df
from app.paper.broker import PaperBroker
from app.performance import evaluator
from app.performance.benchmarks import period_return
from app.risk.risk_manager import OpenPositionView, PortfolioState
from app.strategies.engine import StrategyEngine

log = get_logger(__name__)

MIN_TRADES_FOR_QUALIFICATION = 100
WARMUP_BARS = 200  # need MA200 before any decision


@dataclass
class BacktestResult:
    performance: evaluator.PerformanceResult
    total_decisions: int = 0
    buys: int = 0
    rejected_opportunities: int = 0
    closed_trades: int = 0
    audit_logs: int = 0
    qualified: bool = False
    equity_curve: List[float] = field(default_factory=list)
    start: Optional[datetime] = None
    end: Optional[datetime] = None
    benchmark_symbol: str = "SPY"

    def summary(self) -> str:
        p = self.performance
        return (
            f"Backtest {self.start} -> {self.end}\n"
            f"  Decisions: {self.total_decisions} (buys {self.buys}, "
            f"rejected {self.rejected_opportunities})\n"
            f"  Closed trades: {self.closed_trades} (qualified>=100: {self.qualified})\n"
            f"  Total return: {p.total_return*100:.2f}%  SPY: {p.spy_return*100:.2f}%\n"
            f"  Sharpe: {p.sharpe_ratio:.2f}  MaxDD: {p.max_drawdown*100:.2f}%  "
            f"WinRate: {p.win_rate*100:.1f}%  AvgDQS: {p.average_dqs}\n"
            f"  Audit logs: {self.audit_logs}"
        )


class Backtester:
    def __init__(self, config: Config, strategy_engine: Optional[StrategyEngine] = None):
        self.config = config
        self.engine = strategy_engine or StrategyEngine()
        self.benchmark = config.benchmark.symbol

    def run(
        self,
        data: Dict[str, pd.DataFrame],
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        database_url: str = "sqlite:///:memory:",
    ) -> BacktestResult:
        if self.benchmark not in data:
            raise ValueError(f"Benchmark {self.benchmark} not present in data.")

        # Compute features once per symbol (backward-looking only).
        spy_close = data[self.benchmark]["close"]
        features: Dict[str, pd.DataFrame] = {}
        for sym, df in data.items():
            if df is None or df.empty:
                continue
            features[sym] = compute_features(df, spy_close=spy_close)

        spy_feat = features[self.benchmark]
        calendar = spy_feat.index
        if start is not None:
            calendar = calendar[calendar >= pd.Timestamp(start)]
        if end is not None:
            calendar = calendar[calendar <= pd.Timestamp(end)]

        # Fresh, isolated DB for this backtest run.
        database.init_engine(database_url)
        database.create_all()
        session = database.new_session()
        try:
            return self._simulate(session, features, spy_feat, calendar)
        finally:
            session.commit()
            session.close()

    # ------------------------------------------------------------------ #
    def _portfolio_state(self, broker: PaperBroker, snapshots_repo) -> PortfolioState:
        total = float(broker.calculate_total_value({}))
        cash = float(broker.calculate_cash())
        snaps = snapshots_repo.all()
        weekly = self._trailing_return(snaps, 5)
        monthly = self._trailing_return(snaps, 21)
        dd = self._current_drawdown([float(s.total_value) for s in snaps] + [total])
        return PortfolioState(
            total_value=total,
            cash=cash,
            weekly_return_pct=weekly,
            monthly_return_pct=monthly,
            current_drawdown_pct=dd,
        )

    @staticmethod
    def _trailing_return(snaps, lookback: int) -> float:
        if len(snaps) <= lookback:
            return 0.0
        prev = float(snaps[-lookback].total_value)
        cur = float(snaps[-1].total_value)
        return (cur / prev - 1.0) if prev else 0.0

    @staticmethod
    def _current_drawdown(values: List[float]) -> float:
        if not values:
            return 0.0
        peak = max(values)
        return (values[-1] / peak - 1.0) if peak else 0.0

    def _price_on(self, feat: pd.DataFrame, ts, col: str) -> Optional[float]:
        try:
            val = feat.at[ts, col]
        except (KeyError, IndexError):
            return None
        if pd.isna(val):
            return None
        return float(val)

    def _simulate(self, session, features, spy_feat, calendar) -> BacktestResult:
        broker = PaperBroker(session, self.config)
        snaps_repo = PortfolioSnapshotRepository(session)
        positions_repo = PositionRepository(session)

        tradeable = [s for s in features if s != self.benchmark] or list(features)
        n = len(calendar)
        result_start = calendar[0] if n else None
        result_end = calendar[-1] if n else None

        for i in range(WARMUP_BARS, n - 1):
            day = calendar[i]
            next_day = calendar[i + 1]

            regime = analyze_from_features_df(spy_feat.loc[:day])
            ks_active = self._defensive(regime)
            decision_engine = DecisionEngine(session, self.config, broker, kill_switch_active=False)

            # 1) Exits first (decision on close of `day`, fill next open).
            for pos in list(positions_repo.open_positions()):
                if pos.ticker not in features:
                    continue
                feat = features[pos.ticker]
                if day not in feat.index:
                    continue
                snap = build_feature_snapshot(pos.ticker, feat.loc[:day])
                exit_sig = self.engine.evaluate_exit(
                    pos.strategy or "trend", snap, regime, pos.entry_price, pos.stop_loss, pos.target_price
                )
                if exit_sig is not None:
                    fill = self._price_on(feat, next_day, "open")
                    if fill is None:
                        continue
                    decision_engine.record_exit(pos, exit_sig, snap, regime, fill, ts=next_day)

            # 2) Entries (skip when defensive regime suppresses new risk).
            if not ks_active:
                snapshots = {}
                for sym in tradeable:
                    feat = features.get(sym)
                    if feat is None or day not in feat.index:
                        continue
                    snap = build_feature_snapshot(sym, feat.loc[:day])
                    if snap.is_complete():
                        snapshots[sym] = snap

                signals = self.engine.generate_signals(snapshots, regime)
                open_views = [OpenPositionView(p.ticker, float(p.quantity)) for p in positions_repo.open_positions()]
                pstate = self._portfolio_state(broker, snaps_repo)

                for sig in signals:
                    feat = features[sig.ticker]
                    # Fill the entry at next open; require the bar to exist.
                    fill = self._price_on(feat, next_day, "open")
                    snap = snapshots[sig.ticker]
                    if sig.action.value == "buy" and fill is None:
                        continue
                    # Decision uses day `i` features; execution price = next open.
                    if sig.action.value == "buy":
                        snap_exec = build_feature_snapshot(sig.ticker, feat.loc[:day])
                        snap_exec.close = fill  # execute/size at the realistic fill
                    else:
                        snap_exec = snap
                    decision_engine.process_signal(sig, snap_exec, regime, pstate, open_views, ts=next_day)
                    open_views = [OpenPositionView(p.ticker, float(p.quantity)) for p in positions_repo.open_positions()]
                    pstate = self._portfolio_state(broker, snaps_repo)

            # 3) Mark to market at next-day close and snapshot.
            prices = {}
            for sym in features:
                px = self._price_on(features[sym], next_day, "close")
                if px is not None:
                    prices[sym] = px
            broker.update_portfolio_snapshot(prices, ts=next_day)

        session.flush()
        return self._build_result(session, broker, spy_feat, calendar, result_start, result_end)

    def _defensive(self, regime: RegimeResult) -> bool:
        from app.domain import Regime

        if regime.regime == Regime.PANIC:
            return True
        if regime.spy_ma50 is not None and regime.spy_close < regime.spy_ma50:
            return True
        return False

    def _build_result(self, session, broker, spy_feat, calendar, start, end) -> BacktestResult:
        snaps = PortfolioSnapshotRepository(session).all()
        closed = PositionRepository(session).closed_positions()
        all_decisions = self._all_decisions(session)
        audits = AuditRepository(session).count()

        spy_ret = period_return(spy_feat.loc[calendar[0] : calendar[-1]]["close"]) if len(calendar) else 0.0

        perf = evaluator.evaluate(
            period="backtest",
            snapshots=snaps,
            closed_positions=closed,
            decisions=all_decisions,
            initial_capital=self.config.initial_capital,
            spy_return=spy_ret,
            period_start=start,
            period_end=end,
        )
        buys = sum(1 for d in all_decisions if d.action == "buy")
        rejected = sum(1 for d in all_decisions if d.rejected_opportunity)
        return BacktestResult(
            performance=perf,
            total_decisions=len(all_decisions),
            buys=buys,
            rejected_opportunities=rejected,
            closed_trades=len(closed),
            audit_logs=audits,
            qualified=len(closed) >= MIN_TRADES_FOR_QUALIFICATION,
            equity_curve=[float(s.total_value) for s in snaps],
            start=start,
            end=end,
            benchmark_symbol=self.benchmark,
        )

    @staticmethod
    def _all_decisions(session):
        from sqlalchemy import select

        from app.db.models import Decision

        return list(session.scalars(select(Decision).order_by(Decision.created_at)))
