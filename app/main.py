"""ICS entry point and workflow orchestration.

Subcommands::

    python -m app.main init-db        # create tables + sync watchlist
    python -m app.main daily          # run one daily paper cycle (live data)
    python -m app.main weekly         # build + (optionally) send weekly report
    python -m app.main backtest       # 5y backtest with walk-forward splits
    python -m app.main bot            # run the Telegram bot (long-polling)
    python -m app.main scheduler      # APScheduler: daily + weekly jobs
    python -m app.main demo           # offline synthetic cycle (no network)

Execution conventions:
* Backtest: orders fill at the NEXT trading day's open (conservative, no look-ahead).
* Live daily: orders fill at the latest available close (best paper proxy).
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from typing import Dict, List, Optional

import pandas as pd

from app.config import Config, load_config
from app.db import database
from app.db.repositories import (
    KillSwitchRepository,
    PortfolioSnapshotRepository,
    PositionRepository,
    RegimeRepository,
    SystemConfigRepository,
    WatchlistRepository,
)
from app.decision.decision_engine import DecisionEngine
from app.domain import Regime
from app.features.feature_engine import build_feature_snapshot, compute_features
from app.logging_config import configure_logging, get_logger
from app.market.regime import analyze_from_features_df
from app.db.models import MarketRegime
from app.paper.broker import PaperBroker
from app.risk.kill_switch import KillSwitchManager, evaluate_kill_switch
from app.risk.risk_manager import OpenPositionView, PortfolioState
from app.strategies.engine import StrategyEngine

log = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Setup
# --------------------------------------------------------------------------- #
def init_db(config: Config) -> None:
    database.init_engine(config.env.database_url)
    database.create_all()
    with database.session_scope() as s:
        WatchlistRepository(s).sync(config.watchlist)
    log.info("Database initialised at %s", config.env.database_url)


def _portfolio_state(broker: PaperBroker, snaps_repo, prices: Dict[str, float]) -> PortfolioState:
    total = float(broker.calculate_total_value(prices))
    cash = float(broker.calculate_cash())
    snaps = snaps_repo.all()
    weekly = _trailing_return(snaps, 5)
    monthly = _trailing_return(snaps, 21)
    dd = _drawdown([float(s.total_value) for s in snaps] + [total])
    return PortfolioState(total, cash, weekly, monthly, dd)


def _trailing_return(snaps, lookback: int) -> float:
    if len(snaps) <= lookback:
        return 0.0
    prev = float(snaps[-lookback].total_value)
    cur = float(snaps[-1].total_value)
    return (cur / prev - 1.0) if prev else 0.0


def _drawdown(values: List[float]) -> float:
    if not values:
        return 0.0
    peak = max(values)
    return (values[-1] / peak - 1.0) if peak else 0.0


def _consecutive_losses(closed) -> int:
    count = 0
    for p in sorted(closed, key=lambda x: x.exit_at or datetime.min, reverse=True):
        if (p.realized_pnl or 0) <= 0:
            count += 1
        else:
            break
    return count


# --------------------------------------------------------------------------- #
# Daily workflow
# --------------------------------------------------------------------------- #
def run_daily_workflow(
    config: Config,
    data: Optional[Dict[str, pd.DataFrame]] = None,
    send_report: bool = False,
) -> str:
    """Run one daily paper-trading cycle. Returns the daily report text."""
    database.init_engine(config.env.database_url)
    database.create_all()

    benchmark = config.benchmark.symbol
    symbols = list(dict.fromkeys(config.watchlist + [benchmark]))

    # 1-2. Fetch + clean (or use injected data for tests/demo).
    if data is None:
        from app.data.market_data import fetch_watchlist_history

        period = f"{config.market.history_years}y"
        data = fetch_watchlist_history(symbols, period=period)

    # 3. Features.
    spy_df = data.get(benchmark)
    if spy_df is None or spy_df.empty:
        raise RuntimeError(f"No benchmark ({benchmark}) data; cannot determine regime.")
    spy_close = spy_df["close"]
    features = {
        sym: compute_features(df, spy_close=spy_close)
        for sym, df in data.items()
        if df is not None and not df.empty
    }

    spy_feat = features[benchmark]
    spy_today_pct = float(spy_feat["close"].pct_change().iloc[-1]) if len(spy_feat) > 1 else 0.0

    with database.session_scope() as s:
        # 4. Regime.
        regime = analyze_from_features_df(spy_feat)
        RegimeRepository(s).add(
            MarketRegime(
                regime=regime.regime.value,
                spy_close=regime.spy_close,
                spy_ma50=regime.spy_ma50,
                spy_ma200=regime.spy_ma200,
                drawdown_20d=regime.drawdown_20d,
                reason=regime.reason,
            )
        )

        broker = PaperBroker(s, config)
        snaps_repo = PortfolioSnapshotRepository(s)
        positions_repo = PositionRepository(s)
        ks = KillSwitchManager(s)

        # 13(pre). Evaluate kill switch from current metrics.
        closed = positions_repo.closed_positions()
        latest_prices = {sym: float(f["close"].iloc[-1]) for sym, f in features.items()}
        pstate = _portfolio_state(broker, snaps_repo, latest_prices)
        ks_eval = evaluate_kill_switch(
            weekly_return_pct=pstate.weekly_return_pct,
            monthly_return_pct=pstate.monthly_return_pct,
            drawdown_pct=pstate.current_drawdown_pct,
            consecutive_losses=_consecutive_losses(closed),
        )
        if ks_eval.active:
            ks.trigger(ks_eval)
        ks_active = ks.is_active() or ks_eval.active

        engine = StrategyEngine()
        de = DecisionEngine(s, config, broker, kill_switch_active=ks_active)

        # 9(exits). Close positions whose exit rules fired (fill at latest close).
        for pos in list(positions_repo.open_positions()):
            feat = features.get(pos.ticker)
            if feat is None:
                continue
            snap = build_feature_snapshot(pos.ticker, feat)
            exit_sig = engine.evaluate_exit(
                pos.strategy or "trend", snap, regime, pos.entry_price, pos.stop_loss, pos.target_price
            )
            if exit_sig is not None:
                de.record_exit(pos, exit_sig, snap, regime, float(feat["close"].iloc[-1]))

        # If kill switch L2/L3 requests closing positions, do so (paper only).
        if ks_eval.close_fraction >= 1.0:
            for pos in list(positions_repo.open_positions()):
                feat = features.get(pos.ticker)
                px = float(feat["close"].iloc[-1]) if feat is not None else pos.entry_price
                broker.close_position(pos, px, reason=f"Kill switch L{ks_eval.level.value}")

        # 5-9(entries). Generate, score, validate, execute — unless defensive.
        defensive = ks_active or regime.regime == Regime.PANIC or (
            regime.spy_ma50 is not None and regime.spy_close < regime.spy_ma50
        )
        if not defensive:
            snapshots = {}
            for sym in config.watchlist:
                feat = features.get(sym)
                if feat is None:
                    continue
                snap = build_feature_snapshot(sym, feat)
                if snap.is_complete():
                    snapshots[sym] = snap
            signals = engine.generate_signals(snapshots, regime)
            open_views = [OpenPositionView(p.ticker, float(p.quantity)) for p in positions_repo.open_positions()]
            for sig in signals:
                de.process_signal(sig, snapshots[sig.ticker], regime, pstate, open_views)
                open_views = [OpenPositionView(p.ticker, float(p.quantity)) for p in positions_repo.open_positions()]
                pstate = _portfolio_state(broker, snaps_repo, latest_prices)

        # 11-12. Mark to market + snapshot.
        broker.update_portfolio_snapshot(latest_prices)

        # v1.1: record the decision-cycle timestamp for /health.
        SystemConfigRepository(s).set(
            "last_decision_cycle_at",
            datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        )

    # 14. Daily report.
    from app.telegram.commands import CommandService

    report = CommandService(config).today(spy_today_pct=spy_today_pct)
    log.info("Daily workflow complete.\n%s", report)
    if send_report:
        _send_telegram(config, report)
    return report


def run_weekly_workflow(config: Config, spy_weekly: float = 0.0, send_report: bool = False) -> str:
    database.init_engine(config.env.database_url)
    database.create_all()
    from app.telegram.commands import CommandService

    report = CommandService(config).weekly(spy_weekly=spy_weekly)
    log.info("Weekly workflow complete.\n%s", report)
    if send_report:
        _send_telegram(config, report)
    return report


def _send_telegram(config: Config, text: str) -> None:
    token = config.env.telegram_bot_token
    if not token or not config.telegram.allowed_user_ids:
        log.warning("Telegram not configured; report not sent.")
        return
    import asyncio

    from telegram import Bot

    async def _send():
        bot = Bot(token)
        for uid in config.telegram.allowed_user_ids:
            await bot.send_message(chat_id=uid, text=text)

    asyncio.run(_send())


# --------------------------------------------------------------------------- #
# Backtest
# --------------------------------------------------------------------------- #
def run_backtest(config: Config, period: Optional[str] = None) -> None:
    from app.backtesting.backtester import Backtester
    from app.backtesting.walk_forward import split_dates
    from app.data.market_data import fetch_watchlist_history

    period = period or f"{config.market.history_years}y"
    symbols = list(dict.fromkeys(config.watchlist + [config.benchmark.symbol]))
    log.info("Fetching %s of history for %d symbols...", period, len(symbols))
    data = fetch_watchlist_history(symbols, period=period)

    splits = split_dates(data[config.benchmark.symbol].index)
    print("Walk-forward splits:")
    for name, (a, b) in splits.items():
        print(f"  {name:12s}: {a} -> {b}")

    bt = Backtester(config)
    result = bt.run(data)
    print("\n" + result.summary())


# --------------------------------------------------------------------------- #
# Scheduler / Bot
# --------------------------------------------------------------------------- #
def run_scheduler(config: Config) -> None:
    from apscheduler.schedulers.blocking import BlockingScheduler

    from app.utils.time import KSA, parse_hhmm

    hh, mm = parse_hhmm(config.telegram.daily_report_time_ksa)
    sched = BlockingScheduler(timezone=KSA)
    sched.add_job(lambda: run_daily_workflow(config, send_report=True), "cron", hour=hh, minute=mm)
    sched.add_job(
        lambda: run_weekly_workflow(config, send_report=True),
        "cron",
        day_of_week=config.telegram.weekly_report_day.lower()[:3],
        hour=hh,
        minute=mm + 5,
    )
    log.info("Scheduler started (daily %02d:%02d KSA, weekly %s).", hh, mm, config.telegram.weekly_report_day)
    sched.start()


def run_bot(config: Config) -> None:
    from app.telegram.bot import ICSBot

    init_db(config)
    ICSBot(config).run()


# --------------------------------------------------------------------------- #
# Offline demo (no network)
# --------------------------------------------------------------------------- #
def run_demo(config: Config) -> None:
    import numpy as np

    def gen(n=320, start=100.0, drift=0.0009, vol=0.012, seed=1):
        rng = np.random.default_rng(seed)
        close = start * np.cumprod(1 + rng.normal(drift, vol, n))
        idx = pd.bdate_range("2023-01-02", periods=n)
        high = close * (1 + np.abs(rng.normal(0, 0.004, n)))
        low = close * (1 - np.abs(rng.normal(0, 0.004, n)))
        openp = np.concatenate([[close[0]], close[:-1]])
        vol_ = rng.integers(1_000_000, 3_000_000, n).astype(float)
        return pd.DataFrame(
            {"open": openp, "high": high, "low": low, "close": close, "adjusted_close": close, "volume": vol_},
            index=idx,
        )

    data = {"SPY": gen(seed=7, drift=0.0010)}
    for i, sym in enumerate(["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "COST", "JNJ"]):
        data[sym] = gen(seed=20 + i, drift=0.0009 + 0.0001 * i)

    config.env.database_url = "sqlite:///./ics_demo.db"
    database.init_engine(config.env.database_url)
    database.drop_all()
    database.create_all()
    report = run_daily_workflow(config, data=data, send_report=False)
    print(report)


# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ics", description="Investment Command System (paper-only).")
    p.add_argument("command", choices=["init-db", "daily", "weekly", "backtest", "bot", "scheduler", "demo"])
    p.add_argument("--config", default=None, help="Path to config.yaml")
    p.add_argument("--send", action="store_true", help="Send report to Telegram")
    p.add_argument("--period", default=None, help="Backtest history period, e.g. 5y")
    return p


def main(argv: Optional[List[str]] = None) -> None:
    configure_logging()
    args = build_parser().parse_args(argv)
    config = load_config(args.config)

    if args.command == "init-db":
        init_db(config)
    elif args.command == "daily":
        run_daily_workflow(config, send_report=args.send)
    elif args.command == "weekly":
        run_weekly_workflow(config, send_report=args.send)
    elif args.command == "backtest":
        run_backtest(config, period=args.period)
    elif args.command == "bot":
        run_bot(config)
    elif args.command == "scheduler":
        run_scheduler(config)
    elif args.command == "demo":
        run_demo(config)


if __name__ == "__main__":
    main()
