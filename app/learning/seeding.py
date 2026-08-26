"""Seeding the learning tables from historical decisions.

The live system starts from zero: at ~10 rejections a day it needs months before
the calibration has anything to say. A five-year backtest produces >10,000 scored
rejections in minutes, and they are not synthetic — the *same* pipeline made them
on *real* market data with no look-ahead. For calibrating filters and the DQS
cut-off they are arguably better evidence than two months of a single bull
market, because they span several regimes.

**Only outcome rows are imported** — :class:`RejectedOutcome` for the
calibration and :class:`DecisionOutcome` for the weight loop. No decisions, no
orders, no positions, no portfolio snapshots. Seeding therefore cannot alter the
live portfolio, the cash balance, or the audit invariant; it only gives the two
learning tracks something to read.

Both tracks matter. Rejections calibrate the filters and the cut-off; closed
trades drive the DQS component weights, and that loop needs 30 of them before it
will move at all — a live system produces about 39 a year, so without seeding it
sits idle for months while the other half learns.

Imported rows carry ``decision_id = None`` (and ``position_id = None``): the
originating rows live in the backtest database and importing their ids would
dangle a foreign key. That null doubles as the marker for "historical, not live".

**The multiplicity risk is real.** One historical dataset is one experiment. The
threshold tuner's new-evidence guard is what stops the loop re-reading this seed
every week and walking the cut-off to its bound on a single body of evidence.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional

from sqlalchemy import func, select

from app.config import Config
from app.db.models import DecisionOutcome, RejectedOutcome
from app.db.repositories import SystemConfigRepository
from app.logging_config import get_logger

log = get_logger(__name__)

SEEDED_KEY = "learning_seeded_at"
SEEDED_COUNT_KEY = "learning_seeded_count"


@dataclass
class SeedReport:
    imported: int = 0
    closed_trades: int = 0
    skipped: bool = False
    reason: str = ""
    baseline_return: Optional[float] = None
    total_after: int = 0

    def summary(self) -> str:
        if self.skipped:
            return f"لم يُنفَّذ البذر — {self.reason}"
        return (
            f"بُذرت {self.imported:,} نتيجة مرفوضة و{self.closed_trades:,} صفقة مغلقة "
            f"(إجمالي الرفضات الآن {self.total_after:,})"
        )


def already_seeded(session) -> bool:
    return bool(SystemConfigRepository(session).get(SEEDED_KEY))


def seed_from_backtest(
    target_session,
    config: Config,
    data: Dict[str, object],
    *,
    force: bool = False,
    horizon_days: int = 10,
    backtest_db: str = "sqlite:////tmp/ics_seed_backtest.db",
) -> SeedReport:
    """Run a historical backtest in isolation and import its scored rejections.

    ``data`` is the price history to replay — passed in rather than fetched here
    so the caller controls the dataset and can reuse one fetch for the seed and
    for the baseline.
    """
    from app.backtesting.backtester import Backtester
    from app.db import database
    from app.learning.counterfactuals import (
        compute_baseline,
        load_baseline,
        record_rejected_outcomes,
        save_baseline,
    )
    from app.learning.outcomes import record_outcomes

    if already_seeded(target_session) and not force:
        # Re-running must not re-import, but it should still repair a missing
        # baseline: without one every verdict degrades to "nothing to compare
        # against", and measuring it here costs one pass over data already in
        # hand — no backtest, no refetch.
        mean, hit = load_baseline(target_session)
        if mean is None:
            mean, hit = compute_baseline(data, horizon=horizon_days)
            if mean is not None:
                save_baseline(target_session, mean, hit)
                log.info("Baseline was missing; measured and stored it.")
        return SeedReport(
            skipped=True,
            reason="سبق البذر؛ استخدم --force لإعادته.",
            baseline_return=mean,
            total_after=int(target_session.scalar(func.count(RejectedOutcome.id)) or 0),
        )

    target_url = str(target_session.get_bind().url)

    # 1) Replay history in a database of its own, so the live one is untouched.
    log.info("Seeding: replaying history into an isolated database...")
    Backtester(config).run(data, database_url=backtest_db)

    # 2) Score every rejection it produced, then read the rows back out.
    database.init_engine(backtest_db, force_reset=True)
    with database.session_scope() as bt:
        record_rejected_outcomes(
            bt, horizon_days=horizon_days, price_provider=lambda t: data.get(t)
        )
        # The other half of the loop: what the trades it DID take actually did.
        record_outcomes(bt, price_provider=lambda tkr, a, b: data.get(tkr))
        taken = [
            {
                "ticker": o.ticker,
                "strategy": o.strategy,
                "entry_at": o.entry_at,
                "exit_at": o.exit_at,
                "realized_return": o.realized_return,
                "holding_period_days": o.holding_period_days,
                "mfe": o.mfe,
                "mae": o.mae,
                "dqs_at_entry": o.dqs_at_entry,
                "dqs_components_json": o.dqs_components_json,
            }
            for o in bt.scalars(select(DecisionOutcome))
            if o.realized_return is not None and o.dqs_components_json
        ]
        rows = [
            {
                "ticker": r.ticker,
                "strategy": r.strategy,
                "rejected_at": r.rejected_at,
                "category": r.category,
                "rejection_reason": r.rejection_reason,
                "dqs_score": r.dqs_score,
                "dqs_components_json": r.dqs_components_json,
                "horizon_days": r.horizon_days,
                "forward_return": r.forward_return,
                "forward_mfe": r.forward_mfe,
                "forward_mae": r.forward_mae,
                "rejection_helped": r.rejection_helped,
            }
            for r in bt.scalars(select(RejectedOutcome))
            if r.forward_return is not None
        ]

    baseline, _hit = compute_baseline(data, horizon=horizon_days)

    # 3) Back to the live database and import — outcomes only, nothing tradeable.
    database.init_engine(target_url, force_reset=True)
    with database.session_scope() as live:
        for row in rows:
            live.add(RejectedOutcome(decision_id=None, **row))
        for row in taken:
            live.add(DecisionOutcome(decision_id=None, position_id=None, **row))
        if baseline is not None:
            save_baseline(live, baseline, _hit)
        cfg = SystemConfigRepository(live)
        cfg.set(SEEDED_KEY, datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"))
        cfg.set(SEEDED_COUNT_KEY, str(len(rows)))
        live.flush()
        total = int(live.scalar(func.count(RejectedOutcome.id)) or 0)

    log.info(
        "Seeding complete: %d rejections and %d closed trades imported.",
        len(rows), len(taken),
    )
    return SeedReport(
        imported=len(rows), closed_trades=len(taken),
        baseline_return=baseline, total_after=total,
    )
