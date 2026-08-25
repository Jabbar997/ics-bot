"""Seeding the learning tables from historical decisions.

The live system starts from zero: at ~10 rejections a day it needs months before
the calibration has anything to say. A five-year backtest produces >10,000 scored
rejections in minutes, and they are not synthetic — the *same* pipeline made them
on *real* market data with no look-ahead. For calibrating filters and the DQS
cut-off they are arguably better evidence than two months of a single bull
market, because they span several regimes.

**Only :class:`RejectedOutcome` rows are imported.** No decisions, no orders, no
positions, no portfolio snapshots. Seeding therefore cannot alter the live
portfolio, the cash balance, or the audit invariant — it only gives the
calibration something to read.

Imported rows carry ``decision_id = None``: the originating decision lives in the
backtest database and importing the id would dangle a foreign key. That null
doubles as the marker for "historical, not live".

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
from app.db.models import RejectedOutcome
from app.db.repositories import SystemConfigRepository
from app.logging_config import get_logger

log = get_logger(__name__)

SEEDED_KEY = "learning_seeded_at"
SEEDED_COUNT_KEY = "learning_seeded_count"


@dataclass
class SeedReport:
    imported: int = 0
    skipped: bool = False
    reason: str = ""
    baseline_return: Optional[float] = None
    total_after: int = 0

    def summary(self) -> str:
        if self.skipped:
            return f"لم يُنفَّذ البذر — {self.reason}"
        return (
            f"بُذرت {self.imported:,} نتيجة مرفوضة تاريخية "
            f"(الإجمالي الآن {self.total_after:,})"
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
    from app.learning.counterfactuals import compute_baseline, record_rejected_outcomes

    if already_seeded(target_session) and not force:
        return SeedReport(
            skipped=True,
            reason="سبق البذر؛ استخدم --force لإعادته.",
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
        cfg = SystemConfigRepository(live)
        cfg.set(SEEDED_KEY, datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"))
        cfg.set(SEEDED_COUNT_KEY, str(len(rows)))
        live.flush()
        total = int(live.scalar(func.count(RejectedOutcome.id)) or 0)

    log.info("Seeding complete: %d historical outcomes imported.", len(rows))
    return SeedReport(
        imported=len(rows), baseline_return=baseline, total_after=total
    )
