"""Seeding the calibration tables from a historical backtest.

The guarantee that matters: seeding gives the learner evidence without touching
anything tradeable. Nothing about the portfolio, the cash, the positions or the
audit invariant may change.
"""
from sqlalchemy import func, select

from app.config import load_config
from app.db.database import session_scope
from app.db.models import (
    AuditLog,
    Decision,
    PaperOrder,
    PortfolioSnapshot,
    Position,
    RejectedOutcome,
)
from app.learning.seeding import already_seeded, seed_from_backtest


def _counts(s):
    return {
        m.__name__: int(s.scalar(func.count(m.id)) or 0)
        for m in (Decision, AuditLog, Position, PaperOrder, PortfolioSnapshot)
    }


def test_seeding_imports_outcomes_and_touches_nothing_tradeable(db_url, synthetic_market, tmp_path):
    cfg = load_config()
    with session_scope() as s:
        before = _counts(s)
        assert int(s.scalar(func.count(RejectedOutcome.id)) or 0) == 0

    with session_scope() as s:
        report = seed_from_backtest(
            s, cfg, synthetic_market,
            backtest_db=f"sqlite:///{tmp_path / 'seed_bt.db'}",
        )
    assert report.imported > 0
    assert not report.skipped

    with session_scope() as s:
        after = _counts(s)
        outcomes = list(s.scalars(select(RejectedOutcome)))

    # The learner gained evidence...
    assert len(outcomes) == report.imported
    # ...and nothing tradeable moved.
    assert after == before


def test_seeded_rows_are_marked_historical(db_url, synthetic_market, tmp_path):
    cfg = load_config()
    with session_scope() as s:
        seed_from_backtest(s, cfg, synthetic_market,
                           backtest_db=f"sqlite:///{tmp_path / 'b.db'}")
    with session_scope() as s:
        rows = list(s.scalars(select(RejectedOutcome)))
    # decision_id is null: the originating decision lives in the backtest DB, and
    # importing its id would dangle a foreign key.
    assert rows and all(r.decision_id is None for r in rows)
    assert all(r.forward_return is not None for r in rows)


def test_seeding_is_one_shot_unless_forced(db_url, synthetic_market, tmp_path):
    cfg = load_config()
    with session_scope() as s:
        first = seed_from_backtest(s, cfg, synthetic_market,
                                   backtest_db=f"sqlite:///{tmp_path / 'c.db'}")
    with session_scope() as s:
        assert already_seeded(s) is True
        second = seed_from_backtest(s, cfg, synthetic_market,
                                    backtest_db=f"sqlite:///{tmp_path / 'd.db'}")
    assert second.skipped is True
    assert second.imported == 0
    with session_scope() as s:
        assert int(s.scalar(func.count(RejectedOutcome.id)) or 0) == first.imported


def test_seeding_reports_the_market_baseline(db_url, synthetic_market, tmp_path):
    """The baseline must come with the seed — without it the tuner won't move."""
    cfg = load_config()
    with session_scope() as s:
        report = seed_from_backtest(s, cfg, synthetic_market,
                                    backtest_db=f"sqlite:///{tmp_path / 'e.db'}")
    assert report.baseline_return is not None


def test_reseeding_repairs_a_missing_baseline_without_reimporting(db_url, synthetic_market, tmp_path):
    """The live database was seeded before baselines were persisted.

    Running the command again must not duplicate 10k rows, but it must fill in
    the baseline — otherwise every verdict stays stuck on "no baseline".
    """
    from app.learning.counterfactuals import BASELINE_RETURN_KEY, load_baseline
    from app.db.repositories import SystemConfigRepository

    cfg = load_config()
    with session_scope() as s:
        first = seed_from_backtest(s, cfg, synthetic_market,
                                   backtest_db=f"sqlite:///{tmp_path / 'f.db'}")
    # Simulate the pre-fix state: rows present, baseline absent.
    with session_scope() as s:
        SystemConfigRepository(s).set(BASELINE_RETURN_KEY, "")
    with session_scope() as s:
        assert load_baseline(s)[0] is None

    with session_scope() as s:
        again = seed_from_backtest(s, cfg, synthetic_market,
                                   backtest_db=f"sqlite:///{tmp_path / 'g.db'}")
    assert again.skipped is True
    assert again.imported == 0
    with session_scope() as s:
        assert load_baseline(s)[0] is not None            # repaired
        assert int(s.scalar(func.count(RejectedOutcome.id)) or 0) == first.imported


def test_seeding_unblocks_the_weight_loop_too(db_url, synthetic_market, tmp_path):
    """Rejections alone only start half the system.

    The weight loop needs 30 closed trades before it moves, and a live system
    produces ~39 a year — so without seeding these it sits idle for months.
    """
    from app.db.models import DecisionOutcome
    from app.learning.feedback_loop import MIN_CLOSED_TRADES
    from app.learning.outcomes import load_outcomes

    cfg = load_config()
    with session_scope() as s:
        report = seed_from_backtest(s, cfg, synthetic_market,
                                    backtest_db=f"sqlite:///{tmp_path / 'h.db'}")

    assert report.closed_trades > 0
    with session_scope() as s:
        outcomes = load_outcomes(s)
        rows = list(s.scalars(select(DecisionOutcome)))

    assert len(rows) == report.closed_trades
    # Usable by the loop: a realised return AND the components scored at entry.
    # (How many the synthetic fixture yields depends on its length; the real
    # five-year dataset produces 164, comfortably past MIN_CLOSED_TRADES.)
    assert len(outcomes) == report.closed_trades
    assert MIN_CLOSED_TRADES > 0
    # Historical marker on both sides of the link.
    assert all(o.decision_id is None and o.position_id is None for o in rows)


def test_seeded_closed_trades_do_not_create_positions(db_url, synthetic_market, tmp_path):
    """The weight loop gets its evidence; the portfolio stays untouched."""
    cfg = load_config()
    with session_scope() as s:
        before = _counts(s)
    with session_scope() as s:
        seed_from_backtest(s, cfg, synthetic_market,
                           backtest_db=f"sqlite:///{tmp_path / 'i.db'}")
    with session_scope() as s:
        assert _counts(s) == before
