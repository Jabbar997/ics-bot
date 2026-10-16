"""Seeding the calibration tables from a historical backtest.

The guarantee that matters: seeding gives the learner evidence without touching
anything tradeable. Nothing about the portfolio, the cash, the positions or the
audit invariant may change.
"""
from datetime import datetime

from sqlalchemy import func, select

from app.config import load_config
from app.db.database import session_scope
from app.db.models import (
    AuditLog,
    Decision,
    DecisionOutcome,
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


# --------------------------------------------------------------------------- #
# a database seeded by the older, rejections-only version (the live Render state)
# --------------------------------------------------------------------------- #
def _seed(s, cfg, market, tmp_path, name, **kw):
    return seed_from_backtest(s, cfg, market, backtest_db=f"sqlite:///{tmp_path / name}", **kw)


def _rejection_and_trade_counts(s):
    rej = int(s.scalar(func.count(RejectedOutcome.id)) or 0)
    trades = int(s.scalar(func.count(DecisionOutcome.id)) or 0)
    return rej, trades


def _simulate_old_seed(s):
    """Leave the flag and the rejections, drop the seeded trades: what the version
    before 'seed the closed trades too' left behind."""
    from sqlalchemy import delete
    s.execute(delete(DecisionOutcome).where(DecisionOutcome.decision_id.is_(None)))


def test_a_rejections_only_database_gets_its_missing_trades_without_duplicates(
    db_url, synthetic_market, tmp_path
):
    cfg = load_config()
    with session_scope() as s:
        full = _seed(s, cfg, synthetic_market, tmp_path, "a.db")
    assert full.imported > 0 and full.closed_trades > 0
    with session_scope() as s:
        _simulate_old_seed(s)
        assert _rejection_and_trade_counts(s) == (full.imported, 0)
        assert already_seeded(s) is True          # the flag that used to cause a skip

    with session_scope() as s:
        report = _seed(s, cfg, synthetic_market, tmp_path, "b.db")   # no --force
    assert report.skipped is False
    assert report.closed_trades == full.closed_trades
    assert report.imported == 0                    # rejections were already there
    with session_scope() as s:
        assert _rejection_and_trade_counts(s) == (full.imported, full.closed_trades)


def test_force_replaces_seeded_rows_instead_of_duplicating_them(db_url, synthetic_market, tmp_path):
    cfg = load_config()
    with session_scope() as s:
        first = _seed(s, cfg, synthetic_market, tmp_path, "c.db")
    with session_scope() as s:
        again = _seed(s, cfg, synthetic_market, tmp_path, "d.db", force=True)
    assert not again.skipped
    with session_scope() as s:
        assert _rejection_and_trade_counts(s) == (first.imported, first.closed_trades)


def test_force_never_deletes_live_outcomes(db_url, synthetic_market, tmp_path):
    """Live rows always carry a decision id; only the seeded (NULL) ones may be replaced."""
    cfg = load_config()
    with session_scope() as s:
        _seed(s, cfg, synthetic_market, tmp_path, "e.db")
    with session_scope() as s:
        live_decision = Decision(
            ticker="AAPL", action="buy", strategy="trend", price=100.0, dqs_score=80,
            confidence=0.5, reason="live", market_regime="bull",
        )
        s.add(live_decision)
        s.flush()
        s.add(DecisionOutcome(
            decision_id=live_decision.id, ticker="AAPL", strategy="trend",
            realized_return=0.02, dqs_components_json="{}",
        ))
        s.add(RejectedOutcome(
            decision_id=live_decision.id, ticker="AAPL", strategy="trend",
            category="strategy_filter", forward_return=0.01, horizon_days=10,
            rejected_at=datetime(2026, 1, 5),
        ))
    with session_scope() as s:
        _seed(s, cfg, synthetic_market, tmp_path, "f.db", force=True)
    with session_scope() as s:
        assert s.scalar(select(func.count(DecisionOutcome.id)).where(DecisionOutcome.decision_id.is_not(None))) == 1
        assert s.scalar(select(func.count(RejectedOutcome.id)).where(RejectedOutcome.decision_id.is_not(None))) == 1


def _add_live_outcome_without_decision(s):
    """What record_outcomes writes for a live trade whose BUY decision is not found
    (e.g. a core-allocation position): decision_id NULL, but a real position_id."""
    s.add(DecisionOutcome(
        decision_id=None, position_id="live-core-position", ticker="SPY", strategy="core",
        realized_return=0.03, dqs_components_json={"strategy_alignment": 20},
    ))


def test_a_live_outcome_without_a_decision_is_not_mistaken_for_a_seeded_trade(
    db_url, synthetic_market, tmp_path
):
    """Otherwise it would count as 'trades already seeded' and block the import."""
    cfg = load_config()
    with session_scope() as s:
        _add_live_outcome_without_decision(s)
    with session_scope() as s:
        report = _seed(s, cfg, synthetic_market, tmp_path, "g.db")
    assert report.closed_trades > 0           # the missing seed was still imported
    with session_scope() as s:
        live = s.scalar(select(func.count(DecisionOutcome.id)).where(
            DecisionOutcome.position_id == "live-core-position"))
        assert live == 1


def test_force_keeps_a_live_outcome_that_has_no_decision(db_url, synthetic_market, tmp_path):
    cfg = load_config()
    with session_scope() as s:
        first = _seed(s, cfg, synthetic_market, tmp_path, "h.db")
        _add_live_outcome_without_decision(s)
    with session_scope() as s:
        _seed(s, cfg, synthetic_market, tmp_path, "i.db", force=True)
    with session_scope() as s:
        assert s.scalar(select(func.count(DecisionOutcome.id)).where(
            DecisionOutcome.position_id == "live-core-position")) == 1
        seeded = s.scalar(select(func.count(DecisionOutcome.id)).where(
            DecisionOutcome.decision_id.is_(None), DecisionOutcome.position_id.is_(None)))
        assert seeded == first.closed_trades
