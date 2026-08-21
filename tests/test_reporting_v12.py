"""v1.2 — reporting correctness fixes.

Covers the three defects found in production plus the weekend/stale-bar skip:
1. weekly return was cumulative-since-inception, not weekly
2. manual /weekly always reported SPY as +0.00%
3. risk-manager blocks were labelled "rule violations"
4. the daily cycle re-ran on stale (weekend) data
"""
from datetime import datetime, timedelta
from types import SimpleNamespace

from app.config import load_config
from app.performance import evaluator
from app.telegram.commands import CommandService
from app.telegram.reports import format_weekly_report


def _snap(value, ts):
    return SimpleNamespace(total_value=value, daily_pnl_pct=None, ts=ts)


# -- 1. period vs cumulative return ------------------------------------------ #
def test_period_return_is_not_cumulative_return():
    # Started at 266, was 300 at the start of this week, 303 now.
    snaps = [_snap(300.0, datetime(2026, 8, 14)), _snap(303.0, datetime(2026, 8, 21))]
    res = evaluator.evaluate(
        period="weekly", snapshots=snaps, closed_positions=[], decisions=[],
        initial_capital=266.0,
    )
    # Weekly = 303/300 - 1 = +1.0%
    assert abs(res.period_return - 0.01) < 1e-9
    # Cumulative = 303/266 - 1 ≈ +13.9%  (must NOT be reported as "weekly")
    assert abs(res.total_return - (303.0 / 266.0 - 1.0)) < 1e-9
    assert res.period_return != res.total_return


def test_weekly_report_shows_both_returns():
    res = evaluator.PerformanceResult(
        starting_value=300.0, ending_value=303.0, period_return=0.01, total_return=0.139
    )
    text = format_weekly_report(res, spy_weekly=0.005, risk_blocks=0)
    assert "العائد الأسبوعي:" in text and "+1.00%" in text
    assert "العائد التراكمي (منذ الانطلاق):" in text and "+13.90%" in text


# -- 2. SPY weekly is computed, not hard-zero -------------------------------- #
def test_weekly_uses_supplied_spy_return():
    res = evaluator.PerformanceResult(starting_value=100.0, ending_value=101.0)
    text = format_weekly_report(res, spy_weekly=0.011, risk_blocks=0)
    assert "+1.10%" in text  # SPY value actually rendered


def test_manual_weekly_computes_spy_without_crashing(db_url, monkeypatch):
    """/weekly with no argument must try to compute SPY (not silently use 0)."""
    cfg = load_config()
    svc = CommandService(cfg)
    called = {"n": 0}

    def fake_spy(days):
        called["n"] += 1
        return 0.0123

    monkeypatch.setattr(svc, "_spy_period_return", fake_spy)
    text = svc.weekly()  # no spy_weekly passed — the production bug path
    assert called["n"] == 1, "manual /weekly must compute the benchmark return"
    assert "+1.23%" in text


def test_spy_period_return_is_safe_on_failure(monkeypatch):
    """Network failure must degrade to 0.0, never raise into the bot."""
    cfg = load_config()
    svc = CommandService(cfg)
    import app.data.market_data as md

    def boom(*a, **k):
        raise RuntimeError("network down")

    monkeypatch.setattr(md, "fetch_history", boom)
    assert svc._spy_period_return(7) == 0.0


# -- 3. risk blocks are not called "violations" ------------------------------ #
def test_risk_blocks_labelled_correctly_not_as_violations():
    res = evaluator.PerformanceResult(starting_value=266.0, ending_value=268.0)
    text = format_weekly_report(res, spy_weekly=0.0, risk_blocks=143, actual_violations=0)
    assert "قرارات أوقفها مدير المخاطر:" in text
    assert "143" in text
    # The alarming/incorrect label must be gone, and real violations shown as 0.
    assert "مخالفات القواعد:" not in text
    assert "مخالفات فعلية للقواعد:" in text
    assert "0 ✅" in text


# -- 4. stale-bar (weekend) skip --------------------------------------------- #
def test_daily_workflow_skips_when_no_new_bar(db_url, synthetic_market, monkeypatch):
    from app.db.database import session_scope
    from app.db.repositories import SystemConfigRepository
    from app.main import run_daily_workflow
    from sqlalchemy import func, select
    from app.db.models import Decision

    cfg = load_config()
    cfg.env.database_url = db_url

    # First run processes the latest bar and records decisions.
    run_daily_workflow(cfg, data=synthetic_market, send_report=False)
    with session_scope() as s:
        first_count = s.scalar(select(func.count()).select_from(Decision))
        stamp1 = SystemConfigRepository(s).get("last_decision_cycle_at")
        bar = SystemConfigRepository(s).get("last_processed_bar_date")
    assert first_count > 0 and bar is not None

    # Second run on the SAME data (weekend) must not add duplicate decisions.
    run_daily_workflow(cfg, data=synthetic_market, send_report=False)
    with session_scope() as s:
        second_count = s.scalar(select(func.count()).select_from(Decision))
        stamp2 = SystemConfigRepository(s).get("last_decision_cycle_at")
    assert second_count == first_count, "stale bar must not create new decisions"
    assert stamp2 == stamp1, "decision-cycle stamp must not advance on a skip"

    # ...but --force overrides the skip.
    run_daily_workflow(cfg, data=synthetic_market, send_report=False, force=True)
    with session_scope() as s:
        forced = s.scalar(select(func.count()).select_from(Decision))
    assert forced > second_count, "force=True must run the cycle anyway"
