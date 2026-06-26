"""Kill switch trigger tests (all 3 levels + manual + priority)."""
from app.domain import KillSwitchLevel
from app.risk.kill_switch import evaluate_kill_switch, manual_stop_evaluation


def test_no_trigger_when_calm():
    ev = evaluate_kill_switch(weekly_return_pct=0.01, monthly_return_pct=0.02, drawdown_pct=-0.01)
    assert ev.level == KillSwitchLevel.NONE
    assert ev.active is False
    assert ev.blocks_new_entries is False


def test_level1_weekly_loss():
    ev = evaluate_kill_switch(weekly_return_pct=-0.05)
    assert ev.level == KillSwitchLevel.LEVEL_1
    assert ev.blocks_new_entries is True
    assert ev.cooldown_hours == 48
    assert ev.close_fraction == 0.0


def test_level1_consecutive_losses():
    ev = evaluate_kill_switch(consecutive_losses=3)
    assert ev.level == KillSwitchLevel.LEVEL_1


def test_level2_monthly_loss():
    ev = evaluate_kill_switch(monthly_return_pct=-0.08)
    assert ev.level == KillSwitchLevel.LEVEL_2
    assert ev.close_fraction == 0.5


def test_level2_severe_event():
    ev = evaluate_kill_switch(severe_event=True)
    assert ev.level == KillSwitchLevel.LEVEL_2


def test_level3_monthly_loss():
    ev = evaluate_kill_switch(monthly_return_pct=-0.12)
    assert ev.level == KillSwitchLevel.LEVEL_3
    assert ev.close_fraction == 1.0


def test_level3_drawdown():
    ev = evaluate_kill_switch(drawdown_pct=-0.15)
    assert ev.level == KillSwitchLevel.LEVEL_3


def test_level3_takes_priority_over_level1_and_2():
    ev = evaluate_kill_switch(
        weekly_return_pct=-0.20, monthly_return_pct=-0.30, drawdown_pct=-0.40, consecutive_losses=10
    )
    assert ev.level == KillSwitchLevel.LEVEL_3


def test_manual_stop():
    ev = manual_stop_evaluation()
    assert ev.level == KillSwitchLevel.MANUAL
    assert ev.blocks_new_entries is True
    assert ev.close_fraction == 0.0  # paper only — no real positions closed


def test_kill_switch_manager_freeze_and_resume(db_url):
    from app.db.database import session_scope
    from app.risk.kill_switch import KillSwitchManager

    with session_scope() as s:
        ks = KillSwitchManager(s)
        assert ks.is_active() is False
        ks.manual_stop()
    with session_scope() as s:
        ks = KillSwitchManager(s)
        assert ks.is_frozen() is True
        assert ks.is_active() is True
        cleared = ks.resume()
        assert cleared >= 1
    with session_scope() as s:
        assert KillSwitchManager(s).is_active() is False
