"""ICS-DOC-004 Phase 1 — DSR and the multi-window activation gate.

These guard the *stopping rule*: they must make it hard, not easy, for the ML
layer to leave shadow mode.
"""
import random

import pytest

from app.ml.evaluation import (
    BASELINE,
    DSR_PASS,
    GateResult,
    WindowResult,
    beats_baseline,
    deflated_sharpe_ratio,
    evaluate_gate,
    expected_max_sharpe,
    sharpe_ratio,
    trial_sharpe_variance,
)

# Realistic spread of daily Sharpe across optuna trials of one model family.
TRIAL_VAR = 0.01  # sd = 0.1


# --------------------------------------------------------------------------- #
# DSR mechanics
# --------------------------------------------------------------------------- #
def test_sharpe_of_constant_series_is_zero():
    """Float noise makes sd ~3e-18 rather than 0; that must not become a huge Sharpe."""
    assert sharpe_ratio([0.01] * 50) == 0.0


def test_trial_sharpe_variance_from_the_trial_log():
    assert trial_sharpe_variance([0.5]) == 0.0
    assert trial_sharpe_variance([0.1, 0.2, 0.3]) == pytest.approx(0.01)


def test_expected_max_sharpe_grows_with_trials():
    """The more configurations you try, the higher the bar luck alone clears."""
    e1 = expected_max_sharpe(10)
    e2 = expected_max_sharpe(100)
    e3 = expected_max_sharpe(1000)
    assert 0 < e1 < e2 < e3


def test_expected_max_sharpe_is_zero_for_a_single_trial():
    assert expected_max_sharpe(1) == 0.0


def test_more_trials_deflate_the_same_returns():
    """Identical performance must score lower when it took more tries to find."""
    rng = random.Random(7)
    rets = [rng.gauss(0.002, 0.01) for _ in range(500)]
    few = deflated_sharpe_ratio(rets, n_trials=2, variance_of_trial_sharpes=TRIAL_VAR)
    many = deflated_sharpe_ratio(rets, n_trials=100, variance_of_trial_sharpes=TRIAL_VAR)
    assert few["sharpe"] == pytest.approx(many["sharpe"])   # same raw Sharpe
    assert many["benchmark_sharpe"] > few["benchmark_sharpe"]
    assert many["dsr"] < few["dsr"]                          # ...but deflated harder


def test_strong_edge_survives_deflation():
    rng = random.Random(11)
    rets = [rng.gauss(0.004, 0.005) for _ in range(750)]  # a genuinely large edge
    res = deflated_sharpe_ratio(rets, n_trials=100, variance_of_trial_sharpes=TRIAL_VAR)
    assert res["dsr"] > 0.95


def test_noise_does_not_survive_deflation():
    rng = random.Random(3)
    rets = [rng.gauss(0.0, 0.01) for _ in range(500)]  # no edge at all
    res = deflated_sharpe_ratio(rets, n_trials=100, variance_of_trial_sharpes=TRIAL_VAR)
    assert res["dsr"] < 0.5


def test_dsr_handles_tiny_samples():
    res = deflated_sharpe_ratio([0.01, -0.01], n_trials=10, variance_of_trial_sharpes=TRIAL_VAR)
    assert res["dsr"] == 0.0


# --------------------------------------------------------------------------- #
# Baseline comparison
# --------------------------------------------------------------------------- #
def _win(**kw):
    base = dict(name="w", total_return=0.30, max_drawdown=-0.04, sharpe=1.5,
                win_rate=0.45, average_dqs=86.0, dsr=0.99)
    base.update(kw)
    return WindowResult(**base)


def test_window_must_beat_return_drawdown_and_sharpe_together():
    assert beats_baseline(_win()) is True
    assert beats_baseline(_win(total_return=0.10)) is False       # return too low
    assert beats_baseline(_win(max_drawdown=-0.20)) is False      # worse drawdown
    assert beats_baseline(_win(sharpe=0.5)) is False              # worse Sharpe


def test_baseline_matches_the_documented_numbers():
    assert BASELINE["total_return"] == pytest.approx(0.2188)
    assert BASELINE["max_drawdown"] == pytest.approx(-0.0542)
    assert BASELINE["sharpe"] == pytest.approx(1.01)


# --------------------------------------------------------------------------- #
# The gate
# --------------------------------------------------------------------------- #
def test_gate_requires_at_least_three_windows():
    res = evaluate_gate([_win(name="a"), _win(name="b")])
    assert res.passed is False
    assert "الحد الأدنى" in res.reason


def test_gate_requires_a_majority_not_just_one_good_window():
    """The whole point: one cherry-picked window must not pass."""
    res = evaluate_gate([
        _win(name="good"),
        _win(name="bad1", total_return=0.05),
        _win(name="bad2", sharpe=0.4),
    ])
    assert res.passed is False
    assert res.windows_won == 1


def test_gate_blocks_when_dsr_is_below_threshold():
    res = evaluate_gate([_win(name=f"w{i}", dsr=0.80) for i in range(3)])
    assert res.passed is False
    assert "DSR" in res.reason


def test_gate_passes_only_when_majority_and_dsr_both_hold():
    res = evaluate_gate([_win(name=f"w{i}") for i in range(3)])
    assert res.passed is True
    assert res.windows_won == 3
    assert res.dsr_min >= DSR_PASS


def test_gate_passes_on_a_two_of_three_majority():
    res = evaluate_gate([
        _win(name="w1"), _win(name="w2"), _win(name="w3", total_return=0.05),
    ])
    assert res.passed is True
    assert res.windows_won == 2


def test_gate_result_summary_is_readable():
    res = evaluate_gate([_win(name=f"w{i}") for i in range(3)])
    assert isinstance(res, GateResult)
    assert "✅" in res.summary()
