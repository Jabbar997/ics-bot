"""Tiered risk profiles and regime scaling.

The measured effect comes from one narrow place: sideways markets. Bear, panic
and weak-bull are already blocked by the defensive gate, and bull runs at the
full ceiling — so scaling only ever bites in a choppy, directionless tape. These
tests pin that behaviour down so the mechanism cannot silently drift.
"""
import pytest

from app.domain import Regime
from app.risk.risk_profiles import (
    PROFILES,
    REGIME_SCALE,
    effective_max_positions,
    profile_ceiling,
)


# --------------------------------------------------------------------------- #
# profiles
# --------------------------------------------------------------------------- #
def test_profiles_are_ordered_low_to_high():
    assert PROFILES["conservative"] < PROFILES["balanced"] < PROFILES["aggressive"]


def test_unknown_profile_falls_back_to_the_safest():
    assert profile_ceiling("nonsense") == PROFILES["conservative"]
    assert profile_ceiling("") == PROFILES["conservative"]
    assert profile_ceiling("AGGRESSIVE") == PROFILES["aggressive"]  # case-insensitive


# --------------------------------------------------------------------------- #
# scaling off
# --------------------------------------------------------------------------- #
def test_scaling_off_always_returns_the_profile_ceiling():
    for regime in Regime:
        b = effective_max_positions("balanced", regime, regime_scaling=False)
        assert b.max_positions == PROFILES["balanced"]


def test_unknown_regime_does_not_scale():
    b = effective_max_positions("balanced", None)
    assert b.max_positions == PROFILES["balanced"]


# --------------------------------------------------------------------------- #
# scaling on
# --------------------------------------------------------------------------- #
def test_bull_runs_at_the_full_ceiling():
    for name, ceiling in PROFILES.items():
        b = effective_max_positions(name, Regime.BULL)
        assert b.max_positions == ceiling


def test_bear_and_panic_allow_nothing():
    for regime in (Regime.BEAR, Regime.PANIC):
        b = effective_max_positions("aggressive", regime)
        assert b.max_positions == 0
        assert "لا دخول" in b.reason


def test_sideways_is_where_scaling_actually_bites():
    """The 67 sideways days are the entire source of the measured improvement."""
    b = effective_max_positions("conservative", Regime.SIDEWAYS)
    assert b.base_max_positions == 3
    assert b.max_positions == 1          # round(3 * 0.40)
    assert b.max_positions < b.base_max_positions


def test_weak_bull_narrows_but_does_not_close():
    b = effective_max_positions("conservative", Regime.WEAK_BULL)
    assert b.max_positions == 2          # round(3 * 0.60)


def test_scaling_never_exceeds_the_profile_ceiling():
    for name in PROFILES:
        for regime in Regime:
            b = effective_max_positions(name, regime)
            assert b.max_positions <= PROFILES[name]


def test_scaling_never_returns_a_negative_budget():
    for name in PROFILES:
        for regime in Regime:
            assert effective_max_positions(name, regime).max_positions >= 0


def test_a_permitted_regime_keeps_at_least_one_position():
    """Rounding must not silently close a regime the scale says is tradeable."""
    for name in PROFILES:
        for regime in (Regime.BULL, Regime.WEAK_BULL, Regime.SIDEWAYS):
            assert effective_max_positions(name, regime).max_positions >= 1


def test_aggressive_still_shrinks_in_a_choppy_tape():
    b = effective_max_positions("aggressive", Regime.SIDEWAYS)
    assert b.max_positions == 3          # round(8 * 0.40)
    assert b.max_positions < PROFILES["aggressive"]


def test_regime_scale_table_is_monotonic():
    """Risk appetite must never rise as conditions deteriorate."""
    order = [Regime.BULL, Regime.WEAK_BULL, Regime.SIDEWAYS, Regime.BEAR, Regime.PANIC]
    scales = [REGIME_SCALE[r] for r in order]
    assert scales == sorted(scales, reverse=True)


def test_budget_reason_is_explanatory():
    b = effective_max_positions("balanced", Regime.SIDEWAYS)
    assert "balanced" in b.reason and b.regime == "sideways"
