"""Tiered risk profiles with regime scaling.

Two ideas, kept separate on purpose:

* **Profile** — how much risk the operator is willing to run at all
  (conservative / balanced / aggressive). A human choice.
* **Regime scaling** — how much of that budget the market currently deserves.
  The regime analyser already exists but is only used as an on/off gate; this
  turns it into a dial, so the system expands in a bull market and contracts as
  conditions weaken, instead of holding one fixed size everywhere.

Only the *number of concurrent tactical positions* is scaled. Position size,
stop-loss, loss limits and the kill switch are untouched: this widens or narrows
exposure, it never loosens a safety rule.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from app.domain import Regime

# Named profiles: the ceiling on concurrent tactical positions.
PROFILES: Dict[str, int] = {
    "conservative": 3,
    "balanced": 5,
    "aggressive": 8,
}

# Share of the profile's ceiling that each regime is allowed to use.
REGIME_SCALE: Dict[Regime, float] = {
    Regime.BULL: 1.00,
    Regime.WEAK_BULL: 0.60,
    Regime.SIDEWAYS: 0.40,
    Regime.BEAR: 0.0,
    Regime.PANIC: 0.0,
}


@dataclass
class RiskBudget:
    profile: str
    regime: str
    base_max_positions: int
    max_positions: int
    reason: str


def profile_ceiling(profile: str) -> int:
    return PROFILES.get((profile or "").lower(), PROFILES["conservative"])


def effective_max_positions(
    profile: str,
    regime: Optional[Regime],
    *,
    regime_scaling: bool = True,
    floor: int = 1,
) -> RiskBudget:
    """Concurrent tactical positions allowed right now.

    With scaling off this is just the profile ceiling. With it on, a weakening
    regime shrinks the ceiling; bear and panic already block entries elsewhere,
    and return 0 here so the two agree.
    """
    base = profile_ceiling(profile)
    regime_name = regime.value if regime is not None else "unknown"

    if not regime_scaling or regime is None:
        return RiskBudget(profile, regime_name, base, base, "بلا تدرّج حسب السوق.")

    scale = REGIME_SCALE.get(regime, 0.40)
    if scale <= 0:
        return RiskBudget(profile, regime_name, base, 0, f"سوق {regime_name}: لا دخول.")

    scaled = int(round(base * scale))
    scaled = max(floor, min(base, scaled))
    return RiskBudget(
        profile, regime_name, base, scaled,
        f"ملف {profile} ({base}) × سوق {regime_name} ({scale:.0%}) = {scaled} مركزًا.",
    )
