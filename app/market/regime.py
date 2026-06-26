"""Market Regime Analyzer.

The regime is derived from SPY only and gates the whole strategy layer.

Classification rules (Panic always wins):

* **Panic**     — SPY is down more than 5% from its 20-day high.
* **Bull**      — close > MA50 and close > MA200.
* **Bear**      — close < MA200.
* **Weak Bull** — MA200 < close < MA50 while MA50 >= MA200 (healthy structure,
  shallow dip below the fast MA).
* **Sideways**  — price wedged between the MAs with MA50 < MA200 (crossed /
  choppy), or sitting exactly on an MA.

The Weak-Bull vs Sideways split uses the MA50/MA200 relationship to make the
two mutually exclusive, since the spec's bare definitions overlap.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd

from app.domain import FeatureSnapshot, Regime, RegimeResult

PANIC_DRAWDOWN_THRESHOLD = -0.05  # -5% from the 20-day high


def classify_regime(
    spy_close: float,
    spy_ma50: Optional[float],
    spy_ma200: Optional[float],
    drawdown_20d: float,
) -> RegimeResult:
    """Classify the market regime from SPY levels.

    ``drawdown_20d`` is a fraction (e.g. -0.06 for -6%).
    """
    # Panic overrides everything.
    if drawdown_20d is not None and drawdown_20d <= PANIC_DRAWDOWN_THRESHOLD:
        return RegimeResult(
            regime=Regime.PANIC,
            spy_close=spy_close,
            spy_ma50=spy_ma50,
            spy_ma200=spy_ma200,
            drawdown_20d=drawdown_20d,
            reason=f"SPY down {drawdown_20d * 100:.1f}% from its 20-day high (panic).",
        )

    if spy_ma50 is None or spy_ma200 is None:
        return RegimeResult(
            regime=Regime.SIDEWAYS,
            spy_close=spy_close,
            spy_ma50=spy_ma50,
            spy_ma200=spy_ma200,
            drawdown_20d=drawdown_20d,
            reason="Insufficient moving-average history; defaulting to sideways.",
        )

    # Evaluated in the spec's listed priority: Bull -> Weak Bull -> Sideways -> Bear.
    lo, hi = min(spy_ma50, spy_ma200), max(spy_ma50, spy_ma200)
    if spy_close > spy_ma50 and spy_close > spy_ma200:
        regime = Regime.BULL
        reason = "SPY above MA50 and MA200."
    elif spy_close > spy_ma200 and spy_close < spy_ma50:
        regime = Regime.WEAK_BULL
        reason = "SPY above MA200 but below MA50, MA structure still positive."
    elif lo <= spy_close <= hi:
        regime = Regime.SIDEWAYS
        reason = "SPY wedged between MA50 and MA200 (choppy)."
    else:
        regime = Regime.BEAR
        reason = "SPY below MA200."

    return RegimeResult(
        regime=regime,
        spy_close=spy_close,
        spy_ma50=spy_ma50,
        spy_ma200=spy_ma200,
        drawdown_20d=drawdown_20d,
        reason=reason,
    )


def analyze_from_snapshot(snapshot: FeatureSnapshot) -> RegimeResult:
    """Classify regime directly from a SPY :class:`FeatureSnapshot`."""
    return classify_regime(
        spy_close=snapshot.close,
        spy_ma50=snapshot.ma50,
        spy_ma200=snapshot.ma200,
        drawdown_20d=snapshot.drawdown_20d if snapshot.drawdown_20d is not None else 0.0,
    )


def analyze_from_features_df(spy_features: pd.DataFrame, as_of_index: int = -1) -> RegimeResult:
    """Classify regime from a row of SPY's computed-features frame."""
    row = spy_features.iloc[as_of_index]

    def _val(key):
        v = row.get(key)
        return None if (v is None or pd.isna(v)) else float(v)

    return classify_regime(
        spy_close=_val("close") or 0.0,
        spy_ma50=_val("ma50"),
        spy_ma200=_val("ma200"),
        drawdown_20d=_val("drawdown_20d") or 0.0,
    )
