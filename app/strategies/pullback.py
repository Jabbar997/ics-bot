"""Pullback strategy (buy-the-dip within an up-trend).

Entry: close > MA200, close near MA50 (within 3%), RSI 35-50, regime not Panic.
Exit: failed rebound, close breaks below MA200, stop, or target.
"""
from __future__ import annotations

from typing import Optional

from app.domain import FeatureSnapshot, Regime, RegimeResult, Signal, StrategyName
from app.strategies.base import Strategy, clamp, rsi_centeredness

RSI_LOW, RSI_HIGH = 35.0, 50.0
NEAR_MA50_TOLERANCE = 0.03  # within 3% of MA50 counts as "near"


class PullbackStrategy(Strategy):
    name = StrategyName.PULLBACK

    def evaluate(self, s: FeatureSnapshot, regime: RegimeResult) -> Signal:
        conds = {
            "ticker": s.ticker,
            "close": s.close,
            "ma50": s.ma50,
            "ma200": s.ma200,
            "rsi14": s.rsi14,
            "regime": regime.regime.value,
        }
        if not s.is_complete():
            return self._hold(s.ticker, "بيانات غير مكتملة.", conds)

        if regime.regime == Regime.PANIC:
            return self._hold(s.ticker, "الارتداد مُعطّل في حالة الذعر.", conds)

        if s.close <= s.ma200:
            return self._hold(s.ticker, "دون MA200 — ليس اتجاهًا صاعدًا صحّيًا.", conds)

        near_ma50 = abs(s.close / s.ma50 - 1.0) <= NEAR_MA50_TOLERANCE
        conds["distance_to_ma50_pct"] = round((s.close / s.ma50 - 1.0) * 100, 2)
        if not near_ma50:
            return self._hold(s.ticker, "السعر ليس قريبًا من MA50؛ لا ارتداد.", conds)

        rsi_ok = RSI_LOW <= s.rsi14 <= RSI_HIGH
        if not rsi_ok:
            why = "RSI لم يبلغ التشبّع البيعي بعد" if s.rsi14 > RSI_HIGH else "RSI عميق جدًا"
            return self._reject(s.ticker, f"منطقة ارتداد لكن {why} ({s.rsi14:.0f}).", conds)

        confidence = 0.55
        confidence += 0.2 * rsi_centeredness(s.rsi14, RSI_LOW, RSI_HIGH)
        confidence += 0.15 * (1.0 - abs(s.close / s.ma50 - 1.0) / NEAR_MA50_TOLERANCE)
        confidence = clamp(confidence)
        reason = (
            f"ارتداد: الإغلاق {s.close:.2f} قرب MA50 {s.ma50:.2f}، فوق "
            f"MA200 {s.ma200:.2f}، RSI {s.rsi14:.0f} (تصحيح ضمن اتجاه صاعد)."
        )
        return self._buy(s.ticker, reason, conds, confidence)

    def should_exit(self, s, regime, entry_price, stop_loss, target_price) -> Optional[Signal]:
        base = super().should_exit(s, regime, entry_price, stop_loss, target_price)
        if base:
            return base
        if s.ma200 is not None and s.close < s.ma200:
            conds = {"ticker": s.ticker, "close": s.close, "ma200": s.ma200}
            return self._exit("كسر الإغلاق أسفل MA200 (ارتداد فاشل).", conds, confidence=0.75)
        return None
