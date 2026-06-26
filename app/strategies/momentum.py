"""Momentum strategy.

Entry: close > MA20, close > MA50, RSI 50-70, volume > Volume_MA20,
regime is Bull or Weak Bull.
Exit: RSI deteriorates, close breaks below MA20 then MA50, stop, or target.
"""
from __future__ import annotations

from typing import Optional

from app.domain import FeatureSnapshot, Regime, RegimeResult, Signal, StrategyName
from app.strategies.base import Strategy, clamp, rsi_centeredness

RSI_LOW, RSI_HIGH = 50.0, 70.0


class MomentumStrategy(Strategy):
    name = StrategyName.MOMENTUM

    def evaluate(self, s: FeatureSnapshot, regime: RegimeResult) -> Signal:
        conds = {
            "ticker": s.ticker,
            "close": s.close,
            "ma20": s.ma20,
            "ma50": s.ma50,
            "rsi14": s.rsi14,
            "volume": s.volume,
            "volume_ma20": s.volume_ma20,
            "regime": regime.regime.value,
        }
        if not s.is_complete() or s.ma20 is None:
            return self._hold(s.ticker, "بيانات غير مكتملة.", conds)

        if regime.regime not in (Regime.BULL, Regime.WEAK_BULL):
            return self._hold(s.ticker, "الزخم يتداول فقط في السوق الصاعد/الصاعد الضعيف.", conds)

        core = s.close > s.ma20 and s.close > s.ma50
        if not core:
            return self._hold(s.ticker, "السعر ليس فوق MA20/MA50.", conds)

        rsi_ok = RSI_LOW <= s.rsi14 <= RSI_HIGH
        volume_ok = s.volume > s.volume_ma20
        if not rsi_ok:
            why = "RSI في تشبّع شرائي" if s.rsi14 > RSI_HIGH else "RSI ضعيف جدًا"
            return self._reject(s.ticker, f"إعداد زخم لكن {why} ({s.rsi14:.0f}).", conds)
        if not volume_ok:
            return self._reject(s.ticker, "إعداد زخم لكن الحجم لا يتوسّع.", conds)

        confidence = 0.58
        confidence += 0.18 * rsi_centeredness(s.rsi14, RSI_LOW, RSI_HIGH)
        confidence += clamp((s.volume / s.volume_ma20) - 1.0, 0.0, 1.0) * 0.12
        if regime.regime == Regime.BULL:
            confidence += 0.08
        reason = (
            f"زخم: الإغلاق {s.close:.2f} > MA20 {s.ma20:.2f} > MA50 {s.ma50:.2f}، "
            f"RSI {s.rsi14:.0f}، حجم متوسّع في سوق {regime.regime.value}."
        )
        return self._buy(s.ticker, reason, conds, confidence)

    def should_exit(self, s, regime, entry_price, stop_loss, target_price) -> Optional[Signal]:
        base = super().should_exit(s, regime, entry_price, stop_loss, target_price)
        if base:
            return base
        if s.ma20 is not None and s.ma50 is not None and s.close < s.ma20 and s.close < s.ma50:
            conds = {"ticker": s.ticker, "close": s.close, "ma20": s.ma20, "ma50": s.ma50}
            return self._exit("كسر الإغلاق أسفل MA20 ثم MA50.", conds, confidence=0.72)
        if s.rsi14 is not None and s.rsi14 < 45:
            conds = {"ticker": s.ticker, "rsi14": s.rsi14}
            return self._exit("تدهور الزخم (RSI) دون 45.", conds, confidence=0.6)
        return None
