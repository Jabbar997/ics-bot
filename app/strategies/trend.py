"""Trend-following strategy.

Entry: close > MA50, close > MA200, RSI 40-65, volume >= Volume_MA20,
regime not Bear/Panic.
Exit: close breaks below MA50, stop, target, or panic regime.
"""
from __future__ import annotations

from typing import Optional

from app.domain import FeatureSnapshot, Regime, RegimeResult, Signal, StrategyName
from app.strategies.base import Strategy, clamp, rsi_centeredness

RSI_LOW, RSI_HIGH = 40.0, 65.0


class TrendStrategy(Strategy):
    name = StrategyName.TREND

    def evaluate(self, s: FeatureSnapshot, regime: RegimeResult) -> Signal:
        conds = {
            "ticker": s.ticker,
            "close": s.close,
            "ma50": s.ma50,
            "ma200": s.ma200,
            "rsi14": s.rsi14,
            "volume": s.volume,
            "volume_ma20": s.volume_ma20,
            "regime": regime.regime.value,
        }
        if not s.is_complete():
            return self._hold(s.ticker, "بيانات غير مكتملة.", conds)

        if regime.regime in (Regime.BEAR, Regime.PANIC):
            return self._hold(s.ticker, "استراتيجية الاتجاه تتنحّى في سوق هابط/ذعر.", conds)

        core_trend = s.close > s.ma50 and s.close > s.ma200
        if not core_trend:
            return self._hold(s.ticker, "لا يوجد اتجاه صاعد مؤكّد (السعر دون المتوسطات).", conds)

        # Core trend present — now check quality gates (near-miss => REJECT).
        rsi_ok = RSI_LOW <= s.rsi14 <= RSI_HIGH
        volume_ok = s.volume >= s.volume_ma20
        if not rsi_ok:
            why = "RSI مرتفع" if s.rsi14 > RSI_HIGH else "RSI منخفض"
            return self._reject(s.ticker, f"اتجاه صاعد لكن {why} ({s.rsi14:.0f}).", conds)
        if not volume_ok:
            return self._reject(s.ticker, "اتجاه صاعد لكن الحجم دون متوسط 20 يومًا.", conds)

        confidence = 0.55
        if s.ma50 > s.ma200:
            confidence += 0.12  # golden-cross structure
        confidence += 0.18 * rsi_centeredness(s.rsi14, RSI_LOW, RSI_HIGH)
        confidence += clamp((s.volume / s.volume_ma20) - 1.0, 0.0, 1.0) * 0.10
        reason = (
            f"اتجاه صاعد: الإغلاق {s.close:.2f} > MA50 {s.ma50:.2f} > "
            f"MA200 {s.ma200:.2f}، RSI {s.rsi14:.0f}، الحجم يؤكّد."
        )
        return self._buy(s.ticker, reason, conds, confidence)

    def should_exit(self, s, regime, entry_price, stop_loss, target_price) -> Optional[Signal]:
        base = super().should_exit(s, regime, entry_price, stop_loss, target_price)
        if base:
            return base
        if s.ma50 is not None and s.close < s.ma50:
            conds = {"ticker": s.ticker, "close": s.close, "ma50": s.ma50}
            return self._exit("كسر الإغلاق أسفل MA50.", conds, confidence=0.7)
        return None
