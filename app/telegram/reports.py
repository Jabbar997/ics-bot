"""Report builders (Arabic UI).

Pure string formatting (``format_daily_report`` / ``format_weekly_report``) is
kept separate from any Telegram or DB code so the exact report text can be reused
by the bot, the scheduler, and the CLI. All user-facing text is Arabic; command
names and ticker symbols stay in Latin script.
"""
from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from app.config import Config
from app.db.repositories import (
    DecisionRepository,
    KillSwitchRepository,
    PortfolioSnapshotRepository,
    PositionRepository,
    RegimeRepository,
)
from app.performance.evaluator import PerformanceResult
from app.utils.money import money_str, pct_str, signed_money_str

# Arabic display maps for regimes and actions.
AR_REGIME = {
    "bull": "صاعد",
    "weak_bull": "صاعد ضعيف",
    "sideways": "عرضي",
    "bear": "هابط",
    "panic": "ذعر",
    "unknown": "غير معروف",
}
AR_ACTION = {
    "buy": "شراء",
    "sell": "بيع",
    "hold": "احتفاظ",
    "cash": "نقد",
    "reject": "رفض",
}


def ar_regime(value: str) -> str:
    return AR_REGIME.get((value or "").lower(), value or "غير معروف")


def ar_action(value: str) -> str:
    return AR_ACTION.get((value or "").lower(), (value or "").upper())


def _pos_block(positions: List[dict]) -> str:
    if not positions:
        return "المراكز المفتوحة:\nلا يوجد"
    lines = ["المراكز المفتوحة:"]
    for i, p in enumerate(positions, 1):
        lines.append(f"{i}. {p['ticker']}")
        lines.append(f"   الدخول: {money_str(p['entry'])}")
        lines.append(f"   الحالي: {money_str(p['current'])}")
        lines.append(f"   ر/خ: {pct_str(p['pnl_pct'])}")
        lines.append(f"   DQS: {p.get('dqs', '-')}")
        lines.append(f"   وقف الخسارة: {money_str(p['stop']) if p.get('stop') else '-'}")
    return "\n".join(lines)


def _decisions_block(decisions: List[dict]) -> str:
    if not decisions:
        return "قرارات اليوم:\nلا يوجد"
    lines = ["قرارات اليوم:"]
    for d in decisions:
        action = ar_action(d["action"])
        base = f"- {action} {d['ticker']} | DQS {d['dqs']}"
        if d.get("reason") and d["action"].lower() in ("reject", "sell"):
            base += f" | السبب: {d['reason']}"
        lines.append(base)
    return "\n".join(lines)


def format_daily_report(
    *,
    capital: float,
    portfolio_value: float,
    daily_pnl: float,
    daily_pnl_pct: float,
    spy_today_pct: float,
    regime: str,
    positions: List[dict],
    decisions: List[dict],
    kill_switch: str = "غير مفعّل",
) -> str:
    return "\n".join(
        [
            "📊 تقرير ICS اليومي",
            "",
            "رأس المال:",
            money_str(capital),
            "",
            "المحفظة:",
            money_str(portfolio_value),
            "",
            "ربح/خسارة اليوم:",
            f"{signed_money_str(daily_pnl)} | {pct_str(daily_pnl_pct)}",
            "",
            "SPY اليوم:",
            pct_str(spy_today_pct),
            "",
            "حالة السوق:",
            ar_regime(regime),
            "",
            _pos_block(positions),
            "",
            _decisions_block(decisions),
            "",
            "مفتاح الإيقاف:",
            kill_switch,
            "",
            "حالة النظام:",
            "تداول ورقي فقط",
        ]
    )


def format_weekly_report(
    perf: PerformanceResult,
    spy_weekly: float,
    risk_blocks: int = 0,
    actual_violations: int = 0,
) -> str:
    """Weekly report.

    v1.2: ``العائد الأسبوعي`` is the return over the week only
    (``period_return``); the cumulative figure since inception is shown
    separately. ``risk_blocks`` counts orders the risk manager *prevented* —
    that is the system enforcing its rules, NOT rules being broken.
    """
    return "\n".join(
        [
            "📈 تقرير ICS الأسبوعي",
            "",
            "المحفظة الافتتاحية (بداية الأسبوع):",
            money_str(perf.starting_value),
            "",
            "المحفظة الختامية:",
            money_str(perf.ending_value),
            "",
            "العائد الأسبوعي:",
            pct_str(perf.period_return),
            "",
            "العائد التراكمي (منذ الانطلاق):",
            pct_str(perf.total_return),
            "",
            "SPY الأسبوعي:",
            pct_str(spy_weekly),
            "",
            "شارب (Sharpe):",
            f"{perf.sharpe_ratio:.2f}",
            "",
            "أقصى تراجع:",
            pct_str(perf.max_drawdown),
            "",
            "الصفقات:",
            f"رابحة: {perf.wins}",
            f"خاسرة: {perf.losses}",
            f"مرفوضة: {perf.rejected_opportunities}",
            "",
            "متوسط DQS:",
            f"{perf.average_dqs}",
            "",
            "أفضل قرار:",
            perf.best_decision or "-",
            "",
            "أسوأ قرار:",
            perf.worst_decision or "-",
            "",
            "قرارات أوقفها مدير المخاطر:",
            f"{risk_blocks} (النظام طبّق قواعده)",
            "",
            "مخالفات فعلية للقواعد:",
            f"{actual_violations} ✅" if actual_violations == 0 else f"{actual_violations} ⚠️",
            "",
            "التوصية:",
            perf.recommendation or "استمر في التداول الورقي. لا ترقية بعد.",
        ]
    )


# --------------------------------------------------------------------------- #
# DB-backed builders
# --------------------------------------------------------------------------- #
def build_daily_report(
    session,
    config: Config,
    spy_today_pct: float = 0.0,
    day: Optional[datetime] = None,
) -> str:
    snaps = PortfolioSnapshotRepository(session)
    latest = snaps.latest()
    positions = PositionRepository(session).open_positions()
    regime_row = RegimeRepository(session).latest()
    ks = KillSwitchRepository(session).active_event()

    portfolio_value = float(latest.total_value) if latest else config.initial_capital
    daily_pnl = float(latest.daily_pnl) if latest and latest.daily_pnl is not None else 0.0
    daily_pnl_pct = float(latest.daily_pnl_pct) if latest and latest.daily_pnl_pct is not None else 0.0

    pos_dicts = [
        {
            "ticker": p.ticker,
            "entry": float(p.entry_price),
            "current": float(p.current_price or p.entry_price),
            "pnl_pct": (float(p.current_price or p.entry_price) / float(p.entry_price) - 1.0),
            "dqs": p.dqs_at_entry,
            "stop": float(p.stop_loss) if p.stop_loss else None,
        }
        for p in positions
    ]

    day = day or (latest.ts if latest else datetime.utcnow())
    decisions = DecisionRepository(session).for_day(day)
    dec_dicts = [
        {"action": d.action, "ticker": d.ticker, "dqs": d.dqs_score, "reason": d.rejection_reason or d.reason}
        for d in decisions
    ]

    ks_status = "غير مفعّل" if not ks else f"المستوى {ks.level} مفعّل"
    return format_daily_report(
        capital=config.initial_capital,
        portfolio_value=portfolio_value,
        daily_pnl=daily_pnl,
        daily_pnl_pct=daily_pnl_pct,
        spy_today_pct=spy_today_pct,
        regime=regime_row.regime if regime_row else "unknown",
        positions=pos_dicts,
        decisions=dec_dicts,
        kill_switch=ks_status,
    )
