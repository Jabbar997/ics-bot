"""Command service — the text/business logic behind every Telegram command.

Kept free of any python-telegram-bot types so it can be unit-tested directly and
reused by the CLI. ``bot.py`` is a thin adapter that maps Telegram updates to
these methods and enforces authorization.

All user-facing text is Arabic; command names and ticker symbols stay in Latin
script.
"""
from __future__ import annotations

from app.config import Config
from app.db.database import session_scope
from app.db.repositories import (
    AuditRepository,
    DecisionRepository,
    PositionRepository,
)
from app.performance import evaluator
from app.risk.kill_switch import KillSwitchManager
from app.telegram.reports import ar_action, build_daily_report, format_weekly_report
from app.utils.money import money_str, pct_str

SAFETY = "تداول ورقي فقط — محاكاة تعليمية. ليست نصيحة مالية."


class CommandService:
    def __init__(self, config: Config):
        self.config = config

    def start(self) -> str:
        return (
            "ICS — نظام قيادة الاستثمار (وضع المحلّل/المتدرّب).\n"
            f"{SAFETY}\n\n"
            "الأوامر: /start /status /portfolio /positions /today /weekly /rules "
            "/watchlist /audit /rejected /performance /kill /stop /resume /commands /menu"
        )

    def commands(self) -> str:
        return self.start()

    # /help يُعترض من بعض عملاء تيليجرام قبل وصوله للبوت، فـ /commands و /menu
    # هي البدائل المدعومة لعرض قائمة الأوامر.
    menu = commands
    help = commands  # alias داخلي غير ضار

    def status(self) -> str:
        with session_scope() as s:
            ks = KillSwitchManager(s)
            frozen = ks.is_frozen()
            level = ks.active_level()
        return (
            "حالة ICS\n"
            f"الوضع: {self.config.mode}\n"
            f"النظام مُجمّد: {'نعم' if frozen else 'لا'}\n"
            f"مفتاح الإيقاف: {'المستوى ' + str(level.value) if level.value else 'غير مفعّل'}\n"
            f"{SAFETY}"
        )

    def portfolio(self) -> str:
        from app.paper.portfolio import Portfolio

        with session_scope() as s:
            pf = Portfolio(s, self.config.initial_capital)
            cash = pf.calculate_cash()
            invested = pf.calculate_invested_value()
            total = pf.calculate_total_value()
            ret = pf.total_return_pct()
        return (
            "محفظة ICS\n"
            f"النقد: {money_str(cash)}\n"
            f"المستثمَر: {money_str(invested)}\n"
            f"الإجمالي: {money_str(total)}\n"
            f"العائد الإجمالي: {pct_str(ret)}\n"
            f"رأس المال الابتدائي: {money_str(self.config.initial_capital)}"
        )

    def positions(self) -> str:
        with session_scope() as s:
            positions = PositionRepository(s).open_positions()
            if not positions:
                return "لا توجد مراكز مفتوحة."
            lines = ["المراكز المفتوحة:"]
            for i, p in enumerate(positions, 1):
                cur = float(p.current_price or p.entry_price)
                pnl = cur / float(p.entry_price) - 1.0
                lines.append(
                    f"{i}. {p.ticker} | الدخول {money_str(p.entry_price)} | "
                    f"الآن {money_str(cur)} | {pct_str(pnl)} | DQS {p.dqs_at_entry}"
                )
            return "\n".join(lines)

    def today(self, spy_today_pct: float = 0.0) -> str:
        with session_scope() as s:
            return build_daily_report(s, self.config, spy_today_pct=spy_today_pct)

    def weekly(self, spy_weekly: float = 0.0) -> str:
        from app.db.repositories import PortfolioSnapshotRepository

        with session_scope() as s:
            snaps = PortfolioSnapshotRepository(s).all()
            closed = PositionRepository(s).closed_positions()
            from sqlalchemy import select

            from app.db.models import Decision

            decisions = list(s.scalars(select(Decision)))
            perf = evaluator.evaluate(
                period="weekly",
                snapshots=snaps,
                closed_positions=closed,
                decisions=decisions,
                initial_capital=self.config.initial_capital,
                spy_return=spy_weekly,
            )
            violations = sum(1 for d in decisions if d.rule_violated)
            return format_weekly_report(perf, spy_weekly, rule_violations=violations)

    def rules(self) -> str:
        r = self.config.risk
        return (
            "قواعد المخاطر — ICS\n"
            f"أقصى عدد للمراكز المفتوحة: {r.max_open_positions}\n"
            f"أقصى حجم للمركز: {r.max_position_size_pct}%\n"
            f"حد الخسارة الأسبوعي: -{r.weekly_loss_limit_pct}%\n"
            f"حد الخسارة الشهري: -{r.monthly_loss_limit_pct}%\n"
            f"حد أقصى التراجع: -{r.max_drawdown_limit_pct}%\n"
            f"أدنى DQS: {r.minimum_dqs} (المتوسط المستهدف {r.target_average_dqs})\n"
            f"وقف الخسارة: min({r.stop_loss_atr_multiplier}×ATR، {r.absolute_stop_loss_pct}%)\n"
            "ممنوع: الرافعة، البيع المكشوف، الخيارات، العملات الرقمية، العقود الآجلة، الفوركس، الأسهم الرخيصة."
        )

    def watchlist(self) -> str:
        return "قائمة المتابعة (" + str(len(self.config.watchlist)) + "):\n" + "، ".join(self.config.watchlist)

    def audit(self, limit: int = 10) -> str:
        with session_scope() as s:
            rows = AuditRepository(s).recent(limit)
            if not rows:
                return "سجل التدقيق فارغ."
            lines = [f"سجل التدقيق (آخر {len(rows)}):"]
            for a in rows:
                lines.append(
                    f"- {a.timestamp:%Y-%m-%d} {ar_action(a.action)} {a.ticker} "
                    f"| DQS {a.dqs_score} | {a.strategy}"
                )
            return "\n".join(lines)

    def rejected(self, limit: int = 15) -> str:
        with session_scope() as s:
            rows = DecisionRepository(s).rejected_opportunities(limit)
            if not rows:
                return "لا توجد فرص مرفوضة مسجّلة."
            lines = [f"الفرص المرفوضة (آخر {len(rows)}):"]
            for d in rows:
                lines.append(f"- {d.ticker} | DQS {d.dqs_score} | {d.rejection_reason or d.reason}")
            return "\n".join(lines)

    def performance(self) -> str:
        return self.weekly()

    # -- control commands ------------------------------------------------- #
    def stop(self) -> str:
        with session_scope() as s:
            KillSwitchManager(s).manual_stop()
        return (
            "🛑 تم تفعيل الإيقاف اليدوي.\n"
            "تجميد فوري. إلغاء أي دخول جديد. لا مراكز حقيقية لإغلاقها (تداول ورقي فقط).\n"
            "(النظام مُجمّد — لا دخول جديد.)"
        )

    def kill(self) -> str:
        with session_scope() as s:
            ks = KillSwitchManager(s)
            level = ks.active_level()
            frozen = ks.is_frozen()
        if frozen or level.value:
            return (
                f"مفتاح الإيقاف مفعّل (المستوى {level.value}، مُجمّد={'نعم' if frozen else 'لا'}). "
                "استخدم /resume لإلغائه."
            )
        return "مفتاح الإيقاف غير مفعّل. استخدم /stop لتجميد النظام فورًا."

    def resume(self) -> str:
        with session_scope() as s:
            cleared = KillSwitchManager(s).resume()
        return f"تمت إعادة تشغيل النظام. تم مسح {cleared} حدث إيقاف. الدخول مسموح مجددًا."
