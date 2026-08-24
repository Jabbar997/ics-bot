"""Command service — the text/business logic behind every Telegram command.

Kept free of any python-telegram-bot types so it can be unit-tested directly and
reused by the CLI. ``bot.py`` is a thin adapter that maps Telegram updates to
these methods and enforces authorization.

All user-facing text is Arabic; command names and ticker symbols stay in Latin
script.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional

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

    def weekly(self, spy_weekly: Optional[float] = None, days: int = 7) -> str:
        """Weekly report over the last ``days`` days (v1.2).

        Previously this aggregated the ENTIRE history and labelled it "weekly".
        Now the window is filtered, and ``spy_weekly`` is computed when not
        supplied so the manual /weekly no longer shows SPY as +0.00%.
        """
        from sqlalchemy import select

        from app.db.models import Decision
        from app.db.repositories import PortfolioSnapshotRepository

        if spy_weekly is None:
            spy_weekly = self._spy_period_return(days)

        cutoff = datetime.utcnow() - timedelta(days=days)

        with session_scope() as s:
            snap_repo = PortfolioSnapshotRepository(s)
            snaps = [x for x in snap_repo.all() if x.ts and x.ts >= cutoff]
            if len(snaps) < 2:
                # Not enough history in the window — fall back to everything so
                # the report is still meaningful rather than empty.
                snaps = snap_repo.all()

            closed = [
                p for p in PositionRepository(s).closed_positions()
                if p.exit_at and p.exit_at >= cutoff
            ]
            decisions = [
                d for d in s.scalars(select(Decision))
                if d.created_at and d.created_at >= cutoff
            ]
            perf = evaluator.evaluate(
                period="weekly",
                snapshots=snaps,
                closed_positions=closed,
                decisions=decisions,
                initial_capital=self.config.initial_capital,
                spy_return=spy_weekly,
            )
            # `rule_violated` marks orders the RISK MANAGER BLOCKED — i.e. rules
            # enforced, not broken. Actual violations (an order that executed
            # despite breaking a rule) are structurally impossible here.
            risk_blocks = sum(1 for d in decisions if d.rule_violated)
            return format_weekly_report(
                perf, spy_weekly, risk_blocks=risk_blocks, actual_violations=0
            )

    def _spy_period_return(self, days: int) -> float:
        """Benchmark return over the window. Never raises (0.0 on failure)."""
        try:
            from app.data.market_data import fetch_history
            from app.performance.benchmarks import spy_return_between

            spy = fetch_history(self.config.benchmark.symbol, period="3mo")
            return spy_return_between(spy, start=datetime.utcnow() - timedelta(days=days))
        except Exception:
            return 0.0

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

    def learning(self, limit: int = 5) -> str:
        """ICS-DOC-004 Phase 0 — current DQS weights and recent learning events."""
        from sqlalchemy import select

        from app.db.models import DecisionOutcome, LearningEvent
        from app.learning.feedback_loop import MAX_SHIFT_PCT, MIN_CLOSED_TRADES
        from app.learning.weights import load_weights

        with session_scope() as s:
            weights = load_weights(s)
            outcomes = list(s.scalars(select(DecisionOutcome)))
            n_outcomes = len(outcomes)
            events = list(
                s.scalars(select(LearningEvent).order_by(LearningEvent.timestamp.desc()).limit(limit))
            )
            rows = [
                (e.timestamp, e.event_type, e.applied, e.trades_considered, e.reason)
                for e in events
            ]

        lines = [
            "🧠 حلقة التعلّم — ICS",
            "",
            "أوزان DQS الحالية:",
        ]
        for name, w in weights.items():
            lines.append(f"  • {name}: {w:.2f}")
        lines.append(f"  المجموع: {sum(weights.values()):.2f}")
        lines += [
            "",
            f"صفقات مغلقة مُسجَّلة: {n_outcomes} (الحد الأدنى للتعديل: {MIN_CLOSED_TRADES})",
            f"سقف التعديل لكل مكوّن في الدورة: ±{MAX_SHIFT_PCT}%",
            "",
        ]
        if not rows:
            lines.append("لا توجد دورات تعلّم مسجّلة بعد.")
        else:
            lines.append(f"آخر {len(rows)} دورة:")
            for ts, etype, applied, trades, reason in rows:
                mark = "✅" if applied else "⏸"
                lines.append(f"  {mark} {ts:%Y-%m-%d} | {etype} | صفقات {trades}")
                if reason:
                    lines.append(f"     {reason}")
        lines += self._calibration_lines()
        lines += ["", "تداول ورقي فقط — التعلّم يعدّل التقييم لا المخاطر."]
        return "\n".join(lines)

    # -- filter calibration from rejected decisions ----------------------- #
    AR_CATEGORY = {
        "strategy_filter": "فلاتر الاستراتيجيات",
        "dqs_below_threshold": "عتبة DQS",
        "slots_full": "امتلاء الخانات",
        "already_holding": "مركز مفتوح",
        "risk_limit": "حدود المخاطر",
        "other": "أخرى",
    }

    def _calibration_lines(self) -> list:
        """What the market did after the system said no."""
        from app.learning.counterfactuals import LEARNABLE, analyze_calibration

        try:
            with session_scope() as s:
                rep = analyze_calibration(s)
        except Exception:
            return []

        if not rep.total:
            return ["", "معايرة الفلاتر: لا قرارات مرفوضة مُقاسة بعد."]

        out = [
            "",
            f"📐 معايرة الفلاتر ({rep.total:,} قرارًا مرفوضًا، أفق {rep.horizon_days} أيام):",
        ]
        for cat, stat in sorted(rep.by_category.items(), key=lambda kv: -kv[1].n):
            if cat not in LEARNABLE:
                continue
            name = self.AR_CATEGORY.get(cat, cat)
            out.append(
                f"  • {name}: {stat.n:,} | صواب {stat.hit_rate*100:.0f}% | "
                f"متوسط الحركة بعده {stat.mean_forward_return*100:+.2f}%"
            )
            out.append(f"     {stat.verdict}")

        if rep.slots_full_n:
            out += [
                "",
                f"تكلفة سقف المراكز: {rep.slots_full_n:,} فرصة فاتت، "
                f"متوسط حركتها {rep.slots_full_mean_return*100:+.2f}%",
            ]
        out.append("  (قياس معايرة — ليس ربحًا فائتًا: أخذها كان سيغيّر ما بعدها)")
        return out

    def health(self) -> str:
        """v1.1 health check. Never includes the DB URL or any secret."""
        from sqlalchemy import func, select

        from app.db import database
        from app.db.models import AuditLog, Decision
        from app.db.repositories import SystemConfigRepository

        db_ok = database.ping()
        db_line = f"🟢 متصلة ({database.dialect_name()})" if db_ok else "🔴 غير متصلة"

        with session_scope() as s:
            cfg = SystemConfigRepository(s)
            scheduler = cfg.get("scheduler_status", "غير معروف")
            bot_started = cfg.get("bot_started_at", "—")
            last_cycle = cfg.get("last_decision_cycle_at", "لم تُشغّل بعد")
            last_daily = cfg.get("last_daily_report_at", "لم يُرسل بعد")
            missing = cfg.get("last_missing_symbols", "") or ""
            n_dec = s.scalar(select(func.count()).select_from(Decision)) or 0
            n_aud = s.scalar(select(func.count()).select_from(AuditLog)) or 0
            ks = KillSwitchManager(s)
            level = ks.active_level()
            frozen = ks.is_frozen()

        invariant = "✅ متطابق" if n_dec == n_aud else "⚠️ غير متطابق"
        ks_line = f"المستوى {level.value}" if level.value else "غير مفعّل"
        if frozen:
            ks_line += " (مُجمّد)"

        return "\n".join(
            [
                "🩺 فحص صحة ICS",
                "",
                "البوت: 🟢 يعمل",
                f"المجدول: {scheduler}",
                f"قاعدة البيانات: {db_line}",
                f"الوضع: {self.config.mode} ✅",
                f"مفتاح الإيقاف: {ks_line}",
                "",
                f"آخر دورة قرار: {last_cycle}",
                f"آخر تقرير يومي: {last_daily}",
                ("بيانات مفقودة آخر دورة: " + missing) if missing
                else "اكتمال البيانات: ✅ كل الرموز",
                f"بدء البوت: {bot_started}",
                "",
                f"تطابق السجلات (قرارات=تدقيق): {invariant} ({n_dec}={n_aud})",
            ]
        )

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
