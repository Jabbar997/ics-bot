"""Telegram bot adapter (python-telegram-bot v21+).

Security:
* Only ``telegram.allowed_user_ids`` may use the bot. Everyone else gets
  exactly "Unauthorized.".
* Errors are logged server-side; users never see stack traces or tokens.

python-telegram-bot is imported lazily so the rest of ICS (and the test suite)
does not require it at import time.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Callable, List

from app.config import Config
from app.logging_config import get_logger
from app.telegram.commands import CommandService

log = get_logger(__name__)


def _now_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


class ICSBot:
    def __init__(self, config: Config):
        self.config = config
        self.service = CommandService(config)
        self.allowed: List[int] = list(config.telegram.allowed_user_ids)
        self._app = None        # the running telegram Application (set in post_init)
        self._scheduler = None  # AsyncIOScheduler for daily / weekly reports

    # -- auth ------------------------------------------------------------- #
    def is_authorized(self, user_id: int | None) -> bool:
        # Empty allow-list = locked down (no one authorized) for safety.
        return user_id is not None and user_id in self.allowed

    # -- application wiring ---------------------------------------------- #
    def build_application(self):
        from telegram.ext import (
            Application,
            CommandHandler,
            MessageHandler,
            filters,
        )

        token = self.config.env.telegram_bot_token
        if not token:
            raise RuntimeError(
                "TELEGRAM_BOT_TOKEN is not set. Configure .env before running the bot."
            )

        app = Application.builder().token(token).post_init(self._post_init).build()

        # Map each command name to a CommandService method.
        command_map: dict[str, Callable[[], str]] = {
            "start": self.service.start,
            "commands": self.service.commands,
            "menu": self.service.commands,
            "health": self.service.health,
            "status": self.service.status,
            "portfolio": self.service.portfolio,
            "positions": self.service.positions,
            "today": self.service.today,
            "weekly": self.service.weekly,
            "rules": self.service.rules,
            "watchlist": self.service.watchlist,
            "audit": self.service.audit,
            "rejected": self.service.rejected,
            "performance": self.service.performance,
            "kill": self.service.kill,
            "stop": self.service.stop,
            "resume": self.service.resume,
        }
        for name, fn in command_map.items():
            app.add_handler(CommandHandler(name, self._make_handler(fn)))

        # Plain-text "STOP" also triggers the manual kill switch.
        app.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, self._text_handler)
        )
        app.add_error_handler(self._on_error)
        return app

    def _make_handler(self, fn: Callable[[], str]):
        async def handler(update, context):
            user_id = update.effective_user.id if update.effective_user else None
            if not self.is_authorized(user_id):
                await update.message.reply_text("⛔ غير مصرّح لك باستخدام هذا البوت.")
                return
            try:
                text = fn()
            except Exception:  # never leak internals to the user
                log.exception("Command failed for user %s", user_id)
                text = "حدث خطأ داخلي. يُرجى مراجعة سجلّات الخادم."
            await update.message.reply_text(text)

        return handler

    async def _text_handler(self, update, context):
        user_id = update.effective_user.id if update.effective_user else None
        if not self.is_authorized(user_id):
            await update.message.reply_text("⛔ غير مصرّح لك باستخدام هذا البوت.")
            return
        msg = (update.message.text or "").strip().upper()
        # Plain "STOP" (English) or "توقف"/"إيقاف" (Arabic) triggers the manual kill switch.
        if msg in ("STOP", "توقف", "إيقاف", "ايقاف"):
            await update.message.reply_text(self.service.stop())
        else:
            await update.message.reply_text(
                "رسالة غير معروفة. أرسل /start لعرض قائمة الأوامر."
            )

    async def _post_init(self, app) -> None:
        """Populate the Telegram command menu and start the report scheduler."""
        from telegram import BotCommand

        self._app = app
        await app.bot.set_my_commands(
            [
                BotCommand("start", "ترحيب + تنبيه الأمان"),
                BotCommand("commands", "عرض كل الأوامر"),
                BotCommand("menu", "عرض كل الأوامر"),
                BotCommand("health", "فحص صحة النظام"),
                BotCommand("status", "حالة النظام"),
                BotCommand("portfolio", "قيمة المحفظة والعائد"),
                BotCommand("positions", "المراكز المفتوحة"),
                BotCommand("today", "التقرير اليومي"),
                BotCommand("weekly", "التقرير الأسبوعي"),
                BotCommand("performance", "تقرير الأداء"),
                BotCommand("rules", "قواعد المخاطر"),
                BotCommand("watchlist", "قائمة المتابعة"),
                BotCommand("audit", "سجل التدقيق الأخير"),
                BotCommand("rejected", "الفرص المرفوضة"),
                BotCommand("kill", "حالة مفتاح الإيقاف"),
                BotCommand("stop", "تجميد النظام (إيقاف يدوي)"),
                BotCommand("resume", "إعادة تشغيل النظام"),
            ]
        )
        log.info("Telegram command menu registered (%d commands).", 17)
        self._set_config("bot_started_at", _now_str())

        # Start the in-process scheduler so the SAME worker also pushes the
        # daily/weekly reports automatically (no separate service needed).
        if self.config.telegram.enabled and self.allowed:
            self._start_scheduler()
        else:
            self._set_config("scheduler_status", "غير مفعّل")
            log.warning("Scheduler not started (telegram disabled or no authorized users).")

    def _set_config(self, key: str, value: str) -> None:
        """Best-effort write of a runtime status flag (never raises into the loop)."""
        from app.db.database import session_scope
        from app.db.repositories import SystemConfigRepository

        try:
            with session_scope() as s:
                SystemConfigRepository(s).set(key, value)
        except Exception:
            log.exception("Failed to record runtime status '%s'", key)

    # -- scheduled reports ------------------------------------------------ #
    def _start_scheduler(self) -> None:
        from apscheduler.schedulers.asyncio import AsyncIOScheduler
        from apscheduler.triggers.cron import CronTrigger

        from app.utils.time import KSA, parse_hhmm

        hh, mm = parse_hhmm(self.config.telegram.daily_report_time_ksa)
        weekday = self.config.telegram.weekly_report_day.lower()[:3]
        weekly_minute = (mm + 5) % 60
        sh, sm = parse_hhmm(self.config.telegram.status_report_time_ksa)

        self._scheduler = AsyncIOScheduler(timezone=KSA)
        self._scheduler.add_job(
            self._run_daily_job,
            CronTrigger(hour=hh, minute=mm, timezone=KSA),
            id="daily_report",
            replace_existing=True,
        )
        self._scheduler.add_job(
            self._run_weekly_job,
            CronTrigger(day_of_week=weekday, hour=hh, minute=weekly_minute, timezone=KSA),
            id="weekly_report",
            replace_existing=True,
        )
        # v1.1: short daily status / state-backup heartbeat.
        self._scheduler.add_job(
            self._run_status_job,
            CronTrigger(hour=sh, minute=sm, timezone=KSA),
            id="status_report",
            replace_existing=True,
        )
        self._scheduler.start()
        self._set_config("scheduler_status", "🟢 فعّال (٣ مهام)")
        log.info(
            "Report scheduler started: daily %02d:%02d, weekly %s %02d:%02d, status %02d:%02d KSA.",
            hh, mm, weekday, hh, weekly_minute, sh, sm,
        )

    async def _broadcast(self, text: str) -> None:
        for uid in self.allowed:
            try:
                await self._app.bot.send_message(chat_id=uid, text=text)
            except Exception:
                log.exception("Failed to send scheduled report to %s", uid)

    async def _run_daily_job(self) -> None:
        import asyncio

        from app.main import run_daily_workflow

        log.info("Running scheduled DAILY workflow...")
        loop = asyncio.get_running_loop()
        try:
            # Blocking (network + DB) work runs in a thread so the bot keeps
            # answering commands; the workflow itself does not send (we broadcast).
            text = await loop.run_in_executor(
                None, lambda: run_daily_workflow(self.config, send_report=False)
            )
            await self._broadcast(text)
            self._set_config("last_daily_report_at", _now_str())
            log.info("Daily report sent to %d user(s).", len(self.allowed))
        except Exception:
            log.exception("Scheduled daily workflow failed")

    async def _run_weekly_job(self) -> None:
        import asyncio

        log.info("Running scheduled WEEKLY workflow...")
        loop = asyncio.get_running_loop()
        try:
            text = await loop.run_in_executor(None, self._build_weekly_text)
            await self._broadcast(text)
            log.info("Weekly report sent to %d user(s).", len(self.allowed))
        except Exception:
            log.exception("Scheduled weekly workflow failed")

    def _build_weekly_text(self) -> str:
        """Compute SPY's weekly return then build the weekly report (sync)."""
        from app.main import run_weekly_workflow

        spy_weekly = 0.0
        try:
            from app.data.market_data import fetch_history
            from app.performance.benchmarks import period_return

            spy = fetch_history(self.config.benchmark.symbol, period="1mo")
            spy_weekly = period_return(spy["close"].tail(6))
        except Exception:
            log.warning("Could not fetch SPY weekly return; using 0.")
        return run_weekly_workflow(self.config, spy_weekly=spy_weekly, send_report=False)

    async def _run_status_job(self) -> None:
        """v1.1 daily heartbeat: short status report + lightweight state backup."""
        import asyncio

        log.info("Running scheduled STATUS/backup heartbeat...")
        loop = asyncio.get_running_loop()
        try:
            text = await loop.run_in_executor(None, self._build_status_text)
            await self._broadcast(text)
            self._set_config("last_status_report_at", _now_str())
            log.info("Status/backup heartbeat sent to %d user(s).", len(self.allowed))
        except Exception:
            log.exception("Scheduled status heartbeat failed")

    def _build_status_text(self) -> str:
        """Build the short status text and persist a state-backup checkpoint (sync)."""
        import json

        from sqlalchemy import func, select

        from app.db import database
        from app.db.database import session_scope
        from app.db.models import AuditLog, Decision
        from app.db.repositories import PositionRepository, SystemConfigRepository
        from app.paper.portfolio import Portfolio

        db_ok = database.ping()
        with session_scope() as s:
            pf = Portfolio(s, self.config.initial_capital)
            total = float(pf.calculate_total_value())
            cash = float(pf.calculate_cash())
            open_positions = len(PositionRepository(s).open_positions())
            n_dec = s.scalar(select(func.count()).select_from(Decision)) or 0
            n_aud = s.scalar(select(func.count()).select_from(AuditLog)) or 0
            # Lightweight state-backup checkpoint (no secrets, just paper state).
            SystemConfigRepository(s).set(
                "last_state_backup",
                json.dumps(
                    {
                        "ts": _now_str(),
                        "total_value": total,
                        "cash": cash,
                        "open_positions": open_positions,
                        "decisions": n_dec,
                        "audit_logs": n_aud,
                    }
                ),
            )

        inv = "✅" if n_dec == n_aud else "⚠️"
        db_line = f"🟢 {database.dialect_name()}" if db_ok else "🔴 غير متصلة"
        return "\n".join(
            [
                "🩺 ICS — تقرير حالة يومي مختصر",
                f"الوضع: {self.config.mode}",
                f"قاعدة البيانات: {db_line}",
                f"المحفظة: ${total:,.2f} (نقد ${cash:,.2f})",
                f"مراكز مفتوحة: {open_positions}",
                f"قرارات/تدقيق: {n_dec}/{n_aud} {inv}",
                "تداول ورقي فقط.",
            ]
        )

    async def _on_error(self, update, context):
        log.exception("Unhandled bot error: %s", context.error)

    def run(self):
        """Start long-polling. Blocks until interrupted."""
        if not self.allowed:
            log.warning("No allowed_user_ids configured — every request will be Unauthorized.")
        app = self.build_application()
        log.info("ICS Telegram bot starting (paper-only, %d authorized users).", len(self.allowed))
        app.run_polling()
