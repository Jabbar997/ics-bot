"""Telegram bot adapter (python-telegram-bot v21+).

Security:
* Only ``telegram.allowed_user_ids`` may use the bot. Everyone else gets
  exactly "Unauthorized.".
* Errors are logged server-side; users never see stack traces or tokens.

python-telegram-bot is imported lazily so the rest of ICS (and the test suite)
does not require it at import time.
"""
from __future__ import annotations

from typing import Callable, List

from app.config import Config
from app.logging_config import get_logger
from app.telegram.commands import CommandService

log = get_logger(__name__)


class ICSBot:
    def __init__(self, config: Config):
        self.config = config
        self.service = CommandService(config)
        self.allowed: List[int] = list(config.telegram.allowed_user_ids)

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
        """Populate the Telegram command menu (the blue 'Menu' button)."""
        from telegram import BotCommand

        await app.bot.set_my_commands(
            [
                BotCommand("start", "ترحيب + تنبيه الأمان"),
                BotCommand("commands", "عرض كل الأوامر"),
                BotCommand("menu", "عرض كل الأوامر"),
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
        log.info("Telegram command menu registered (%d commands).", 16)

    async def _on_error(self, update, context):
        log.exception("Unhandled bot error: %s", context.error)

    def run(self):
        """Start long-polling. Blocks until interrupted."""
        if not self.allowed:
            log.warning("No allowed_user_ids configured — every request will be Unauthorized.")
        app = self.build_application()
        log.info("ICS Telegram bot starting (paper-only, %d authorized users).", len(self.allowed))
        app.run_polling()
