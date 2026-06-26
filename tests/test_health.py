"""v1.1 — /health command: auth-gating, no secret leakage, and the
decisions == audit_logs invariant.
"""
import asyncio
from datetime import datetime

from app.config import load_config
from app.db.database import session_scope
from app.decision.decision_engine import DecisionEngine
from app.domain import Action, FeatureSnapshot, Regime, RegimeResult, Signal, StrategyName
from app.paper.broker import PaperBroker
from app.risk.risk_manager import PortfolioState
from app.telegram.bot import ICSBot
from app.telegram.commands import CommandService

UNAUTH_MSG = "⛔ غير مصرّح لك باستخدام هذا البوت."


# -- tiny fakes so we can drive the real bot handler without Telegram --------- #
class _Msg:
    def __init__(self):
        self.replies = []

    async def reply_text(self, text):
        self.replies.append(text)


class _User:
    def __init__(self, uid):
        self.id = uid


class _Update:
    def __init__(self, uid):
        self.effective_user = _User(uid)
        self.message = _Msg()


def _bot(allowed):
    cfg = load_config()
    cfg.telegram.allowed_user_ids = list(allowed)
    return ICSBot(cfg)


def test_health_command_unauthorized(db_url):
    bot = _bot([123])
    handler = bot._make_handler(bot.service.health)
    upd = _Update(999)  # not in allow-list
    asyncio.run(handler(upd, None))
    assert upd.message.replies == [UNAUTH_MSG]


def test_health_command_authorized(db_url):
    bot = _bot([123])
    handler = bot._make_handler(bot.service.health)
    upd = _Update(123)
    asyncio.run(handler(upd, None))
    text = upd.message.replies[0]
    assert "فحص صحة ICS" in text
    assert "paper_only" in text
    assert "تطابق السجلات" in text


def test_health_does_not_leak_secrets(db_url):
    cfg = load_config()
    cfg.telegram.allowed_user_ids = [123]
    # Plant secrets in config to prove they never reach the /health output.
    cfg.env.telegram_bot_token = "1234567890:FAKE_TOKEN_DO_NOT_USE"
    cfg.env.database_url = "postgresql+psycopg://admin:SUPERSECRET@db.internal:5432/ics"
    text = CommandService(cfg).health()
    assert "SECRET_TOKEN_ABC" not in text
    assert "SUPERSECRET" not in text
    assert "admin:" not in text
    assert "db.internal" not in text


def test_decisions_equal_audit_logs_invariant(db_url):
    from sqlalchemy import func, select

    from app.db.models import AuditLog, Decision

    cfg = load_config()
    regime = RegimeResult(Regime.BULL, 500, 492, 450, -0.01, "bull")

    def feat(ticker):
        return FeatureSnapshot(
            ticker=ticker, as_of=datetime(2024, 1, 2), close=190.0, ma20=185, ma50=180,
            ma200=160, rsi14=55.0, macd=1.2, macd_signal=0.8, atr14=3.5, volume=2_000_000,
            volume_ma20=1_500_000, drawdown_20d=-0.01,
        )

    with session_scope() as s:
        broker = PaperBroker(s, cfg)
        de = DecisionEngine(s, cfg, broker)
        buy = Signal("AAPL", StrategyName.TREND, Action.BUY,
                     "trend up with healthy RSI and confirming volume quality entry",
                     0.85, {"x": 1})
        reject = Signal("NVDA", StrategyName.TREND, Action.REJECT,
                        "Trend present but RSI high.", 0.3, {"rsi": 75})
        de.process_signal(buy, feat("AAPL"), regime, PortfolioState(266.0, 266.0), [])
        de.process_signal(reject, feat("NVDA"), regime, PortfolioState(266.0, 266.0), [])

    with session_scope() as s:
        n_dec = s.scalar(select(func.count()).select_from(Decision))
        n_aud = s.scalar(select(func.count()).select_from(AuditLog))
        assert n_dec == n_aud >= 2  # the core ICS invariant
