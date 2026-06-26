"""Audit log + rejected-opportunity logging tests.

Core invariant: every Decision the engine records also creates exactly one
AuditLog. A decision without an audit log is invalid.
"""
from datetime import datetime

from app.config import load_config
from app.db.database import session_scope
from app.db.repositories import AuditRepository, DecisionRepository
from app.decision.decision_engine import DecisionEngine
from app.domain import Action, FeatureSnapshot, Regime, RegimeResult, Signal, StrategyName
from app.paper.broker import PaperBroker
from app.risk.risk_manager import PortfolioState


def _bull():
    return RegimeResult(Regime.BULL, 500, 492, 450, -0.01, "bull")


def _good_features(ticker="AAPL"):
    return FeatureSnapshot(
        ticker=ticker, as_of=datetime(2024, 1, 2), close=190.0, ma20=185, ma50=180,
        ma200=160, rsi14=55.0, macd=1.2, macd_signal=0.8, atr14=3.5, volume=2_000_000,
        volume_ma20=1_500_000, high_20d=192, drawdown_20d=-0.01,
    )


def _buy_signal(ticker="AAPL"):
    return Signal(
        ticker=ticker, strategy=StrategyName.TREND, action=Action.BUY,
        reason="Up-trend with healthy RSI and confirming volume across the board.",
        confidence=0.85, raw_conditions={"close": 190, "ma50": 180, "rsi14": 55},
    )


def test_every_decision_creates_one_audit_log(db_url):
    cfg = load_config()
    with session_scope() as s:
        broker = PaperBroker(s, cfg)
        de = DecisionEngine(s, cfg, broker)
        de.process_signal(_buy_signal(), _good_features(), _bull(), PortfolioState(266.0, 266.0), [])
        decisions = list(DecisionRepository(s).for_day(datetime(2024, 1, 2)))
    with session_scope() as s:
        from sqlalchemy import func, select

        from app.db.models import AuditLog, Decision

        n_decisions = s.scalar(select(func.count()).select_from(Decision))
        n_audits = s.scalar(select(func.count()).select_from(AuditLog))
        assert n_decisions == n_audits == 1
        audit = s.scalars(select(AuditLog)).first()
        assert audit.decision_id is not None
        assert audit.raw_context_json is not None  # full context captured


def test_rejected_opportunity_is_logged(db_url):
    cfg = load_config()
    with session_scope() as s:
        broker = PaperBroker(s, cfg)
        de = DecisionEngine(s, cfg, broker)
        # A near-miss REJECT signal must be recorded as a rejected opportunity.
        reject = Signal(
            ticker="NVDA", strategy=StrategyName.TREND, action=Action.REJECT,
            reason="Trend present but RSI high (75).", confidence=0.3,
            raw_conditions={"rsi14": 75},
        )
        de.process_signal(reject, _good_features("NVDA"), _bull(), PortfolioState(266.0, 266.0), [])
        rejected = DecisionRepository(s).rejected_opportunities()
        assert len(rejected) == 1
        assert rejected[0].ticker == "NVDA"
        assert rejected[0].rejected_opportunity is True
        assert rejected[0].action == "reject"
        # And it still produced an audit log.
        assert AuditRepository(s).count() == 1


def test_low_dqs_buy_is_rejected_and_logged(db_url):
    cfg = load_config()
    with session_scope() as s:
        broker = PaperBroker(s, cfg)
        de = DecisionEngine(s, cfg, broker)
        # Bear regime + bad timing pushes DQS under 70 even for a BUY action.
        bad_feat = FeatureSnapshot(
            ticker="AAPL", as_of=datetime(2024, 1, 2), close=10.0, ma20=9, ma50=9,
            ma200=8, rsi14=78.0, macd=-1, macd_signal=0.5, atr14=1.0, volume=100,
            volume_ma20=900, drawdown_20d=-0.04,
        )
        weak_buy = Signal(
            ticker="AAPL", strategy=StrategyName.TREND, action=Action.BUY,
            reason="weak", confidence=0.25, raw_conditions={"x": 1},
        )
        de.process_signal(weak_buy, bad_feat, RegimeResult(Regime.BEAR, 1, 1, 1, -0.04, "bear"),
                          PortfolioState(266.0, 266.0), [])
        rejected = DecisionRepository(s).rejected_opportunities()
        assert len(rejected) == 1
        assert rejected[0].action == "reject"
        # No position should have been opened.
        from app.db.repositories import PositionRepository

        assert PositionRepository(s).get_open_by_ticker("AAPL") is None
