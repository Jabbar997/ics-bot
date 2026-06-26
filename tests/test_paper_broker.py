"""Paper broker execution tests (virtual orders only)."""
from app.config import load_config
from app.db.database import session_scope
from app.db.repositories import OrderRepository, PositionRepository
from app.paper.broker import PaperBroker


def _cfg():
    return load_config()


def test_buy_reduces_cash_and_creates_position(db_url):
    cfg = _cfg()
    with session_scope() as s:
        broker = PaperBroker(s, cfg)
        assert float(broker.calculate_cash()) == cfg.initial_capital
        broker.place_buy_order("AAPL", quantity=0.1, price=190.0, strategy="trend", dqs=80)
        cash = float(broker.calculate_cash())
        assert cash < cfg.initial_capital
        assert abs(cash - (cfg.initial_capital - 0.1 * 190.0)) < 1e-6
        pos = PositionRepository(s).get_open_by_ticker("AAPL")
        assert pos is not None and pos.is_open and float(pos.quantity) == 0.1
        assert OrderRepository(s).all()[0].side == "buy"


def test_buy_caps_quantity_to_available_cash(db_url):
    cfg = _cfg()
    with session_scope() as s:
        broker = PaperBroker(s, cfg)
        # Request way more than $266 of notional; broker must cap to cash.
        broker.place_buy_order("AAPL", quantity=100, price=190.0)
        cash = float(broker.calculate_cash())
        assert cash >= -1e-6  # never goes negative (no leverage / margin)
        invested = float(broker.calculate_invested_value({"AAPL": 190.0}))
        total = float(broker.calculate_total_value({"AAPL": 190.0}))
        assert abs(total - cfg.initial_capital) < 1e-2


def test_close_realizes_profit_and_restores_cash(db_url):
    cfg = _cfg()
    with session_scope() as s:
        broker = PaperBroker(s, cfg)
        broker.place_buy_order("AAPL", quantity=0.5, price=100.0, strategy="trend")
        pos = PositionRepository(s).get_open_by_ticker("AAPL")
        broker.close_position(pos, price=120.0, reason="target")
        assert pos.is_open is False
        assert pos.realized_pnl > 0
        assert abs(pos.realized_pnl - (0.5 * (120.0 - 100.0))) < 1e-6
        assert abs(pos.realized_pnl_pct - 0.2) < 1e-6
        # Cash should reflect the realised gain.
        assert float(broker.calculate_cash()) > cfg.initial_capital


def test_total_value_constant_when_only_marking_to_market(db_url):
    cfg = _cfg()
    with session_scope() as s:
        broker = PaperBroker(s, cfg)
        broker.place_buy_order("AAPL", quantity=0.5, price=100.0)
        # Marking to market at the same price keeps total value unchanged.
        total = float(broker.calculate_total_value({"AAPL": 100.0}))
        assert abs(total - cfg.initial_capital) < 1e-2
