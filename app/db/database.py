"""Database engine and session management.

SQLite for the MVP. The repository layer (``repositories.py``) is the only thing
the rest of the app talks to, so migrating to PostgreSQL later is a matter of
changing ``DATABASE_URL`` — no business code changes.
"""
from __future__ import annotations

import json
from contextlib import contextmanager
from datetime import date, datetime
from decimal import Decimal
from typing import Any, Iterator, Optional

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from app.db.models import Base

_engine: Optional[Engine] = None
_SessionFactory: Optional[sessionmaker] = None


def _json_default(obj: Any):
    """Make JSON columns tolerant of Decimal / datetime (common in our context)."""
    if isinstance(obj, Decimal):
        return float(obj)
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    return str(obj)


def _json_serializer(obj: Any) -> str:
    return json.dumps(obj, default=_json_default)


def init_engine(database_url: str = "sqlite:///./ics.db", echo: bool = False) -> Engine:
    """Create the global engine and session factory.

    Idempotent: if an engine for the same URL already exists it is reused, so a
    workflow that calls ``init_engine`` again (e.g. the scheduled report job)
    never tears down a pool that a live command handler is using.
    """
    global _engine, _SessionFactory

    if _engine is not None and str(_engine.url) == database_url:
        return _engine

    connect_args = {}
    if database_url.startswith("sqlite"):
        # Allow use across threads (APScheduler / Telegram run in their own).
        connect_args = {"check_same_thread": False}

    _engine = create_engine(
        database_url,
        echo=echo,
        future=True,
        connect_args=connect_args,
        json_serializer=_json_serializer,
    )
    _SessionFactory = sessionmaker(bind=_engine, expire_on_commit=False, future=True)
    return _engine


def get_engine() -> Engine:
    if _engine is None:
        init_engine()
    assert _engine is not None
    return _engine


def create_all() -> None:
    """Create all tables (idempotent)."""
    Base.metadata.create_all(get_engine())


def drop_all() -> None:
    Base.metadata.drop_all(get_engine())


@contextmanager
def session_scope() -> Iterator[Session]:
    """Provide a transactional scope around a series of operations."""
    if _SessionFactory is None:
        init_engine()
    assert _SessionFactory is not None
    session = _SessionFactory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def new_session() -> Session:
    """Return a raw session (caller manages commit/close)."""
    if _SessionFactory is None:
        init_engine()
    assert _SessionFactory is not None
    return _SessionFactory()
