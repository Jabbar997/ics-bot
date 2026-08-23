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
_engine_url: Optional[str] = None


def normalize_database_url(url: str) -> str:
    """Normalise a DATABASE_URL to a SQLAlchemy-compatible driver URL.

    Managed Postgres providers (Render, Heroku, ...) hand out ``postgres://`` or
    ``postgresql://`` URLs. SQLAlchemy 2.x needs an explicit driver, so we map
    both to the psycopg (v3) driver. SQLite URLs are returned unchanged.
    """
    if url.startswith("postgres://"):
        return "postgresql+psycopg://" + url[len("postgres://") :]
    if url.startswith("postgresql://"):
        return "postgresql+psycopg://" + url[len("postgresql://") :]
    return url


def _json_default(obj: Any):
    """Make JSON columns tolerant of Decimal / datetime (common in our context)."""
    if isinstance(obj, Decimal):
        return float(obj)
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    return str(obj)


def _json_serializer(obj: Any) -> str:
    return json.dumps(obj, default=_json_default)


def init_engine(
    database_url: str = "sqlite:///./ics.db",
    echo: bool = False,
    force_reset: bool = False,
) -> Engine:
    """Create the global engine and session factory.

    Idempotent: if an engine for the same URL already exists it is reused, so a
    workflow that calls ``init_engine`` again (e.g. the scheduled report job)
    never tears down a pool that a live command handler is using.
    """
    global _engine, _SessionFactory, _engine_url

    url = normalize_database_url(database_url)
    # `force_reset` exists for the backtester: reusing a cached engine for the
    # same URL (especially sqlite:///:memory:) silently carried one run's rows
    # into the next, so successive scenarios accumulated instead of starting
    # clean. The live bot still wants the idempotent path.
    if _engine is not None and _engine_url == url and not force_reset:
        return _engine
    if _engine is not None:
        _engine.dispose()

    is_sqlite = url.startswith("sqlite")
    # SQLite: allow cross-thread use (APScheduler / Telegram run in their own).
    connect_args = {"check_same_thread": False} if is_sqlite else {}

    engine_kwargs: dict = dict(
        echo=echo,
        future=True,
        connect_args=connect_args,
        json_serializer=_json_serializer,
        pool_pre_ping=True,  # drop dead connections instead of erroring (stability)
    )
    if not is_sqlite:
        # Recycle Postgres connections before managed providers time them out.
        engine_kwargs["pool_recycle"] = 1800

    _engine = create_engine(url, **engine_kwargs)
    _SessionFactory = sessionmaker(bind=_engine, expire_on_commit=False, future=True)
    _engine_url = url
    return _engine


def dialect_name() -> str:
    """Return the active database dialect ('sqlite' / 'postgresql'). No secrets."""
    return get_engine().dialect.name


def ping() -> bool:
    """Best-effort connectivity check used by /health."""
    from sqlalchemy import text

    try:
        with get_engine().connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


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
