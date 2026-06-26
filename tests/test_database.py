"""v1.1 — database URL selection, SQLite fallback, and idempotent init-db."""
from app.db import database
from app.db.database import session_scope
from app.db.repositories import SystemConfigRepository


def test_normalize_postgres_urls():
    assert database.normalize_database_url("postgres://u:p@h:5432/d") == (
        "postgresql+psycopg://u:p@h:5432/d"
    )
    assert database.normalize_database_url("postgresql://u:p@h:5432/d") == (
        "postgresql+psycopg://u:p@h:5432/d"
    )
    # SQLite untouched.
    assert database.normalize_database_url("sqlite:///./ics.db") == "sqlite:///./ics.db"


def test_database_url_selection_postgres_dialect():
    # Engine is created lazily — no connection is made, so this is safe offline.
    eng = database.init_engine("postgresql+psycopg://u:p@localhost:5432/db")
    try:
        assert eng.dialect.name == "postgresql"
        assert eng.dialect.driver == "psycopg"
    finally:
        database.init_engine("sqlite:///:memory:")  # restore global state


def test_database_url_selection_from_render_style_url():
    eng = database.init_engine("postgres://u:p@localhost:5432/db")
    try:
        assert eng.dialect.name == "postgresql"  # scheme normalised + driver added
    finally:
        database.init_engine("sqlite:///:memory:")


def test_sqlite_fallback_works_locally(tmp_path):
    url = f"sqlite:///{tmp_path / 'fallback.db'}"
    database.init_engine(url)
    database.create_all()
    assert database.dialect_name() == "sqlite"
    assert database.ping() is True
    # Round-trip a row to prove the local DB is fully usable.
    with session_scope() as s:
        SystemConfigRepository(s).set("hello", "world")
    with session_scope() as s:
        assert SystemConfigRepository(s).get("hello") == "world"


def test_init_db_idempotent_does_not_wipe_existing(tmp_path):
    url = f"sqlite:///{tmp_path / 'idem.db'}"
    database.init_engine(url)
    database.create_all()  # empty DB
    with session_scope() as s:
        SystemConfigRepository(s).set("keep", "1")

    # Re-running create_all (what `init-db` does) on an existing DB must be a
    # no-op that preserves data, not a reset.
    database.create_all()
    with session_scope() as s:
        assert SystemConfigRepository(s).get("keep") == "1"
