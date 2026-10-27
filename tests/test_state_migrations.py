"""Failed session-store migrations must abort startup, not degrade to 500s.

The column-backfill migrations in ``docint.core.state.base`` used to log a
warning and continue on failure. On a hardened deployment (deploy ADR 0001)
whose sessions volume was still owned by another uid, that left the app
serving with a stale schema: every ORM query selecting the new columns
failed with ``no such column`` — a wall of per-request 500s instead of one
actionable startup error.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Engine, create_engine, text

from docint.core.state import base as base_module
from docint.core.state import session_manager as session_manager_module
from docint.core.state.base import (
    SessionStoreMigrationError,
    _ensure_collection_owner_activity_column,
    _ensure_conversation_owner_column,
    _ensure_conversation_scope_columns,
    _ensure_report_columns,
    _ensure_turn_validation_columns,
    _make_session_maker,
)


def _readonly_url(db_path: Path) -> str:
    """Build a URL that has SQLite open an existing DB file read-only.

    Root ignores file permission bits, so ``chmod 0o444`` cannot make the file
    read-only for a test run as root. ``mode=ro`` makes SQLite refuse writes for
    every uid, with the same ``attempt to write a readonly database`` error a
    non-writable file raises.

    Args:
        db_path (Path): The SQLite database file.

    Returns:
        str: A SQLAlchemy URL for a read-only connection to ``db_path``.
    """
    return f"sqlite:///file:{db_path}?mode=ro&uri=true"


def _readonly_engine(tmp_path: Path, ddl: str) -> Engine:
    """Create a DB holding one pre-existing table and return an engine that can only read it.

    Args:
        tmp_path (Path): Directory to create ``sessions.sqlite3`` in.
        ddl (str): Statement that creates the table.

    Returns:
        Engine: A read-only engine over the new DB.
    """
    db_path = tmp_path / "sessions.sqlite3"
    engine = create_engine(f"sqlite:///{db_path}")
    with engine.begin() as conn:
        conn.execute(text(ddl))
    engine.dispose()
    return create_engine(_readonly_url(db_path))


@pytest.mark.parametrize(
    ("migrate", "ddl"),
    [
        pytest.param(
            _ensure_conversation_scope_columns,
            "CREATE TABLE conversations (id TEXT PRIMARY KEY, owner TEXT)",
            id="scope-columns",
        ),
        pytest.param(
            _ensure_conversation_owner_column,
            "CREATE TABLE conversations (id TEXT PRIMARY KEY)",
            id="owner-column",
        ),
        pytest.param(
            _ensure_turn_validation_columns,
            "CREATE TABLE turns (id TEXT PRIMARY KEY)",
            id="turn-validation-columns",
        ),
        pytest.param(
            _ensure_report_columns,
            "CREATE TABLE reports (id TEXT PRIMARY KEY)",
            id="report-columns",
        ),
        pytest.param(
            _ensure_collection_owner_activity_column,
            "CREATE TABLE collection_owners (physical_name TEXT PRIMARY KEY)",
            id="collection-owner-activity-column",
        ),
    ],
)
def test_migration_raises_loudly_on_a_readonly_db(
    tmp_path: Path,
    migrate: Callable[[Engine], None],
    ddl: str,
) -> None:
    """A migration that cannot write must raise, never warn-and-continue.

    The ORM models already select the new columns, so continuing serves
    guaranteed 500s on every query touching the table.
    """
    engine = _readonly_engine(tmp_path, ddl)

    with pytest.raises(SessionStoreMigrationError) as exc_info:
        migrate(engine)

    message = str(exc_info.value)
    assert "readonly" in message
    assert "sessions.sqlite3" in message


def test_readonly_failure_names_the_volume_ownership_fix(tmp_path: Path) -> None:
    """The readonly case is the hardened-deployment trap — say how to fix it."""
    engine = _readonly_engine(tmp_path, "CREATE TABLE conversations (id TEXT PRIMARY KEY, owner TEXT)")

    with pytest.raises(SessionStoreMigrationError) as exc_info:
        _ensure_conversation_scope_columns(engine)

    assert "chown" in str(exc_info.value)


def test_missing_table_is_still_not_an_error(tmp_path: Path) -> None:
    """A fresh DB has no tables yet; ``create_all`` handles that path."""
    engine = create_engine(f"sqlite:///{tmp_path / 'sessions.sqlite3'}")

    _ensure_conversation_scope_columns(engine)  # must not raise


def test_make_session_maker_propagates_a_failed_migration(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The store factory must not hand out a session maker over a stale schema."""
    db_path = tmp_path / "sessions.sqlite3"
    db_url = f"sqlite:///{db_path}"
    _make_session_maker(db_url)
    engine = create_engine(db_url)
    with engine.begin() as conn:
        conn.execute(text("ALTER TABLE conversations DROP COLUMN scope_chunk_ids"))
    engine.dispose()

    def _readonly_create_engine(_url: str, **kwargs: Any) -> Engine:
        return create_engine(_readonly_url(db_path), **kwargs)

    # Patch the engine, not the URL: ``_ensure_sqlite_parent_dir`` reads a
    # ``file:`` URI as a relative path and would create it under the working directory.
    monkeypatch.setattr(base_module, "create_engine", _readonly_create_engine)

    with pytest.raises(SessionStoreMigrationError):
        _make_session_maker(db_url)


def test_lifespan_startup_initializes_the_session_store(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Startup runs the store init (and thus the migrations) eagerly.

    The store used to initialize lazily on the first session request, so a
    broken migration surfaced minutes after boot as per-request 500s.
    """
    from docint.core import api as api_module
    from docint.core import rag as rag_module

    monkeypatch.setattr(rag_module.RAG, "probe_qdrant", lambda self: True)
    calls: list[bool] = []

    def _record(self: session_manager_module.SessionManager) -> None:
        calls.append(True)

    monkeypatch.setattr(session_manager_module.SessionManager, "init_session_store_if_needed", _record)

    with TestClient(api_module.app):
        assert calls == [True]


def test_lifespan_startup_fails_on_a_broken_session_store(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed migration aborts startup instead of serving 500s."""
    from docint.core import api as api_module
    from docint.core import rag as rag_module

    monkeypatch.setattr(rag_module.RAG, "probe_qdrant", lambda self: True)

    def _boom(self: session_manager_module.SessionManager) -> None:
        raise SessionStoreMigrationError("conversations scope-columns migration failed")

    monkeypatch.setattr(session_manager_module.SessionManager, "init_session_store_if_needed", _boom)

    with pytest.raises(SessionStoreMigrationError):
        with TestClient(api_module.app):
            pass
