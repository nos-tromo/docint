"""Tests for session-pinned search scopes."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock

import pytest
from sqlalchemy import create_engine, inspect, text

from docint.core.state.base import _ensure_conversation_scope_columns
from docint.core.state.session_manager import SessionManager


def test_scope_columns_are_added_to_a_pre_existing_conversations_table(tmp_path: Path) -> None:
    """An existing sessions DB must upgrade in place.

    ``Base.metadata.create_all`` only creates missing tables, never adds
    columns, so a DB created before scopes shipped would fail every insert
    that touches the new columns.
    """
    engine = create_engine(f"sqlite:///{tmp_path / 'sessions.db'}")
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE conversations (id TEXT PRIMARY KEY, owner TEXT)"))

    _ensure_conversation_scope_columns(engine)

    columns = {col["name"] for col in inspect(engine).get_columns("conversations")}
    assert "scope_chunk_ids" in columns
    assert "scope_set_at" in columns


def test_scope_migration_is_idempotent(tmp_path: Path) -> None:
    """Running it twice must not raise — it runs on every startup."""
    engine = create_engine(f"sqlite:///{tmp_path / 'sessions.db'}")
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE conversations (id TEXT PRIMARY KEY, owner TEXT)"))

    _ensure_conversation_scope_columns(engine)
    _ensure_conversation_scope_columns(engine)

    columns = {col["name"] for col in inspect(engine).get_columns("conversations")}
    assert "scope_chunk_ids" in columns


def test_scope_migration_ignores_a_missing_table(tmp_path: Path) -> None:
    """A fresh DB has no conversations table yet; create_all handles it."""
    engine = create_engine(f"sqlite:///{tmp_path / 'sessions.db'}")

    _ensure_conversation_scope_columns(engine)  # must not raise


@pytest.fixture
def manager(tmp_path: Path) -> SessionManager:
    """A SessionManager backed by a throwaway SQLite file."""
    rag_mock = MagicMock()
    # start_session persists the pinned collection; a MagicMock is not a
    # bindable SQLite parameter.
    rag_mock.qdrant_collection = "test"
    mgr = SessionManager(rag=cast(Any, rag_mock))
    mgr.init_session_store(f"sqlite:///{tmp_path / 'sessions.db'}")
    return mgr


def test_scope_round_trips_for_its_owner(manager: SessionManager) -> None:
    """What was set is what comes back."""
    session_id = manager.start_session(owner="alice")

    assert manager.set_scope(session_id, "alice", ["a", "b"]) is True
    assert manager.get_scope(session_id, "alice") == ["a", "b"]


def test_scope_is_invisible_to_another_owner(manager: SessionManager) -> None:
    """Cross-owner reads must not leak, matching get_session_collection."""
    session_id = manager.start_session(owner="alice")
    manager.set_scope(session_id, "alice", ["a"])

    assert manager.get_scope(session_id, "bob") == []
    assert manager.set_scope(session_id, "bob", ["x"]) is False
    assert manager.get_scope(session_id, "alice") == ["a"]


def test_clearing_a_scope_returns_the_session_to_normal_retrieval(manager: SessionManager) -> None:
    """An empty scope must mean unscoped, not "scoped to nothing"."""
    session_id = manager.start_session(owner="alice")
    manager.set_scope(session_id, "alice", ["a"])

    assert manager.clear_scope(session_id, "alice") is True
    assert manager.get_scope(session_id, "alice") == []


def test_unknown_session_has_no_scope(manager: SessionManager) -> None:
    """A missing session and a cross-owner one look the same."""
    assert manager.get_scope("nope", "alice") == []
