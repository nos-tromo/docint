"""Tests for the topic a chat session keeps from "Ask in chat"."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock

import pytest
from sqlalchemy import create_engine, inspect, text

from docint.core.state.base import _ensure_conversation_topic_column
from docint.core.state.conversation import Conversation
from docint.core.state.session_manager import SessionManager

TOPIC = {"run_id": "20260102-030405-a1b2c3", "topic_id": 3, "label": "Harbour cranes"}


def test_topic_column_is_added_to_a_pre_existing_conversations_table(tmp_path: Path) -> None:
    """An existing sessions DB must upgrade in place, as for the scope columns."""
    engine = create_engine(f"sqlite:///{tmp_path / 'sessions.db'}")
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE conversations (id TEXT PRIMARY KEY, owner TEXT)"))

    _ensure_conversation_topic_column(engine)

    columns = {col["name"] for col in inspect(engine).get_columns("conversations")}
    assert "topic_focus" in columns


def test_topic_migration_is_idempotent(tmp_path: Path) -> None:
    """Running it twice must not raise — it runs on every startup."""
    engine = create_engine(f"sqlite:///{tmp_path / 'sessions.db'}")
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE conversations (id TEXT PRIMARY KEY, owner TEXT)"))

    _ensure_conversation_topic_column(engine)
    _ensure_conversation_topic_column(engine)

    columns = {col["name"] for col in inspect(engine).get_columns("conversations")}
    assert "topic_focus" in columns


def test_topic_migration_ignores_a_missing_table(tmp_path: Path) -> None:
    """A fresh DB has no conversations table yet; create_all handles it."""
    engine = create_engine(f"sqlite:///{tmp_path / 'sessions.db'}")

    _ensure_conversation_topic_column(engine)  # must not raise


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


def test_topic_round_trips_for_its_owner(manager: SessionManager) -> None:
    """What was set is what comes back."""
    session_id = manager.start_session(owner="alice")

    assert manager.set_topic_focus(session_id, "alice", TOPIC) is True
    assert manager.get_topic_focus(session_id, "alice") == TOPIC


def test_a_new_session_has_no_topic(manager: SessionManager) -> None:
    """A chat opened without "Ask in chat" belongs to no topic."""
    session_id = manager.start_session(owner="alice")

    assert manager.get_topic_focus(session_id, "alice") is None


def test_topics_belong_to_their_own_session(manager: SessionManager) -> None:
    """Setting one chat's topic leaves every other chat's as it was."""
    first = manager.start_session(owner="alice")
    second = manager.start_session(owner="alice")
    other = {**TOPIC, "topic_id": 7, "label": "Grain silos"}

    manager.set_topic_focus(first, "alice", TOPIC)
    manager.set_topic_focus(second, "alice", other)

    assert manager.get_topic_focus(first, "alice") == TOPIC
    assert manager.get_topic_focus(second, "alice") == other


def test_topic_is_invisible_to_another_owner(manager: SessionManager) -> None:
    """Cross-owner reads must not leak, matching get_scope."""
    session_id = manager.start_session(owner="alice")
    manager.set_topic_focus(session_id, "alice", TOPIC)

    assert manager.get_topic_focus(session_id, "bob") is None
    assert manager.set_topic_focus(session_id, "bob", {**TOPIC, "topic_id": 9}) is False
    assert manager.clear_topic_focus(session_id, "bob") is False
    assert manager.get_topic_focus(session_id, "alice") == TOPIC


def test_clearing_a_topic_leaves_the_session(manager: SessionManager) -> None:
    """Removing the topic keeps the chat and only drops what narrowed it."""
    session_id = manager.start_session(owner="alice")
    manager.set_topic_focus(session_id, "alice", TOPIC)

    assert manager.clear_topic_focus(session_id, "alice") is True
    assert manager.get_topic_focus(session_id, "alice") is None
    assert manager.get_session_collection(session_id, "alice") == "test"


def test_unknown_session_has_no_topic(manager: SessionManager) -> None:
    """A missing session and a cross-owner one look the same."""
    assert manager.get_topic_focus("nope", "alice") is None
    assert manager.set_topic_focus("nope", "alice", TOPIC) is False


def test_deleting_a_session_deletes_its_topic(manager: SessionManager) -> None:
    """The topic lives on the session row, so it goes with the chat."""
    session_id = manager.start_session(owner="alice")
    manager.set_topic_focus(session_id, "alice", TOPIC)

    assert manager.delete_session(session_id, "alice") is True
    assert manager.get_topic_focus(session_id, "alice") is None


@pytest.mark.parametrize(
    "raw",
    [
        "{not json",
        '["20260102-030405-a1b2c3", 3]',
        '{"run_id": "20260102-030405-a1b2c3"}',
        '{"run_id": "20260102-030405-a1b2c3", "topic_id": true}',
        '{"run_id": 7, "topic_id": 3}',
    ],
)
def test_an_unreadable_stored_topic_reads_as_none(manager: SessionManager, raw: str) -> None:
    """A row that does not hold a topic must not break the chat that owns it."""
    session_id = manager.start_session(owner="alice")
    with manager._session_scope() as s:
        conv = s.query(Conversation).filter_by(id=session_id).one()
        conv.topic_focus = cast(Any, raw)
        s.commit()

    assert manager.get_topic_focus(session_id, "alice") is None
