"""Tests for validation persistence on the session manager."""

from typing import Any, Generator, cast
from unittest.mock import MagicMock

import pytest
from sqlalchemy import (
    Column,
    DateTime,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    create_engine,
    inspect,
    text,
)
from sqlalchemy.orm import sessionmaker

from docint.core.state.base import Base, _ensure_turn_validation_columns
from docint.core.state.session_manager import SessionManager


@pytest.fixture
def session_manager() -> Generator[SessionManager, None, None]:
    """SessionManager bound to an in-memory SQLite DB.

    Mirrors the fixture in ``test_session_manager.py`` so tests share the
    same RAG mock surface.
    """
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    SessionMaker = sessionmaker(bind=engine)

    rag_mock = MagicMock()
    rag_mock.index = None
    rag_mock.qdrant_collection = "test_collection"
    rag_mock.qdrant_host = "http://qdrant:6333"
    rag_mock.embed_model_id = "embed-model"
    rag_mock.sparse_model_id = "sparse-model"
    rag_mock.text_model_id = "text-model"
    rag_mock.retrieve_similarity_top_k = 20
    rag_mock.rerank_top_n = 5
    rag_mock.conversation_summary_prompt = "Summarize turns:\n"
    rag_mock.rewrite_retrieval_query.return_value = "rewritten question"
    rag_mock._infer_collection_profile.return_value = {"coverage_unit": "documents"}
    mode = MagicMock()
    mode.value = "compact"
    rag_mock._resolve_chat_response_mode.return_value = mode
    cast(Any, rag_mock.get_source_by_node_id).return_value = None

    sm = SessionManager(rag=rag_mock)
    sm._SessionMaker = SessionMaker
    yield sm
    engine.dispose()


def _persist_dummy_turn(sm: SessionManager, session_id: str) -> int:
    """Persist a single empty turn and return its assigned idx."""
    resp_mock = MagicMock()
    resp_mock.metadata = {}
    resp_mock.source_nodes = []
    return sm._persist_turn(
        session_id,
        "hello",
        resp_mock,
        {"response": "world", "reasoning": None},
    )


def test_persist_then_update_validation_round_trips(
    session_manager: SessionManager,
) -> None:
    """update_turn_validation should attach fields read back by get_session_history."""
    session_id = "sess-validate"
    idx = _persist_dummy_turn(session_manager, session_id)

    session_manager.update_turn_validation(
        session_id,
        idx,
        validation_checked=True,
        validation_mismatch=False,
        validation_reason=None,
    )

    history = session_manager.get_session_history(session_id)
    assistant = history[1]
    assert assistant["role"] == "assistant"
    assert assistant["validation_checked"] is True
    assert assistant["validation_mismatch"] is False
    assert assistant["validation_reason"] is None


def test_persist_returns_increasing_idx(session_manager: SessionManager) -> None:
    """_persist_turn must hand back a 0-based idx so the API can target it."""
    session_id = "sess-idx"
    first = _persist_dummy_turn(session_manager, session_id)
    second = _persist_dummy_turn(session_manager, session_id)
    assert first == 0
    assert second == 1


def test_update_turn_validation_unknown_idx_logs_and_returns(
    session_manager: SessionManager,
) -> None:
    """Targeting a missing turn must not raise — only warn — so the SSE handler stays alive."""
    session_manager.update_turn_validation(
        "no-such-session",
        99,
        validation_checked=True,
        validation_mismatch=False,
        validation_reason=None,
    )


def test_get_session_history_omits_validation_when_all_null(
    session_manager: SessionManager,
) -> None:
    """Legacy turns (never updated) must not surface validation keys at all,
    so the frontend's unvalidated-fallback keeps handling them.
    """
    session_id = "sess-legacy"
    _persist_dummy_turn(session_manager, session_id)

    history = session_manager.get_session_history(session_id)
    assistant = history[1]
    assert "validation_checked" not in assistant
    assert "validation_mismatch" not in assistant
    assert "validation_reason" not in assistant


def test_idempotent_column_migration_adds_missing_columns_on_legacy_db() -> None:
    """An existing turns table without validation_* columns must be upgraded."""
    engine = create_engine("sqlite:///:memory:")
    metadata = MetaData()
    # Mirror only the columns that exist in the pre-validation schema.
    # FK to ``conversations`` is intentionally omitted so the legacy
    # turns table can be created standalone for the migration assertion.
    Table(
        "turns",
        metadata,
        Column("id", Integer, primary_key=True, autoincrement=True),
        Column("conversation_id", String),
        Column("idx", Integer, nullable=False),
        Column("user_text", Text, nullable=False),
        Column("model_response", Text, nullable=False),
        Column("created_at", DateTime, nullable=False),
    )
    metadata.create_all(engine)
    with engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO turns (conversation_id, idx, user_text, "
                "model_response, created_at) "
                "VALUES ('c1', 0, 'hi', 'hello', '2026-01-01 00:00:00')"
            )
        )

    pre_columns = {c["name"] for c in inspect(engine).get_columns("turns")}
    assert "validation_checked" not in pre_columns

    _ensure_turn_validation_columns(engine)
    _ensure_turn_validation_columns(engine)  # second call must be a no-op

    post_columns = {c["name"] for c in inspect(engine).get_columns("turns")}
    assert {"validation_checked", "validation_mismatch", "validation_reason"} <= (
        post_columns
    )

    with engine.begin() as conn:
        legacy = conn.execute(text("SELECT validation_checked FROM turns")).scalar()
    assert legacy is None  # legacy row keeps NULL — not False
    engine.dispose()


def test_migration_helper_no_op_when_table_missing() -> None:
    """Helper must short-circuit cleanly if the turns table doesn't exist yet."""
    engine = create_engine("sqlite:///:memory:")
    _ensure_turn_validation_columns(engine)  # no exception
    assert "turns" not in inspect(engine).get_table_names()
    engine.dispose()
