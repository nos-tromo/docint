"""Tests for the session manager citation persistence."""

import json
from pathlib import Path
from typing import Any, Generator, cast
from unittest.mock import MagicMock

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from docint.core.state.session_manager import SessionManager
from docint.core.state.base import Base


@pytest.fixture
def session_manager() -> Generator[SessionManager, None, None]:
    """Fixture to create a SessionManager with an in-memory SQLite database.

    Returns:
        Generator[SessionManager, None, None]: The SessionManager instance.
    """
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    SessionMaker = sessionmaker(bind=engine)

    rag_mock = MagicMock()
    rag_mock.index = None  # Simulate no index initially
    rag_mock.qdrant_collection = "test_collection"
    rag_mock.qdrant_host = "http://qdrant:6333"
    rag_mock.embed_model_id = "embed-model"
    rag_mock.sparse_model_id = "sparse-model"
    rag_mock.text_model_id = "text-model"
    rag_mock.retrieve_similarity_top_k = 20
    rag_mock.rerank_top_n = 5
    cast(Any, rag_mock.get_source_by_node_id).return_value = None

    sm = SessionManager(rag=rag_mock)
    sm._SessionMaker = SessionMaker
    yield sm
    engine.dispose()


def test_persist_and_retrieve_citation_with_hash(
    session_manager: SessionManager,
) -> None:
    """Test persisting and retrieving a turn with citations that include file_hash.

    Args:
        session_manager (SessionManager): The session manager fixture.
    """
    session_id = "test_session"
    user_msg = "hello"

    # Mock response object
    resp_mock = MagicMock()
    resp_mock.metadata = {}
    resp_mock.source_nodes = []

    # Create a source node with file_hash
    node_mock = MagicMock()
    node_mock.node_id = "node1"
    node_mock.metadata = {
        "filename": "test.pdf",
        "file_hash": "hash123",
        "page_label": "1",
        "source": "document",
    }

    src_node_mock = MagicMock()
    src_node_mock.node = node_mock
    src_node_mock.score = 0.9

    resp_mock.source_nodes = [src_node_mock]

    data = {"response": "Hi there", "reasoning": "None"}

    # Persist
    session_manager._persist_turn(session_id, user_msg, resp_mock, data)

    # Retrieve
    # We don't mock _get_node_text_by_id, so it will return None/empty string

    history = session_manager.get_session_history(session_id)
    assert len(history) == 2  # User + Assistant

    assistant_msg = history[1]
    assert assistant_msg["role"] == "assistant"
    assert len(assistant_msg["sources"]) == 1

    source = assistant_msg["sources"][0]
    assert source["filename"] == "test.pdf"
    assert source["file_hash"] == "hash123"
    assert source["page"] == 1
    # assert source["preview_text"] == "Preview text" # This will be empty string


def test_get_session_history_enriches_sources_from_node_lookup(
    session_manager: SessionManager,
) -> None:
    """Session history should reuse normalized source payloads when available.
    
    Args:
        session_manager (SessionManager): The session manager fixture.
    """
    cast(Any, session_manager.rag.get_source_by_node_id).return_value = {
        "text": "Loaded text",
        "preview_text": "Loaded text",
        "filename": "social.csv",
        "filetype": "text/csv",
        "source": "table",
        "score": 0.8,
        "row": 4,
        "file_hash": "hash-social",
        "reference_metadata": {
            "network": "Telegram",
            "type": "comment",
            "timestamp": "2026-01-02T10:00:00Z",
            "author": "Alice",
            "author_id": "a1",
            "vanity": "alice-v",
            "text": "Loaded text",
            "text_id": "c1",
        },
    }

    resp_mock = MagicMock()
    resp_mock.metadata = {}
    node_mock = MagicMock()
    node_mock.node_id = "node-social"
    node_mock.metadata = {
        "filename": "social.csv",
        "file_hash": "hash-social",
        "source": "table",
    }
    src_node_mock = MagicMock()
    src_node_mock.node = node_mock
    src_node_mock.score = 0.8
    resp_mock.source_nodes = [src_node_mock]

    session_manager._persist_turn(
        "session-social",
        "hello",
        resp_mock,
        {"response": "Hi", "reasoning": None},
    )

    history = session_manager.get_session_history("session-social")

    source = history[1]["sources"][0]
    assert source["filename"] == "social.csv"
    assert source["reference_metadata"]["text_id"] == "c1"
    assert source["row"] == 4


def test_export_session_omits_stale_host_dir_and_succeeds(
    session_manager: SessionManager, tmp_path: Path
) -> None:
    """Session export should not depend on removed Qdrant collection path state.

    Args:
        session_manager (SessionManager): The session manager fixture.
        tmp_path (Path): Temporary directory provided by pytest.
    """
    session_id = "export-session"
    resp_mock = MagicMock()
    resp_mock.metadata = {}
    resp_mock.source_nodes = []

    session_manager._persist_turn(
        session_id,
        "hello",
        resp_mock,
        {"response": "Hi there", "reasoning": None},
    )

    export_dir = session_manager.export_session(session_id, tmp_path)
    session_meta = json.loads((export_dir / "session.json").read_text(encoding="utf-8"))

    assert session_meta["vector_store"]["type"] == "qdrant"
    assert session_meta["vector_store"]["url"] == "http://qdrant:6333"
    assert session_meta["vector_store"]["collection"] == "test_collection"
    assert "host_dir" not in session_meta["vector_store"]
