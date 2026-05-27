"""Tests for SessionManager prior-turn injection into the generation template."""

from __future__ import annotations

from collections.abc import Generator
from typing import Any, cast
from unittest.mock import MagicMock

import pytest
from llama_index.core.prompts import PromptTemplate
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from docint.agents.types import PriorTurn
from docint.core.state.base import Base
from docint.core.state.session_manager import SessionManager


@pytest.fixture
def session_manager() -> Generator[SessionManager, None, None]:
    """SessionManager with an in-memory SQLite store and a fully mocked RAG."""
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
    rag_mock.rewrite_retrieval_query.return_value = "should-not-be-used"
    rag_mock._infer_collection_profile.return_value = {
        "coverage_unit": "documents",
        "is_social_table": False,
    }
    mode = MagicMock()
    mode.value = "compact"
    rag_mock._resolve_chat_response_mode.return_value = mode
    cast(Any, rag_mock.get_source_by_node_id).return_value = None
    rag_mock._build_grounded_text_qa_template.return_value = PromptTemplate(
        "QA template. continuity={prior_turn_context} ctx={context_str} q={query_str}"
    ).partial_format(prior_turn_context="(no prior turn)")
    rag_mock._build_grounded_refine_template.return_value = PromptTemplate(
        "Refine template. continuity={prior_turn_context} q={query_str} a={existing_answer} m={context_msg}"
    ).partial_format(prior_turn_context="(no prior turn)")

    sm = SessionManager(rag=rag_mock)
    sm._SessionMaker = SessionMaker
    yield sm
    engine.dispose()


def test_chat_skips_rewrite_when_prior_turn_supplied(
    session_manager: SessionManager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``rewrite_retrieval_query`` must NOT be invoked when ``prior_turn`` is set."""
    engine = MagicMock()
    filtered_response = MagicMock()
    filtered_response.metadata = {}
    filtered_response.source_nodes = []
    engine.query.return_value = filtered_response

    session_manager.rag.query_engine = engine
    session_manager.rag.expand_query_with_graph_with_debug.return_value = (  # type: ignore[attr-defined]
        "expanded",
        {"applied": False},
    )
    session_manager.rag._normalize_response_data.return_value = {  # type: ignore[attr-defined]
        "response": "ok",
        "sources": [],
    }
    session_manager.session_id = "s1"
    session_manager.chat_engine = object()  # type: ignore[assignment]

    monkeypatch.setattr(SessionManager, "_persist_turn", lambda *args: None)
    monkeypatch.setattr(SessionManager, "_maybe_update_summary", lambda *args: None)

    prior = PriorTurn(
        user_text="Which interpretation is correct?",
        assistant_text="The text mentions the UN Security Council.",
    )
    session_manager.chat("Please elaborate.", prior_turn=prior)

    session_manager.rag.rewrite_retrieval_query.assert_not_called()  # type: ignore[attr-defined]
    engine.query.assert_called_once_with("expanded")
    session_manager.rag.expand_query_with_graph_with_debug.assert_called_once_with(  # type: ignore[attr-defined]
        "Please elaborate."
    )


def test_chat_binds_prior_turn_context_via_update_prompts(
    session_manager: SessionManager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``engine.update_prompts`` is called with QA + refine templates bound to the prior turn."""
    engine = MagicMock()
    filtered_response = MagicMock()
    filtered_response.metadata = {}
    filtered_response.source_nodes = []
    engine.query.return_value = filtered_response

    session_manager.rag.query_engine = engine
    session_manager.rag.expand_query_with_graph_with_debug.return_value = (  # type: ignore[attr-defined]
        "expanded",
        {"applied": False},
    )
    session_manager.rag._normalize_response_data.return_value = {  # type: ignore[attr-defined]
        "response": "ok",
        "sources": [],
    }
    session_manager.session_id = "s1"
    session_manager.chat_engine = object()  # type: ignore[assignment]

    monkeypatch.setattr(SessionManager, "_persist_turn", lambda *args: None)
    monkeypatch.setattr(SessionManager, "_maybe_update_summary", lambda *args: None)

    prior = PriorTurn(
        user_text="Which interpretation is correct?",
        assistant_text="The text mentions the UN Security Council.",
    )
    session_manager.chat("Please elaborate.", prior_turn=prior)

    engine.update_prompts.assert_called_once()
    payload = engine.update_prompts.call_args.args[0]
    qa_tmpl: PromptTemplate = payload["response_synthesizer:text_qa_template"]
    refine_tmpl: PromptTemplate = payload["response_synthesizer:refine_template"]
    qa_rendered = qa_tmpl.format(context_str="CTX", query_str="Q")
    refine_rendered = refine_tmpl.format(context_msg="NEW", query_str="Q", existing_answer="A0")
    assert "UN Security Council" in qa_rendered
    assert "Which interpretation is correct?" in qa_rendered
    assert "UN Security Council" in refine_rendered


def test_chat_default_path_still_rewrites(
    session_manager: SessionManager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without ``prior_turn`` the legacy double-rewrite remains in effect."""
    engine = MagicMock()
    filtered_response = MagicMock()
    filtered_response.metadata = {}
    filtered_response.source_nodes = []
    engine.query.return_value = filtered_response

    session_manager.rag.query_engine = engine
    session_manager.rag.rewrite_retrieval_query.return_value = "rewritten"  # type: ignore[attr-defined]
    session_manager.rag.expand_query_with_graph_with_debug.return_value = (  # type: ignore[attr-defined]
        "expanded",
        {"applied": False},
    )
    session_manager.rag._normalize_response_data.return_value = {  # type: ignore[attr-defined]
        "response": "ok",
        "sources": [],
    }
    session_manager.session_id = "s1"
    session_manager.chat_engine = object()  # type: ignore[assignment]

    monkeypatch.setattr(SessionManager, "_persist_turn", lambda *args: None)
    monkeypatch.setattr(SessionManager, "_maybe_update_summary", lambda *args: None)

    session_manager.chat("Please elaborate.")

    session_manager.rag.rewrite_retrieval_query.assert_called_once()  # type: ignore[attr-defined]
    engine.update_prompts.assert_not_called()


def test_default_grounded_qa_template_renders_without_prior_turn() -> None:
    """The default-bound sentinel must let the template render with only context/query."""
    from docint.core.rag import DEFAULT_GROUNDED_TEXT_QA_PROMPT

    tmpl = PromptTemplate(DEFAULT_GROUNDED_TEXT_QA_PROMPT).partial_format(prior_turn_context="(no prior turn)")
    rendered = tmpl.format(context_str="CTX", query_str="Q")
    assert "(no prior turn)" in rendered
    assert "CTX" in rendered
    assert "Q" in rendered
