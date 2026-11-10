"""Tests for SessionManager prior-turn injection into the generation template."""

from __future__ import annotations

from collections.abc import Generator
from typing import Any, cast
from unittest.mock import MagicMock

import pytest
from llama_index.core.llms import MockLLM
from llama_index.core.prompts import PromptTemplate
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import ResponseMode, get_response_synthesizer
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from typing_extensions import override

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
    filtered_response.metadata = cast(dict[str, Any], {})
    filtered_response.source_nodes = cast(list[Any], [])
    engine.synthesize.return_value = filtered_response

    shared = MagicMock()
    session_manager.rag.query_engine = shared
    session_manager.rag.build_query_engine.return_value = engine  # type: ignore[attr-defined]
    session_manager.rag.expand_query_with_graph_with_debug.return_value = (  # type: ignore[attr-defined]
        "expanded",
        {"applied": False},
    )
    session_manager.rag._normalize_response_data.return_value = {  # type: ignore[attr-defined]
        "response": "ok",
        "sources": [],
    }
    monkeypatch.setattr(SessionManager, "_persist_turn", lambda *args, **kwargs: None)
    monkeypatch.setattr(SessionManager, "_maybe_update_summary", lambda *args: None)

    prior = PriorTurn(
        user_text="Which interpretation is correct?",
        assistant_text="The text mentions the UN Security Council.",
    )
    session_manager.chat("Please elaborate.", prior_turn=prior)

    session_manager.rag.rewrite_retrieval_query.assert_not_called()  # type: ignore[attr-defined]
    session_manager.rag.expand_query_with_graph_with_debug.assert_called_once_with(  # type: ignore[attr-defined]
        "Please elaborate."
    )
    # Retrieval runs on the expanded query; the answer is asked the message alone.
    engine.retrieve.assert_called_once()
    assert engine.retrieve.call_args.args[0].query_str == "expanded"
    engine.synthesize.assert_called_once()
    assert engine.synthesize.call_args.args[0].query_str == "Please elaborate."
    shared.retrieve.assert_not_called()


def test_chat_binds_prior_turn_context_via_update_prompts(
    session_manager: SessionManager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``engine.update_prompts`` is called with QA + refine templates bound to the prior turn.

    The binding lands on an engine built for the turn, never on the shared cached one.
    """
    engine = MagicMock()
    filtered_response = MagicMock()
    filtered_response.metadata = cast(dict[str, Any], {})
    filtered_response.source_nodes = cast(list[Any], [])
    engine.synthesize.return_value = filtered_response

    shared = MagicMock()
    session_manager.rag.query_engine = shared
    session_manager.rag.build_query_engine.return_value = engine  # type: ignore[attr-defined]
    session_manager.rag.expand_query_with_graph_with_debug.return_value = (  # type: ignore[attr-defined]
        "expanded",
        {"applied": False},
    )
    session_manager.rag._normalize_response_data.return_value = {  # type: ignore[attr-defined]
        "response": "ok",
        "sources": [],
    }
    monkeypatch.setattr(SessionManager, "_persist_turn", lambda *args, **kwargs: None)
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
    shared.update_prompts.assert_not_called()


def test_chat_default_path_still_rewrites(
    session_manager: SessionManager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without ``prior_turn`` the legacy double-rewrite remains in effect."""
    engine = MagicMock()
    filtered_response = MagicMock()
    filtered_response.metadata = cast(dict[str, Any], {})
    filtered_response.source_nodes = cast(list[Any], [])
    engine.synthesize.return_value = filtered_response

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
    monkeypatch.setattr(SessionManager, "_persist_turn", lambda *args, **kwargs: None)
    monkeypatch.setattr(SessionManager, "_maybe_update_summary", lambda *args: None)

    session_manager.chat("Please elaborate.")

    session_manager.rag.rewrite_retrieval_query.assert_called_once()  # type: ignore[attr-defined]
    engine.update_prompts.assert_not_called()
    # With nothing to bind, the cached engine answers.
    session_manager.rag.build_query_engine.assert_not_called()  # type: ignore[attr-defined]
    engine.retrieve.assert_called_once()


def test_default_grounded_qa_template_renders_without_prior_turn() -> None:
    """The default-bound sentinel must let the template render with only context/query."""
    from docint.core.rag import DEFAULT_GROUNDED_TEXT_QA_PROMPT

    tmpl = PromptTemplate(DEFAULT_GROUNDED_TEXT_QA_PROMPT).partial_format(prior_turn_context="(no prior turn)")
    rendered = tmpl.format(context_str="CTX", query_str="Q")
    assert "(no prior turn)" in rendered
    assert "CTX" in rendered
    assert "Q" in rendered


def _stub_turn_plumbing(session_manager: SessionManager, monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub everything a plain ``chat`` turn touches besides its engine.

    Args:
        session_manager (SessionManager): The session manager fixture.
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
    """
    session_manager.rag.expand_query_with_graph_with_debug.return_value = (  # type: ignore[attr-defined]
        "expanded",
        {"applied": False},
    )
    session_manager.rag._normalize_response_data.return_value = {  # type: ignore[attr-defined]
        "response": "ok",
        "sources": [],
    }
    monkeypatch.setattr(SessionManager, "_persist_turn", lambda *args, **kwargs: None)
    monkeypatch.setattr(SessionManager, "_maybe_update_summary", lambda *args: None)


def test_chat_keeps_the_cached_engine_when_nothing_source_backed_is_left_to_bind(
    session_manager: SessionManager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An assessment with no sourced part binds nothing, so no engine of its own is built.

    Args:
        session_manager (SessionManager): The session manager fixture.
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
    """
    engine = MagicMock()
    session_manager.rag.query_engine = engine
    _stub_turn_plumbing(session_manager, monkeypatch)
    prior = PriorTurn(
        user_text="Is the schedule reliable?",
        assistant_text="**Model assessment**\n\nIt reads as unreliable.",
        evidence_chars=0,
    )

    session_manager.chat("Please elaborate.", prior_turn=prior)

    session_manager.rag.build_query_engine.assert_not_called()  # type: ignore[attr-defined]
    engine.update_prompts.assert_not_called()
    engine.retrieve.assert_called_once()


class _FixedRetriever(BaseRetriever):
    """Retrieve one fixed passage, whatever the query."""

    @override
    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        """Return the fixed passage.

        Args:
            query_bundle (QueryBundle): The query, ignored.

        Returns:
            list[NodeWithScore]: The one passage.
        """
        return [NodeWithScore(node=TextNode(text="The harbour festival opens on Friday."), score=1.0)]


def _echo_engine(session_manager: SessionManager) -> RetrieverQueryEngine:
    """Build a real query engine whose model answers with the prompt it was given.

    Args:
        session_manager (SessionManager): Supplies the grounded QA and refine templates.

    Returns:
        RetrieverQueryEngine: A refine engine over :class:`_FixedRetriever`.
    """
    rag = cast(Any, session_manager.rag)
    synthesizer = get_response_synthesizer(
        llm=MockLLM(),
        response_mode=ResponseMode.REFINE,
        text_qa_template=rag._build_grounded_text_qa_template(),
        refine_template=rag._build_grounded_refine_template(),
    )
    return RetrieverQueryEngine(retriever=_FixedRetriever(), response_synthesizer=synthesizer)


def test_a_prior_turn_bound_by_chat_never_reaches_a_later_lookup_on_the_cached_engine(
    session_manager: SessionManager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A prior turn bound for one session must not stay on the engine every lookup shares.

    ``/agent/chat`` hands ``chat`` the session's prior turn on the default path, which
    answers on the collection's cached engine; ``run_query`` and ``/query`` later answer
    on that same engine for any session.

    Args:
        session_manager (SessionManager): The session manager fixture.
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
    """
    cached = _echo_engine(session_manager)
    session_manager.rag.query_engine = cached
    session_manager.rag.build_query_engine.side_effect = lambda **_: _echo_engine(session_manager)  # type: ignore[attr-defined]
    _stub_turn_plumbing(session_manager, monkeypatch)
    prior = PriorTurn(
        user_text="When does the harbour festival open?",
        assistant_text="It opens on Friday [1].",
    )

    session_manager.chat("Please elaborate.", prior_turn=prior, session_id="session-a")

    # The turn itself was answered with its prior exchange in view.
    turn_response = session_manager.rag._normalize_response_data.call_args.args[1]  # type: ignore[attr-defined]
    assert "Previous user question: When does the harbour festival open?" in str(turn_response)

    lookup_prompt = str(cached.query("Who organises the festival?"))

    assert "Previous user question" not in lookup_prompt
    assert "It opens on Friday [1]." not in lookup_prompt
    assert "continuity=(no prior turn)" in lookup_prompt
