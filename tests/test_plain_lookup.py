"""The plain lookup answers the user's own words.

Retrieval runs on the history-resolved rewrite with the graph's terms appended; synthesis is asked the
message itself with the rewrite labelled beside it. The engines are real, their model echoes its prompt,
and every message, passage and name here is invented.
"""

from __future__ import annotations

from collections.abc import Generator, Sequence
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock

import pytest
from agent_fakes import ThumbnailStore, frame_passage, thumbnail_url
from assessment_fakes import scored_node
from llama_index.core import PromptTemplate
from llama_index.core.base.llms.types import ChatMessage, ChatResponse, ImageBlock, MessageRole
from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.llms import MockLLM
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.llms.openai.utils import to_openai_message_dict
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from typing_extensions import override

from docint.agents.turn.history import RESOLVED_QUESTION_LABEL
from docint.core.rag import (
    CitationNumberingPostprocessor,
    StreamingCompactAndRefine,
    VisualStreamingCompactAndRefine,
    _ScopedRetriever,
)
from docint.core.retrieval.visual import VISUAL_IMAGES_ATTACHED_KEY
from docint.core.state.base import Base
from docint.core.state.session_manager import SessionManager

#: The physical name of the request's collection.
PHYSICAL = "u0123456789ab__hafen"
#: What the user typed: a follow-up that means nothing without the conversation before it.
FOLLOW_UP = "Are those all?"
#: What the history-resolved rewrite makes of it.
REWRITE = "Which stalls open at the harbour festival on Sunday?"
#: What the entity graph appends to the rewrite for retrieval.
TERMS = "Related entities for retrieval: Person A, Person B"
#: A grounded template that shows the question exactly as the synthesizer is asked it.
ECHO_TEMPLATE = "CTX<<{context_str}>> Q<<{query_str}>>"
#: How the question reads in the echo: the user's words, then the rewrite under its label.
ASKED = f"Q<<{FOLLOW_UP}\n{RESOLVED_QUESTION_LABEL}: {REWRITE}>>"
#: What a model that is shown pictures answers.
SEEN_ANSWER = "Two people under umbrellas [1]."

ENTRIES = ["chat", "stream_chat"]


def _normalize(query: str, result: Any, reason: str | None = None, **kwargs: Any) -> dict[str, Any]:
    """Mirror the normalizer's shape: the answer text, numbered sources and the routing fields.

    Args:
        query (str): The user's message.
        result (Any): The response the engine produced.
        reason (str | None): The reasoning the caller hands over.
        **kwargs (Any): The normalizer's keyword arguments.

    Returns:
        dict[str, Any]: A normalized payload.
    """
    return {
        "query": query,
        "response": result.response,
        "sources": [{"id": scored.node.node_id} for scored in result.source_nodes],
        "reasoning": reason,
        "retrieval_query": kwargs["retrieval_query"],
        "coverage_unit": kwargs["coverage_unit"],
        "retrieval_mode": kwargs["retrieval_mode"],
        "rerank": None,
        "visual": None,
    }


def _expand(query: str) -> tuple[str, dict[str, Any]]:
    """Expand a query the way the entity graph does when it finds neighbours.

    Args:
        query (str): The unexpanded retrieval query.

    Returns:
        tuple[str, dict[str, Any]]: The query with the terms appended, and the debug payload.
    """
    expanded = f"{query}\n\n{TERMS}"
    return expanded, {"applied": True, "original_query": query, "expanded_query": expanded}


@pytest.fixture
def session_manager() -> Generator[SessionManager, None, None]:
    """Provide a SessionManager on an in-memory DB over a RAG double whose engines each test brings.

    Returns:
        Generator[SessionManager, None, None]: The session manager.
    """
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    rag = MagicMock()
    rag.index = object()
    rag.qdrant_collection = PHYSICAL
    rag.rewrite_retrieval_query.return_value = REWRITE
    rag.expand_query_with_graph_with_debug.side_effect = _expand
    rag.graph_debug_skipped.side_effect = lambda query, reason: {
        "applied": False,
        "reason": reason,
        "original_query": query,
    }
    rag._infer_collection_profile.return_value = {"coverage_unit": "documents"}
    mode = MagicMock()
    mode.value = "compact"
    rag._resolve_chat_response_mode.return_value = mode
    rag._normalize_response_data.side_effect = _normalize
    rag.get_source_by_node_id.return_value = None
    sm = SessionManager(rag=rag)
    sm._SessionMaker = sessionmaker(bind=engine)
    yield sm
    engine.dispose()


class _Passages(BaseRetriever):
    """Return fixed passages for any query, keeping the queries it was asked."""

    def __init__(self, passages: Sequence[NodeWithScore]) -> None:
        """Hold the passages.

        Args:
            passages (Sequence[NodeWithScore]): What every retrieval returns.
        """
        super().__init__()
        self._passages = list(passages)
        self.asked: list[str] = []

    @override
    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        """Record the query and return the passages.

        Args:
            query_bundle (QueryBundle): The query.

        Returns:
            list[NodeWithScore]: The passages.
        """
        self.asked.append(query_bundle.query_str)
        return list(self._passages)


def _passages() -> list[NodeWithScore]:
    """Build the invented passages a text lookup retrieves.

    Returns:
        list[NodeWithScore]: Two passages of one invented programme.
    """
    return [
        scored_node("n1", "Stalls open at ten on Sunday.", filename="programme.pdf", page=2, file_hash="h1"),
        scored_node("n2", "Music starts at noon.", filename="programme.pdf", page=3, file_hash="h1"),
    ]


def _engine(
    retriever: BaseRetriever,
    *,
    streaming: bool,
    postprocessors: Sequence[Any] = (CitationNumberingPostprocessor(),),
) -> RetrieverQueryEngine:
    """Build a real engine whose model echoes the prompt it is asked.

    Args:
        retriever (BaseRetriever): The engine's retriever.
        streaming (bool): Whether the synthesizer streams.
        postprocessors (Sequence[Any]): The engine's node postprocessors.

    Returns:
        RetrieverQueryEngine: The engine.
    """
    return RetrieverQueryEngine(
        retriever=retriever,
        node_postprocessors=list(postprocessors),
        response_synthesizer=StreamingCompactAndRefine(
            llm=MockLLM(), text_qa_template=PromptTemplate(ECHO_TEMPLATE), streaming=streaming
        ),
    )


def _arm(sm: SessionManager, engine: RetrieverQueryEngine) -> None:
    """Hand the turn's engine to the session manager, as the cached and the built one alike.

    Args:
        sm (SessionManager): The session manager under test.
        engine (RetrieverQueryEngine): The engine.
    """
    rag = cast(Any, sm.rag)
    rag.query_engine = engine
    rag.build_query_engine.return_value = engine


def _answer(sm: SessionManager, entry: str, message: str = FOLLOW_UP, **kwargs: Any) -> dict[str, Any]:
    """Run one turn of session ``s1`` through ``chat`` or ``stream_chat`` and return its final payload.

    Args:
        sm (SessionManager): The session manager under test.
        entry (str): ``chat`` or ``stream_chat``.
        message (str): The user's message.
        **kwargs (Any): Further keyword arguments for the entry point.

    Returns:
        dict[str, Any]: The final payload.
    """
    if entry == "chat":
        return sm.chat(message, session_id="s1", **kwargs)
    final = list(sm.stream_chat(message, session_id="s1", **kwargs))[-1]
    assert isinstance(final, dict)
    return final


@pytest.mark.parametrize("entry", ENTRIES)
def test_the_plain_lookup_answers_the_users_own_words_with_the_rewrite_labelled(
    session_manager: SessionManager, entry: str
) -> None:
    """The follow-up is asked as typed, with the rewrite it was resolved to labelled beside it.

    Args:
        session_manager (SessionManager): The session manager fixture.
        entry (str): The session manager method under test.
    """
    _arm(session_manager, _engine(_Passages(_passages()), streaming=entry == "stream_chat"))

    answer = _answer(session_manager, entry)["response"]

    assert ASKED in answer


@pytest.mark.parametrize("entry", ENTRIES)
def test_graph_terms_never_reach_the_synthesis_prompt(session_manager: SessionManager, entry: str) -> None:
    """The graph's terms widen the search and stop there.

    Args:
        session_manager (SessionManager): The session manager fixture.
        entry (str): The session manager method under test.
    """
    retriever = _Passages(_passages())
    _arm(session_manager, _engine(retriever, streaming=entry == "stream_chat"))

    final = _answer(session_manager, entry)

    assert retriever.asked == [f"{REWRITE}\n\n{TERMS}"]
    assert TERMS not in final["response"]
    assert final["graph_debug"]["applied"] is True
    assert final["retrieval_query"] == REWRITE


@pytest.mark.parametrize("entry", ENTRIES)
def test_a_message_the_rewrite_adds_nothing_to_is_asked_alone(session_manager: SessionManager, entry: str) -> None:
    """A rewrite holding the message's words is not shown beside it.

    Args:
        session_manager (SessionManager): The session manager fixture.
        entry (str): The session manager method under test.
    """
    message = "Which stalls open on Sunday?"
    cast(Any, session_manager.rag).rewrite_retrieval_query.return_value = "which stalls open on sunday"
    _arm(session_manager, _engine(_Passages(_passages()), streaming=entry == "stream_chat"))

    answer = _answer(session_manager, entry, message)["response"]

    assert f"Q<<{message}>>" in answer


@pytest.mark.parametrize("entry", ENTRIES)
def test_a_retrieval_query_handed_in_as_the_message_is_asked_as_it_stands(
    session_manager: SessionManager, entry: str
) -> None:
    """The corrective retry hands in its reformulated query and skips the rewrite: that query is the question.

    Args:
        session_manager (SessionManager): The session manager fixture.
        entry (str): The session manager method under test.
    """
    reformulated = "harbour festival stall opening hours Sunday"
    retriever = _Passages(_passages())
    _arm(session_manager, _engine(retriever, streaming=entry == "stream_chat"))

    final = _answer(session_manager, entry, reformulated, skip_query_rewrite=True)

    cast(Any, session_manager.rag).rewrite_retrieval_query.assert_not_called()
    assert retriever.asked == [f"{reformulated}\n\n{TERMS}"]
    assert f"Q<<{reformulated}>>" in final["response"]
    assert final["retrieval_query"] == reformulated


@pytest.mark.parametrize("entry", ENTRIES)
def test_a_scoped_lookup_answers_the_users_own_words_from_exactly_the_pinned_chunks(
    session_manager: SessionManager, entry: str
) -> None:
    """A pinned scope retrieves by id whatever the query, and its answer is asked the user's words all the same.

    Args:
        session_manager (SessionManager): The session manager fixture.
        entry (str): The session manager method under test.
    """
    chunks = {"c1": "Stalls open at ten on Sunday.", "c2": "Music starts at noon.", "c3": "An unpinned chunk."}

    def _retrieve(**kwargs: Any) -> list[Any]:
        """Return the points asked for, as Qdrant would.

        Args:
            **kwargs (Any): The call's keywords; ``ids`` names the points.

        Returns:
            list[Any]: One point per id.
        """
        return [SimpleNamespace(id=point, payload={"text": chunks[point]}) for point in kwargs["ids"]]

    rag = SimpleNamespace(qdrant_collection=PHYSICAL, qdrant_client=SimpleNamespace(retrieve=_retrieve))
    retriever = _ScopedRetriever(rag=cast(Any, rag), node_ids=["c2", "c1"])
    _arm(session_manager, _engine(retriever, streaming=entry == "stream_chat"))

    final = _answer(session_manager, entry, scoped_node_ids=["c2", "c1"])

    answer = final["response"]
    assert ASKED in answer
    assert answer.index(chunks["c2"]) < answer.index(chunks["c1"])
    assert chunks["c3"] not in answer
    cast(Any, session_manager.rag).expand_query_with_graph_with_debug.assert_not_called()
    assert final["graph_debug"]["reason"] == "scoped"
    assert final["retrieval_mode"] == "scoped"
    assert final["retrieval_query"] == REWRITE


class _SeeingLLM(MockLLM):
    """A model that is sent chat messages and keeps them, pictures included, and answers ``SEEN_ANSWER``."""

    _seen: list[list[ChatMessage]] = PrivateAttr(default_factory=list)

    @property
    def seen(self) -> list[list[ChatMessage]]:
        """Return the messages of every chat call, oldest first.

        Returns:
            list[list[ChatMessage]]: One list of messages per call.
        """
        return self._seen

    @override
    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        """Keep the messages and answer.

        Args:
            messages (Sequence[ChatMessage]): The rendered messages.
            **kwargs (Any): Unused.

        Returns:
            ChatResponse: The answer.
        """
        self._seen.append(list(messages))
        return ChatResponse(message=ChatMessage(role=MessageRole.ASSISTANT, content=SEEN_ANSWER))

    @override
    def stream_chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> Any:
        """Keep the messages and stream the answer in one piece.

        Args:
            messages (Sequence[ChatMessage]): The rendered messages.
            **kwargs (Any): Unused.

        Returns:
            Any: A generator of chat responses.
        """
        self._seen.append(list(messages))

        def _pieces() -> Generator[ChatResponse, None, None]:
            """Yield the answer.

            Yields:
                ChatResponse: The one piece.
            """
            yield ChatResponse(message=ChatMessage(role=MessageRole.ASSISTANT, content=SEEN_ANSWER), delta=SEEN_ANSWER)

        return _pieces()


@pytest.mark.parametrize("entry", ENTRIES)
def test_a_visual_lookup_still_shows_the_model_its_images_and_asks_the_users_words(
    session_manager: SessionManager, entry: str
) -> None:
    """The visual synthesizer collects its images inside ``synthesize``, which the plain path now calls itself.

    Args:
        session_manager (SessionManager): The session manager fixture.
        entry (str): The session manager method under test.
    """
    llm = _SeeingLLM(max_tokens=None)
    store = ThumbnailStore(1)
    synthesizer = VisualStreamingCompactAndRefine(
        llm=llm, text_qa_template=PromptTemplate(ECHO_TEMPLATE), streaming=entry == "stream_chat"
    )
    synthesizer._fetch_thumbnails = store
    synthesizer._max_images = 4
    synthesizer._legend_template = "Images:\n{image_legend}"
    synthesizer._attached = []
    synthesizer._stamped = []
    retriever = _Passages([frame_passage(1)])
    _arm(
        session_manager,
        RetrieverQueryEngine(
            retriever=retriever,
            node_postprocessors=[CitationNumberingPostprocessor()],
            response_synthesizer=synthesizer,
        ),
    )

    final = _answer(session_manager, entry, retrieval_target="visual")

    assert final["response"] == SEEN_ANSWER
    assert store.fetches == [["frame-1"]]
    message = llm.seen[0][-1]
    assert ASKED in (message.content or "")
    assert sum(isinstance(block, ImageBlock) for block in message.blocks) == 1
    parts = cast(list[dict[str, Any]], to_openai_message_dict(message)["content"])
    assert [part["image_url"]["url"] for part in parts if part.get("type") == "image_url"] == [thumbnail_url(1)]
    shown = cast(Any, session_manager.rag)._normalize_response_data.call_args.args[1].source_nodes
    assert [scored.node.metadata.get(VISUAL_IMAGES_ATTACHED_KEY) for scored in shown] == [True]
    assert retriever.asked == [REWRITE]
    assert final["graph_debug"]["reason"] == "visual"
    cast(Any, session_manager.rag).expand_query_with_graph_with_debug.assert_not_called()
