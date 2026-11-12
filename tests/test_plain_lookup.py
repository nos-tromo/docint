"""The plain lookup answers the user's own words, in a session and without one.

Retrieval runs on the history-resolved rewrite; synthesis is asked the message itself with the rewrite
labelled beside it, and a stateless query, which has no rewrite, retrieves with and is asked its question
alone. The engines are real, their model echoes its prompt, and every message, passage and
name here is invented.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Generator, Sequence
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock

import pytest
from agent_fakes import ThumbnailStore, frame_passage, thumbnail_url
from assessment_fakes import scored_node
from fastapi.testclient import TestClient
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

import docint.core.api as api_module
from docint.agents.turn.history import RESOLVED_QUESTION_LABEL
from docint.core.rag import (
    RAG,
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
#: A grounded template that shows the question exactly as the synthesizer is asked it.
ECHO_TEMPLATE = "CTX<<{context_str}>> Q<<{query_str}>>"
#: How the question reads in the echo: the user's words, then the rewrite under its label.
ASKED = f"Q<<{FOLLOW_UP}\n{RESOLVED_QUESTION_LABEL}: {REWRITE}>>"
#: What a model that is shown pictures answers.
SEEN_ANSWER = "Two people under umbrellas [1]."

#: The chunks of the invented collection a scope picks from.
CHUNKS = {"c1": "Stalls open at ten on Sunday.", "c2": "Music starts at noon.", "c3": "An unpinned chunk."}

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


def _scoped_retriever(node_ids: Sequence[str]) -> _ScopedRetriever:
    """Build the real scoped retriever over a Qdrant stand-in that holds ``CHUNKS``.

    Args:
        node_ids (Sequence[str]): The pinned chunk ids, in scope order.

    Returns:
        _ScopedRetriever: The retriever.
    """

    def _retrieve(**kwargs: Any) -> list[Any]:
        """Return the points asked for, as Qdrant would.

        Args:
            **kwargs (Any): The call's keywords; ``ids`` names the points.

        Returns:
            list[Any]: One point per id.
        """
        return [SimpleNamespace(id=point, payload={"text": CHUNKS[point]}) for point in kwargs["ids"]]

    rag = SimpleNamespace(qdrant_collection=PHYSICAL, qdrant_client=SimpleNamespace(retrieve=_retrieve))
    return _ScopedRetriever(rag=cast(Any, rag), node_ids=node_ids)


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
def test_the_plain_lookup_retrieves_with_the_rewrite_alone(session_manager: SessionManager, entry: str) -> None:
    """Retrieval runs on the rewrite as it stands; nothing is appended to it and no graph payload is reported.

    Args:
        session_manager (SessionManager): The session manager fixture.
        entry (str): The session manager method under test.
    """
    retriever = _Passages(_passages())
    _arm(session_manager, _engine(retriever, streaming=entry == "stream_chat"))

    final = _answer(session_manager, entry)

    assert retriever.asked == [REWRITE]
    assert ASKED in final["response"]
    assert "graph_debug" not in final
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
    assert retriever.asked == [reformulated]
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
    _arm(session_manager, _engine(_scoped_retriever(["c2", "c1"]), streaming=entry == "stream_chat"))

    final = _answer(session_manager, entry, scoped_node_ids=["c2", "c1"])

    answer = final["response"]
    assert ASKED in answer
    assert answer.index(CHUNKS["c2"]) < answer.index(CHUNKS["c1"])
    assert CHUNKS["c3"] not in answer
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
    async def achat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        """Keep the messages and answer, asynchronously.

        Args:
            messages (Sequence[ChatMessage]): The rendered messages.
            **kwargs (Any): Unused.

        Returns:
            ChatResponse: The answer.
        """
        return self.chat(messages)

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


def _visual_engine(
    retriever: BaseRetriever, llm: _SeeingLLM, store: ThumbnailStore, *, streaming: bool
) -> RetrieverQueryEngine:
    """Build a real engine whose synthesizer shows the model the stored thumbnails of its images.

    Args:
        retriever (BaseRetriever): The engine's retriever.
        llm (_SeeingLLM): The model, which keeps what it is sent.
        store (ThumbnailStore): The thumbnails the synthesizer may fetch, by point id.
        streaming (bool): Whether the synthesizer streams.

    Returns:
        RetrieverQueryEngine: The engine.
    """
    synthesizer = VisualStreamingCompactAndRefine(
        llm=llm, text_qa_template=PromptTemplate(ECHO_TEMPLATE), streaming=streaming
    )
    synthesizer._fetch_thumbnails = store
    synthesizer._max_images = 4
    synthesizer._legend_template = "Images:\n{image_legend}"
    synthesizer._attached = []
    synthesizer._stamped = []
    return RetrieverQueryEngine(
        retriever=retriever,
        node_postprocessors=[CitationNumberingPostprocessor()],
        response_synthesizer=synthesizer,
    )


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
    retriever = _Passages([frame_passage(1)])
    _arm(session_manager, _visual_engine(retriever, llm, store, streaming=entry == "stream_chat"))

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


#: What a stateless user asked: there is no history, so nothing to rewrite.
QUESTION = "Which stalls open on Sunday?"
#: The stateless entry points: the sync query, and the async one the SSE endpoint awaits.
QUERY_ENTRIES = ["run_query", "run_query_async"]
#: The retrieval settings a stateless query reports; the real ones read the collection.
RETRIEVAL_SETTINGS: dict[str, Any] = {
    "label": "default",
    "parent_context_enabled": False,
    "vector_store_query_mode": SimpleNamespace(value="default"),
}


@pytest.fixture
def stateless_rag(monkeypatch: pytest.MonkeyPatch) -> RAG:
    """Provide a RAG whose engine each test brings and whose retrieval settings need no collection.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.

    Returns:
        RAG: The instance.
    """
    monkeypatch.setattr(RAG, "_resolve_runtime_retrieval_settings", lambda self, **kwargs: RETRIEVAL_SETTINGS)
    return RAG(qdrant_collection=PHYSICAL)


def _bring(monkeypatch: pytest.MonkeyPatch, engine: RetrieverQueryEngine) -> None:
    """Hand the stateless query its engine, whether it is built lazily or for the request.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        engine (RetrieverQueryEngine): The engine.
    """
    monkeypatch.setattr(RAG, "build_query_engine", lambda self, **kwargs: engine)


def _ask(rag: RAG, entry: str, **kwargs: Any) -> dict[str, Any]:
    """Run one stateless query for ``QUESTION`` through the sync or the async entry point.

    Args:
        rag (RAG): The RAG under test.
        entry (str): ``run_query`` or ``run_query_async``.
        **kwargs (Any): Further keyword arguments for the entry point.

    Returns:
        dict[str, Any]: The normalized payload.
    """
    if entry == "run_query":
        return rag.run_query(QUESTION, **kwargs)
    return asyncio.run(rag.run_query_async(QUESTION, **kwargs))


@pytest.mark.parametrize("entry", QUERY_ENTRIES)
def test_the_stateless_answer_is_asked_the_users_question(
    stateless_rag: RAG, monkeypatch: pytest.MonkeyPatch, entry: str
) -> None:
    """A stateless turn has no rewrite, so the answer is asked the question alone.

    Args:
        stateless_rag (RAG): The RAG fixture.
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        entry (str): The entry point under test.
    """
    _bring(monkeypatch, _engine(_Passages(_passages()), streaming=False))

    result = _ask(stateless_rag, entry)

    assert f"Q<<{QUESTION}>>" in result["response"]
    assert RESOLVED_QUESTION_LABEL not in result["response"]


@pytest.mark.parametrize("entry", QUERY_ENTRIES)
def test_a_stateless_query_retrieves_with_the_prompt(
    stateless_rag: RAG, monkeypatch: pytest.MonkeyPatch, entry: str
) -> None:
    """A stateless query retrieves with and asks the same words.

    Args:
        stateless_rag (RAG): The RAG fixture.
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        entry (str): The entry point under test.
    """
    retriever = _Passages(_passages())
    _bring(monkeypatch, _engine(retriever, streaming=False))

    result = _ask(stateless_rag, entry)

    assert retriever.asked == [QUESTION]
    assert f"Q<<{QUESTION}>>" in result["response"]


@pytest.mark.parametrize("entry", QUERY_ENTRIES)
def test_a_scoped_stateless_lookup_answers_the_question_from_exactly_the_pinned_chunks(
    stateless_rag: RAG, monkeypatch: pytest.MonkeyPatch, entry: str
) -> None:
    """A pinned scope retrieves by id whatever the query, and its answer is asked the question all the same.

    Args:
        stateless_rag (RAG): The RAG fixture.
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        entry (str): The entry point under test.
    """
    _bring(monkeypatch, _engine(_scoped_retriever(["c2", "c1"]), streaming=False))

    result = _ask(stateless_rag, entry, scoped_node_ids=["c2", "c1"])

    answer = result["response"]
    assert f"Q<<{QUESTION}>>" in answer
    assert answer.index(CHUNKS["c2"]) < answer.index(CHUNKS["c1"])
    assert CHUNKS["c3"] not in answer
    assert (result["retrieval_mode"], result["scoped_chunk_count"]) == ("scoped", 2)


@pytest.mark.parametrize("entry", QUERY_ENTRIES)
def test_a_visual_stateless_lookup_still_shows_the_model_its_images_and_asks_the_question(
    stateless_rag: RAG, monkeypatch: pytest.MonkeyPatch, entry: str
) -> None:
    """The visual synthesizer collects its images in ``synthesize`` and ``asynthesize``, which the query calls itself.

    Args:
        stateless_rag (RAG): The RAG fixture.
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        entry (str): The entry point under test.
    """
    llm = _SeeingLLM(max_tokens=None)
    store = ThumbnailStore(1)
    retriever = _Passages([frame_passage(1)])
    _bring(monkeypatch, _visual_engine(retriever, llm, store, streaming=False))

    result = _ask(stateless_rag, entry, retrieval_target="visual")

    assert result["response"] == SEEN_ANSWER
    assert store.fetches == [["frame-1"]]
    message = llm.seen[0][-1]
    assert f"Q<<{QUESTION}>>" in (message.content or "")
    assert sum(isinstance(block, ImageBlock) for block in message.blocks) == 1
    parts = cast(list[dict[str, Any]], to_openai_message_dict(message)["content"])
    assert [part["image_url"]["url"] for part in parts if part.get("type") == "image_url"] == [thumbnail_url(1)]
    assert (result["retrieval_target"], result["visual"]) == ("visual", {"images_attached": 1})
    assert retriever.asked == [QUESTION]


class _Owners:
    """Collection-owner stand-in naming each owner's physical collection."""

    def resolve(self, owner: str | None, logical: str) -> str:
        """Return the physical name an owner's logical collection has.

        Args:
            owner (str | None): The calling principal.
            logical (str): The logical collection name.

        Returns:
            str: A name unique to the owner and the collection.
        """
        return f"u_{owner}__{logical}"


@dataclass(frozen=True)
class _Served:
    """The real API served over a real RAG whose engine echoes its prompt.

    Attributes:
        client (TestClient): Calls the API.
        retriever (_Passages): The engine's retriever, which keeps the queries it was asked.
    """

    client: TestClient
    retriever: _Passages


@pytest.fixture
def served(monkeypatch: pytest.MonkeyPatch) -> _Served:
    """Serve the real API over a real RAG whose engine echoes its prompt.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.

    Returns:
        _Served: The client and the retriever.
    """
    retriever = _Passages(_passages())
    engine = _engine(retriever, streaming=False)
    monkeypatch.setattr(RAG, "build_query_engine", lambda self, **kwargs: engine)
    monkeypatch.setattr(RAG, "create_index", lambda self: None)
    monkeypatch.setattr(RAG, "create_query_engine", lambda self: None)
    monkeypatch.setattr(RAG, "_resolve_runtime_retrieval_settings", lambda self, **kwargs: RETRIEVAL_SETTINGS)
    monkeypatch.setattr(RAG, "ensure_collection_owner_manager", lambda self: _Owners())
    monkeypatch.setattr(api_module, "rag", RAG(qdrant_collection=""))
    monkeypatch.setattr(api_module, "_validation_payload", lambda **kwargs: {})
    monkeypatch.setattr(api_module, "_agent_turn_allowed", lambda *args: False)
    monkeypatch.setattr(api_module, "build_metadata_filters", lambda rules: object())
    monkeypatch.setattr(api_module, "build_qdrant_filter", lambda rules: None)
    monkeypatch.setenv("DOCINT_DEFAULT_IDENTITY", "operator")
    return _Served(client=TestClient(api_module.app), retriever=retriever)


#: A stateless request: no session, the collection named by its logical name.
STATELESS_REQUEST = {"question": QUESTION, "collection": "docs", "retrieval_mode": "stateless"}


def test_the_stateless_query_endpoint_retrieves_with_and_asks_the_question(served: _Served) -> None:
    """``/query`` retrieves with the question as typed and reports no graph payload.

    Args:
        served (_Served): The served API.
    """
    response = served.client.post("/query", json=STATELESS_REQUEST, headers={"X-Auth-User": "operator"})

    assert response.status_code == 200, response.text
    body = response.json()
    assert served.retriever.asked == [QUESTION]
    assert f"Q<<{QUESTION}>>" in body["answer"]
    assert "graph_debug" not in body


def test_the_stateless_stream_endpoint_retrieves_with_and_asks_the_question(served: _Served) -> None:
    """``/stream_query`` does the same through the async query the SSE body awaits.

    Args:
        served (_Served): The served API.
    """
    with served.client.stream(
        "POST", "/stream_query", json=STATELESS_REQUEST, headers={"X-Auth-User": "operator"}
    ) as response:
        assert response.status_code == 200
        text = "".join(chunk.decode() for chunk in response.iter_raw())

    frames = [json.loads(line.removeprefix("data: ")) for line in text.splitlines() if line.startswith("data: ")]
    final = next(frame for frame in frames if "response" in frame)
    assert served.retriever.asked == [QUESTION]
    assert f"Q<<{QUESTION}>>" in final["response"]
    assert "graph_debug" not in final
