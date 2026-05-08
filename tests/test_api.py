"""Tests for the FastAPI application endpoints."""

import types
from collections.abc import Generator
from pathlib import Path
from typing import Any, cast

import pytest
from fastapi.testclient import TestClient

import docint.core.api as api_module
from docint.agents.types import IntentAnalysis, OrchestratorResult, RetrievalResult


class DummySessionManager:
    """Dummy session manager for testing purposes."""

    def list_sessions(self) -> list[dict[str, Any]]:
        """List all sessions.

        Returns:
            list[dict[str, Any]]: A list of session dictionaries.
        """
        return [{"id": "123", "created_at": "2023-01-01", "title": "Test Chat"}]

    def get_session_history(self, session_id: str) -> list[dict[str, Any]]:
        """Get the message history for a session.

        Args:
            session_id (str): The ID of the session.

        Returns:
            list[dict[str, Any]]: A list of message dictionaries.
        """
        return [{"role": "user", "content": "hi"}]

    def delete_session(self, session_id: str) -> bool:
        """Delete a session by ID.

        Args:
            session_id (str): The ID of the session.

        Returns:
            bool: True if the session was successfully deleted, False otherwise.
        """
        return True

    def get_agent_context(self, session_id: str) -> Any:
        """
        Get the agent context for a session.

        Args:
            session_id (str): The ID of the session.

        Returns:
            Any: The agent context for the session.
        """

        class Ctx:
            """Dummy context object for testing purposes."""

            clarifications = 0

        return Ctx()


class DummyRAG:
    """Dummy Retrieval-Augmented Generation (RAG) class for testing purposes."""

    def __init__(self) -> None:
        """Initialize the DummyRAG instance."""
        self.qdrant_collection = "alpha"
        self.summarize_prompt = "Summarize collection"
        self.index = object()
        self.query_engine = object()
        self.selected: list[str] = []
        self.sessions = DummySessionManager()
        self.chats: list[str] = []
        self.stateless_queries: list[str] = []
        self.entity_occurrence_queries: list[str] = []
        self.entity_occurrence_filters: list[Any] = []
        self.entity_occurrence_merge_modes: list[str] = []
        self.multi_entity_occurrence_queries: list[str] = []
        self.multi_entity_occurrence_filters: list[Any] = []
        self.multi_entity_occurrence_merge_modes: list[str] = []
        self.chat_filters: list[Any] = []
        self.stream_filters: list[Any] = []
        self.created_index = 0  # Tracks the number of times an index is created
        self.created_query_engine = 0
        self.ner_sources: list[dict[str, Any]] = []
        self.ner_refresh_calls: list[bool] = []
        self.ner_stats_merge_modes: list[str] = []
        self.ner_search_merge_modes: list[str] = []
        self.hate_speech_rows: list[dict[str, Any]] = []
        self.summary_refresh_calls: list[bool] = []
        self.summary_stream_refresh_calls: list[bool] = []
        self.summary_payload: dict[str, Any] = {
            "response": "summary",
            "sources": [{"id": "s1"}],
            "summary_diagnostics": {
                "total_documents": 2,
                "covered_documents": 2,
                "coverage_ratio": 1.0,
                "uncovered_documents": [],
                "coverage_target": 0.7,
                "coverage_unit": "documents",
                "candidate_count": 2,
                "deduped_count": 2,
                "sampled_count": 2,
            },
        }

    def list_collections(self) -> list[str]:
        """List all available collections.

        Returns:
            list[str]: A list of collection names.
        """
        return ["alpha", "beta"]

    def select_collection(self, name: str) -> None:
        """Select a collection by name.

        Args:
            name (str): The name of the collection to select.
        """
        self.selected.append(name)
        self.qdrant_collection = name
        self.index = None
        self.query_engine = None

    def create_index(self) -> None:
        """Create a new index for the selected collection."""
        self.created_index += 1
        self.index = object()

    def create_query_engine(self) -> None:
        """Create a new query engine for the selected collection."""
        self.created_query_engine += 1
        self.query_engine = object()

    def ensure_session_manager(self) -> DummySessionManager:
        """Return the SessionManager stub, mirroring RAG.ensure_session_manager.

        Returns:
            DummySessionManager: The pre-initialized session manager stub.
        """
        return self.sessions

    def start_session(self, session_id: str | None = None) -> str:
        """Start a new session or resume an existing one.

        Args:
            session_id (str | None, optional): The ID of the session to resume. Defaults to None.

        Returns:
            str: The session ID.
        """
        return session_id or "generated-session"

    def chat(
        self,
        question: str,
        *,
        metadata_filters: Any = None,
        metadata_filters_active: bool = False,
        metadata_filter_rules: Any = None,
        vector_store_kwargs: Any = None,
    ) -> dict[str, Any]:
        """Chat with the RAG system.

        Args:
            question (str): The question to ask the RAG system.
            metadata_filters (Any): Optional compiled metadata filters.
            metadata_filters_active (bool): Whether request filters were active.
            metadata_filter_rules (Any): Optional raw request filter rules.
            vector_store_kwargs (Any): Optional native vector-store query kwargs.

        Returns:
            dict[str, Any]: The response from the RAG system.
        """
        self.chats.append(question)
        self.chat_filters.append(
            {
                "filters": metadata_filters,
                "active": metadata_filters_active,
                "rules": metadata_filter_rules,
                "vector_store_kwargs": vector_store_kwargs,
            }
        )
        return {
            "response": "answer",
            "sources": [{"id": 1}],
            "retrieval_query": f"rewritten::{question}",
            "coverage_unit": "documents",
            "retrieval_mode": "rewrite_compact_graph",
            "graph_debug": {
                "enabled": True,
                "applied": True,
                "original_query": question,
                "expanded_query": f"{question}\n\nRelated entities for retrieval: Acme",
                "anchor_entities": ["Acme"],
                "neighbor_entities": ["Widget"],
            },
        }

    def stream_chat(
        self,
        question: str,
        *,
        metadata_filters: Any = None,
        metadata_filters_active: bool = False,
        metadata_filter_rules: Any = None,
        vector_store_kwargs: Any = None,
    ) -> Generator[str | dict[str, Any], None, None]:
        """
        Stream chat responses from the RAG system.

        Args:
            question (str): The question to ask the RAG system.
            metadata_filters (Any): Optional compiled metadata filters.
            metadata_filters_active (bool): Whether request filters were active.
            metadata_filter_rules (Any): Optional raw request filter rules.
            vector_store_kwargs (Any): Optional native vector-store query kwargs.

        Yields:
            str | dict[str, Any]: Chunks of the chat response as they are generated.
        """
        self.stream_filters.append(
            {
                "filters": metadata_filters,
                "active": metadata_filters_active,
                "rules": metadata_filter_rules,
                "vector_store_kwargs": vector_store_kwargs,
            }
        )
        yield "chunk"
        yield {
            "response": "answer",
            "sources": [{"id": 1}],
            "session_id": "generated-session",
            "retrieval_query": f"rewritten::{question}",
            "coverage_unit": "documents",
            "retrieval_mode": "rewrite_compact_graph",
            "graph_debug": {
                "enabled": True,
                "applied": True,
                "original_query": question,
                "expanded_query": f"{question}\n\nRelated entities for retrieval: Acme",
                "anchor_entities": ["Acme"],
                "neighbor_entities": ["Widget"],
            },
        }

    def run_query(
        self,
        prompt: str,
        *,
        metadata_filters: Any = None,
        metadata_filter_rules: Any = None,
        vector_store_kwargs: Any = None,
    ) -> dict[str, Any]:
        """Run a stateless retrieval query.

        Args:
            prompt: Query prompt.
            metadata_filters: Optional compiled metadata filters.
            metadata_filter_rules: Optional raw request filter rules.
            vector_store_kwargs: Optional native vector-store query kwargs.

        Returns:
            dict[str, Any]: Response payload.
        """
        _ = metadata_filters
        _ = metadata_filter_rules
        _ = vector_store_kwargs
        self.stateless_queries.append(prompt)
        return {
            "response": "answer",
            "sources": [{"id": 1}],
        }

    def run_entity_occurrence_query(
        self,
        prompt: str,
        *,
        qdrant_filter: Any = None,
        limit: int = 100,
        refresh: bool = False,
        entity_merge_mode: str = "orthographic",
    ) -> dict[str, Any]:
        """Return canned entity-occurrence results for tests.

        Args:
            prompt: Query prompt.
            qdrant_filter: Optional native Qdrant filter.
            limit: Maximum number of occurrence rows to return.
            refresh: Whether cache refresh was requested.

        Returns:
            dict[str, Any]: Canned occurrence payload.
        """
        _ = (limit, refresh)
        self.entity_occurrence_queries.append(prompt)
        self.entity_occurrence_filters.append(qdrant_filter)
        self.entity_occurrence_merge_modes.append(entity_merge_mode)
        if prompt == "Ambiguous":
            return {
                "response": "Your query matches multiple entities equally well.",
                "sources": [],
                "retrieval_query": prompt,
                "coverage_unit": "entity_mentions",
                "retrieval_mode": "entity_occurrence_ambiguous",
                "entity_match_candidates": [
                    {"text": "Acme", "type": "ORG", "mentions": 3},
                    {"text": "Acme", "type": "PRODUCT", "mentions": 2},
                ],
                "entity_match_groups": [],
            }
        return {
            "response": "Found 3 occurrence(s) of 'Acme' across 2 chunk(s) in 2 document(s).",
            "sources": [{"id": "occ-1"}, {"id": "occ-2"}],
            "retrieval_query": prompt,
            "coverage_unit": "entity_mentions",
            "retrieval_mode": "entity_occurrence",
            "entity_match_candidates": [{"text": "Acme", "type": "ORG", "mentions": 3}],
            "entity_match_groups": [
                {
                    "entity": {"text": "Acme", "type": "ORG", "mentions": 3},
                    "sources": [{"id": "occ-1"}, {"id": "occ-2"}],
                    "chunk_count": 2,
                    "document_count": 2,
                    "truncated": False,
                }
            ],
        }

    def run_multi_entity_occurrence_query(
        self,
        prompt: str,
        *,
        qdrant_filter: Any = None,
        limit: int = 100,
        refresh: bool = False,
        entity_merge_mode: str = "orthographic",
    ) -> dict[str, Any]:
        """Return canned multi-entity occurrence results for tests."""
        _ = (limit, refresh)
        self.multi_entity_occurrence_queries.append(prompt)
        self.multi_entity_occurrence_filters.append(qdrant_filter)
        self.multi_entity_occurrence_merge_modes.append(entity_merge_mode)
        return {
            "response": "Found 2 equally strong entity match(es) for 'Acme', covering 2 chunk(s) across 2 document(s).",
            "sources": [{"id": "occ-org"}, {"id": "occ-product"}],
            "retrieval_query": prompt,
            "coverage_unit": "entity_mentions",
            "retrieval_mode": "entity_occurrence_multi",
            "entity_match_candidates": [
                {"text": "Acme", "type": "ORG", "mentions": 3},
                {"text": "Acme", "type": "PRODUCT", "mentions": 2},
            ],
            "entity_match_groups": [
                {
                    "entity": {"text": "Acme", "type": "ORG", "mentions": 3},
                    "sources": [{"id": "occ-org"}],
                    "chunk_count": 1,
                    "document_count": 1,
                    "truncated": False,
                },
                {
                    "entity": {"text": "Acme", "type": "PRODUCT", "mentions": 2},
                    "sources": [{"id": "occ-product"}],
                    "chunk_count": 1,
                    "document_count": 1,
                    "truncated": False,
                },
            ],
        }

    def expand_query_with_graph_with_debug(
        self, query: str
    ) -> tuple[str, dict[str, Any]]:
        """Return deterministic GraphRAG expansion metadata for tests.

        Args:
            query: Input query.

        Returns:
            tuple[str, dict[str, Any]]: Expanded query and debug metadata.
        """
        return (
            f"{query}\n\nRelated entities for retrieval: Acme",
            {
                "enabled": True,
                "applied": True,
                "original_query": query,
                "expanded_query": f"{query}\n\nRelated entities for retrieval: Acme",
                "anchor_entities": ["Acme"],
                "neighbor_entities": ["Widget"],
            },
        )

    def summarize_collection(self, refresh: bool = False) -> dict[str, Any]:
        """Return canned summarize payload.

        Args:
            refresh (bool, optional): Whether to bypass summary cache.

        Returns:
            dict[str, Any]: A dictionary containing the summary response, sources, and diagnostics.
        """
        self.summary_refresh_calls.append(bool(refresh))
        return self.summary_payload

    def stream_summarize_collection(
        self,
        refresh: bool = False,
    ) -> Generator[str | dict[str, Any], None, None]:
        """Stream canned summary payload.

        Args:
            refresh (bool, optional): Whether to bypass summary cache.

        Returns:
            Generator[str | dict[str, Any], None, None]: A generator that yields chunks of the summary response and the final payload.

        Yields:
            str | dict[str, Any]: Chunks of the summary response as they are generated, followed by the final summary payload.
        """
        self.summary_stream_refresh_calls.append(bool(refresh))
        yield "sum"
        yield self.summary_payload

    def get_collection_ner(self, refresh: bool = False) -> list[dict[str, Any]]:
        """Get information extraction data for the selected collection.

        Args:
            refresh (bool, optional): Whether to bypass cached NER rows.

        Returns:
            list[dict[str, Any]]: Information extraction data for the selected collection.
        """
        self.ner_refresh_calls.append(bool(refresh))
        return self.ner_sources

    def get_collection_hate_speech(self) -> list[dict[str, Any]]:
        """Get hate-speech findings for the selected collection.

        Returns:
            list[dict[str, Any]]: A list of dictionaries containing metadata about hate-speech
            findings, such as chunk ID, text, category, confidence, reason, source reference,
            and page number.
        """
        return self.hate_speech_rows

    def get_collection_ner_stats(
        self,
        *,
        top_k: int = 15,
        min_mentions: int = 2,
        entity_type: str | None = None,
        include_relations: bool = True,
        entity_merge_mode: str = "orthographic",
    ) -> dict[str, Any]:
        """Return canned NER stats payload.

        Args:
            top_k (int, optional): The number of top entities to return. Defaults to 15.
            min_mentions (int, optional): The minimum number of mentions for an entity to be included. Defaults to 2.
            entity_type (str | None, optional): Filter entities by type. Defaults to None.
            include_relations (bool, optional): Whether to include relation statistics. Defaults to True.

        Returns:
            dict[str, Any]: A dictionary containing information extraction statistics, including totals, top entities, entity types, top relations, and document-level stats.
        """
        _ = (top_k, min_mentions, entity_type, include_relations)
        self.ner_stats_merge_modes.append(entity_merge_mode)
        return {
            "totals": {
                "unique_entities": 1,
                "entity_mentions": 3,
                "unique_relations": 1,
            },
            "top_entities": [
                {
                    "text": "Acme",
                    "type": "ORG",
                    "mentions": 3,
                    "best_score": 0.9,
                    "source_count": 2,
                }
            ],
            "entity_types": [{"type": "ORG", "mentions": 3, "unique_entities": 1}],
            "top_relations": [
                {"head": "Acme", "label": "owns", "tail": "Widget", "mentions": 2}
            ],
            "documents": [
                {
                    "filename": "doc1.pdf",
                    "entity_mentions": 3,
                    "unique_entities": 1,
                    "ie_source_count": 2,
                    "entity_density": 1.5,
                }
            ],
        }

    def search_collection_ner_entities(
        self,
        *,
        q: str = "",
        entity_type: str | None = None,
        limit: int = 100,
        entity_merge_mode: str = "orthographic",
    ) -> list[dict[str, Any]]:
        """Return simple entity search results.

        Args:
            q (str, optional): The search query string. Defaults to "".
            entity_type (str | None, optional): Filter entities by type. Defaults to None.
            limit (int, optional): The maximum number of results to return. Defaults to 100.

        Returns:
            list[dict[str, Any]]: A list of entity dictionaries that match the search criteria.
        """
        _ = (entity_type, limit)
        self.ner_search_merge_modes.append(entity_merge_mode)
        if q and q.lower() not in "acme":
            return []
        return [
            {
                "text": "Acme",
                "type": "ORG",
                "mentions": 3,
                "best_score": 0.9,
                "source_count": 2,
            }
        ]


@pytest.fixture(autouse=True)
def _patch_rag(monkeypatch: pytest.MonkeyPatch) -> Any | None:
    """Patch the RAG instance for testing.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.

    Returns:
        Any | None: Yields None after patching.
    """
    dummy = DummyRAG()
    monkeypatch.setattr(api_module, "rag", dummy)
    yield


@pytest.fixture
def client() -> TestClient:
    """Create a TestClient for testing the FastAPI application.

    Returns:
        TestClient: The TestClient instance.
    """
    return TestClient(api_module.app)


def test_collections_list_success(client: TestClient) -> None:
    """Test the successful retrieval of the collections list.

    Args:
        client (TestClient): The TestClient instance.
    """
    response = client.get("/collections/list")
    assert response.status_code == 200
    assert response.json() == ["alpha", "beta"]


def test_collections_list_failure(
    monkeypatch: pytest.MonkeyPatch, client: TestClient
) -> None:
    """Test the failed retrieval of the collections list.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        client (TestClient): The TestClient instance.
    """

    def raiser() -> list[str]:
        raise RuntimeError("boom")

    monkeypatch.setattr(api_module.rag, "list_collections", raiser)
    response = client.get("/collections/list")
    assert response.status_code == 500
    assert response.json()["detail"] == "boom"


def test_collections_select_success(
    monkeypatch: pytest.MonkeyPatch, client: TestClient
) -> None:
    """Selecting a collection must succeed without warming the query engine.

    Regression guard for the same OOM pattern that commit 18a47a6 removed
    from ``/ingest`` and ``/ingest/upload``: ``/collections/select``
    previously called ``rag.create_index`` + ``rag.create_query_engine`` +
    ``rag.get_collection_ner(refresh=True)`` immediately after
    ``rag.select_collection``. That chain loads bge-m3 (~2 GB),
    bge-reranker-v2-m3 (~1 GB), and GLiNER on every collection switch,
    causing OOM-kill on CPU Docker. The query engine is now built lazily
    on the first chat query.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        client (TestClient): The TestClient instance.
    """
    response = client.post("/collections/select", json={"name": " gamma "})
    assert response.status_code == 200
    payload = response.json()
    assert payload == {"ok": True, "name": "gamma"}
    rag = cast(Any, api_module.rag)
    assert rag.qdrant_collection == "gamma"
    # Neither the index nor the query engine may be built eagerly on select.
    assert rag.created_index == 0, (
        "rag.create_index() must NOT be called from /collections/select; "
        "it triggers bge-m3 load and OOM-kills CPU Docker on collection switch."
    )
    assert rag.created_query_engine == 0, (
        "rag.create_query_engine() must NOT be called from /collections/select; "
        "it triggers reranker + embedding loads and OOM-kills CPU Docker."
    )
    # NER pre-warm must also be skipped — it previously triggered GLiNER on select.
    assert rag.ner_refresh_calls == [], (
        "get_collection_ner() must NOT be called from /collections/select; "
        "it triggers GLiNER load and compounds warmup memory pressure."
    )


def test_collections_select_blank_name(client: TestClient) -> None:
    """Test the selection of a collection with a blank name.

    Args:
        client (TestClient): The TestClient instance.
    """
    response = client.post("/collections/select", json={"name": "   "})
    assert response.status_code == 500
    assert "Collection name required" in response.json()["detail"]


def test_collections_select_raises_on_nonexistent_collection(
    monkeypatch: pytest.MonkeyPatch, client: TestClient
) -> None:
    """Non-``HTTPException`` errors from ``select_collection`` surface as 500.

    ``/collections/select`` only catches ``HTTPException`` in its outer
    handler. If ``rag.select_collection`` raises ``ValueError`` (collection
    missing) or any other non-HTTP exception, it propagates to FastAPI's
    default handler and becomes an HTTP 500. This test locks that contract
    so any future refactor that silently swallows the error (or wraps it
    as a 200) fails this assertion.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        client (TestClient): The TestClient instance.
    """

    def raise_missing(name: str) -> None:
        """Simulate the production-code ``ValueError`` path.

        Args:
            name (str): The collection name (ignored).

        Raises:
            ValueError: Always, to mimic a missing-collection scenario.
        """
        raise ValueError(f"Collection '{name}' does not exist")

    monkeypatch.setattr(api_module.rag, "select_collection", raise_missing)
    # TestClient's default raise_server_exceptions=True re-raises uncaught
    # server-side exceptions to the caller. In production (uvicorn), FastAPI's
    # default handler would turn this into an HTTP 500 response. Asserting the
    # exception propagates captures the same contract — the endpoint does not
    # silently swallow or coerce the error.
    with pytest.raises(ValueError, match="does not exist"):
        client.post("/collections/select", json={"name": "ghost"})


def test_collections_ner_success(client: TestClient) -> None:
    """Test the successful retrieval of information extraction data.

    Args:
        client (TestClient): The TestClient instance.
    """
    api_module.rag.ner_sources = [
        {"filename": "doc1.pdf", "page": 1, "row": 2, "entities": [], "relations": []}
    ]
    response = client.get("/collections/ner")
    assert response.status_code == 200
    assert response.json() == {"sources": api_module.rag.ner_sources}
    assert cast(DummyRAG, api_module.rag).ner_refresh_calls[-1] is False


def test_collections_ner_refresh_success(client: TestClient) -> None:
    """NER endpoint should forward explicit refresh requests.

    Args:
        client (TestClient): The TestClient instance.
    """
    dummy_rag = cast(DummyRAG, api_module.rag)
    dummy_rag.ner_sources = [{"filename": "doc1.pdf", "entities": [], "relations": []}]

    response = client.get("/collections/ner", params={"refresh": "true"})

    assert response.status_code == 200
    assert response.json()["sources"] == dummy_rag.ner_sources
    assert dummy_rag.ner_refresh_calls[-1] is True


def test_collections_ner_stats_success(client: TestClient) -> None:
    """Stats endpoint should return NER summary payload.

    Args:
        client (TestClient): The TestClient instance.
    """
    response = client.get("/collections/ner/stats")
    assert response.status_code == 200
    payload = response.json()
    assert payload["totals"]["unique_entities"] == 1
    assert payload["top_entities"][0]["text"] == "Acme"
    assert cast(DummyRAG, api_module.rag).ner_stats_merge_modes[-1] == "orthographic"


def test_collections_ner_stats_support_exact_merge_mode(client: TestClient) -> None:
    """Stats endpoint should forward explicit merge-mode overrides."""
    response = client.get(
        "/collections/ner/stats", params={"entity_merge_mode": "exact"}
    )
    assert response.status_code == 200
    assert cast(DummyRAG, api_module.rag).ner_stats_merge_modes[-1] == "exact"


def test_collections_hate_speech_success(client: TestClient) -> None:
    """Hate-speech endpoint should return flagged rows.

    Args:
        client (TestClient): The TestClient instance.
    """
    dummy_rag = cast(DummyRAG, api_module.rag)
    dummy_rag.hate_speech_rows = [
        {
            "chunk_id": "c1",
            "chunk_text": "flagged text",
            "category": "ethnicity",
            "confidence": "high",
            "reason": "Contains slur",
            "source_ref": "doc1.pdf",
            "page": 2,
        }
    ]
    response = client.get("/collections/hate-speech")
    assert response.status_code == 200
    payload = response.json()
    assert payload["results"][0]["chunk_id"] == "c1"


def test_collections_ner_search_success(client: TestClient) -> None:
    """Search endpoint should return matching entities.

    Args:
        client (TestClient): The TestClient instance.
    """
    response = client.get("/collections/ner/search", params={"q": "ac"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["results"][0]["text"] == "Acme"
    assert cast(DummyRAG, api_module.rag).ner_search_merge_modes[-1] == "orthographic"


def test_collections_ner_search_support_exact_merge_mode(client: TestClient) -> None:
    """Search endpoint should forward explicit merge-mode overrides."""
    response = client.get(
        "/collections/ner/search",
        params={"q": "ac", "entity_merge_mode": "exact"},
    )
    assert response.status_code == 200
    assert cast(DummyRAG, api_module.rag).ner_search_merge_modes[-1] == "exact"


def test_agent_chat_answers(
    monkeypatch: pytest.MonkeyPatch, client: TestClient
) -> None:
    """Agent chat should return an answer when confidence is sufficient.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        client (TestClient): The TestClient instance.
    """

    def fake_chat(question: str, **_: Any) -> dict[str, Any]:
        """
        Fake implementation of the RAG chat method for testing purposes.

        Args:
            question (str): The question to ask the RAG system.

        Returns:
            dict[str, Any]: The response from the RAG system, including an answer and sources.
        """
        return {"response": f"echo:{question}", "sources": [{"id": 1}]}

    monkeypatch.setattr(api_module.rag, "chat", fake_chat)

    payload = {"message": "hello"}
    response = client.post("/agent/chat", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "answer"
    assert data["answer"] == "echo:hello"
    assert data["sources"] == [{"id": 1}]
    assert data["session_id"] == "generated-session"
    assert data["intent"] is not None
    assert data["confidence"] is not None


def test_agent_chat_clarifies(
    monkeypatch: pytest.MonkeyPatch, client: TestClient
) -> None:
    """Agent chat should request clarification when policy requires it.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        client (TestClient): The TestClient instance.
    """

    monkeypatch.setattr(
        api_module,
        "_clarification_policy",
        api_module.ClarificationPolicy(
            api_module.ClarificationConfig(
                confidence_threshold=1.0, require_entities=True
            )
        ),
    )

    payload = {"message": "hello"}
    response = client.post("/agent/chat", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "clarification"
    assert data["message"]
    assert data["intent"] is not None
    assert data["confidence"] is not None


def test_agent_chat_returns_validation_alert(
    monkeypatch: pytest.MonkeyPatch, client: TestClient
) -> None:
    """Agent chat should surface response-validation metadata.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        client (TestClient): The TestClient instance.
    """

    class _StubOrchestrator:
        """Stub orchestrator that returns a canned retrieval result with validation metadata for testing purposes."""

        def handle_turn(self, turn, context=None) -> OrchestratorResult:
            """Handle a turn by returning a canned retrieval result with validation metadata.

            Args:
                turn (_type_): The user turn to process.
                context (_type_, optional): The context for the turn. Defaults to None.

            Returns:
                OrchestratorResult: The result of processing the turn.
            """
            _ = turn, context
            analysis = IntentAnalysis(
                intent="qa", confidence=0.9, entities={"query": "hello"}
            )
            retrieval = RetrievalResult(
                answer="answer",
                sources=[{"id": 1}],
                session_id="generated-session",
                validation_checked=True,
                validation_mismatch=True,
                validation_reason="mismatch",
            )
            return OrchestratorResult(
                clarification=None, retrieval=retrieval, analysis=analysis
            )

    monkeypatch.setattr(api_module, "_build_orchestrator", lambda: _StubOrchestrator())

    response = client.post("/agent/chat", json={"message": "hello"})

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "answer"
    assert data["validation_checked"] is True
    assert data["validation_mismatch"] is True
    assert data["validation_reason"] == "mismatch"


def test_agent_chat_stream_clarifies(
    monkeypatch: pytest.MonkeyPatch, client: TestClient
) -> None:
    """Streaming endpoint should emit clarification event when policy demands it.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        client (TestClient): The TestClient instance.
    """

    monkeypatch.setattr(
        api_module,
        "_clarification_policy",
        api_module.ClarificationPolicy(
            api_module.ClarificationConfig(
                confidence_threshold=1.0, require_entities=True
            )
        ),
    )

    with client.stream("POST", "/agent/chat/stream", json={"message": "hello"}) as resp:
        assert resp.status_code == 200
        text = "".join([chunk.decode() for chunk in resp.iter_raw()])
    assert "clarification" in text
    assert "status" in text


def test_stream_query_includes_validation_metadata(client: TestClient) -> None:
    """Streaming query endpoint should emit validation metadata in final payload.

    Args:
        client (TestClient): The TestClient instance.
    """
    with client.stream("POST", "/stream_query", json={"question": "hello"}) as resp:
        assert resp.status_code == 200
        text = "".join([chunk.decode() for chunk in resp.iter_raw()])
    assert '"validation_checked"' in text
    assert '"validation_mismatch"' in text
    assert '"graph_debug"' in text
    assert '"retrieval_query"' in text
    assert '"retrieval_mode"' in text
    assert '"response": "answer"' in text


def test_query_stateless_mode_skips_session_chat(client: TestClient) -> None:
    """Stateless query mode should use direct retrieval without chat session state.

    Args:
        client: The TestClient instance.
    """
    rag = cast(DummyRAG, api_module.rag)
    before_chats = len(rag.chats)

    response = client.post(
        "/query",
        json={"question": "What?", "retrieval_mode": "stateless"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "answer"
    assert body["session_id"] == "stateless"
    assert len(rag.chats) == before_chats
    assert rag.stateless_queries[-1].startswith("What?")
    assert body["graph_debug"]["applied"] is True


def test_stream_query_stateless_mode_emits_tokens(client: TestClient) -> None:
    """Stateless stream mode should emit token events and final metadata payload.

    Args:
        client: The TestClient instance.
    """
    with client.stream(
        "POST",
        "/stream_query",
        json={"question": "hello", "retrieval_mode": "stateless"},
    ) as resp:
        assert resp.status_code == 200
        text = "".join([chunk.decode() for chunk in resp.iter_raw()])

    assert '"token"' in text
    assert '"session_id": "stateless"' in text
    assert '"graph_debug"' in text


@pytest.mark.anyio
async def test_stream_simulated_text_applies_visible_pacing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Synthetic token replay should keep a small delay between chunks.

    Args:
        monkeypatch (pytest.MonkeyPatch): The pytest monkeypatch fixture.
    """
    delays: list[float] = []

    async def _fake_sleep(delay: float) -> None:
        delays.append(delay)

    monkeypatch.setattr(api_module.asyncio, "sleep", _fake_sleep)

    events: list[str] = []
    async for event in api_module._stream_simulated_text("hello world"):
        events.append(event)

    assert len(events) == 2
    assert all('"token"' in event for event in events)
    assert delays == [
        api_module.SIMULATED_STREAM_TOKEN_DELAY_SECONDS,
        api_module.SIMULATED_STREAM_TOKEN_DELAY_SECONDS,
    ]


def test_query_entity_occurrence_mode_skips_chat_and_uses_ner_lookup(
    client: TestClient,
) -> None:
    """Entity occurrence mode should bypass chat and return mention rows.

    Args:
        client: The TestClient instance.
    """
    rag = cast(DummyRAG, api_module.rag)
    before_chats = len(rag.chats)

    response = client.post(
        "/query",
        json={"question": "Acme", "query_mode": "entity_occurrence"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["answer"].startswith("Found 3 occurrence")
    assert body["sources"] == [{"id": "occ-1"}, {"id": "occ-2"}]
    assert body["session_id"] == "stateless"
    assert len(rag.chats) == before_chats
    assert rag.entity_occurrence_queries[-1] == "Acme"
    assert body["retrieval_mode"] == "entity_occurrence"


def test_query_entity_occurrence_mode_passes_native_filters(
    client: TestClient,
) -> None:
    """Entity occurrence mode should reuse native Qdrant metadata filters.

    Args:
        client: The TestClient instance.
    """
    response = client.post(
        "/query",
        json={
            "question": "Acme",
            "query_mode": "entity_occurrence",
            "metadata_filters": [
                {
                    "field": "hate_speech.hate_speech",
                    "operator": "eq",
                    "value": True,
                }
            ],
        },
    )

    assert response.status_code == 200
    rag = cast(DummyRAG, api_module.rag)
    assert rag.entity_occurrence_filters[-1] is not None


def test_stream_query_entity_occurrence_mode_emits_tokens(client: TestClient) -> None:
    """Streaming occurrence mode should stream the synthesized occurrence summary.

    Args:
        client: The TestClient instance.
    """
    with client.stream(
        "POST",
        "/stream_query",
        json={"question": "Acme", "query_mode": "entity_occurrence"},
    ) as resp:
        assert resp.status_code == 200
        text = "".join([chunk.decode() for chunk in resp.iter_raw()])

    assert '"token"' in text
    assert '"retrieval_mode": "entity_occurrence"' in text
    assert '"coverage_unit": "entity_mentions"' in text


def test_query_entity_occurrence_mode_returns_ambiguity_candidates(
    client: TestClient,
) -> None:
    """Ambiguous single-entity occurrence lookups should expose candidates."""
    response = client.post(
        "/query",
        json={"question": "Ambiguous", "query_mode": "entity_occurrence"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["retrieval_mode"] == "entity_occurrence_ambiguous"
    assert body["sources"] == []
    assert len(body["entity_match_candidates"]) == 2


def test_query_entity_occurrence_multi_mode_returns_groups(client: TestClient) -> None:
    """Multi-entity occurrence mode should return grouped strong matches."""
    response = client.post(
        "/query",
        json={"question": "Acme", "query_mode": "entity_occurrence_multi"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["retrieval_mode"] == "entity_occurrence_multi"
    assert len(body["entity_match_groups"]) == 2
    assert body["entity_match_groups"][0]["entity"]["text"] == "Acme"


def test_stream_query_entity_occurrence_multi_mode_emits_groups(
    client: TestClient,
) -> None:
    """Streaming multi-entity occurrence mode should include grouped results."""
    with client.stream(
        "POST",
        "/stream_query",
        json={"question": "Acme", "query_mode": "entity_occurrence_multi"},
    ) as resp:
        assert resp.status_code == 200
        text = "".join([chunk.decode() for chunk in resp.iter_raw()])

    assert '"retrieval_mode": "entity_occurrence_multi"' in text
    assert '"entity_match_groups"' in text


def test_summarize_includes_summary_diagnostics(client: TestClient) -> None:
    """Summarize endpoint should expose summary diagnostics and validation metadata.

    Args:
        client (TestClient): The TestClient instance.
    """
    response = client.post("/summarize")
    assert response.status_code == 200
    payload = response.json()
    assert payload["summary"] == "summary"
    assert payload["sources"] == [{"id": "s1"}]
    assert payload["summary_diagnostics"]["total_documents"] == 2
    assert payload["summary_diagnostics"]["covered_documents"] == 2
    assert payload["summary_diagnostics"]["coverage_unit"] == "documents"
    assert "validation_checked" in payload
    assert "validation_mismatch" in payload
    assert "validation_reason" in payload


def test_summarize_refresh_flag_passthrough(client: TestClient) -> None:
    """Summarize endpoint should pass refresh query parameter to RAG.

    Args:
        client (TestClient): The TestClient instance.
    """
    response = client.post("/summarize?refresh=true")
    assert response.status_code == 200
    rag = cast(DummyRAG, api_module.rag)
    assert rag.summary_refresh_calls[-1] is True


def test_summarize_stream_includes_summary_diagnostics(client: TestClient) -> None:
    """Streaming summarize endpoint should emit diagnostics in final payload.

    Args:
        client (TestClient): The TestClient instance.
    """
    with client.stream("POST", "/summarize/stream") as resp:
        assert resp.status_code == 200
        text = "".join([chunk.decode() for chunk in resp.iter_raw()])
    assert '"summary_diagnostics"' in text
    assert '"coverage_ratio"' in text
    assert '"coverage_unit"' in text


def test_summarize_stream_refresh_flag_passthrough(client: TestClient) -> None:
    """Streaming summarize endpoint should pass refresh query parameter to RAG.

    Args:
        client (TestClient): The TestClient instance.
    """
    with client.stream("POST", "/summarize/stream?refresh=true") as resp:
        assert resp.status_code == 200
        _ = "".join([chunk.decode() for chunk in resp.iter_raw()])
    rag = cast(DummyRAG, api_module.rag)
    assert rag.summary_stream_refresh_calls[-1] is True


def test_collections_ner_requires_selection(client: TestClient) -> None:
    """Test that information extraction requires a collection to be selected.

    Args:
        client (TestClient): The TestClient instance.
    """
    api_module.rag.qdrant_collection = ""
    response = client.get("/collections/ner")
    assert response.status_code == 400
    assert "No collection selected" in response.json()["detail"]


def test_collections_ner_stats_requires_selection(client: TestClient) -> None:
    """Stats endpoint should require active collection selection.

    Args:
        client (TestClient): The TestClient instance.
    """
    api_module.rag.qdrant_collection = ""
    response = client.get("/collections/ner/stats")
    assert response.status_code == 400
    assert "No collection selected" in response.json()["detail"]


def test_collections_ner_search_requires_selection(client: TestClient) -> None:
    """Search endpoint should require active collection selection.

    Args:
        client (TestClient): The TestClient instance.
    """
    api_module.rag.qdrant_collection = ""
    response = client.get("/collections/ner/search", params={"q": "acme"})
    assert response.status_code == 400
    assert "No collection selected" in response.json()["detail"]


def test_collections_hate_speech_requires_selection(client: TestClient) -> None:
    """Hate-speech endpoint should require active collection selection.

    Args:
        client (TestClient): The TestClient instance.
    """
    api_module.rag.qdrant_collection = ""
    response = client.get("/collections/hate-speech")
    assert response.status_code == 400
    assert "No collection selected" in response.json()["detail"]


def test_collections_ner_failure(
    monkeypatch: pytest.MonkeyPatch, client: TestClient
) -> None:
    """Test the failed retrieval of information extraction data.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        client (TestClient): The TestClient instance.
    """

    def raiser(*, refresh: bool = False) -> list[dict[str, Any]]:
        """Get information extraction data for the selected collection.

        Args:
            refresh (bool, optional): Whether to bypass cached NER rows.

        Returns:
            list[dict[str, Any]]: Information extraction data for the selected collection.

        Raises:
            RuntimeError: If there is an error retrieving the information extraction data.
        """
        _ = refresh
        raise RuntimeError("boom")

    monkeypatch.setattr(api_module.rag, "get_collection_ner", raiser)
    response = client.get("/collections/ner")
    assert response.status_code == 500
    assert response.json()["detail"] == "boom"


def test_collections_ner_stats_failure(
    monkeypatch: pytest.MonkeyPatch, client: TestClient
) -> None:
    """Stats endpoint should surface backend failures.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        client (TestClient): The TestClient instance.
    """

    def raiser(**kwargs) -> dict[str, Any]:
        """
        Fake implementation of get_collection_ner_stats that raises an error for testing purposes.

        Returns:
            dict[str, Any]: Information extraction statistics for the selected collection.

        Raises:
            RuntimeError: If there is an error retrieving the information extraction stats.
        """
        _ = kwargs
        raise RuntimeError("boom")

    monkeypatch.setattr(api_module.rag, "get_collection_ner_stats", raiser)
    response = client.get("/collections/ner/stats")
    assert response.status_code == 500
    assert response.json()["detail"] == "boom"


def test_collections_ner_search_failure(
    monkeypatch: pytest.MonkeyPatch, client: TestClient
) -> None:
    """Search endpoint should surface backend failures.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        client (TestClient): The TestClient instance.
    """

    def raiser(**kwargs) -> list[dict[str, Any]]:
        """
        Fake implementation of search_collection_ner_entities that raises an error for testing purposes.

        Returns:
            list[dict[str, Any]]: The search results.

        Raises:
            RuntimeError: If there is an error during the search.
        """
        _ = kwargs
        raise RuntimeError("boom")

    monkeypatch.setattr(api_module.rag, "search_collection_ner_entities", raiser)
    response = client.get("/collections/ner/search", params={"q": "ac"})
    assert response.status_code == 500
    assert response.json()["detail"] == "boom"


def test_collections_hate_speech_failure(
    monkeypatch: pytest.MonkeyPatch, client: TestClient
) -> None:
    """Hate-speech endpoint should surface backend failures.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        client (TestClient): The TestClient instance.
    """

    def raiser() -> list[dict[str, Any]]:
        """Fake implementation of get_collection_hate_speech that raises an error for testing purposes.

        Returns:
            list[dict[str, Any]]: A list of dictionaries containing metadata about hate-speech
            findings, such as chunk ID, text, category, confidence, reason, source reference,
            and page number.

        Raises:
            RuntimeError: If there is an error retrieving the hate-speech findings.
        """
        raise RuntimeError("boom")

    monkeypatch.setattr(api_module.rag, "get_collection_hate_speech", raiser)
    response = client.get("/collections/hate-speech")
    assert response.status_code == 500
    assert response.json()["detail"] == "boom"


def test_query_requires_collection(
    monkeypatch: pytest.MonkeyPatch, client: TestClient
) -> None:
    """Test the query endpoint requires a collection to be selected.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        client (TestClient): The TestClient instance.
    """
    api_module.rag.qdrant_collection = ""
    response = client.post("/query", json={"question": "hi"})
    assert response.status_code == 500
    assert "No collection selected" in response.json()["detail"]


def test_query_success(monkeypatch: pytest.MonkeyPatch, client: TestClient) -> None:
    """Test the successful query execution.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        client (TestClient): The TestClient instance.
    """
    api_module.rag.index = None
    api_module.rag.query_engine = None
    response = client.post(
        "/query",
        json={"question": "What?", "session_id": "abc"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "answer"
    assert body["sources"] == [{"id": 1}]
    assert body["session_id"] == "abc"
    assert body["graph_debug"]["applied"] is True
    assert body["graph_debug"]["anchor_entities"] == ["Acme"]


def test_query_builds_and_passes_metadata_filters(client: TestClient) -> None:
    """Query endpoint should compile request filters and pass them to RAG chat.

    Args:
        client (TestClient): The TestClient instance.
    """
    response = client.post(
        "/query",
        json={
            "question": "What?",
            "metadata_filters": [
                {
                    "field": "mimetype",
                    "operator": "mime_match",
                    "value": "image/*",
                },
                {
                    "field": "reference_metadata.timestamp",
                    "operator": "date_on_or_after",
                    "value": "2026-01-01",
                },
            ],
        },
    )

    assert response.status_code == 200
    rag = cast(DummyRAG, api_module.rag)
    last_filters = rag.chat_filters[-1]
    assert last_filters["active"] is True
    assert [rule.model_dump() for rule in last_filters["rules"]] == [
        {
            "field": "mimetype",
            "operator": "mime_match",
            "value": "image/*",
            "values": [],
        },
        {
            "field": "reference_metadata.timestamp",
            "operator": "date_on_or_after",
            "value": "2026-01-01",
            "values": [],
        },
    ]
    compiled = last_filters["filters"]
    assert compiled is not None
    assert len(compiled.filters) == 2
    assert last_filters["vector_store_kwargs"]["qdrant_filters"] is not None


def test_stream_query_passes_metadata_filters(client: TestClient) -> None:
    """Streaming query endpoint should compile and pass request filters.

    Args:
        client (TestClient): The TestClient instance.
    """
    with client.stream(
        "POST",
        "/stream_query",
        json={
            "question": "hello",
            "metadata_filters": [
                {
                    "field": "hate_speech.hate_speech",
                    "operator": "eq",
                    "value": True,
                }
            ],
        },
    ) as resp:
        assert resp.status_code == 200
        assert any(line for line in resp.iter_lines())

    rag = cast(DummyRAG, api_module.rag)
    last_filters = rag.stream_filters[-1]
    assert last_filters["active"] is True
    assert [rule.model_dump() for rule in last_filters["rules"]] == [
        {
            "field": "hate_speech.hate_speech",
            "operator": "eq",
            "value": True,
            "values": [],
        }
    ]
    assert last_filters["vector_store_kwargs"]["qdrant_filters"] is not None


def test_query_handles_missing_sources(
    monkeypatch: pytest.MonkeyPatch, client: TestClient
) -> None:
    """Test the query handles missing sources.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        client (TestClient): The TestClient instance.
    """

    def fake_chat(question: str, **_: Any) -> str:
        """Fake chat function for testing.

        Args:
            question (str): The question to ask.

        Returns:
            str: The response from the chat.
        """
        return "plain response"

    monkeypatch.setattr(api_module.rag, "chat", fake_chat)
    response = client.post("/query", json={"question": "What?"})
    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == ""
    assert body["sources"] == []
    assert body["session_id"] == "generated-session"


def test_ingest_success(
    monkeypatch: pytest.MonkeyPatch, client: TestClient, tmp_path: Path
) -> None:
    """Test the successful ingestion of documents.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        client (TestClient): The TestClient instance.
        tmp_path (Path): The temporary path fixture.
    """
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    called = types.SimpleNamespace(args=None)

    def fake_ingest(
        collection: str,
        path,
        hybrid: bool = True,
        progress_callback=None,
    ) -> None:
        """
        Fake implementation of the ingest_docs function for testing purposes.

        Args:
            collection (str): The name of the collection to ingest into.
            path (Path): The path to the data to ingest.
            hybrid (bool, optional): Whether to use hybrid ingestion. Defaults to True.
            progress_callback (callable, optional): A callback function for progress updates. Defaults to None.
        """
        called.args = (
            collection,
            path,
            hybrid,
            progress_callback,
        )

    monkeypatch.setattr(api_module, "_resolve_data_dir", lambda: data_dir)
    monkeypatch.setattr(api_module.ingest_module, "ingest_docs", fake_ingest)

    response = client.post(
        "/ingest",
        json={"collection": "docs", "hybrid": False},
    )
    assert response.status_code == 200
    body = response.json()
    assert body == {
        "ok": True,
        "collection": "docs",
        "data_dir": str(data_dir),
        "hybrid": False,
        "empty": False,
    }
    assert called.args[0:3] == ("docs", data_dir, False)


def test_sessions_endpoints(client: TestClient) -> None:
    """Test session management endpoints.

    Args:
        client (TestClient): The TestClient instance.
    """
    # List
    resp = client.get("/sessions/list")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["sessions"]) == 1
    assert data["sessions"][0]["id"] == "123"

    # History
    resp = client.get("/sessions/123/history")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["messages"]) == 1
    assert data["messages"][0]["content"] == "hi"

    # Delete
    resp = client.delete("/sessions/123")
    assert resp.status_code == 200
    assert resp.json()["ok"] is True


def test_ingest_missing_directory(
    monkeypatch: pytest.MonkeyPatch, client: TestClient, tmp_path: Path
) -> None:
    """Test the ingestion of documents when the data directory is missing.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        client (TestClient): The TestClient instance.
        tmp_path (Path): The temporary path fixture.
    """
    missing = tmp_path / "missing"
    monkeypatch.setattr(api_module, "_resolve_data_dir", lambda: missing)
    response = client.post("/ingest", json={"collection": "abc"})
    assert response.status_code == 500
    assert "Data directory does not exist" in response.json()["detail"]


def test_ingest_sync_generic_exception_propagates_as_500(
    monkeypatch: pytest.MonkeyPatch, client: TestClient, tmp_path: Path
) -> None:
    """Non-``EmptyIngestionError`` failures in sync ``/ingest`` surface as 500.

    Commit e1060fd added an explicit ``except EmptyIngestionError`` around
    the ``ingest_docs`` call that returns 200 with ``empty=true``. Generic
    runtime errors (Qdrant unreachable, disk full, OOM) must still
    propagate to FastAPI's default handler as an HTTP 500. This test
    guards against an over-broad exception catch accidentally swallowing
    real failures into the soft-empty response.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        client (TestClient): The TestClient instance.
        tmp_path (Path): The temporary path fixture, used as the data dir.
    """
    monkeypatch.setattr(api_module, "_resolve_data_dir", lambda: tmp_path)

    def exploding_ingest(
        collection: str,
        path: Path,
        hybrid: bool = True,
        progress_callback: Any = None,
    ) -> None:
        """Simulate a hard runtime failure during ingestion.

        Args:
            collection (str): Collection name (ignored).
            path (Path): Source directory path (ignored).
            hybrid (bool): Whether hybrid retrieval was requested (ignored).
            progress_callback (Any): Optional progress callback (ignored).

        Raises:
            RuntimeError: Always, to mimic an infrastructure failure.
        """
        _ = (collection, path, hybrid, progress_callback)
        raise RuntimeError("Qdrant unreachable")

    monkeypatch.setattr(api_module.ingest_module, "ingest_docs", exploding_ingest)

    # TestClient's default re-raises to the caller — in production, FastAPI's
    # default handler turns an uncaught ``RuntimeError`` into HTTP 500. The
    # assertion captures the same contract: the endpoint does NOT silently
    # swallow or coerce a generic runtime failure into the soft-empty 200
    # response that only ``EmptyIngestionError`` maps to.
    with pytest.raises(RuntimeError, match="Qdrant unreachable"):
        client.post("/ingest", json={"collection": "col", "hybrid": True})


def test_ingest_upload_empty_emits_warning_and_completes(
    monkeypatch: pytest.MonkeyPatch, client: TestClient, tmp_path: Path
) -> None:
    """Empty ingestion via /ingest/upload yields a warning + empty completion event.

    Verifies the API translates :class:`EmptyIngestionError` into an SSE
    ``warning`` event followed by an ``ingestion_complete`` event with
    ``"empty": true`` and skips ``rag.select_collection`` (which would
    otherwise raise ``ValueError`` because the collection was never
    created), instead of surfacing a generic ``"Ingestion failed"``.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        client (TestClient): The TestClient instance.
        tmp_path (Path): The temporary path fixture.
    """
    monkeypatch.setattr(api_module, "_resolve_qdrant_src_dir", lambda: tmp_path)

    def fake_ingest(
        collection: str,
        path: Path,
        hybrid: bool = True,
        progress_callback: Any = None,
    ) -> None:
        """Simulate an ingestion run that produced no documents.

        Args:
            collection (str): Collection name.
            path (Path): Source directory path (ignored).
            hybrid (bool): Whether hybrid retrieval was requested (ignored).
            progress_callback (Any): Optional progress callback (ignored).
        """
        _ = (path, hybrid, progress_callback)
        raise api_module.EmptyIngestionError(collection)

    monkeypatch.setattr(api_module.ingest_module, "ingest_docs", fake_ingest)

    response = client.post(
        "/ingest/upload",
        data={"collection": "silence-test", "hybrid": "false"},
        files={"files": ("silence.m4a", b"\x00" * 32, "audio/mp4")},
    )

    assert response.status_code == 200
    body = response.text

    # The SSE payload should contain a warning event referencing the collection
    # and an ingestion_complete event flagged as empty=true. It should NOT
    # contain a generic "Ingestion failed" error event.
    assert "event: warning" in body
    assert "silence-test" in body
    assert "event: ingestion_complete" in body
    assert '"empty": true' in body
    assert "Ingestion failed" not in body

    # select_collection must NOT have been called — DummyRAG.selected stays empty.
    assert api_module.rag.selected == []


def test_ingest_upload_success_does_not_warm_query_engine(
    monkeypatch: pytest.MonkeyPatch, client: TestClient, tmp_path: Path
) -> None:
    """Successful ``/ingest/upload`` must not eagerly warm the query engine.

    Regression guard for an OOM-kill observed on CPU Docker containers with
    the default 8 GB limit: the ingest SSE handler previously called
    ``rag.create_index()`` and ``rag.create_query_engine()`` on the
    module-level ``api.rag`` singleton immediately after a successful
    ingestion. That warmup triggered the reranker (bge-reranker-v2-m3,
    roughly 1 GB) and the embedding model (bge-m3, roughly 2 GB) to load
    on top of the PyTorch allocator memory still held by the just-finished
    ingest pipeline, blowing past the container memory cap and producing
    exit 137.

    The query engine is still built lazily on the next chat query, so the
    only user-visible effect of removing the warmup is a slower first-query
    TTFB. There is no correctness regression. This test pins down the
    behavioral defect (eager warmup) that reproduces the OOM, so that
    deleting the warmup block keeps the test green while any future
    reintroduction would fail it.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        client (TestClient): The TestClient instance.
        tmp_path (Path): The temporary path fixture, used as the uploads dir.
    """
    # Route uploads into a temp dir so the endpoint does not touch real state.
    monkeypatch.setattr(api_module, "_resolve_qdrant_src_dir", lambda: tmp_path)

    def fake_ingest(
        collection: str,
        path: Path,
        hybrid: bool = True,
        progress_callback: Any = None,
    ) -> None:
        """Simulate a successful, no-op ingestion run.

        Returning cleanly causes the SSE handler to enqueue a ``None``
        sentinel and break out of the progress loop, which is exactly the
        code path that previously reached the eager warmup block.

        Args:
            collection (str): Collection name (ignored).
            path (Path): Source directory path (ignored).
            hybrid (bool): Whether hybrid retrieval was requested (ignored).
            progress_callback (Any): Optional progress callback (ignored).
        """
        _ = (collection, path, hybrid, progress_callback)

    monkeypatch.setattr(api_module.ingest_module, "ingest_docs", fake_ingest)

    # Sanity: the autouse _patch_rag fixture installs a fresh DummyRAG whose
    # create_index / create_query_engine counters start at zero.
    dummy_rag = cast(DummyRAG, api_module.rag)
    assert dummy_rag.created_index == 0
    assert dummy_rag.created_query_engine == 0

    response = client.post(
        "/ingest/upload",
        data={"collection": "warmup-guard", "hybrid": "true"},
        files={"files": ("hello.txt", b"hello world", "text/plain")},
    )

    # The ingest itself must still succeed and emit the completion event —
    # we are not regressing success signalling, only removing the warmup.
    assert response.status_code == 200
    body = response.text
    assert "event: ingestion_complete" in body
    assert "Ingestion failed" not in body

    # Core assertion: neither the index nor the query engine may be built
    # eagerly during a successful ingest. Both counters must remain zero.
    # Under the buggy code path (now removed), the ingest SSE success path
    # called rag.create_index() and rag.create_query_engine() immediately
    # after the progress loop, loading reranker + embedding and OOM-killing
    # the backend on CPU Docker with default 8 GB limit.
    assert dummy_rag.created_query_engine == 0, (
        "rag.create_query_engine() must NOT be called from the ingest "
        "success path; it triggers reranker + embedding model loads that "
        "OOM-kill the backend on CPU Docker with default 8 GB limit."
    )
    assert dummy_rag.created_index == 0, (
        "rag.create_index() must NOT be called from the ingest success "
        "path either; the next chat query will build the index lazily."
    )


def test_ingest_upload_cancels_awaiter_on_client_disconnect(
    monkeypatch: pytest.MonkeyPatch, client: TestClient, tmp_path: Path
) -> None:
    """Client disconnect mid-ingest cancels the awaiter and exits cleanly.

    The SSE ``/ingest/upload`` endpoint now polls
    ``request.is_disconnected()`` while waiting on the worker queue so
    that a disconnected client doesn't leave an orphan coroutine
    blocked on a queue no one will read. On disconnect, the awaiter is
    cancelled, a warning is logged, and the generator returns without
    emitting ``ingestion_complete``. The worker thread itself runs to
    completion (Python has no safe thread kill), but its output is
    safely discarded.

    Implementation detail: we monkeypatch
    ``INGEST_DISCONNECT_POLL_INTERVAL_S`` to a tiny value so the test
    finishes in well under a second, and override ``is_disconnected``
    on the Starlette ``Request`` class to simulate an immediate
    disconnect. A brief ``time.sleep`` in ``fake_ingest`` guarantees
    the poll path is reached before the worker enqueues the success
    sentinel.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        client (TestClient): The TestClient instance.
        tmp_path (Path): The temporary path fixture, used as the uploads dir.
    """
    import time

    from starlette.requests import Request as StarletteRequest

    monkeypatch.setattr(api_module, "_resolve_qdrant_src_dir", lambda: tmp_path)
    monkeypatch.setattr(api_module, "INGEST_DISCONNECT_POLL_INTERVAL_S", 0.05)

    def blocking_fake_ingest(
        collection: str,
        path: Path,
        hybrid: bool = True,
        progress_callback: Any = None,
    ) -> None:
        """Block long enough for the disconnect poll to fire first.

        Args:
            collection (str): Collection name (ignored).
            path (Path): Source directory path (ignored).
            hybrid (bool): Whether hybrid retrieval was requested (ignored).
            progress_callback (Any): Optional progress callback (ignored).
        """
        _ = (collection, path, hybrid, progress_callback)
        time.sleep(0.3)

    monkeypatch.setattr(api_module.ingest_module, "ingest_docs", blocking_fake_ingest)

    async def always_disconnected(_self: StarletteRequest) -> bool:
        """Simulate an immediate client disconnect.

        Args:
            _self (StarletteRequest): The Request instance (ignored).

        Returns:
            bool: Always ``True``.
        """
        return True

    monkeypatch.setattr(StarletteRequest, "is_disconnected", always_disconnected)

    response = client.post(
        "/ingest/upload",
        data={"collection": "disconnect-guard", "hybrid": "true"},
        files={"files": ("hello.txt", b"hello world", "text/plain")},
    )

    # Connection opened successfully (SSE headers sent)...
    assert response.status_code == 200
    body = response.text
    # ... but the stream was cut short — no completion event, no error event.
    assert "event: ingestion_complete" not in body, (
        "ingestion_complete must NOT be emitted when the awaiter is cancelled; "
        "the worker thread still runs but its output is discarded."
    )
    assert "Ingestion failed" not in body, (
        "Cancellation must not be surfaced as a generic ingestion failure."
    )


def test_ingest_upload_poll_continues_when_still_connected(
    monkeypatch: pytest.MonkeyPatch, client: TestClient, tmp_path: Path
) -> None:
    """Poll timeouts without disconnect must not prematurely exit the stream.

    The ``asyncio.wait_for(queue.get())`` timeout branch has two outcomes:
    ``is_disconnected() -> True`` (cancel & return) and
    ``is_disconnected() -> False`` (continue polling).
    ``test_ingest_upload_cancels_awaiter_on_client_disconnect`` exercises
    the first branch. This test exercises the "still connected → continue"
    branch: the fake worker blocks longer than a single poll interval so
    the timeout fires at least once, but ``is_disconnected`` always
    returns ``False`` — the stream must keep draining until the worker's
    completion sentinel arrives, and ``ingestion_complete`` must still be
    emitted.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        client (TestClient): The TestClient instance.
        tmp_path (Path): The temporary path fixture, used as the uploads dir.
    """
    import time

    from starlette.requests import Request as StarletteRequest

    monkeypatch.setattr(api_module, "_resolve_qdrant_src_dir", lambda: tmp_path)
    monkeypatch.setattr(api_module, "INGEST_DISCONNECT_POLL_INTERVAL_S", 0.05)

    def slow_fake_ingest(
        collection: str,
        path: Path,
        hybrid: bool = True,
        progress_callback: Any = None,
    ) -> None:
        """Block long enough for the poll to fire at least once.

        Args:
            collection (str): Collection name (ignored).
            path (Path): Source directory path (ignored).
            hybrid (bool): Whether hybrid retrieval was requested (ignored).
            progress_callback (Any): Optional progress callback (ignored).
        """
        _ = (collection, path, hybrid, progress_callback)
        time.sleep(0.15)  # ~3× poll interval

    monkeypatch.setattr(api_module.ingest_module, "ingest_docs", slow_fake_ingest)

    poll_count: dict[str, int] = {"n": 0}

    async def always_connected(_self: StarletteRequest) -> bool:
        """Simulate a stable connection — client stays the full duration.

        Args:
            _self (StarletteRequest): The Request instance (ignored).

        Returns:
            bool: Always ``False``.
        """
        poll_count["n"] += 1
        return False

    monkeypatch.setattr(StarletteRequest, "is_disconnected", always_connected)

    response = client.post(
        "/ingest/upload",
        data={"collection": "connected-guard", "hybrid": "true"},
        files={"files": ("hello.txt", b"hello world", "text/plain")},
    )

    assert response.status_code == 200
    body = response.text
    assert "event: ingestion_complete" in body, (
        "With is_disconnected always False, the poll must not cancel the "
        "awaiter — the worker's completion sentinel must reach the client."
    )
    assert "Ingestion failed" not in body
    assert poll_count["n"] >= 1, (
        "Poll interval (0.05 s) is shorter than fake_ingest's sleep (0.15 s), "
        "so is_disconnected must have been consulted at least once."
    )


def test_ingest_sync_empty_returns_empty_flag(
    monkeypatch: pytest.MonkeyPatch, client: TestClient, tmp_path: Path
) -> None:
    """Empty ingestion via sync ``POST /ingest`` returns 200 with ``empty=true``.

    Matches the SSE ``/ingest/upload`` behaviour: ``EmptyIngestionError``
    is a soft-empty outcome (no content parsed), not a server error. The
    sync endpoint previously let the exception propagate to FastAPI's
    default handler, yielding an HTTP 500 which forced SDK/REST/CLI
    consumers to parse tracebacks to distinguish an empty upload from a
    real failure. The endpoint now catches the exception and returns
    ``{"ok": true, "empty": true, ...}``.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        client (TestClient): The TestClient instance.
        tmp_path (Path): The temporary path fixture, used as the data dir.
    """
    monkeypatch.setattr(api_module, "_resolve_data_dir", lambda: tmp_path)

    def fake_ingest(
        collection: str,
        path: Path,
        hybrid: bool = True,
        progress_callback: Any = None,
    ) -> None:
        """Simulate an ingestion run that produced no documents.

        Args:
            collection (str): Collection name.
            path (Path): Source directory path (ignored).
            hybrid (bool): Whether hybrid retrieval was requested (ignored).
            progress_callback (Any): Optional progress callback (ignored).
        """
        _ = (path, hybrid, progress_callback)
        raise api_module.EmptyIngestionError(collection)

    monkeypatch.setattr(api_module.ingest_module, "ingest_docs", fake_ingest)

    response = client.post(
        "/ingest",
        json={"collection": "silence-sync-test", "hybrid": True},
    )

    # Empty ingestion must NOT surface as a 500; the SSE path already
    # handles this gracefully, and the sync path now mirrors it.
    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is True
    assert body["empty"] is True
    assert body["collection"] == "silence-sync-test"
    assert body["data_dir"] == str(tmp_path)

    # Collection was never created, so no select_collection should have fired.
    assert api_module.rag.selected == []


def test_ingest_sync_success_does_not_warm_query_engine(
    monkeypatch: pytest.MonkeyPatch, client: TestClient, tmp_path: Path
) -> None:
    """Successful synchronous ``POST /ingest`` must not eagerly warm the query engine.

    Companion guard to :func:`test_ingest_upload_success_does_not_warm_query_engine`:
    the synchronous ``/ingest`` endpoint contained the same warmup pattern
    (``rag.select_collection`` + ``create_index`` + ``create_query_engine``
    + NER pre-warm) and had the same OOM-kill potential on CPU Docker
    containers. This test pins down the behavioral defect so that any
    future reintroduction of a post-ingest warmup on the sync route fails
    the suite.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        client (TestClient): The TestClient instance.
        tmp_path (Path): The temporary path fixture, used as the data dir.
    """
    # Point the endpoint at a real, empty temp directory so _resolve_data_dir
    # does not blow up the request before the warmup would have fired.
    monkeypatch.setattr(api_module, "_resolve_data_dir", lambda: tmp_path)

    def fake_ingest(
        collection: str,
        path: Path,
        hybrid: bool = True,
        progress_callback: Any = None,
    ) -> None:
        """Simulate a successful, no-op synchronous ingestion run.

        Returning cleanly is exactly the code path that previously fell
        through to the warmup block on the ``/ingest`` endpoint.

        Args:
            collection (str): Collection name (ignored).
            path (Path): Source directory path (ignored).
            hybrid (bool): Whether hybrid retrieval was requested (ignored).
            progress_callback (Any): Optional progress callback (ignored).
        """
        _ = (collection, path, hybrid, progress_callback)

    monkeypatch.setattr(api_module.ingest_module, "ingest_docs", fake_ingest)

    dummy_rag = cast(DummyRAG, api_module.rag)
    assert dummy_rag.created_index == 0
    assert dummy_rag.created_query_engine == 0

    response = client.post(
        "/ingest",
        json={"collection": "warmup-guard-sync", "hybrid": True},
    )

    # The sync endpoint must still return a successful payload.
    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is True
    assert body["collection"] == "warmup-guard-sync"

    # Core assertion: neither the index nor the query engine may be built
    # eagerly during a successful sync ingest. Both counters must remain zero.
    assert dummy_rag.created_query_engine == 0, (
        "rag.create_query_engine() must NOT be called from the sync "
        "/ingest success path; it triggers reranker + embedding model "
        "loads that OOM-kill the backend on CPU Docker."
    )
    assert dummy_rag.created_index == 0, (
        "rag.create_index() must NOT be called from the sync /ingest "
        "success path either; the next chat query will build the index "
        "lazily."
    )
    # select_collection must also not run eagerly — DummyRAG.selected stays empty.
    assert api_module.rag.selected == []


def test_query_forwards_retrieval_query_to_validation_payload(
    monkeypatch: pytest.MonkeyPatch, client: TestClient
) -> None:
    """The /query endpoint must forward retrieval_query to _validation_payload.

    The DummyRAG.chat stub returns ``retrieval_query`` in its response dict.
    This test monkeypatches ``_validation_payload`` in the api module and
    captures the kwargs it receives, then asserts the retrieval query is
    forwarded unchanged. ``retrieval_mode`` is the session-routing mode and
    is NOT forwarded as ``tool_used`` (those are semantically distinct).

    Args:
        monkeypatch (pytest.MonkeyPatch): Pytest monkeypatch fixture.
        client (TestClient): The TestClient instance.
    """
    captured: dict[str, Any] = {}

    def _stub_validation_payload(**kwargs: Any) -> dict[str, Any]:
        """Capture kwargs forwarded from the /query handler.

        Args:
            **kwargs: All keyword arguments forwarded to the real helper.

        Returns:
            dict[str, Any]: A minimal valid validation payload.
        """
        captured.update(kwargs)
        return {
            "validation_checked": None,
            "validation_mismatch": None,
            "validation_reason": None,
        }

    monkeypatch.setattr(api_module, "_validation_payload", _stub_validation_payload)

    response = client.post("/query", json={"question": "What?"})

    assert response.status_code == 200
    # DummyRAG.chat returns retrieval_query="rewritten::What?" and retrieval_mode="rewrite_compact_graph".
    assert captured.get("retrieval_query") == "rewritten::What?"
    # retrieval_mode is the session-routing mode and must not be passed as tool_used.
    assert captured.get("tool_used") is None
    assert captured.get("question") == "What?"


def test_stream_query_context_window_overflow_surfaces_descriptive_error(
    monkeypatch: pytest.MonkeyPatch, client: TestClient
) -> None:
    """Context-window overflow should surface a descriptive error instead of 'Internal server error'.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
        client: The TestClient instance.
    """
    original_run_query = type(api_module.rag).run_query

    def _exploding_run_query(self: Any, *a: Any, **kw: Any) -> Any:
        raise ValueError(
            "The query and retrieved context exceed the configured "
            "context window (4096 tokens). Increase OPENAI_CTX_WINDOW "
            "to match your model's actual context length or reduce the "
            "retrieval top-k."
        )

    monkeypatch.setattr(type(api_module.rag), "run_query", _exploding_run_query)

    try:
        with client.stream(
            "POST",
            "/stream_query",
            json={"question": "hello", "retrieval_mode": "stateless"},
        ) as resp:
            text = "".join(chunk.decode() for chunk in resp.iter_raw())
    finally:
        monkeypatch.setattr(type(api_module.rag), "run_query", original_run_query)

    assert "OPENAI_CTX_WINDOW" in text
    assert "Internal server error" not in text
