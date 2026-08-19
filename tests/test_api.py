"""Tests for the FastAPI application endpoints."""

import asyncio
import contextlib
import io
import json
import types
from collections.abc import Generator, Iterator
from pathlib import Path
from typing import Any, cast

import pytest
from conftest import run_ingest
from fastapi.testclient import TestClient

import docint.core.api as api_module
from docint.agents.types import IntentAnalysis, OrchestratorResult, PriorTurn, RetrievalResult
from docint.agents.understanding import ContextualUnderstandingAgent
from docint.core.entities.resolution import ResolutionSummary
from docint.core.ingest.ingestion_pipeline import NoSupportedFilesError


class DummySessionManager:
    """Dummy session manager for testing purposes."""

    scope_owned = True

    def __init__(self) -> None:
        """Start with no scope pinned, per instance."""
        self.scope: list[str] = []
        self.validation_updates: list[dict[str, Any]] = []

    def init_session_store_if_needed(self) -> None:
        """Satisfy the lifespan's eager store init without touching a DB."""
        return None

    def update_turn_validation(
        self,
        *,
        session_id: str,
        turn_idx: int,
        validation_checked: bool | None,
        validation_mismatch: bool | None,
        validation_reason: str | None,
        retried: bool | None = None,
        retry_query: str | None = None,
    ) -> None:
        """Record the validation back-write the streaming path performs.

        Args:
            session_id (str): Conversation the turn belongs to.
            turn_idx (int): Index of the turn being stamped.
            validation_checked (bool | None): Validation outcome flag.
            validation_mismatch (bool | None): Mismatch flag.
            validation_reason (str | None): Validator explanation.
            retried (bool | None): Whether a corrective retry produced the answer.
            retry_query (str | None): The reformulated query the retry used.
        """
        self.validation_updates.append(
            {
                "session_id": session_id,
                "turn_idx": turn_idx,
                "validation_checked": validation_checked,
                "validation_mismatch": validation_mismatch,
                "validation_reason": validation_reason,
                "retried": retried,
                "retry_query": retry_query,
            }
        )

    def set_scope(self, session_id: str, owner: str | None, chunk_ids: Any) -> bool:
        """Record a scope, honouring the owner gate the real manager applies.

        Args:
            session_id (str): The session to scope.
            owner (str | None): The requesting principal.
            chunk_ids (Any): Chunk ids to answer from.

        Returns:
            bool: Whether the scope was stored.
        """
        if not self.scope_owned:
            return False
        self.scope = [str(entry) for entry in chunk_ids]
        return True

    def get_scope(self, session_id: str, owner: str | None) -> list[str]:
        """Return the recorded scope.

        Args:
            session_id (str): The session to read.
            owner (str | None): The requesting principal.

        Returns:
            list[str]: The scoped chunk ids.
        """
        return list(self.scope)

    def clear_scope(self, session_id: str, owner: str | None) -> bool:
        """Clear the recorded scope.

        Args:
            session_id (str): The session to unscope.
            owner (str | None): The requesting principal.

        Returns:
            bool: Whether the session was found.
        """
        if not self.scope_owned:
            return False
        self.scope = []
        return True

    def list_sessions(self, owner: str | None = None) -> list[dict[str, Any]]:
        """List the caller's sessions.

        Args:
            owner (str | None): The owning principal.

        Returns:
            list[dict[str, Any]]: A list of session dictionaries.
        """
        return [{"id": "123", "created_at": "2023-01-01", "title": "Test Chat"}]

    def get_session_history(self, session_id: str, owner: str | None = None) -> list[dict[str, Any]]:
        """Get the message history for a session.

        Args:
            session_id (str): The ID of the session.
            owner (str | None): The owning principal.

        Returns:
            list[dict[str, Any]]: A list of message dictionaries.
        """
        return [{"role": "user", "content": "hi"}]

    def get_session_collection(self, session_id: str, owner: str | None = None) -> str | None:
        """Return the collection a session is pinned to (stub: unpinned).

        Args:
            session_id (str): The ID of the session.
            owner (str | None): The owning principal.

        Returns:
            str | None: ``None`` -- the stub never pins, so the endpoint's
                collection-mismatch pre-flight is a no-op here.
        """
        return None

    def delete_session(self, session_id: str, owner: str | None = None) -> bool:
        """Delete a session by ID.

        Args:
            session_id (str): The ID of the session.
            owner (str | None): The owning principal.

        Returns:
            bool: True if the session was successfully deleted, False otherwise.
        """
        return True

    def get_agent_context(self, session_id: str) -> Any:
        """Get the agent context for a session.

        Args:
            session_id (str): The ID of the session.

        Returns:
            Any: The agent context for the session.
        """

        class Ctx:
            """Dummy context object for testing purposes."""

            clarifications = 0

        return Ctx()


class _DummyOwners:
    """Passthrough collection-ownership manager for endpoint tests.

    Ownership is a no-op here (physical name == logical name, every name is
    "owned"), so these tests stay focused on endpoint behavior; real owner
    scoping is covered in ``test_api_collections_ownership.py``. ``list_for``
    delegates to ``list_collections`` so failure-injection on the RAG still
    surfaces through ``/collections/list``.
    """

    def __init__(self, rag: "DummyRAG") -> None:
        self._rag = rag

    def register(self, owner: str | None, logical: str) -> str:
        return logical

    def resolve(self, owner: str | None, logical: str) -> str | None:
        return logical

    def list_for(self, owner: str | None) -> list[str]:
        return self._rag.list_collections()

    def delete(self, owner: str | None, logical: str) -> str | None:
        return logical

    def backfill_legacy(self, physical_names: list[str], default_owner: str | None) -> None:
        return None


class DummyRAG:
    """Dummy Retrieval-Augmented Generation (RAG) class for testing purposes."""

    def measure_scope(self, chunk_ids: Any) -> dict[str, Any]:
        """Report a fixed budget measurement for the requested scope.

        Args:
            chunk_ids (Any): Candidate chunk ids.

        Returns:
            dict[str, Any]: Measurement mirroring ``RAG.measure_scope``.
        """
        self.measured_scopes.append([str(entry) for entry in chunk_ids])
        return {
            "chunks": len(list(chunk_ids)),
            "est_tokens": 10,
            "usable_tokens": 100,
            "missing": 0,
            "fits": self.scope_fits,
        }

    def get_chunk_text(self, chunk_id: str) -> str | None:
        """Return the stored text for a chunk, or None when it is gone.

        Args:
            chunk_id (str): Qdrant point id.

        Returns:
            str | None: The chunk text, or ``None`` when absent.
        """
        return "the whole chunk text" if chunk_id == "c1" else None

    def search_fulltext(self, query: str, **kwargs: Any) -> dict[str, Any]:
        """Record the call and return an empty result set.

        Args:
            query (str): The raw query text.
            **kwargs (Any): Paging and filter arguments from the endpoint.

        Returns:
            dict[str, Any]: An empty, well-formed search result.
        """
        self.search_calls.append({"query": query, **kwargs})
        return {
            "status": "ok",
            "hits": [],
            "total": 0,
            "next_cursor": None,
            "index_status": {
                "indexed": True,
                "total": 1,
                "with_search_text": 1,
                "missing": 0,
                "complete": True,
            },
        }

    def search_aggregate(self, query: str, **kwargs: Any) -> dict[str, Any]:
        """Record the call and return one synthetic group.

        Args:
            query (str): Raw keywords, possibly blank.
            **kwargs (Any): ``group_by``, filter and sizing arguments.

        Returns:
            dict[str, Any]: A well-formed grouped result.
        """
        self.last_aggregate = {"query": query, **kwargs}
        return {
            "status": "ok",
            "group_by": kwargs["group_by"],
            "total": 2,
            "unassigned": 0,
            "groups": [{"value": "acme_news", "count": 2, "samples": []}],
            "limit": kwargs.get("limit_groups", 100),
            "index_status": {"indexed": True, "total": 2, "with_search_text": 2, "missing": 0, "complete": True},
        }

    def probe_rerank_endpoint(self) -> None:
        """Satisfy the lifespan rerank probe without touching the network."""
        return None

    def probe_qdrant(self) -> bool:
        """Satisfy the lifespan startup probe without touching the network.

        Returns:
            bool: Always ``True`` — the stand-in is always "reachable".
        """
        return True

    def reconcile_quantization(self) -> int:
        """Satisfy the lifespan quantization reconcile without touching Qdrant.

        Returns:
            int: Always ``0`` — nothing to reconcile in the stand-in.
        """
        return 0

    def __init__(self) -> None:
        """Initialize the DummyRAG instance."""
        self.qdrant_collection = "alpha"
        self._owners = _DummyOwners(self)
        self.summarize_prompt = "Summarize collection"
        self.index = object()
        self.query_engine = object()
        self.selected: list[str] = []
        self.sessions = DummySessionManager()
        self.chats: list[str] = []
        self.search_calls: list[dict[str, Any]] = []
        self.scope_fits = True
        # Chunk ids each retrieval entry point was asked to answer from, so
        # tests can assert a request-carried scope reaches the engine.
        self.scoped_ids: list[list[str] | None] = []
        # Scopes handed to ``measure_scope``, so tests can assert an unchanged
        # selection is not re-measured on every turn.
        self.measured_scopes: list[list[str]] = []
        # Physical collection observed (via the active scope) at call time, so
        # tests can assert /query and /stream_query thread the resolved name in.
        self.seen_collections: list[str] = []
        self.stateless_queries: list[str] = []
        self.stateless_query_filters: list[dict[str, Any]] = []
        self.chat_filters: list[Any] = []
        self.stream_filters: list[Any] = []
        # One entry per stream_chat call, so the corrective-retry tests can
        # assert what the second pass was asked for.
        self.stream_calls: list[dict[str, Any]] = []
        self.created_index = 0  # Tracks the number of times an index is created
        self.created_query_engine = 0
        self.ner_sources: list[dict[str, Any]] = []
        self.ner_refresh_calls: list[bool] = []
        self.ner_stats_merge_modes: list[str] = []
        self.ner_search_merge_modes: list[str] = []
        self.ner_graph_merge_modes: list[str] = []
        self.ner_graph_top_ks: list[int] = []
        self.hate_speech_rows: list[dict[str, Any]] = []
        self.documents: list[dict[str, Any]] = []
        self.cached_summary_calls = 0
        self.build_tree_summary_calls = 0
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

    @contextlib.contextmanager
    def collection_scope(self, physical: str) -> Iterator[None]:
        """Mirror :meth:`RAG.collection_scope` for the stub.

        Sets the active collection for the duration of the block and restores
        the previous value on exit, so endpoint code that scopes a request sees
        the resolved physical name on this stub.

        Args:
            physical (str): The physical collection name to make active.

        Yields:
            None: Control returns with the scope active.
        """
        prev = self.qdrant_collection
        self.qdrant_collection = physical
        try:
            yield
        finally:
            self.qdrant_collection = prev

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

    def ensure_collection_owner_manager(self) -> "_DummyOwners":
        """Return the passthrough ownership manager stub.

        Returns:
            _DummyOwners: The no-op ownership manager.
        """
        return self._owners

    def start_session(self, session_id: str | None = None, owner: str | None = None) -> str:
        """Start a new session or resume an existing one.

        Args:
            session_id (str | None, optional): The ID of the session to resume. Defaults to None.
            owner (str | None, optional): The owning principal. Defaults to None.

        Returns:
            str: The session ID.
        """
        _ = owner
        return session_id or "generated-session"

    def chat(
        self,
        question: str,
        *,
        session_id: str | None = None,
        owner: str | None = None,
        metadata_filters: Any = None,
        metadata_filters_active: bool = False,
        metadata_filter_rules: Any = None,
        vector_store_kwargs: Any = None,
        scoped_node_ids: Any = None,
    ) -> dict[str, Any]:
        """Chat with the RAG system.

        Args:
            question (str): The question to ask the RAG system.
            session_id (str | None): The threaded conversation id (ignored by the stub).
            owner (str | None): The threaded owning principal (ignored by the stub).
            metadata_filters (Any): Optional compiled metadata filters.
            metadata_filters_active (bool): Whether request filters were active.
            metadata_filter_rules (Any): Optional raw request filter rules.
            vector_store_kwargs (Any): Optional native vector-store query kwargs.
            scoped_node_ids (Any): Hand-picked chunk ids to answer from; recorded by the stub.

        Returns:
            dict[str, Any]: The response from the RAG system.
        """
        _ = session_id, owner
        self.seen_collections.append(self.qdrant_collection)
        self.scoped_ids.append([str(entry) for entry in scoped_node_ids] if scoped_node_ids else None)
        self.chats.append(question)
        self.chat_filters.append(
            {
                "filters": metadata_filters,
                "active": metadata_filters_active,
                "rules": metadata_filter_rules,
                "vector_store_kwargs": vector_store_kwargs,
            }
        )
        if scoped_node_ids:
            # Mirror SessionManager: a scoped turn names itself, so endpoint
            # tests can assert the report survives the response model.
            return {
                "response": "answer",
                "sources": [{"id": 1}],
                "retrieval_query": f"rewritten::{question}",
                "coverage_unit": "documents",
                "retrieval_mode": "scoped",
                "scoped_chunk_count": len(list(scoped_node_ids)),
            }
        return {
            "response": "answer",
            "sources": [{"id": 1}],
            "retrieval_query": f"rewritten::{question}",
            "coverage_unit": "documents",
            "retrieval_mode": "rewrite_compact_graph",
            # Mirror SessionManager: the engine reports whether the sources
            # were re-ranked, so endpoint tests can assert it reaches the wire.
            "rerank": {"applied": False, "error": "upstream down"},
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
        session_id: str | None = None,
        owner: str | None = None,
        metadata_filters: Any = None,
        metadata_filters_active: bool = False,
        metadata_filter_rules: Any = None,
        vector_store_kwargs: Any = None,
        prior_turn: Any = None,
        skip_query_rewrite: Any = None,
        scoped_node_ids: Any = None,
        replace_turn_idx: int | None = None,
    ) -> Generator[str | dict[str, Any], None, None]:
        """Stream chat responses from the RAG system.

        Args:
            question (str): The question to ask the RAG system.
            session_id (str | None): The threaded conversation id (ignored by the stub).
            owner (str | None): The threaded owning principal (ignored by the stub).
            metadata_filters (Any): Optional compiled metadata filters.
            metadata_filters_active (bool): Whether request filters were active.
            metadata_filter_rules (Any): Optional raw request filter rules.
            vector_store_kwargs (Any): Optional native vector-store query kwargs.
            prior_turn (Any): Optional prior user/assistant exchange for context.
            skip_query_rewrite (Any): Accepted for parity with RAG.stream_chat; ignored by the stub.
            scoped_node_ids (Any): Hand-picked chunk ids to answer from; recorded by the stub.
            replace_turn_idx (int | None): Turn to overwrite; recorded by the stub.

        Yields:
            str | dict[str, Any]: Chunks of the chat response as they are generated.
        """
        _ = session_id, owner
        self.scoped_ids.append([str(entry) for entry in scoped_node_ids] if scoped_node_ids else None)
        self.stream_calls.append(
            {
                "question": question,
                "skip_query_rewrite": skip_query_rewrite,
                "replace_turn_idx": replace_turn_idx,
                "prior_turn": prior_turn,
            }
        )
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
            "turn_idx": 0,
            "retrieval_query": f"rewritten::{question}",
            "coverage_unit": "documents",
            "retrieval_mode": "rewrite_compact_graph",
            "rerank": {"applied": False, "error": "upstream down"},
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
        scoped_node_ids: Any = None,
    ) -> dict[str, Any]:
        """Run a stateless retrieval query.

        Args:
            prompt: Query prompt.
            metadata_filters: Optional compiled metadata filters.
            metadata_filter_rules: Optional raw request filter rules.
            vector_store_kwargs: Optional native vector-store query kwargs.
            scoped_node_ids: Hand-picked chunk ids to answer from; recorded by the stub.

        Returns:
            dict[str, Any]: Response payload.
        """
        _ = metadata_filters
        _ = metadata_filter_rules
        _ = vector_store_kwargs
        self.seen_collections.append(self.qdrant_collection)
        self.scoped_ids.append([str(entry) for entry in scoped_node_ids] if scoped_node_ids else None)
        self.stateless_queries.append(prompt)
        return {
            "response": "answer",
            "sources": [{"id": 1}],
        }

    async def run_query_async(
        self,
        prompt: str,
        *,
        metadata_filters: Any = None,
        metadata_filter_rules: Any = None,
        vector_store_kwargs: Any = None,
        retrieval_options: Any = None,
        scoped_node_ids: Any = None,
    ) -> dict[str, Any]:
        """Async stateless retrieval query mirroring :meth:`run_query`.

        Args:
            prompt: Query prompt.
            metadata_filters: Optional compiled metadata filters.
            metadata_filter_rules: Optional raw request filter rules.
            vector_store_kwargs: Optional native vector-store query kwargs.
            retrieval_options: Optional runtime retrieval overrides.
            scoped_node_ids: Hand-picked chunk ids to answer from.

        Returns:
            dict[str, Any]: Response payload.
        """
        _ = retrieval_options
        self.stateless_query_filters.append(
            {
                "filters": metadata_filters,
                "rules": metadata_filter_rules,
                "vector_store_kwargs": vector_store_kwargs,
            }
        )
        return self.run_query(
            prompt,
            metadata_filters=metadata_filters,
            metadata_filter_rules=metadata_filter_rules,
            vector_store_kwargs=vector_store_kwargs,
            scoped_node_ids=scoped_node_ids,
        )

    def expand_query_with_graph_with_debug(self, query: str) -> tuple[str, dict[str, Any]]:
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

    def cached_collection_summary(self) -> dict[str, Any] | None:
        """Return the canned cached summary payload, mirroring RAG.cached_collection_summary.

        Returns:
            dict[str, Any] | None: The stub's canned payload.
        """
        self.cached_summary_calls += 1
        return self.summary_payload

    def build_tree_summary(self, progress: Any = None) -> dict[str, Any]:
        """Return the canned summary payload, mirroring RAG.build_tree_summary.

        Args:
            progress (Any, optional): Progress callback (ignored).

        Returns:
            dict[str, Any]: The stub's canned payload.
        """
        self.build_tree_summary_calls += 1
        if progress is not None:
            progress(1, 1)
        return self.summary_payload

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

    def list_documents(self) -> list[dict[str, Any]]:
        """Return the canned list of documents for the active collection."""
        return self.documents

    def iter_documents(
        self,
        *,
        cursor: str | None = None,
        limit: int = 50,
    ) -> tuple[list[dict[str, Any]], str | None]:
        """Stub paginated document iterator that slices a fixed list."""
        from docint.utils.cursor import decode_cursor, encode_cursor

        offset = int(decode_cursor(cursor).get("o") or 0)
        rows = getattr(self, "documents", [])
        end = offset + max(1, int(limit))
        page = rows[offset:end]
        next_cursor = encode_cursor(end) if end < len(rows) else None
        return page, next_cursor

    def iter_hate_speech(
        self,
        *,
        cursor: str | None = None,
        limit: int = 50,
        category: str | None = None,
        min_confidence: str | None = None,
    ) -> tuple[list[dict[str, Any]], str | None]:
        """Stub paginated hate-speech iterator that slices ``hate_speech_rows``."""
        from docint.utils.cursor import decode_cursor, encode_cursor

        offset = int(decode_cursor(cursor).get("o") or 0)
        rows = self.hate_speech_rows
        end = offset + max(1, int(limit))
        page = rows[offset:end]
        next_cursor = encode_cursor(end) if end < len(rows) else None
        return page, next_cursor

    def iter_collection_ner_sources(
        self,
        *,
        cursor: str | None = None,
        limit: int = 50,
        entity_key: str | None = None,
        entity_text: str | None = None,
        entity_type: str | None = None,
        entity_merge_mode: str = "orthographic",
    ) -> tuple[list[dict[str, Any]], str | None]:
        """Stub paginated NER source iterator that slices ``ner_sources``."""
        from docint.utils.cursor import decode_cursor, encode_cursor

        self.last_ner_sources_filter = {
            "entity_key": entity_key,
            "entity_text": entity_text,
            "entity_type": entity_type,
            "entity_merge_mode": entity_merge_mode,
        }
        offset = int(decode_cursor(cursor).get("o") or 0)
        rows = self.ner_sources
        end = offset + max(1, int(limit))
        page = rows[offset:end]
        next_cursor = encode_cursor(end) if end < len(rows) else None
        return page, next_cursor

    def _get_collection_ner_aggregate(self, **_: Any) -> dict[str, Any]:
        """Stub aggregate warm-up that returns an empty payload."""
        self.warm_calls = getattr(self, "warm_calls", 0) + 1
        return {"entities": [], "relations": []}

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
            top_k (int, optional): Number of top entities to return. Defaults to 15.
            min_mentions (int, optional): Minimum mention count for inclusion. Defaults to 2.
            entity_type (str | None, optional): Filter entities by type. Defaults to None.
            include_relations (bool, optional): Whether to include relation statistics.
                Defaults to True.
            entity_merge_mode (str): Entity clustering mode (recorded for assertions).

        Returns:
            dict[str, Any]: NER stats payload with totals, top entities, entity types, top
                relations, and document-level stats.
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
            "top_relations": [{"head": "Acme", "label": "owns", "tail": "Widget", "mentions": 2}],
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
            entity_merge_mode (str): Entity clustering mode (recorded for assertions).

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

    def get_collection_ner_graph(
        self,
        *,
        top_k_nodes: int = 80,
        min_edge_weight: int = 1,
        entity_merge_mode: str = "orthographic",
    ) -> dict[str, Any]:
        """Return a canned entity graph payload.

        Args:
            top_k_nodes (int): Node cap (recorded indirectly via the payload meta).
            min_edge_weight (int): Edge weight threshold (ignored by the stub).
            entity_merge_mode (str): Entity clustering mode (recorded for assertions).

        Returns:
            dict[str, Any]: Graph payload with nodes, edges, and meta counts.
        """
        _ = min_edge_weight
        self.ner_graph_top_ks.append(top_k_nodes)
        self.ner_graph_merge_modes.append(entity_merge_mode)
        return {
            "nodes": [
                {"id": "acme::org", "text": "Acme", "type": "ORG", "mentions": 3},
                {"id": "widget::product", "text": "Widget", "type": "PRODUCT", "mentions": 2},
            ],
            "edges": [
                {
                    "source": "acme::org",
                    "target": "widget::product",
                    "label": "owns",
                    "kind": "relation",
                    "weight": 2,
                }
            ],
            "meta": {"node_count": 2, "edge_count": 1},
        }

    def resolve_entities(self, *, progress_callback: Any = None) -> ResolutionSummary:
        """Record a resolution call and return a fixed summary.

        Args:
            progress_callback (Any): Optional progress sink (ignored).

        Returns:
            ResolutionSummary: Fixed counts for endpoint assertions.
        """
        _ = progress_callback
        self.resolve_called = True
        return ResolutionSummary(processed=4, minted=2, attached=1, skipped=1, entities_touched=2)


@pytest.fixture(autouse=True)
def _patch_rag(monkeypatch: pytest.MonkeyPatch) -> Any | None:
    """Patch the RAG instance for testing.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.

    Returns:
        Any | None: Yields None after patching.
    """
    monkeypatch.delenv("DOCINT_AUTH_HEADER", raising=False)
    monkeypatch.setenv("DOCINT_DEFAULT_IDENTITY", "test-operator")
    dummy = DummyRAG()
    monkeypatch.setattr(api_module, "rag", dummy)
    yield


@pytest.fixture
def client() -> Generator[TestClient, None, None]:
    """Create a TestClient for testing the FastAPI application.

    Entered as a context manager so a single portal (and its background
    event-loop thread) stays alive for the whole test: ingest jobs
    (``docint/core/jobs.py``) run as a detached ``asyncio`` task meant to
    outlive the request that queued them, and a bare, non-context-managed
    ``TestClient`` opens a brand-new throwaway event loop per call — orphaning
    that task the instant the queuing request returns.

    Yields:
        TestClient: The TestClient instance.
    """
    with TestClient(api_module.app) as client:
        yield client


def test_collections_list_success(client: TestClient) -> None:
    """Test the successful retrieval of the collections list.

    Args:
        client (TestClient): The TestClient instance.
    """
    response = client.get("/collections/list")
    assert response.status_code == 200
    assert response.json() == ["alpha", "beta"]


def test_collections_list_failure(monkeypatch: pytest.MonkeyPatch, client: TestClient) -> None:
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
    assert response.json()["detail"] == "Request failed."


def test_collections_select_is_nonmutating(monkeypatch: pytest.MonkeyPatch, client: TestClient) -> None:
    """Selecting a collection is a non-mutating ownership check (WS2).

    ``/collections/select`` no longer changes any server-side state: it neither
    switches the active collection nor warms the index/query engine. Selection
    is purely client-side; the server only confirms ownership (200 with the
    name) so the request path stays stateless and concurrency-safe. This also
    subsumes the prior OOM guard (no eager bge-m3 / reranker / GLiNER load).

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        client (TestClient): The TestClient instance.
    """
    rag = cast(Any, api_module.rag)
    before = rag.qdrant_collection
    response = client.post("/collections/select", json={"name": " gamma "})
    assert response.status_code == 200
    assert response.json() == {"ok": True, "name": "gamma"}
    # No server-side state may change: active collection, selection log, and the
    # index/engine/NER warmup counters must all be untouched.
    assert rag.qdrant_collection == before
    assert rag.selected == []
    assert rag.created_index == 0
    assert rag.created_query_engine == 0
    assert rag.ner_refresh_calls == []


def test_collections_select_blank_name(client: TestClient) -> None:
    """Blank collection names must surface as a structured HTTP 400.

    Regression guard: a prior handler caught ``HTTPException`` and
    re-raised it as 500, collapsing the original 400 into a generic
    server error. The handler must now propagate the original status.

    Args:
        client (TestClient): The TestClient instance.
    """
    response = client.post("/collections/select", json={"name": "   "})
    assert response.status_code == 400
    assert "Collection name required" in response.json()["detail"]


def test_collections_select_returns_404_when_not_owned(monkeypatch: pytest.MonkeyPatch, client: TestClient) -> None:
    """A collection the caller does not own resolves to HTTP 404.

    Selection is now an ownership check: when the owner manager cannot resolve
    the logical name for this principal (unowned or nonexistent), the endpoint
    returns 404 without leaking whether the name exists. This replaces the old
    ``ValueError``-from-``select_collection`` path (select no longer mutates).

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        client (TestClient): The TestClient instance.
    """
    rag = cast(Any, api_module.rag)
    monkeypatch.setattr(rag._owners, "resolve", lambda owner, logical: None)
    response = client.post("/collections/select", json={"name": "ghost"})
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_collections_ner_success(client: TestClient) -> None:
    """Test the successful retrieval of information extraction data.

    Args:
        client (TestClient): The TestClient instance.
    """
    dummy_rag = cast(DummyRAG, api_module.rag)
    dummy_rag.ner_sources = [{"filename": "doc1.pdf", "page": 1, "row": 2, "entities": [], "relations": []}]
    response = client.get("/collections/ner")
    assert response.status_code == 200
    assert response.json() == {"sources": dummy_rag.ner_sources}
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
    response = client.get("/collections/ner/stats", params={"entity_merge_mode": "exact"})
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


def test_collections_ner_stats_support_resolved_merge_mode(client: TestClient) -> None:
    """Stats endpoint should accept and forward the resolved merge mode."""
    response = client.get("/collections/ner/stats", params={"entity_merge_mode": "resolved"})
    assert response.status_code == 200
    assert cast(DummyRAG, api_module.rag).ner_stats_merge_modes[-1] == "resolved"


def test_collections_ner_graph_success(client: TestClient) -> None:
    """Graph endpoint should return nodes, edges, and meta counts.

    Args:
        client (TestClient): The TestClient instance.
    """
    response = client.get("/collections/ner/graph")
    assert response.status_code == 200
    payload = response.json()
    assert payload["nodes"][0]["text"] == "Acme"
    assert payload["edges"][0]["kind"] == "relation"
    assert payload["meta"]["node_count"] == 2
    assert cast(DummyRAG, api_module.rag).ner_graph_merge_modes[-1] == "orthographic"


def test_collections_ner_graph_forwards_merge_mode(client: TestClient) -> None:
    """Graph endpoint should forward explicit merge-mode overrides."""
    response = client.get("/collections/ner/graph", params={"entity_merge_mode": "resolved"})
    assert response.status_code == 200
    assert cast(DummyRAG, api_module.rag).ner_graph_merge_modes[-1] == "resolved"


def test_collections_ner_graph_requires_selection(client: TestClient) -> None:
    """Graph endpoint should 400 when no collection is selected."""
    api_module.rag.qdrant_collection = ""
    response = client.get("/collections/ner/graph")
    assert response.status_code == 400


def test_collections_ner_graph_failure(monkeypatch: pytest.MonkeyPatch, client: TestClient) -> None:
    """Graph endpoint should 500 when the RAG layer raises.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        client (TestClient): The TestClient instance.
    """

    def raiser(**_: Any) -> dict[str, Any]:
        raise RuntimeError("boom")

    monkeypatch.setattr(api_module.rag, "get_collection_ner_graph", raiser)
    response = client.get("/collections/ner/graph")
    assert response.status_code == 500


def test_resolve_entities_success(client: TestClient) -> None:
    """The resolve endpoint returns the resolution summary counts."""
    response = client.post("/collections/entities/resolve")
    assert response.status_code == 200
    payload = response.json()
    assert payload == {
        "processed": 4,
        "minted": 2,
        "attached": 1,
        "skipped": 1,
        "entities_touched": 2,
    }


def test_resolve_entities_requires_selected_collection(client: TestClient) -> None:
    """The resolve endpoint 400s when no collection is selected."""
    api_module.rag.qdrant_collection = ""
    response = client.post("/collections/entities/resolve")
    assert response.status_code == 400


def test_agent_chat_answers(monkeypatch: pytest.MonkeyPatch, client: TestClient) -> None:
    """Agent chat should return an answer when confidence is sufficient.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        client (TestClient): The TestClient instance.
    """

    def fake_chat(question: str, **_: Any) -> dict[str, Any]:
        """Fake implementation of the RAG chat method for testing purposes.

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


def test_agent_chat_clarifies(monkeypatch: pytest.MonkeyPatch, client: TestClient) -> None:
    """Agent chat should request clarification when policy requires it.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        client (TestClient): The TestClient instance.
    """
    monkeypatch.setattr(
        api_module,
        "_clarification_policy",
        api_module.ClarificationPolicy(api_module.ClarificationConfig(confidence_threshold=1.0, require_entities=True)),
    )

    payload = {"message": "hello"}
    response = client.post("/agent/chat", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "clarification"
    assert data["message"]
    assert data["intent"] is not None
    assert data["confidence"] is not None


def test_agent_chat_falls_back_to_clarification_on_weak_validation_mismatch(
    monkeypatch: pytest.MonkeyPatch, client: TestClient
) -> None:
    """A weak (refusal-shaped) answer with validation_mismatch must surface as clarification.

    Exercises the orchestrator's post-responder fallback end-to-end: the
    monkeypatched orchestrator returns a ``RetrievalResult`` shaped exactly
    like the production failure (answer="Evidence insufficient.",
    validation_mismatch=True), and the API must respond with
    ``status="clarification"`` and a helpful nudge instead of echoing the
    refusal back to the user.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        client (TestClient): The TestClient instance.
    """
    # Use a real orchestrator (post-responder fallback lives there).
    monkeypatch.setattr(
        api_module.rag,
        "chat",
        lambda *_, **__: {
            "response": "Evidence insufficient.",
            "sources": [],
        },
    )

    # Force the response validator to flag mismatch by stubbing it.
    from docint.agents.generation import ResultValidationResponseAgent
    from docint.agents.types import RetrievalResult as _RR

    def _flag_mismatch(self: Any, result: _RR, turn: Any) -> _RR:
        """Mark every result as mismatched for this test."""
        _ = self, turn
        result.validation_checked = True
        result.validation_mismatch = True
        result.validation_reason = "no UN content in sources"
        return result

    monkeypatch.setattr(ResultValidationResponseAgent, "finalize", _flag_mismatch)

    response = client.post("/agent/chat", json={"message": "Please elaborate."})

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "clarification"
    assert data["message"]
    assert "previous answer" in data["message"].lower() or "elaborate" in data["message"].lower()


def test_agent_chat_returns_validation_alert(monkeypatch: pytest.MonkeyPatch, client: TestClient) -> None:
    """Agent chat should surface response-validation metadata.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        client (TestClient): The TestClient instance.
    """

    class _StubOrchestrator:
        """Stub orchestrator that returns a canned retrieval result with validation metadata for testing purposes."""

        def handle_turn(self, turn: Any, context: Any = None) -> OrchestratorResult:
            """Handle a turn by returning a canned retrieval result with validation metadata.

            Args:
                turn (_type_): The user turn to process.
                context (_type_, optional): The context for the turn. Defaults to None.

            Returns:
                OrchestratorResult: The result of processing the turn.
            """
            _ = turn, context
            analysis = IntentAnalysis(intent="qa", confidence=0.9, entities={"query": "hello"})
            retrieval = RetrievalResult(
                answer="answer",
                sources=[{"id": 1}],
                session_id="generated-session",
                validation_checked=True,
                validation_mismatch=True,
                validation_reason="mismatch",
            )
            return OrchestratorResult(clarification=None, retrieval=retrieval, analysis=analysis)

    monkeypatch.setattr(api_module, "_build_orchestrator", lambda: _StubOrchestrator())

    response = client.post("/agent/chat", json={"message": "hello"})

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "answer"
    assert data["validation_checked"] is True
    assert data["validation_mismatch"] is True
    assert data["validation_reason"] == "mismatch"


def test_agent_chat_surfaces_corrective_retry_provenance(monkeypatch: pytest.MonkeyPatch, client: TestClient) -> None:
    """An answer produced by a corrective retry must say so in the response.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        client (TestClient): The TestClient instance.
    """

    class _RetriedOrchestrator:
        """Stub orchestrator returning a result the corrective retry produced."""

        def handle_turn(self, turn: Any, context: Any = None) -> OrchestratorResult:
            """Return a canned retried retrieval result.

            Args:
                turn (Any): The user turn to process.
                context (Any): The turn context. Defaults to None.

            Returns:
                OrchestratorResult: The result of processing the turn.
            """
            _ = turn, context
            analysis = IntentAnalysis(intent="qa", confidence=0.9, entities={"query": "hello"})
            retrieval = RetrievalResult(
                answer="A properly grounded answer.",
                sources=[{"id": 1}],
                session_id="generated-session",
                validation_checked=True,
                validation_mismatch=False,
                retried=True,
                retry_query="Security Council resolutions",
                turn_idx=0,
            )
            return OrchestratorResult(clarification=None, retrieval=retrieval, analysis=analysis)

    monkeypatch.setattr(api_module, "_build_orchestrator", lambda: _RetriedOrchestrator())

    data = client.post("/agent/chat", json={"message": "hello"}).json()

    assert data["retried"] is True
    assert data["retry_query"] == "Security Council resolutions"
    # The persisted-turn join key is internal and must never reach a client.
    assert "turn_idx" not in data


def test_agent_chat_stream_clarifies(monkeypatch: pytest.MonkeyPatch, client: TestClient) -> None:
    """Streaming endpoint should emit clarification event when policy demands it.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        client (TestClient): The TestClient instance.
    """
    monkeypatch.setattr(
        api_module,
        "_clarification_policy",
        api_module.ClarificationPolicy(api_module.ClarificationConfig(confidence_threshold=1.0, require_entities=True)),
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


def _sse_frames(text: str) -> list[dict[str, Any]]:
    """Parse an SSE body into its decoded JSON frames.

    Args:
        text (str): The raw SSE response body.

    Returns:
        list[dict[str, Any]]: One dict per ``data:`` frame, in order.
    """
    frames: list[dict[str, Any]] = []
    for line in text.splitlines():
        if line.startswith("data: "):
            frames.append(json.loads(line[6:]))
    return frames


def _arm_corrective_retry(
    monkeypatch: pytest.MonkeyPatch,
    *,
    answers: list[str],
    reformulation: str | None = "reformulated query",
    mismatch: bool = True,
    retrieval_mode: str = "rewrite_compact_graph",
) -> list[dict[str, Any]]:
    """Wire the streaming path so the corrective retry can be exercised.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        answers (list[str]): Answer text per ``stream_chat`` call; the last repeats.
        reformulation (str | None): What the stubbed reformulator returns.
        mismatch (bool): The validation verdict to force.
        retrieval_mode (str): The retrieval mode the stream reports.

    Returns:
        list[dict[str, Any]]: Recorded ``stream_chat`` calls, appended in order.
    """
    calls: list[dict[str, Any]] = []

    def _fake_stream_chat(question: str, **kwargs: Any) -> Generator[str | dict[str, Any], None, None]:
        """Stream a canned answer and record how it was called.

        Args:
            question (str): The query being answered.
            **kwargs (Any): Remaining ``stream_chat`` keyword arguments.

        Yields:
            str | dict[str, Any]: Answer tokens, then the final metadata dict.
        """
        calls.append({"question": question, **kwargs})
        answer = answers[min(len(calls) - 1, len(answers) - 1)]
        yield answer
        yield {
            "response": answer,
            "sources": [{"id": 1}],
            "session_id": "generated-session",
            "turn_idx": 0,
            "retrieval_query": question,
            "retrieval_mode": retrieval_mode,
        }

    def _fake_validation(**kwargs: Any) -> dict[str, Any]:
        """Return a forced validation verdict.

        Args:
            **kwargs (Any): Validation inputs (ignored).

        Returns:
            dict[str, Any]: The forced verdict.
        """
        _ = kwargs
        return {
            "validation_checked": True,
            "validation_mismatch": mismatch,
            "validation_reason": "no UN content in sources",
        }

    def _fake_reformulate(question: str, failed_query: str | None, reason: str | None) -> str | None:
        """Return the canned reformulation.

        Args:
            question (str): The user's original question.
            failed_query (str | None): The query that failed.
            reason (str | None): The validator's reason.

        Returns:
            str | None: The canned reformulation.
        """
        _ = question, failed_query, reason
        return reformulation

    monkeypatch.setattr(api_module.rag, "stream_chat", _fake_stream_chat)
    monkeypatch.setattr(api_module, "_validation_payload", _fake_validation)
    monkeypatch.setattr(api_module, "_reformulated_query", _fake_reformulate)
    return calls


def _stream_text(client: TestClient, question: str = "What did the UN say?") -> str:
    """Run a streaming query and return the raw SSE body.

    Args:
        client (TestClient): The TestClient instance.
        question (str): The question to ask.

    Returns:
        str: The raw SSE response body.
    """
    with client.stream("POST", "/stream_query", json={"question": question}) as resp:
        assert resp.status_code == 200
        return "".join([chunk.decode() for chunk in resp.iter_raw()])


def test_stream_query_retries_a_weak_mismatched_answer(monkeypatch: pytest.MonkeyPatch, client: TestClient) -> None:
    """A rejected weak answer is re-answered once, visibly, on the same stream.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        client (TestClient): The TestClient instance.
    """
    calls = _arm_corrective_retry(
        monkeypatch,
        answers=[
            "Evidence insufficient.",
            "The Security Council adopted three resolutions on the matter in 2019.",
        ],
    )

    frames = _sse_frames(_stream_text(client))

    assert {"retry": {"query": "reformulated query"}} in frames
    assert {"token": "The Security Council adopted three resolutions on the matter in 2019."} in frames
    final = frames[-1]
    assert final["retried"] is True
    assert final["retry_query"] == "reformulated query"
    assert final["response"] == "The Security Council adopted three resolutions on the matter in 2019."
    assert len(calls) == 2


def test_stream_query_retry_replaces_the_first_turn(monkeypatch: pytest.MonkeyPatch, client: TestClient) -> None:
    """The retry overwrites the persisted turn and stamps it once.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        client (TestClient): The TestClient instance.
    """
    calls = _arm_corrective_retry(
        monkeypatch,
        answers=["Evidence insufficient.", "A properly grounded answer about the Security Council."],
    )
    sessions = cast("Any", api_module.rag.sessions)
    sessions.validation_updates.clear()

    _stream_text(client)

    assert calls[0]["replace_turn_idx"] is None
    assert calls[1]["replace_turn_idx"] == 0
    # The reformulation IS the retrieval query; rewriting it again would undo
    # the correction.
    assert calls[1]["skip_query_rewrite"] is True
    assert calls[1]["question"] == "reformulated query"
    updates = sessions.validation_updates
    assert len(updates) == 1
    assert updates[0]["retried"] is True
    assert updates[0]["retry_query"] == "reformulated query"
    assert updates[0]["turn_idx"] == 0


def test_stream_query_retry_reuses_the_first_passes_prior_turn(
    monkeypatch: pytest.MonkeyPatch, client: TestClient
) -> None:
    """The retry must not bind the answer it is replacing as its own context.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        client (TestClient): The TestClient instance.
    """
    calls = _arm_corrective_retry(
        monkeypatch,
        answers=["Evidence insufficient.", "A properly grounded answer about the Security Council."],
    )

    _stream_text(client)

    assert calls[0]["prior_turn"] == calls[1]["prior_turn"]


def test_stream_query_does_not_retry_a_strong_mismatched_answer(
    monkeypatch: pytest.MonkeyPatch, client: TestClient
) -> None:
    """A substantive answer is delivered even when the validator flags it.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        client (TestClient): The TestClient instance.
    """
    calls = _arm_corrective_retry(
        monkeypatch,
        answers=["A long, substantive answer that clears the weak-answer threshold entirely."],
    )

    frames = _sse_frames(_stream_text(client))

    assert len(calls) == 1
    assert not any("retry" in frame for frame in frames)
    assert "retried" not in frames[-1]


def test_stream_query_does_not_retry_a_scoped_turn(monkeypatch: pytest.MonkeyPatch, client: TestClient) -> None:
    """A hand-picked scope runs no retrieval for a new query to change.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        client (TestClient): The TestClient instance.
    """
    calls = _arm_corrective_retry(
        monkeypatch,
        answers=["Evidence insufficient."],
        retrieval_mode="scoped",
    )

    frames = _sse_frames(_stream_text(client))

    assert len(calls) == 1
    assert not any("retry" in frame for frame in frames)


def test_stream_query_skips_retry_when_disabled(monkeypatch: pytest.MonkeyPatch, client: TestClient) -> None:
    """``CORRECTIVE_RETRY_ENABLED=false`` restores the single-pass behaviour.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        client (TestClient): The TestClient instance.
    """
    monkeypatch.setenv("CORRECTIVE_RETRY_ENABLED", "false")
    calls = _arm_corrective_retry(monkeypatch, answers=["Evidence insufficient."])

    frames = _sse_frames(_stream_text(client))

    assert len(calls) == 1
    assert not any("retry" in frame for frame in frames)


def test_stream_query_skips_retry_when_reformulation_declines(
    monkeypatch: pytest.MonkeyPatch, client: TestClient
) -> None:
    """No usable reformulation means no second pass and no retry frame.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        client (TestClient): The TestClient instance.
    """
    calls = _arm_corrective_retry(monkeypatch, answers=["Evidence insufficient."], reformulation=None)

    frames = _sse_frames(_stream_text(client))

    assert len(calls) == 1
    assert not any("retry" in frame for frame in frames)


def test_stream_query_keeps_the_first_answer_when_the_retry_fails(
    monkeypatch: pytest.MonkeyPatch, client: TestClient
) -> None:
    """A failed retry degrades to the delivered answer, never to an error.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        client (TestClient): The TestClient instance.
    """
    calls: list[str] = []

    def _flaky_stream_chat(question: str, **kwargs: Any) -> Generator[str | dict[str, Any], None, None]:
        """Answer weakly the first time, then fail.

        Args:
            question (str): The query being answered.
            **kwargs (Any): Remaining ``stream_chat`` keyword arguments.

        Yields:
            str | dict[str, Any]: Tokens then metadata, on the first call only.

        Raises:
            RuntimeError: On every call after the first.
        """
        _ = kwargs
        calls.append(question)
        if len(calls) > 1:
            raise RuntimeError("retrieval died mid-retry")
        yield "Evidence insufficient."
        yield {
            "response": "Evidence insufficient.",
            "sources": [{"id": 1}],
            "session_id": "generated-session",
            "turn_idx": 0,
            "retrieval_query": question,
            "retrieval_mode": "rewrite_compact_graph",
        }

    _arm_corrective_retry(monkeypatch, answers=["unused"])
    monkeypatch.setattr(api_module.rag, "stream_chat", _flaky_stream_chat)

    frames = _sse_frames(_stream_text(client))

    assert len(calls) == 2
    final = frames[-1]
    assert "error" not in final
    assert final["response"] == "Evidence insufficient."
    assert "retried" not in final


def test_query_stamps_default_identity_on_session_start(monkeypatch: pytest.MonkeyPatch, client: TestClient) -> None:
    """Session-backed query requests must stamp the resolved principal.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        client (TestClient): The TestClient instance.
    """
    monkeypatch.setenv("DOCINT_DEFAULT_IDENTITY", "operator")
    seen: dict[str, Any] = {}

    def record_start_session(session_id: str | None = None, owner: str | None = None) -> str:
        seen["session_id"] = session_id
        seen["owner"] = owner
        return session_id or "generated-session"

    monkeypatch.setattr(api_module.rag, "start_session", record_start_session)

    response = client.post("/query", json={"question": "hello"})

    assert response.status_code == 200
    assert seen == {"session_id": None, "owner": "operator"}


def test_stream_query_stamps_default_identity_on_session_start(
    monkeypatch: pytest.MonkeyPatch, client: TestClient
) -> None:
    """Session-backed stream queries must stamp the resolved principal.

    The frontend uses ``/stream_query`` for chat. If the write path starts
    sessions without the same owner that ``/sessions/list`` later filters by,
    chats persist but never appear in the sidebar.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        client (TestClient): The TestClient instance.
    """
    monkeypatch.setenv("DOCINT_DEFAULT_IDENTITY", "operator")
    seen: dict[str, Any] = {}

    def record_start_session(session_id: str | None = None, owner: str | None = None) -> str:
        seen["session_id"] = session_id
        seen["owner"] = owner
        return session_id or "generated-session"

    monkeypatch.setattr(api_module.rag, "start_session", record_start_session)

    with client.stream("POST", "/stream_query", json={"question": "hello"}) as resp:
        assert resp.status_code == 200
        list(resp.iter_lines())

    assert seen == {"session_id": None, "owner": "operator"}


def test_agent_chat_stamps_default_identity_on_session_start(
    monkeypatch: pytest.MonkeyPatch, client: TestClient
) -> None:
    """Agent chat must stamp the resolved principal on session start.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        client (TestClient): The TestClient instance.
    """
    monkeypatch.setenv("DOCINT_DEFAULT_IDENTITY", "operator")
    seen: dict[str, Any] = {}

    class _StubOrchestrator:
        def handle_turn(self, turn: Any, context: Any = None) -> OrchestratorResult:
            _ = turn, context
            analysis = IntentAnalysis(intent="qa", confidence=0.9, entities={"query": "hello"})
            retrieval = RetrievalResult(answer="answer", sources=[{"id": 1}], session_id="generated-session")
            return OrchestratorResult(clarification=None, retrieval=retrieval, analysis=analysis)

    def record_start_session(session_id: str | None = None, owner: str | None = None) -> str:
        seen["session_id"] = session_id
        seen["owner"] = owner
        return session_id or "generated-session"

    monkeypatch.setattr(api_module, "_build_orchestrator", lambda: _StubOrchestrator())
    monkeypatch.setattr(api_module.rag, "start_session", record_start_session)

    response = client.post("/agent/chat", json={"message": "hello"})

    assert response.status_code == 200
    assert seen == {"session_id": None, "owner": "operator"}


def test_agent_chat_stream_stamps_default_identity_on_session_start(
    monkeypatch: pytest.MonkeyPatch, client: TestClient
) -> None:
    """Streaming agent chat must stamp the resolved principal on session start.

    ``/agent/chat/stream`` resolves the principal eagerly (before the SSE
    generator) and passes it to ``start_session``. Without it, agent chats would
    persist unowned and never surface in ``/sessions/list``. The clarification
    policy is forced so the generator reaches ``start_session`` and returns
    without depending on the chat stream.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        client (TestClient): The TestClient instance.
    """
    monkeypatch.setenv("DOCINT_DEFAULT_IDENTITY", "operator")
    seen: dict[str, Any] = {}

    def record_start_session(session_id: str | None = None, owner: str | None = None) -> str:
        seen["session_id"] = session_id
        seen["owner"] = owner
        return session_id or "generated-session"

    monkeypatch.setattr(api_module.rag, "start_session", record_start_session)
    monkeypatch.setattr(
        api_module,
        "_clarification_policy",
        api_module.ClarificationPolicy(api_module.ClarificationConfig(confidence_threshold=1.0, require_entities=True)),
    )

    with client.stream("POST", "/agent/chat/stream", json={"message": "hello"}) as resp:
        assert resp.status_code == 200
        list(resp.iter_lines())

    assert seen == {"session_id": None, "owner": "operator"}


def test_agent_chat_stream_uses_history_and_prior_turn(monkeypatch: pytest.MonkeyPatch, client: TestClient) -> None:
    """Streaming agent chat must feed prior history into understanding and stream_chat.

    Verifies parity with /agent/chat: prior_turn + history-rewritten query
    are both forwarded to stream_chat.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        client (TestClient): The TestClient instance.
    """
    monkeypatch.setenv("DOCINT_DEFAULT_IDENTITY", "operator")
    seeded_history = [
        {"role": "user", "content": "Who chairs the council?"},
        {"role": "assistant", "content": "The Security Council has a rotating presidency."},
    ]
    monkeypatch.setattr(
        api_module.rag.sessions,
        "get_session_history",
        lambda session_id, owner=None: seeded_history,
    )

    seen: dict[str, Any] = {}

    class _RecordingUnderstanding:
        def analyze(self, turn: Any, context: Any = None) -> IntentAnalysis:
            seen["context_history"] = list(context.history) if context is not None else None
            return IntentAnalysis(
                intent="qa",
                confidence=0.9,
                entities={"query": turn.user_input},
                rewritten_query="REWRITTEN QUERY",
            )

    def record_stream_chat(
        user_msg: str, *, prior_turn: Any = None, **kwargs: Any
    ) -> Generator[str | dict[str, Any], None, None]:
        seen["stream_query"] = user_msg
        seen["prior_turn"] = prior_turn
        yield "token"
        yield {"sources": [], "session_id": "generated-session"}

    monkeypatch.setattr(api_module, "_understanding_agent", _RecordingUnderstanding())
    monkeypatch.setattr(
        api_module,
        "_clarification_policy",
        api_module.ClarificationPolicy(
            api_module.ClarificationConfig(confidence_threshold=0.0, require_entities=False)
        ),
    )
    monkeypatch.setattr(api_module.rag, "stream_chat", record_stream_chat)

    with client.stream("POST", "/agent/chat/stream", json={"message": "And who is she?"}) as resp:
        assert resp.status_code == 200
        list(resp.iter_lines())

    assert seen["context_history"] == seeded_history
    assert seen["stream_query"] == "REWRITTEN QUERY"
    assert isinstance(seen["prior_turn"], PriorTurn)
    assert seen["prior_turn"].user_text == "Who chairs the council?"
    assert seen["prior_turn"].assistant_text == "The Security Council has a rotating presidency."


def test_stream_query_session_mode_feeds_prior_turn(monkeypatch: pytest.MonkeyPatch, client: TestClient) -> None:
    """``/stream_query`` (the endpoint the React SPA actually calls) must feed the prior turn.

    In session mode it should build the immediately preceding user/assistant
    exchange from owner-scoped history and pass it to ``stream_chat`` together
    with ``skip_query_rewrite=False`` — so generation becomes history-aware while
    the endpoint keeps its own internal retrieval-query rewrite.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        client (TestClient): The TestClient instance.
    """
    monkeypatch.setenv("DOCINT_DEFAULT_IDENTITY", "operator")
    seeded_history = [
        {"role": "user", "content": "Was ist im Bild sichtbar?"},
        {"role": "assistant", "content": "Ein grüner Baum auf einer Wiese."},
    ]
    monkeypatch.setattr(
        api_module.rag.sessions,
        "get_session_history",
        lambda session_id, owner=None: seeded_history,
    )

    seen: dict[str, Any] = {}

    def record_stream_chat(
        user_msg: str,
        *,
        prior_turn: Any = None,
        skip_query_rewrite: Any = None,
        **kwargs: Any,
    ) -> Generator[str | dict[str, Any], None, None]:
        seen["user_msg"] = user_msg
        seen["prior_turn"] = prior_turn
        seen["skip_query_rewrite"] = skip_query_rewrite
        yield "tok"
        yield {"response": "answer", "sources": [], "session_id": "generated-session"}

    monkeypatch.setattr(api_module.rag, "stream_chat", record_stream_chat)

    with client.stream("POST", "/stream_query", json={"question": "Enthält es Menschen?"}) as resp:
        assert resp.status_code == 200
        list(resp.iter_lines())

    # The raw user message reaches stream_chat (the internal rewrite still runs there).
    assert seen["user_msg"] == "Enthält es Menschen?"
    assert seen["skip_query_rewrite"] is False
    assert isinstance(seen["prior_turn"], PriorTurn)
    assert seen["prior_turn"].user_text == "Was ist im Bild sichtbar?"
    assert seen["prior_turn"].assistant_text == "Ein grüner Baum auf einer Wiese."


def test_agent_chat_history_is_owner_scoped_on_both_endpoints(
    monkeypatch: pytest.MonkeyPatch, client: TestClient
) -> None:
    """Both agent endpoints load session history scoped to the resolved principal.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        client (TestClient): The TestClient instance.
    """
    monkeypatch.setenv("DOCINT_DEFAULT_IDENTITY", "operator")
    owners: list[str | None] = []

    def record_history(session_id: str, owner: str | None = None) -> list[dict[str, str]]:
        owners.append(owner)
        return [{"role": "user", "content": "hi"}]

    monkeypatch.setattr(api_module.rag.sessions, "get_session_history", record_history)

    class _StubOrchestrator:
        def handle_turn(self, turn: Any, context: Any = None) -> OrchestratorResult:
            _ = turn, context
            analysis = IntentAnalysis(intent="qa", confidence=0.9, entities={"query": "hi"})
            retrieval = RetrievalResult(answer="a", sources=[], session_id="generated-session")
            return OrchestratorResult(clarification=None, retrieval=retrieval, analysis=analysis)

    monkeypatch.setattr(api_module, "_build_orchestrator", lambda: _StubOrchestrator())
    resp1 = client.post("/agent/chat", json={"message": "hi"})
    assert resp1.status_code == 200

    monkeypatch.setattr(
        api_module,
        "_clarification_policy",
        api_module.ClarificationPolicy(api_module.ClarificationConfig(confidence_threshold=1.0, require_entities=True)),
    )
    with client.stream("POST", "/agent/chat/stream", json={"message": "hi"}) as resp2:
        assert resp2.status_code == 200
        list(resp2.iter_lines())

    assert owners == ["operator", "operator"]


def test_select_understanding_agent_falls_back_to_simple_without_llm(
    monkeypatch: pytest.MonkeyPatch, client: TestClient
) -> None:
    """Without a configured LLM, the shared selector returns the module-level simple agent."""
    monkeypatch.setattr(api_module.rag, "text_model_id", None, raising=False)
    assert api_module._select_understanding_agent() is api_module._understanding_agent


def test_select_understanding_agent_prefers_contextual_when_llm_configured(
    monkeypatch: pytest.MonkeyPatch, client: TestClient
) -> None:
    """With an LLM configured, the selector returns the history-aware contextual agent."""
    monkeypatch.setattr(api_module.rag, "text_model_id", "test-model", raising=False)
    monkeypatch.setattr(api_module.rag, "text_model", object(), raising=False)
    assert isinstance(api_module._select_understanding_agent(), ContextualUnderstandingAgent)


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


def test_query_collection_field_resolves_and_scopes_physical(
    monkeypatch: pytest.MonkeyPatch, client: TestClient
) -> None:
    """`/query` resolves the logical `collection` to its physical name and scopes the engine (WS2).

    The owner manager maps the caller's logical name to an owner-namespaced
    physical collection; the engine must run under that physical name, not the
    process-default active collection.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        client (TestClient): The TestClient instance.
    """
    rag = cast(Any, api_module.rag)
    monkeypatch.setattr(rag._owners, "resolve", lambda owner, logical: f"phys__{logical}")
    response = client.post(
        "/query",
        json={"question": "What?", "collection": "alpha", "retrieval_mode": "stateless"},
    )
    assert response.status_code == 200
    # The engine saw the resolved physical name while scoped, not "alpha".
    assert rag.seen_collections[-1] == "phys__alpha"


def test_query_collection_field_404_when_not_owned(monkeypatch: pytest.MonkeyPatch, client: TestClient) -> None:
    """`/query` with a `collection` the caller does not own returns 404 (WS2).

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        client (TestClient): The TestClient instance.
    """
    rag = cast(Any, api_module.rag)
    monkeypatch.setattr(rag._owners, "resolve", lambda owner, logical: None)
    response = client.post(
        "/query",
        json={"question": "What?", "collection": "ghost", "retrieval_mode": "stateless"},
    )
    assert response.status_code == 404


def test_stream_query_collection_field_404_when_not_owned(monkeypatch: pytest.MonkeyPatch, client: TestClient) -> None:
    """`/stream_query` gates the `collection` upfront — an unowned name 404s before streaming (WS2).

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        client (TestClient): The TestClient instance.
    """
    rag = cast(Any, api_module.rag)
    monkeypatch.setattr(rag._owners, "resolve", lambda owner, logical: None)
    response = client.post(
        "/stream_query",
        json={"question": "What?", "collection": "ghost", "retrieval_mode": "stateless"},
    )
    assert response.status_code == 404


def test_stream_query_stateless_mode_emits_tokens(client: TestClient) -> None:
    """Stateless stream mode should emit tokens via the async query path.

    The stateless branch must call the native-async ``run_query_async`` (so the
    event loop is not blocked) and forward the request-scoped filters intact.

    Args:
        client: The TestClient instance.
    """
    rag = cast(DummyRAG, api_module.rag)
    before = len(rag.stateless_query_filters)
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
    # The async query path was exercised and carried the filter kwargs.
    assert len(rag.stateless_query_filters) == before + 1
    recorded = rag.stateless_query_filters[-1]
    assert set(recorded) == {"filters", "rules", "vector_store_kwargs"}


def test_stream_query_answers_from_a_scope_carried_on_the_request(client: TestClient) -> None:
    """The first turn of a new chat must answer from the scope it was sent with.

    ``PUT /sessions/{id}/scope`` can only write to a conversation row that
    exists, and that row is minted by this very turn — so a client holding a
    hand-picked selection has nowhere to put it beforehand. Carrying it on the
    request is the only way the *first* answer can be scoped; before this, the
    selection reached the server after the answer had already been written.

    Args:
        client (TestClient): The TestClient instance.
    """
    rag = cast(DummyRAG, api_module.rag)
    with client.stream(
        "POST",
        "/stream_query",
        json={"question": "What?", "collection": "alpha", "scope_chunk_ids": ["c1", "c2"]},
    ) as resp:
        assert resp.status_code == 200
        resp.read()

    assert rag.scoped_ids[-1] == ["c1", "c2"]


def test_stream_query_stores_a_request_carried_scope_on_the_session(client: TestClient) -> None:
    """A scope that arrives with a turn is pinned, so later turns keep it.

    Args:
        client (TestClient): The TestClient instance.
    """
    rag = cast(DummyRAG, api_module.rag)
    with client.stream(
        "POST",
        "/stream_query",
        json={"question": "What?", "collection": "alpha", "scope_chunk_ids": ["c1", "c2"]},
    ) as resp:
        resp.read()

    assert rag.sessions.scope == ["c1", "c2"]


def test_query_answers_from_a_scope_carried_on_the_request(client: TestClient) -> None:
    """`/query` honours a request-carried scope like `/stream_query` does.

    Args:
        client (TestClient): The TestClient instance.
    """
    rag = cast(DummyRAG, api_module.rag)
    response = client.post(
        "/query",
        json={"question": "What?", "collection": "alpha", "scope_chunk_ids": ["c1", "c2"]},
    )

    assert response.status_code == 200
    assert rag.scoped_ids[-1] == ["c1", "c2"]
    assert rag.sessions.scope == ["c1", "c2"]


def test_query_reports_the_scope_it_answered_from(client: TestClient) -> None:
    """The report must survive the response model, or the client cannot check it.

    A field the engine sets but ``QueryOut`` does not declare is silently
    dropped — which is how a scoped turn came to be indistinguishable from an
    ordinary one on the wire.

    Args:
        client (TestClient): The TestClient instance.
    """
    response = client.post(
        "/query",
        json={"question": "What?", "collection": "alpha", "scope_chunk_ids": ["c1", "c2"]},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["retrieval_mode"] == "scoped"
    assert body["scoped_chunk_count"] == 2


def test_query_reports_whether_sources_were_reranked(client: TestClient) -> None:
    """A turn whose reranker was unreachable must say so on the wire.

    Measured on a live stack: the reranker container had been down for a day
    and every answer shipped its top-5 by raw fusion order, indistinguishable
    from a healthy turn. The field is what lets the client flag it.

    Args:
        client (TestClient): The TestClient instance.
    """
    response = client.post("/query", json={"question": "What?", "collection": "alpha"})

    assert response.status_code == 200
    assert response.json()["rerank"] == {"applied": False, "error": "upstream down"}


def test_stream_query_final_frame_reports_rerank_outcome(client: TestClient) -> None:
    """The streaming path carries the same report in its final frame.

    Args:
        client (TestClient): The TestClient instance.
    """
    with client.stream("POST", "/stream_query", json={"question": "What?", "collection": "alpha"}) as response:
        frames = [line for line in response.iter_lines() if line.startswith("data:")]
    final = json.loads(frames[-1][len("data:") :])
    assert final["rerank"] == {"applied": False, "error": "upstream down"}


def test_stateless_query_answers_from_a_scope_carried_on_the_request(client: TestClient) -> None:
    """A scope is about *which evidence*, not about carrying chat history.

    Stateless mode has no session to pin a scope to, so the request is the only
    place it can come from; dropping it there would answer a hand-picked
    question from the whole collection.

    Args:
        client (TestClient): The TestClient instance.
    """
    rag = cast(DummyRAG, api_module.rag)
    response = client.post(
        "/query",
        json={
            "question": "What?",
            "collection": "alpha",
            "retrieval_mode": "stateless",
            "scope_chunk_ids": ["c1"],
        },
    )

    assert response.status_code == 200
    assert rag.scoped_ids[-1] == ["c1"]


def test_query_without_a_scope_field_still_uses_the_stored_scope(client: TestClient) -> None:
    """Omitting the field keeps the session's pinned scope (sticky, as before).

    Args:
        client (TestClient): The TestClient instance.
    """
    rag = cast(DummyRAG, api_module.rag)
    rag.sessions.scope = ["stored-1"]

    response = client.post("/query", json={"question": "What?", "collection": "alpha"})

    assert response.status_code == 200
    assert rag.scoped_ids[-1] == ["stored-1"]


def test_stream_query_refuses_an_oversize_request_scope_before_streaming(client: TestClient) -> None:
    """An oversize selection is refused up front, never truncated mid-answer.

    Mirrors ``PUT /sessions/{id}/scope``: scoped answering splices the chunks
    straight into the prompt, so a selection that cannot fit is a 422 — and it
    must be raised before the SSE body opens, or the refusal would arrive as an
    in-stream error the client cannot tell from a generation failure.

    Args:
        client (TestClient): The TestClient instance.
    """
    rag = cast(DummyRAG, api_module.rag)
    rag.scope_fits = False

    response = client.post(
        "/stream_query",
        json={"question": "What?", "collection": "alpha", "scope_chunk_ids": ["c1", "c2"]},
    )

    assert response.status_code == 422
    assert rag.sessions.scope == []


def test_an_unchanged_request_scope_is_not_measured_again(client: TestClient) -> None:
    """Re-sending the pinned scope costs no extra Qdrant round trip.

    Args:
        client (TestClient): The TestClient instance.
    """
    rag = cast(DummyRAG, api_module.rag)
    rag.sessions.scope = ["c1", "c2"]

    response = client.post(
        "/query",
        json={
            "question": "What?",
            "collection": "alpha",
            "session_id": "generated-session",
            "scope_chunk_ids": ["c1", "c2"],
        },
    )

    assert response.status_code == 200
    assert rag.scoped_ids[-1] == ["c1", "c2"]
    assert rag.measured_scopes == []


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


def test_summarize_includes_summary_diagnostics(client: TestClient) -> None:
    """Summarize endpoint serves the cached payload with diagnostics and validation metadata.

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
    rag = cast(DummyRAG, api_module.rag)
    assert rag.cached_summary_calls == 1


def test_summarize_refresh_true_queues_a_job(client: TestClient) -> None:
    """``refresh=true`` bypasses the cache and queues a rebuild job instead of a 200.

    Args:
        client (TestClient): The TestClient instance.
    """
    response = client.post("/summarize?collection=alpha&refresh=true")
    assert response.status_code == 202
    assert response.json()["job_id"]


def test_summarize_stream_route_removed(client: TestClient) -> None:
    """``POST /summarize/stream`` no longer exists -- clients follow job SSE instead.

    Args:
        client (TestClient): The TestClient instance.
    """
    response = client.post("/summarize/stream")
    assert response.status_code in (404, 405)


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


def test_collections_ner_failure(monkeypatch: pytest.MonkeyPatch, client: TestClient) -> None:
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
    assert response.json()["detail"] == "Request failed."


def test_collections_ner_stats_failure(monkeypatch: pytest.MonkeyPatch, client: TestClient) -> None:
    """Stats endpoint should surface backend failures.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        client (TestClient): The TestClient instance.
    """

    def raiser(**kwargs: Any) -> dict[str, Any]:
        """Fake implementation of get_collection_ner_stats that raises an error for testing purposes.

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
    assert response.json()["detail"] == "Request failed."


def test_collections_ner_search_failure(monkeypatch: pytest.MonkeyPatch, client: TestClient) -> None:
    """Search endpoint should surface backend failures.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        client (TestClient): The TestClient instance.
    """

    def raiser(**kwargs: Any) -> list[dict[str, Any]]:
        """Fake implementation of search_collection_ner_entities that raises an error for testing purposes.

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
    assert response.json()["detail"] == "Request failed."


def test_collections_hate_speech_failure(monkeypatch: pytest.MonkeyPatch, client: TestClient) -> None:
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
    assert response.json()["detail"] == "Request failed."


def test_query_requires_collection(monkeypatch: pytest.MonkeyPatch, client: TestClient) -> None:
    """Empty active collection must surface as a structured HTTP 400.

    Regression guard for the outer-handler antipattern that collapsed
    a 400 into a 500. The handler must now propagate the 400 raised
    by ``_require_active_collection`` unchanged.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        client (TestClient): The TestClient instance.
    """
    api_module.rag.qdrant_collection = ""
    response = client.post("/query", json={"question": "hi"})
    assert response.status_code == 400
    assert "No collection selected" in response.json()["detail"]


def test_query_returns_404_when_active_collection_missing(monkeypatch: pytest.MonkeyPatch, client: TestClient) -> None:
    """Stale active collection must surface as HTTP 404 with a clean message.

    Regression guard for the chat-after-delete crash: if a collection is
    deleted out-of-band (or the API singleton holds a stale name),
    ``_require_active_collection`` must trip and return a structured 404
    instead of letting llama-index propagate Qdrant's raw 404.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        client (TestClient): The TestClient instance.
    """
    api_module.rag.qdrant_collection = "ghost"
    monkeypatch.setattr(api_module.rag, "list_collections", lambda: ["alpha", "beta"])
    response = client.post("/query", json={"question": "hi"})
    assert response.status_code == 404
    assert "ghost" in response.json()["detail"]
    assert "no longer exists" in response.json()["detail"]
    # Singleton must self-heal so the user can recover via re-select.
    assert api_module.rag.qdrant_collection == ""
    assert api_module.rag.index is None
    assert api_module.rag.query_engine is None


def test_stream_query_returns_404_when_active_collection_missing(
    monkeypatch: pytest.MonkeyPatch, client: TestClient
) -> None:
    """Stream query must gate on collection existence before opening the SSE stream.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        client (TestClient): The TestClient instance.
    """
    api_module.rag.qdrant_collection = "ghost"
    monkeypatch.setattr(api_module.rag, "list_collections", lambda: ["alpha", "beta"])
    response = client.post("/stream_query", json={"question": "hi"})
    assert response.status_code == 404
    assert "ghost" in response.json()["detail"]


def test_stream_query_requires_collection(monkeypatch: pytest.MonkeyPatch, client: TestClient) -> None:
    """Empty active collection on stream_query must surface as HTTP 400.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        client (TestClient): The TestClient instance.
    """
    api_module.rag.qdrant_collection = ""
    response = client.post("/stream_query", json={"question": "hi"})
    assert response.status_code == 400
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
    assert [(rule.field, rule.operator, rule.value) for rule in last_filters["rules"]] == [
        ("mimetype", "mime_match", "image/*"),
        ("reference_metadata.timestamp", "date_on_or_after", "2026-01-01"),
    ]
    # Only the MIME rule compiles to a LlamaIndex filter. A date bound would
    # become Range(gte=<ISO string>) inside QdrantVectorStore, whose bounds are
    # floats, so it is carried by the native filter instead — which is the one
    # that executes, since qdrant_filters overrides the LlamaIndex filters.
    compiled = last_filters["filters"]
    assert compiled is not None
    assert len(compiled.filters) == 1
    assert cast(Any, compiled.filters[0]).key == "mimetype"

    native = last_filters["vector_store_kwargs"]["qdrant_filters"]
    assert native is not None
    assert len(list(native.must)) == 2


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
    assert [(rule.field, rule.operator, rule.value) for rule in last_filters["rules"]] == [
        (
            "hate_speech.hate_speech",
            "eq",
            True,
        )
    ]
    assert last_filters["vector_store_kwargs"]["qdrant_filters"] is not None


def test_query_handles_missing_sources(monkeypatch: pytest.MonkeyPatch, client: TestClient) -> None:
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


def test_ingest_success(monkeypatch: pytest.MonkeyPatch, client: TestClient, tmp_path: Path) -> None:
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
        path: Any,
        hybrid: bool = True,
        progress_callback: Any = None,
        **kwargs: Any,
    ) -> None:
        """Fake implementation of the ingest_docs function for testing purposes.

        Args:
            collection (str): The name of the collection to ingest into.
            path (Path): The path to the data to ingest.
            hybrid (bool, optional): Whether to use hybrid ingestion. Defaults to True.
            progress_callback (callable, optional): A callback function for progress updates. Defaults to None.
            **kwargs: Ignored extra ingest flags (ner / hate_speech).
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


@pytest.mark.parametrize(
    ("payload_hybrid", "expected_hybrid"),
    [
        (None, None),
        (True, True),
        (False, False),
    ],
)
def test_ingest_sync_forwards_hybrid_without_coercion(
    monkeypatch: pytest.MonkeyPatch,
    client: TestClient,
    tmp_path: Path,
    payload_hybrid: bool | None,
    expected_hybrid: bool | None,
) -> None:
    """The sync ``POST /ingest`` must forward ``hybrid`` to ``ingest_docs`` as-is.

    Regression test for a surface of the critical "ingest forces hybrid on"
    defect not covered by the CLI-level or ``/ingest/finalize`` tests: this
    endpoint had its own ``payload.hybrid if payload.hybrid is not None else
    True`` coercion at the ``ingest_docs`` call site. Every pre-existing
    ``/ingest`` test passes ``hybrid`` explicitly, so none of them would
    catch that coercion being reintroduced. Parametrized over all three
    states so an omitted ``hybrid`` (must reach ``ingest_docs`` as ``None``)
    is distinguished from an explicit ``False`` (must not be collapsed into
    ``None`` by an overcorrection either).

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        client (TestClient): The TestClient instance.
        tmp_path (Path): The temporary path fixture.
        payload_hybrid (bool | None): The ``hybrid`` value to send in the
            request body; ``None`` omits the field entirely.
        expected_hybrid (bool | None): The value ``ingest_docs`` must
            receive for the given ``payload_hybrid``.
    """
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    monkeypatch.setattr(api_module, "_resolve_data_dir", lambda: data_dir)

    recorded: dict[str, Any] = {}

    def fake_ingest(
        collection: str, path: Any, hybrid: bool | None = None, progress_callback: Any = None, **kwargs: Any
    ) -> None:
        recorded["hybrid"] = hybrid

    monkeypatch.setattr(api_module.ingest_module, "ingest_docs", fake_ingest)

    payload: dict[str, Any] = {"collection": "docs"}
    if payload_hybrid is not None:
        payload["hybrid"] = payload_hybrid

    response = client.post("/ingest", json=payload)
    assert response.status_code == 200
    assert recorded["hybrid"] is expected_hybrid


def test_sessions_endpoints(client: TestClient) -> None:
    """Test session management endpoints.

    Args:
        client (TestClient): The TestClient instance.
    """
    headers = {"X-Auth-User": "tester"}

    # List
    resp = client.get("/sessions/list", headers=headers)
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["sessions"]) == 1
    assert data["sessions"][0]["id"] == "123"

    # History
    resp = client.get("/sessions/123/history", headers=headers)
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["messages"]) == 1
    assert data["messages"][0]["content"] == "hi"

    # Delete
    resp = client.delete("/sessions/123", headers=headers)
    assert resp.status_code == 200
    assert resp.json()["ok"] is True


def test_ingest_missing_directory(monkeypatch: pytest.MonkeyPatch, client: TestClient, tmp_path: Path) -> None:
    """Test the ingestion of documents when the data directory is missing.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        client (TestClient): The TestClient instance.
        tmp_path (Path): The temporary path fixture.
    """
    missing = tmp_path / "missing"
    monkeypatch.setattr(api_module, "_resolve_data_dir", lambda: missing)
    response = client.post("/ingest", json={"collection": "abc"})
    assert response.status_code == 400
    assert response.json()["detail"] == "Server storage is not available."


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
        **kwargs: Any,
    ) -> None:
        """Simulate a hard runtime failure during ingestion.

        Args:
            collection (str): Collection name (ignored).
            path (Path): Source directory path (ignored).
            hybrid (bool): Whether hybrid retrieval was requested (ignored).
            progress_callback (Any): Optional progress callback (ignored).
            **kwargs: Ignored extra ingest flags (ner / hate_speech).

        Raises:
            RuntimeError: Always, to mimic an infrastructure failure.
        """
        _ = (collection, path, hybrid, progress_callback)
        raise RuntimeError("Qdrant unreachable")

    monkeypatch.setattr(api_module.ingest_module, "ingest_docs", exploding_ingest)

    # The endpoint must convert unexpected runtime failures into a
    # structured HTTP 500 rather than propagating an unhandled exception
    # or coercing them into the soft-empty 200 reserved for
    # ``EmptyIngestionError``.
    response = client.post("/ingest", json={"collection": "col", "hybrid": True})
    assert response.status_code == 500
    assert response.json()["detail"] == "Request failed."


def test_ingest_upload_empty_emits_warning_and_completes(
    monkeypatch: pytest.MonkeyPatch, client: TestClient, tmp_path: Path
) -> None:
    """An empty ingest job completes with a warning message, not an error.

    Verifies the job runner translates :class:`EmptyIngestionError` into a
    ``completed`` snapshot with ``empty: true`` and a warning message
    referencing the collection, and skips ``rag.select_collection`` (which
    would otherwise raise ``ValueError`` because the collection was never
    created) — instead of surfacing a generic "Ingestion failed".

    Ingestion staging (``/ingest/upload``) no longer runs the pipeline itself
    — it only saves files. The actual ingest now runs as a job queued by
    ``/ingest/finalize``.

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
        **kwargs: Any,
    ) -> None:
        """Simulate an ingestion run that produced no documents.

        Args:
            collection (str): Collection name.
            path (Path): Source directory path (ignored).
            hybrid (bool): Whether hybrid retrieval was requested (ignored).
            progress_callback (Any): Optional progress callback (ignored).
            **kwargs: Ignored extra ingest flags (ner / hate_speech).
        """
        _ = (path, hybrid, progress_callback)
        raise api_module.EmptyIngestionError(collection)

    monkeypatch.setattr(api_module.ingest_module, "ingest_docs", fake_ingest)

    staged = client.post(
        "/ingest/upload",
        data={"collection": "silence-test", "hybrid": "false"},
        files={"files": ("silence.m4a", b"\x00" * 32, "audio/mp4")},
    )
    assert staged.status_code == 200

    snapshot = run_ingest(client, "silence-test", {})

    # The completed job must carry a warning message referencing the
    # collection and be flagged empty. It must NOT carry a generic
    # "Ingestion failed" error.
    assert snapshot["status"] == "completed"
    assert snapshot["empty"] is True
    assert snapshot["message"] is not None
    assert "silence-test" in snapshot["message"]
    assert snapshot["error"] is None

    # select_collection must NOT have been called — DummyRAG.selected stays empty.
    assert cast(Any, api_module.rag).selected == []


def test_ingest_empty_warning_omits_physical_collection_name(
    monkeypatch: pytest.MonkeyPatch, client: TestClient, tmp_path: Path
) -> None:
    """The job's warning message never echoes the physical, owner-namespaced name.

    Regression test for a leak where ``EmptyIngestionError``'s ``str()`` (built
    from the physical, owner-namespaced collection) was pushed verbatim as the
    job's warning message. The test above
    (``test_ingest_upload_empty_emits_warning_and_completes``) can't catch this
    on its own — its stub ``register()`` returns the logical name unchanged, so
    physical and logical happen to be equal there. This test uses an owner
    manager stub whose physical name differs from the logical one, matching
    production's ``u{hash}__{logical}`` namespacing.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        client (TestClient): The TestClient instance.
        tmp_path (Path): The temporary path fixture.
    """
    monkeypatch.setattr(api_module, "_resolve_qdrant_src_dir", lambda: tmp_path)

    physical = "u3f2a91bc44d0__mydocs"

    class NamespacingOwners:
        """Owner manager stub whose physical name differs from the logical one."""

        def register(self, owner: str | None, logical: str) -> str:
            return physical

    monkeypatch.setattr(api_module.rag, "ensure_collection_owner_manager", lambda: NamespacingOwners())

    def fake_ingest(
        collection: str,
        path: Path,
        hybrid: bool = True,
        progress_callback: Any = None,
        **kwargs: Any,
    ) -> None:
        """Simulate an ingestion run that produced no documents.

        Args:
            collection (str): Physical collection name passed by the job runner.
            path (Path): Source directory path (ignored).
            hybrid (bool): Whether hybrid retrieval was requested (ignored).
            progress_callback (Any): Optional progress callback (ignored).
            **kwargs: Ignored extra ingest flags (ner / hate_speech).
        """
        _ = (path, hybrid, progress_callback)
        raise api_module.EmptyIngestionError(collection)

    monkeypatch.setattr(api_module.ingest_module, "ingest_docs", fake_ingest)

    # /ingest/finalize points the job at `_resolve_qdrant_src_dir() / physical`;
    # this stub's register() doesn't go through /ingest/upload's own
    # directory-naming, so stage the directory directly.
    (tmp_path / physical).mkdir(parents=True, exist_ok=True)

    snapshot = run_ingest(client, "mydocs", {})

    assert snapshot["status"] == "completed"
    assert snapshot["empty"] is True
    assert snapshot["message"] is not None
    assert "mydocs" in snapshot["message"]
    assert physical not in snapshot["message"]
    assert snapshot["collection"] == "mydocs"


def test_ingest_upload_defer_saves_without_ingesting(
    monkeypatch: pytest.MonkeyPatch, client: TestClient, tmp_path: Path
) -> None:
    """``/ingest/upload`` stages the file(s) but runs no ingestion pass.

    Staging is now unconditional (there is no ``defer_ingest`` toggle anymore
    — every upload only saves files). The SPA uploads a large selection as
    several batches, then triggers a single ingestion via ``/ingest/finalize``,
    which queues a job. A staged batch must emit ``upload_complete`` and never
    call ``ingest_docs``.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        client (TestClient): The TestClient instance.
        tmp_path (Path): The temporary path fixture.
    """
    monkeypatch.setattr(api_module, "_resolve_qdrant_src_dir", lambda: tmp_path)
    calls: list[Path] = []

    def spy_ingest(
        collection: str, path: Path, hybrid: bool = True, progress_callback: Any = None, **kwargs: Any
    ) -> None:
        """Record that ingestion was invoked (it must not be, during upload)."""
        _ = (collection, hybrid, progress_callback)
        calls.append(path)

    monkeypatch.setattr(api_module.ingest_module, "ingest_docs", spy_ingest)

    response = client.post(
        "/ingest/upload",
        data={"collection": "stage-1"},
        files={"files": ("a.txt", b"hello", "text/plain")},
    )

    assert response.status_code == 200
    body = response.text
    assert "event: upload_complete" in body
    assert "event: ingestion_started" not in body
    assert "event: ingestion_complete" not in body
    # No ingestion ran, but the file was staged to disk for a later finalize.
    assert calls == []
    assert (tmp_path / "stage-1" / "a.txt").read_bytes() == b"hello"


def test_ingest_finalize_ingests_staged_files(
    monkeypatch: pytest.MonkeyPatch, client: TestClient, tmp_path: Path
) -> None:
    """``/ingest/finalize`` queues a job that runs one ingestion pass over the staged directory.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        client (TestClient): The TestClient instance.
        tmp_path (Path): The temporary path fixture.
    """
    monkeypatch.setattr(api_module, "_resolve_qdrant_src_dir", lambda: tmp_path)
    calls: list[Path] = []

    def spy_ingest(
        collection: str, path: Path, hybrid: bool = True, progress_callback: Any = None, **kwargs: Any
    ) -> None:
        """Record the directory ingestion ran over."""
        _ = (collection, hybrid, progress_callback)
        calls.append(path)

    monkeypatch.setattr(api_module.ingest_module, "ingest_docs", spy_ingest)

    staged = client.post(
        "/ingest/upload",
        data={"collection": "final-1"},
        files={"files": ("a.txt", b"hello", "text/plain")},
    )
    assert staged.status_code == 200
    assert calls == []  # staged, not yet ingested

    snapshot = run_ingest(client, "final-1", {})

    assert snapshot["status"] == "completed"
    assert snapshot["error"] is None
    # Exactly one ingestion pass, over the whole staged directory.
    assert calls == [tmp_path / "final-1"]


def test_ingest_finalize_missing_dir_completes_empty(
    monkeypatch: pytest.MonkeyPatch, client: TestClient, tmp_path: Path
) -> None:
    """Finalizing a collection with nothing staged completes empty, not error.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        client (TestClient): The TestClient instance.
        tmp_path (Path): The temporary path fixture.
    """
    monkeypatch.setattr(api_module, "_resolve_qdrant_src_dir", lambda: tmp_path)
    called = False

    def spy_ingest(*args: Any, **kwargs: Any) -> None:
        """Fail the test if ingestion is attempted on a missing directory."""
        nonlocal called
        _ = (args, kwargs)
        called = True

    monkeypatch.setattr(api_module.ingest_module, "ingest_docs", spy_ingest)

    snapshot = run_ingest(client, "never-staged", {})

    assert snapshot["status"] == "completed"
    assert snapshot["empty"] is True
    assert snapshot["message"] is not None
    assert snapshot["error"] is None
    assert called is False
    assert called is False


def test_ingest_soft_completes_when_reader_finds_no_supported_files(
    monkeypatch: pytest.MonkeyPatch, client: TestClient, tmp_path: Path
) -> None:
    """A ``NoSupportedFilesError`` becomes a soft-empty completion.

    A finalize job whose staged directory holds nothing ingestable (e.g. only
    audio/video with Nextext unconfigured, which the pre-passes cannot claim)
    makes the pipeline raise the typed ``NoSupportedFilesError``. That must
    surface as a warning + empty completion, not a ``failed`` job.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        client (TestClient): The TestClient instance.
        tmp_path (Path): The temporary path fixture.
    """
    monkeypatch.setattr(api_module, "_resolve_qdrant_src_dir", lambda: tmp_path)

    def raise_no_files(
        collection: str, path: Path, hybrid: bool = True, progress_callback: Any = None, **kwargs: Any
    ) -> None:
        """Simulate the pipeline finding no ingestable files in ``path``."""
        _ = (collection, hybrid, progress_callback)
        raise NoSupportedFilesError(f"No ingestable files in batch directory {path}.")

    monkeypatch.setattr(api_module.ingest_module, "ingest_docs", raise_no_files)

    staged = client.post(
        "/ingest/upload",
        data={"collection": "media-only"},
        files={"files": ("clip.mp4", b"\x00" * 16, "video/mp4")},
    )
    assert staged.status_code == 200

    snapshot = run_ingest(client, "media-only", {})

    assert snapshot["status"] == "completed"
    assert snapshot["empty"] is True
    assert snapshot["message"] is not None
    assert "No ingestable files" in snapshot["message"]
    assert snapshot["error"] is None


def test_ingest_upload_success_does_not_warm_query_engine(
    monkeypatch: pytest.MonkeyPatch, client: TestClient, tmp_path: Path
) -> None:
    """A successful ingest job must not eagerly warm the query engine.

    Regression guard for an OOM-kill observed on CPU Docker containers with
    the default 8 GB limit: the ingest handler previously called
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
        **kwargs: Any,
    ) -> None:
        """Simulate a successful, no-op ingestion run.

        Returning cleanly is exactly the code path that previously reached
        the eager warmup block.

        Args:
            collection (str): Collection name (ignored).
            path (Path): Source directory path (ignored).
            hybrid (bool): Whether hybrid retrieval was requested (ignored).
            progress_callback (Any): Optional progress callback (ignored).
            **kwargs: Ignored extra ingest flags (ner / hate_speech).
        """
        _ = (collection, path, hybrid, progress_callback)

    monkeypatch.setattr(api_module.ingest_module, "ingest_docs", fake_ingest)

    # Sanity: the autouse _patch_rag fixture installs a fresh DummyRAG whose
    # create_index / create_query_engine counters start at zero.
    dummy_rag = cast(DummyRAG, api_module.rag)
    assert dummy_rag.created_index == 0
    assert dummy_rag.created_query_engine == 0

    staged = client.post(
        "/ingest/upload",
        data={"collection": "warmup-guard", "hybrid": "true"},
        files={"files": ("hello.txt", b"hello world", "text/plain")},
    )
    assert staged.status_code == 200

    # The ingest itself must still succeed — we are not regressing success
    # signalling, only removing the warmup.
    snapshot = run_ingest(client, "warmup-guard", {})
    assert snapshot["status"] == "completed"
    assert snapshot["error"] is None

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


def test_ingest_finalize_job_ignores_request_disconnect(
    monkeypatch: pytest.MonkeyPatch, client: TestClient, tmp_path: Path
) -> None:
    """The ingest job runs to completion independent of the initiating request.

    This replaces two obsolete tests
    (``test_ingest_upload_cancels_awaiter_on_client_disconnect`` and
    ``test_ingest_upload_poll_continues_when_still_connected``) that pinned
    down the *old* SSE design: ``/ingest/upload`` used to run ingestion
    in-request and poll ``request.is_disconnected()`` while waiting, cancelling
    the awaiter (and silently skipping entity resolution) on a hangup. That
    polling loop and ``INGEST_DISCONNECT_POLL_INTERVAL_S`` no longer exist —
    ingestion now runs as a job dispatched by ``/ingest/finalize``, decoupled
    from any single request's lifecycle (this is *the* fix motivating the
    server-owned job registry: a reload or hangup no longer discards progress
    or skips resolution).

    Pins the new behavior down: even with ``Request.is_disconnected`` patched
    to always report a disconnect (the worst case under the old design), the
    job still reaches ``completed`` when polled on a separate request — because
    nothing in the new flow ever calls ``is_disconnected`` at all.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        client (TestClient): The TestClient instance.
        tmp_path (Path): The temporary path fixture, used as the uploads dir.
    """
    from starlette.requests import Request as StarletteRequest

    monkeypatch.setattr(api_module, "_resolve_qdrant_src_dir", lambda: tmp_path)
    calls: list[Path] = []

    def fake_ingest(
        collection: str,
        path: Path,
        hybrid: bool = True,
        progress_callback: Any = None,
        **kwargs: Any,
    ) -> None:
        """Record that the run completed.

        Args:
            collection (str): Collection name (ignored).
            path (Path): Source directory path.
            hybrid (bool): Whether hybrid retrieval was requested (ignored).
            progress_callback (Any): Optional progress callback (ignored).
            **kwargs: Ignored extra ingest flags (ner / hate_speech).
        """
        _ = (collection, hybrid, progress_callback)
        calls.append(path)

    monkeypatch.setattr(api_module.ingest_module, "ingest_docs", fake_ingest)

    async def always_disconnected(_self: StarletteRequest) -> bool:
        """Simulate a client that hangs up immediately after every check.

        Args:
            _self (StarletteRequest): The Request instance (ignored).

        Returns:
            bool: Always ``True``.
        """
        return True

    monkeypatch.setattr(StarletteRequest, "is_disconnected", always_disconnected)

    staged = client.post(
        "/ingest/upload",
        data={"collection": "disconnect-guard"},
        files={"files": ("hello.txt", b"hello world", "text/plain")},
    )
    assert staged.status_code == 200

    snapshot = run_ingest(client, "disconnect-guard", {})

    assert snapshot["status"] == "completed"
    assert snapshot["error"] is None
    assert calls == [tmp_path / "disconnect-guard"]


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
        **kwargs: Any,
    ) -> None:
        """Simulate an ingestion run that produced no documents.

        Args:
            collection (str): Collection name.
            path (Path): Source directory path (ignored).
            hybrid (bool): Whether hybrid retrieval was requested (ignored).
            progress_callback (Any): Optional progress callback (ignored).
            **kwargs: Ignored extra ingest flags (ner / hate_speech).
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
    assert cast(Any, api_module.rag).selected == []


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
        **kwargs: Any,
    ) -> None:
        """Simulate a successful, no-op synchronous ingestion run.

        Returning cleanly is exactly the code path that previously fell
        through to the warmup block on the ``/ingest`` endpoint.

        Args:
            collection (str): Collection name (ignored).
            path (Path): Source directory path (ignored).
            hybrid (bool): Whether hybrid retrieval was requested (ignored).
            progress_callback (Any): Optional progress callback (ignored).
            **kwargs: Ignored extra ingest flags (ner / hate_speech).
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
    assert cast(Any, api_module.rag).selected == []


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


def test_stream_query_context_window_overflow_returns_generic_error(
    monkeypatch: pytest.MonkeyPatch, client: TestClient
) -> None:
    """Context-window overflow surfaces the generic stream error, not the raw exception text.

    The descriptive message (env var names, token counts) is logged for
    operators but never sent to the client — only static, i18n-driven text
    may reach the SPA.

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

    assert "OPENAI_CTX_WINDOW" not in text
    assert "Internal server error" in text


# ---------------------------------------------------------------------------
# Paginated reads: /collections/documents, /collections/hate-speech,
# /collections/ner/sources, /collections/ner/warm
# ---------------------------------------------------------------------------


def _select_alpha(client: TestClient) -> None:
    """Drive the API into a state with the canned 'alpha' collection active."""
    response = client.post("/collections/select", json={"name": "alpha"})
    assert response.status_code == 200, response.text


def test_documents_legacy_mode_returns_full_list(client: TestClient) -> None:
    """Without cursor or limit, /collections/documents returns the legacy envelope."""
    _select_alpha(client)
    rag = cast(DummyRAG, api_module.rag)
    rag.documents = [{"filename": f"doc{i}.pdf"} for i in range(3)]

    response = client.get("/collections/documents")
    assert response.status_code == 200
    payload = response.json()
    assert payload == {"documents": rag.documents}


def test_documents_paginated_mode_round_trips_cursor(client: TestClient) -> None:
    """Cursor + limit drives the paginated envelope and round-trips correctly."""
    _select_alpha(client)
    rag = cast(DummyRAG, api_module.rag)
    rag.documents = [{"filename": f"doc{i:03d}.pdf"} for i in range(25)]

    page1 = client.get("/collections/documents", params={"limit": 10}).json()
    assert len(page1["items"]) == 10
    assert page1["next_cursor"] is not None
    assert page1["items"][0]["filename"] == "doc000.pdf"

    page2 = client.get(
        "/collections/documents",
        params={"limit": 10, "cursor": page1["next_cursor"]},
    ).json()
    assert page2["items"][0]["filename"] == "doc010.pdf"
    assert page2["next_cursor"] is not None

    page3 = client.get(
        "/collections/documents",
        params={"limit": 10, "cursor": page2["next_cursor"]},
    ).json()
    assert len(page3["items"]) == 5
    assert page3["next_cursor"] is None


def test_documents_invalid_cursor_returns_400(client: TestClient) -> None:
    """Malformed cursor tokens must surface as HTTP 400, not 500."""
    _select_alpha(client)
    response = client.get(
        "/collections/documents",
        params={"cursor": "not-a-valid-token", "limit": 10},
    )
    assert response.status_code == 400


def test_documents_no_collection_selected_returns_400(client: TestClient) -> None:
    """The paginated endpoint must require an active collection like the legacy one."""
    api_module.rag.qdrant_collection = ""
    response = client.get("/collections/documents", params={"limit": 10})
    assert response.status_code == 400


def test_hate_speech_legacy_mode_returns_results_envelope(client: TestClient) -> None:
    """Without cursor/limit/filter args the response keeps the legacy ``results`` shape."""
    _select_alpha(client)
    rag = cast(DummyRAG, api_module.rag)
    rag.hate_speech_rows = [{"chunk_id": "c1", "category": "X"}]

    response = client.get("/collections/hate-speech")
    assert response.status_code == 200
    assert response.json() == {"results": [{"chunk_id": "c1", "category": "X"}]}


def test_hate_speech_paginated_mode(client: TestClient) -> None:
    """Passing ``limit`` switches the response to the paginated envelope."""
    _select_alpha(client)
    rag = cast(DummyRAG, api_module.rag)
    rag.hate_speech_rows = [{"chunk_id": f"c{i}"} for i in range(7)]

    payload = client.get("/collections/hate-speech", params={"limit": 3}).json()
    assert payload["items"] == rag.hate_speech_rows[:3]
    assert payload["next_cursor"] is not None


def test_ner_sources_paginates_and_forwards_filters(client: TestClient) -> None:
    """Paginated NER sources slice the cached list and forward filter args."""
    _select_alpha(client)
    rag = cast(DummyRAG, api_module.rag)
    rag.ner_sources = [{"chunk_id": f"c{i}", "entities": []} for i in range(8)]

    response = client.get(
        "/collections/ner/sources",
        params={"limit": 5, "entity_key": "Acme::ORG"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["items"] == rag.ner_sources[:5]
    assert payload["next_cursor"] is not None

    forwarded = rag.last_ner_sources_filter
    assert forwarded["entity_key"] == "Acme::ORG"


def test_ner_sources_forwards_resolved_merge_mode(client: TestClient) -> None:
    """The paginated NER-sources endpoint forwards entity_merge_mode=resolved."""
    _select_alpha(client)
    rag = cast(DummyRAG, api_module.rag)
    response = client.get(
        "/collections/ner/sources",
        params={"entity_key": "US::loc", "entity_merge_mode": "resolved"},
    )
    assert response.status_code == 200
    assert rag.last_ner_sources_filter["entity_merge_mode"] == "resolved"


def test_export_ner_sources_csv_forwards_resolved_merge_mode(client: TestClient) -> None:
    """The NER-sources CSV export forwards entity_merge_mode=resolved to the iterator."""
    _select_alpha(client)
    rag = cast(DummyRAG, api_module.rag)
    response = client.get(
        "/collections/alpha/export/ner-sources.csv",
        params={"entity_text": "US", "entity_type": "loc", "entity_merge_mode": "resolved"},
    )
    assert response.status_code == 200
    assert rag.last_ner_sources_filter["entity_merge_mode"] == "resolved"


def test_ner_sources_invalid_cursor_returns_400(client: TestClient) -> None:
    """Malformed cursors on the paginated NER endpoint must return HTTP 400."""
    _select_alpha(client)
    response = client.get("/collections/ner/sources", params={"cursor": "$$$"})
    assert response.status_code == 400


def test_ner_warm_kicks_aggregate(client: TestClient) -> None:
    """POST /collections/ner/warm triggers exactly one aggregate-build."""
    _select_alpha(client)
    response = client.post("/collections/ner/warm")
    assert response.status_code == 200
    assert response.json() == {"ok": True}
    rag = cast(DummyRAG, api_module.rag)
    assert getattr(rag, "warm_calls", 0) == 1


def test_ner_warm_requires_collection(client: TestClient) -> None:
    """Warming with no active collection must surface HTTP 400."""
    api_module.rag.qdrant_collection = ""
    response = client.post("/collections/ner/warm")
    assert response.status_code == 400


# ---------------------------------------------------------------------------
# Streaming CSV exports: /collections/{name}/export/*.csv
# ---------------------------------------------------------------------------


def _parse_csv_body(body: bytes) -> list[list[str]]:
    """Decode a streamed CSV body (BOM-tolerant) into rows."""
    import csv
    import io

    text = body.decode("utf-8-sig")
    return list(csv.reader(io.StringIO(text)))


def test_export_documents_csv_streams(client: TestClient) -> None:
    """Document export streams a UTF-8 BOM-prefixed CSV with RFC 6266 headers."""
    _select_alpha(client)
    rag = cast(DummyRAG, api_module.rag)
    rag.documents = [
        {
            "filename": "doc1.pdf",
            "mimetype": "application/pdf",
            "file_hash": "abc",
            "node_count": 3,
            "page_count": 2,
            "entity_types": ["PERSON", "ORG"],
        }
    ]

    response = client.get("/collections/alpha/export/documents.csv")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/csv")
    disp = response.headers["content-disposition"]
    assert "alpha-documents.csv" in disp
    assert "filename*=UTF-8''alpha-documents.csv" in disp

    rows = _parse_csv_body(response.content)
    assert rows[0] == [
        "filename",
        "mimetype",
        "file_hash",
        "node_count",
        "page_count",
        "max_rows",
        "max_duration",
        "entity_types",
    ]
    assert rows[1][0] == "doc1.pdf"
    assert rows[1][-1] == "PERSON;ORG"


def test_export_404_when_not_owned(monkeypatch: pytest.MonkeyPatch, client: TestClient) -> None:
    """Exporting a collection the caller does not own returns 404 (WS2).

    Exports are now owner-gated on the URL path collection rather than coupled
    to a global active collection (the old 409 "mismatch" path is gone). When
    the owner manager cannot resolve the name for this principal, the endpoint
    404s without leaking existence.
    """
    rag = cast(Any, api_module.rag)
    monkeypatch.setattr(rag._owners, "resolve", lambda owner, logical: None)
    response = client.get("/collections/beta/export/documents.csv")
    assert response.status_code == 404


def test_export_entities_csv_uses_ner_stats(client: TestClient) -> None:
    """Entity export streams the rank/entity/type/mentions schema from get_collection_ner_stats."""
    _select_alpha(client)
    response = client.get("/collections/alpha/export/entities.csv")
    assert response.status_code == 200
    rows = _parse_csv_body(response.content)
    assert rows[0] == ["rank", "entity", "type", "mentions"]
    assert rows[1] == ["1", "Acme", "ORG", "3"]


def test_export_entities_csv_supports_resolved_merge_mode(client: TestClient) -> None:
    """Entity export accepts entity_merge_mode=resolved and forwards it to stats."""
    _select_alpha(client)
    response = client.get(
        "/collections/alpha/export/entities.csv",
        params={"entity_merge_mode": "resolved"},
    )
    assert response.status_code == 200
    assert cast(DummyRAG, api_module.rag).ner_stats_merge_modes[-1] == "resolved"


def test_export_ner_sources_csv_filters_by_entity(client: TestClient) -> None:
    """NER-source export honors entity_text + entity_type filters and embeds the entity label."""
    _select_alpha(client)
    rag = cast(DummyRAG, api_module.rag)
    rag.ner_sources = [
        {
            "chunk_id": "c1",
            "filename": "doc1.pdf",
            "chunk_text": "Acme makes widgets",
            "entities": [{"text": "Acme", "type": "ORG"}],
        }
    ]
    response = client.get(
        "/collections/alpha/export/ner-sources.csv",
        params={"entity_text": "Acme", "entity_type": "ORG"},
    )
    assert response.status_code == 200
    rows = _parse_csv_body(response.content)
    assert rows[0][0] == "entity"
    assert rows[1][0] == "Acme [ORG]"
    assert rows[1][1] == "doc1.pdf"


def test_export_hate_speech_csv_passes_through_rows(client: TestClient) -> None:
    """Hate-speech export streams the per-finding schema with reference_metadata fields."""
    _select_alpha(client)
    rag = cast(DummyRAG, api_module.rag)
    rag.hate_speech_rows = [
        {
            "chunk_id": "c1",
            "filename": "doc.pdf",
            "category": "Hateful",
            "confidence": "high",
            "reason": "tagged",
            "chunk_text": "bad text",
            "page": 2,
        }
    ]
    response = client.get("/collections/alpha/export/hate-speech.csv")
    assert response.status_code == 200
    rows = _parse_csv_body(response.content)
    assert rows[0][0] == "source"
    assert rows[1][0] == "doc.pdf"
    assert rows[1][4] == "Hateful"


def test_export_independent_of_global_active_collection(client: TestClient) -> None:
    """Exports resolve the path collection by ownership, not the global active one (WS2).

    Clearing the process-default active collection must not affect an export of
    an owned collection: the path name is owner-gated and scoped per request, so
    exports are stateless.
    """
    api_module.rag.qdrant_collection = ""
    response = client.get("/collections/alpha/export/documents.csv")
    assert response.status_code == 200


def test_select_does_not_warm_documents_or_hate_speech_caches(client: TestClient) -> None:
    """Select must remain light — paginated caches populate lazily.

    Companion to test_collections_select_success's NER guard. Documents and
    hate-speech caches must not be eagerly populated by /collections/select;
    they populate on first GET to the paginated endpoint.
    """
    rag = cast(DummyRAG, api_module.rag)
    rag.documents = [{"filename": "x.pdf"}]
    rag.hate_speech_rows = [{"chunk_id": "c1"}]

    _select_alpha(client)

    assert getattr(rag, "warm_calls", 0) == 0
    assert rag.ner_refresh_calls == []


# ---------------------------------------------------------------------------
# /collections/documents/count
# ---------------------------------------------------------------------------


def test_documents_count_returns_size_of_document_list(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """The count endpoint returns ``len(list_documents())`` for the active collection."""
    _select_alpha(client)
    rag = cast(DummyRAG, api_module.rag)
    rag.documents = [{"filename": f"d{i}.pdf"} for i in range(7)]

    # The endpoint calls rag.get_document_count(); the dummy doesn't have one,
    # so wire a thin lambda for this test.
    monkeypatch.setattr(rag, "get_document_count", lambda: len(rag.documents), raising=False)

    response = client.get("/collections/documents/count")
    assert response.status_code == 200
    assert response.json() == {"count": 7}


def test_documents_count_requires_active_collection(client: TestClient) -> None:
    """Count endpoint must reject calls with no active collection."""
    api_module.rag.qdrant_collection = ""
    response = client.get("/collections/documents/count")
    assert response.status_code == 400


def test_documents_summary_returns_collection_wide_aggregate(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The summary endpoint returns collection-wide document aggregates.

    This is the fix for the Inspector undercounting file types on a large,
    lazily-paginated collection: the breakdown is computed server-side over the
    whole collection, not just the loaded rows.

    Args:
        client (TestClient): The TestClient instance.
        monkeypatch (pytest.MonkeyPatch): Fixture to override the RAG method.
    """
    _select_alpha(client)
    rag = cast(DummyRAG, api_module.rag)
    aggregate = {
        "document_count": 305,
        "node_count": 305,
        "file_types": [{"label": "JPEG", "count": 304}, {"label": "PNG", "count": 1}],
        "entity_types": ["loc", "org"],
    }
    # DummyRAG has no get_document_summary; wire a thin lambda for this test.
    monkeypatch.setattr(rag, "get_document_summary", lambda: aggregate, raising=False)

    response = client.get("/collections/documents/summary")

    assert response.status_code == 200
    assert response.json() == aggregate


def test_documents_summary_requires_active_collection(client: TestClient) -> None:
    """Summary endpoint must reject calls with no active collection."""
    api_module.rag.qdrant_collection = ""
    response = client.get("/collections/documents/summary")
    assert response.status_code == 400


# ---------------------------------------------------------------------------
# /sessions/{session_id}/sources.zip
# ---------------------------------------------------------------------------


def test_session_sources_zip_streams_unique_files(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Session-ZIP bundles every unique cited file exactly once and streams as application/zip."""
    import zipfile as _zipfile

    src_a = tmp_path / "a.pdf"
    src_b = tmp_path / "b.pdf"
    src_a.write_bytes(b"hello from a")
    src_b.write_bytes(b"hello from b")

    def fake_history(self: Any, sid: str, owner: str) -> list[dict[str, Any]]:
        assert sid == "sess-1"
        assert owner == "tester"
        return [
            {"role": "user", "content": "q"},
            {
                "role": "assistant",
                "content": "a",
                "sources": [
                    {"file_hash": "h-a", "filename": "a.pdf", "collection": "alpha"},
                    {"file_hash": "h-b", "filename": "b.pdf", "collection": "alpha"},
                    # Duplicate hash must not produce a second entry.
                    {"file_hash": "h-a", "filename": "a.pdf", "collection": "alpha"},
                ],
            },
        ]

    def fake_resolve(
        collection: str,
        file_hash: str,
        *,
        filename_hint: str | None = None,
    ) -> Path | None:
        return {"h-a": src_a, "h-b": src_b}.get(file_hash)

    monkeypatch.setattr(DummySessionManager, "get_session_history", fake_history, raising=False)
    monkeypatch.setattr(api_module, "_resolve_source_file_path", fake_resolve)

    response = client.get("/sessions/sess-1/sources.zip", headers={"X-Auth-User": "tester"})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/zip")
    assert "session-sess-1-sources.zip" in response.headers["content-disposition"]

    archive = _zipfile.ZipFile(io.BytesIO(response.content))
    assert sorted(archive.namelist()) == ["a.pdf", "b.pdf"]
    assert archive.read("a.pdf") == b"hello from a"
    assert archive.read("b.pdf") == b"hello from b"


def test_session_sources_zip_returns_404_when_no_files(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """Empty or unresolvable sessions surface as HTTP 404, not an empty ZIP."""
    monkeypatch.setattr(
        DummySessionManager,
        "get_session_history",
        lambda self, sid, owner: [{"role": "user", "content": "q"}],
        raising=False,
    )
    response = client.get("/sessions/empty/sources.zip", headers={"X-Auth-User": "tester"})
    assert response.status_code == 404


def test_session_sources_zip_skips_unresolved_files(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Files the backend can't resolve are dropped silently, not failed loudly."""
    import zipfile as _zipfile

    src = tmp_path / "kept.pdf"
    src.write_bytes(b"kept")

    monkeypatch.setattr(
        DummySessionManager,
        "get_session_history",
        lambda self, sid, owner: [
            {
                "role": "assistant",
                "content": "a",
                "sources": [
                    {"file_hash": "kept", "filename": "kept.pdf", "collection": "alpha"},
                    {"file_hash": "missing", "filename": "missing.pdf", "collection": "alpha"},
                ],
            }
        ],
        raising=False,
    )
    monkeypatch.setattr(
        api_module,
        "_resolve_source_file_path",
        lambda collection, file_hash, **_: src if file_hash == "kept" else None,
    )

    response = client.get("/sessions/partial/sources.zip", headers={"X-Auth-User": "tester"})
    assert response.status_code == 200
    archive = _zipfile.ZipFile(io.BytesIO(response.content))
    assert archive.namelist() == ["kept.pdf"]


def test_sessions_endpoints_pass_principal_and_404_on_cross_owner(
    monkeypatch: pytest.MonkeyPatch, client: TestClient
) -> None:
    """Session endpoints forward the resolved principal and 404 cross-owner.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        client (TestClient): The TestClient instance.
    """
    seen: dict[str, Any] = {}

    class OwnerAwareSessions:
        """Session manager stub recording the owner it was called with."""

        def list_sessions(self, owner: str) -> list[dict[str, Any]]:
            seen["list"] = owner
            return [{"id": "s1", "created_at": "2026-01-01", "title": "t"}]

        def get_session_history(self, session_id: str, owner: str) -> list[dict[str, Any]]:
            seen["history"] = (session_id, owner)
            # Simulate a cross-owner / missing session: empty history.
            return [] if owner == "bob" else [{"role": "user", "content": "hi"}]

        def delete_session(self, session_id: str, owner: str) -> bool:
            seen["delete"] = (session_id, owner)
            return owner == "alice"

    monkeypatch.setattr(api_module.rag, "ensure_session_manager", lambda: OwnerAwareSessions())

    # List forwards the header principal.
    resp = client.get("/sessions/list", headers={"X-Auth-User": "alice"})
    assert resp.status_code == 200
    assert seen["list"] == "alice"
    assert resp.json()["sessions"][0]["id"] == "s1"

    # History for the owner succeeds.
    resp = client.get("/sessions/s1/history", headers={"X-Auth-User": "alice"})
    assert resp.status_code == 200
    assert seen["history"] == ("s1", "alice")
    assert resp.json()["messages"][0]["content"] == "hi"

    # Cross-owner history is 404 (empty -> not found, no existence leak).
    resp = client.get("/sessions/s1/history", headers={"X-Auth-User": "bob"})
    assert resp.status_code == 404

    # Cross-owner delete is 404.
    resp = client.delete("/sessions/s1", headers={"X-Auth-User": "bob"})
    assert resp.status_code == 404
    assert seen["delete"] == ("s1", "bob")

    # Owner delete succeeds.
    resp = client.delete("/sessions/s1", headers={"X-Auth-User": "alice"})
    assert resp.status_code == 200
    assert resp.json()["ok"] is True


def test_sessions_list_401_without_header_or_default(monkeypatch: pytest.MonkeyPatch, client: TestClient) -> None:
    """With no trusted header and no configured default, endpoints 401.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        client (TestClient): The TestClient instance.
    """
    monkeypatch.delenv("DOCINT_AUTH_HEADER", raising=False)
    monkeypatch.delenv("DOCINT_DEFAULT_IDENTITY", raising=False)

    resp = client.get("/sessions/list")
    assert resp.status_code == 401


def test_sessions_list_uses_default_identity_when_no_header(
    monkeypatch: pytest.MonkeyPatch, client: TestClient
) -> None:
    """A configured default identity is used as the owner when no header.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        client (TestClient): The TestClient instance.
    """
    monkeypatch.delenv("DOCINT_AUTH_HEADER", raising=False)
    monkeypatch.setenv("DOCINT_DEFAULT_IDENTITY", "operator")
    seen: dict[str, Any] = {}

    class OwnerAwareSessions:
        """Session manager stub recording the owner it was called with."""

        def list_sessions(self, owner: str) -> list[dict[str, Any]]:
            seen["list"] = owner
            return []

    monkeypatch.setattr(api_module.rag, "ensure_session_manager", lambda: OwnerAwareSessions())

    resp = client.get("/sessions/list")
    assert resp.status_code == 200
    assert seen["list"] == "operator"


def test_sessions_list_scopes_to_collection(monkeypatch: pytest.MonkeyPatch, client: TestClient) -> None:
    """`?collection=` resolves to physical and scopes; unowned -> empty, not 404."""
    seen: dict[str, Any] = {}

    class ScopedSessions:
        def list_sessions(self, owner: str, collection: str | None = None) -> list[dict[str, Any]]:
            seen["args"] = (owner, collection)
            return [{"id": "s1", "created_at": "2026-01-01", "title": "t", "collection": collection}]

    class Owners:
        def resolve(self, owner: str, logical: str) -> str | None:
            return f"u123__{logical}" if logical == "alpha" else None

    monkeypatch.setattr(api_module.rag, "ensure_session_manager", lambda: ScopedSessions())
    monkeypatch.setattr(api_module.rag, "ensure_collection_owner_manager", lambda: Owners())

    resp = client.get("/sessions/list?collection=alpha", headers={"X-Auth-User": "alice"})
    assert resp.status_code == 200
    assert seen["args"] == ("alice", "u123__alpha")
    assert resp.json()["sessions"][0]["id"] == "s1"

    resp = client.get("/sessions/list?collection=ghost", headers={"X-Auth-User": "alice"})
    assert resp.status_code == 200
    assert resp.json()["sessions"] == []


@pytest.mark.anyio
async def test_stream_query_does_not_block_event_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A slow streaming query must not stall concurrent requests.

    Regression test for the PR-195 freeze: the chat stream used to iterate a
    blocking sync generator directly on the event loop, so a single in-flight
    ``/stream_query`` starved every other request (nginx 504s, frozen UI). With
    the generator pumped on a worker thread, a cheap ``/collections/list`` must
    still return promptly while the slow stream is mid-flight.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
    """
    import time

    import httpx
    from httpx import ASGITransport

    rag = cast(DummyRAG, api_module.rag)

    def slow_stream(question: str, **_kwargs: Any) -> Generator[str | dict[str, Any], None, None]:
        """Block before the first yield, modelling retrieval + first-token latency.

        The heavy synchronous work (query rewrite, embedding, Qdrant search,
        rerank, first LLM token) all happens inside the first ``next()`` before
        any chunk is produced — so a single ``time.sleep`` before the first
        ``yield`` is the faithful reproduction of the freeze.
        """
        time.sleep(1.0)
        yield "tok "
        yield {"response": "answer", "sources": [], "session_id": "generated-session"}

    monkeypatch.setattr(rag, "stream_chat", slow_stream)

    # Measure each request's completion time relative to a shared start. The
    # blocking sync sleep cannot be "raced" on a single event loop — what
    # distinguishes the bug from the fix is *when the loop is free*. With the
    # bug, the stream blocks the loop for its full duration, so the cheap GET
    # cannot complete until the stream releases it (both finish ~together).
    # With the fix, the stream parks on a worker thread immediately, so the GET
    # completes near-instantly while the stream is still sleeping.
    transport = ASGITransport(app=api_module.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        start = time.perf_counter()
        timings: dict[str, float] = {}

        async def timed_stream() -> int:
            """Drive the slow streaming query to completion, recording its finish."""
            resp = await ac.post("/stream_query", json={"question": "hi", "session_id": "s"})
            timings["stream"] = time.perf_counter() - start
            return resp.status_code

        async def timed_cheap() -> int:
            """Hit a cheap endpoint concurrently, recording its finish."""
            resp = await ac.get("/collections/list")
            timings["cheap"] = time.perf_counter() - start
            return resp.status_code

        stream_status, cheap_status = await asyncio.gather(timed_stream(), timed_cheap())

        assert stream_status == 200
        assert cheap_status == 200
        # The cheap call must clear well before the ~1s stream finishes; if the
        # loop were blocked it could only complete once the stream released it.
        assert timings["cheap"] < timings["stream"] - 0.5, (
            f"concurrent request was blocked behind the stream: "
            f"cheap={timings['cheap']:.2f}s stream={timings['stream']:.2f}s"
        )


def test_stream_query_disconnect_cancels_awaiter(monkeypatch: pytest.MonkeyPatch, client: TestClient) -> None:
    """A client disconnect mid-stream must stop draining the chat generator.

    Mirrors ``test_ingest_upload_cancels_awaiter_on_client_disconnect``: the
    poll interval is shrunk and ``is_disconnected`` forced ``True`` so the
    disconnect fires before the slow generator yields its first token — no
    token events should reach the client.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        client (TestClient): The TestClient instance.
    """
    import time

    from starlette.requests import Request as StarletteRequest

    monkeypatch.setattr(api_module, "STREAM_DISCONNECT_POLL_INTERVAL_S", 0.05)

    rag = cast(DummyRAG, api_module.rag)

    def slow_stream(question: str, **_kwargs: Any) -> Generator[str | dict[str, Any], None, None]:
        """Block before the first yield so the disconnect poll wins the race."""
        time.sleep(0.3)
        yield "tok "
        yield {"response": "answer", "sources": [], "session_id": "generated-session"}

    monkeypatch.setattr(rag, "stream_chat", slow_stream)

    async def always_disconnected(_self: StarletteRequest) -> bool:
        """Simulate an immediate client disconnect."""
        return True

    monkeypatch.setattr(StarletteRequest, "is_disconnected", always_disconnected)

    with client.stream("POST", "/stream_query", json={"question": "hi", "session_id": "s"}) as resp:
        assert resp.status_code == 200
        body = "".join(chunk.decode() for chunk in resp.iter_raw())

    # The awaiter was cancelled before the generator produced any token.
    assert '"token"' not in body


def test_stream_query_surfaces_generator_error(monkeypatch: pytest.MonkeyPatch, client: TestClient) -> None:
    """An exception raised inside the chat generator becomes an SSE error event.

    The thread-bridge re-raises the worker exception on the loop, where the
    endpoint's existing ``except`` clause converts it to an ``error`` payload.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        client (TestClient): The TestClient instance.
    """
    rag = cast(DummyRAG, api_module.rag)

    def boom_stream(question: str, **_kwargs: Any) -> Generator[str | dict[str, Any], None, None]:
        """Yield one token, then raise to exercise error propagation."""
        yield "tok "
        raise RuntimeError("kaboom")

    monkeypatch.setattr(rag, "stream_chat", boom_stream)

    with client.stream("POST", "/stream_query", json={"question": "hi", "session_id": "s"}) as resp:
        assert resp.status_code == 200
        body = "".join(chunk.decode() for chunk in resp.iter_raw())

    assert '"token": "tok ' in body
    assert '"error"' in body


def test_safe_relative_dest_preserves_subdirs(tmp_path: Path) -> None:
    """_safe_relative_dest preserves subdirectories from browser folder upload."""
    assert api_module._safe_relative_dest(tmp_path, "media/sub/a.jpg") == tmp_path / "media" / "sub" / "a.jpg"


def test_safe_relative_dest_strips_traversal(tmp_path: Path) -> None:
    """_safe_relative_dest neutralizes path traversal attempts."""
    dest = api_module._safe_relative_dest(tmp_path, "../../etc/passwd")
    assert dest == tmp_path / "etc" / "passwd"


def test_safe_relative_dest_drops_absolute_leading_slash(tmp_path: Path) -> None:
    """_safe_relative_dest converts absolute paths to relative."""
    assert api_module._safe_relative_dest(tmp_path, "/abs/x.jpg") == tmp_path / "abs" / "x.jpg"


def test_safe_relative_dest_normalizes_backslashes(tmp_path: Path) -> None:
    """_safe_relative_dest normalizes backslashes to forward slashes."""
    assert api_module._safe_relative_dest(tmp_path, "media\\sub\\b.png") == tmp_path / "media" / "sub" / "b.png"


def test_safe_relative_dest_empty_falls_back(tmp_path: Path) -> None:
    """_safe_relative_dest falls back to 'upload' when given empty string."""
    assert api_module._safe_relative_dest(tmp_path, "") == tmp_path / "upload"


def test_ingest_upload_preserves_subdir_structure(
    monkeypatch: pytest.MonkeyPatch, client: TestClient, tmp_path: Path
) -> None:
    """Endpoint test: /ingest/upload preserves subdirectory structure.

    Verifies that when a file with a relative path (webkitRelativePath)
    is uploaded, the directory structure is preserved in the target directory.

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
        **kwargs: Any,
    ) -> None:
        """Simulate an ingestion run.

        Args:
            collection (str): Collection name.
            path (Path): Source directory path (ignored).
            hybrid (bool): Whether hybrid retrieval was requested (ignored).
            progress_callback (Any): Optional progress callback (ignored).
            **kwargs: Ignored extra ingest flags (ner / hate_speech).
        """
        _ = (collection, path, hybrid, progress_callback)

    monkeypatch.setattr(api_module.ingest_module, "ingest_docs", fake_ingest)

    response = client.post(
        "/ingest/upload",
        data={"collection": "tree", "hybrid": "false"},
        files=[("files", ("media/sub/a.jpg", b"\xff\xd8\xff", "image/jpeg"))],
    )

    assert response.status_code == 200
    assert (tmp_path / "tree" / "media" / "sub" / "a.jpg").is_file()


def test_get_config_returns_graph_settings(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """`GET /config` reports the env-driven graph node defaults.

    Args:
        client (TestClient): The TestClient instance.
        monkeypatch (pytest.MonkeyPatch): Fixture to override env vars.
    """
    monkeypatch.setenv("NER_GRAPH_TOP_K", "120")
    monkeypatch.setenv("NER_GRAPH_MAX_TOP_K", "900")

    response = client.get("/config")

    assert response.status_code == 200
    body = response.json()
    assert body["graph_top_k"] == 120
    assert body["graph_max_top_k"] == 900
    assert "collection_timeout" in body


def test_get_config_reports_upload_ceiling(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """`GET /config` advertises the per-request upload ceiling in bytes.

    The SPA sizes its upload batches from this so a large selection is split
    into sub-ceiling requests instead of one body nginx would 413.

    Args:
        client (TestClient): The TestClient instance.
        monkeypatch (pytest.MonkeyPatch): Fixture to override env vars.
    """
    monkeypatch.setenv("DOCINT_CLIENT_MAX_BODY_SIZE", "4g")

    response = client.get("/config")

    assert response.status_code == 200
    assert response.json()["max_upload_bytes"] == 4 * 1024**3


def test_get_config_includes_language(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """`GET /config` reports the active `RESPONSE_LANGUAGE` locale.

    Args:
        client (TestClient): The TestClient instance.
        monkeypatch (pytest.MonkeyPatch): Fixture to override env vars.
    """
    monkeypatch.setenv("RESPONSE_LANGUAGE", "de")

    response = client.get("/config")

    assert response.status_code == 200
    assert response.json()["language"] == "de"


def test_ner_graph_uses_env_default_when_top_k_omitted(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """Omitting `top_k_nodes` falls back to `NER_GRAPH_TOP_K`.

    Args:
        client (TestClient): The TestClient instance.
        monkeypatch (pytest.MonkeyPatch): Fixture to override env vars.
    """
    monkeypatch.setenv("NER_GRAPH_TOP_K", "150")
    monkeypatch.delenv("NER_GRAPH_MAX_TOP_K", raising=False)

    response = client.get("/collections/ner/graph")

    assert response.status_code == 200
    assert cast(DummyRAG, api_module.rag).ner_graph_top_ks[-1] == 150


def test_ner_graph_accepts_value_above_legacy_500_cap(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """With a raised ceiling, > 500 nodes are honoured (old hard cap removed).

    Args:
        client (TestClient): The TestClient instance.
        monkeypatch (pytest.MonkeyPatch): Fixture to override env vars.
    """
    monkeypatch.setenv("NER_GRAPH_MAX_TOP_K", "800")

    response = client.get("/collections/ner/graph", params={"top_k_nodes": 800})

    assert response.status_code == 200
    assert cast(DummyRAG, api_module.rag).ner_graph_top_ks[-1] == 800


def test_ner_graph_clamps_top_k_to_configured_max(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """A request above the ceiling is clamped down, not rejected.

    Args:
        client (TestClient): The TestClient instance.
        monkeypatch (pytest.MonkeyPatch): Fixture to override env vars.
    """
    monkeypatch.setenv("NER_GRAPH_MAX_TOP_K", "300")

    response = client.get("/collections/ner/graph", params={"top_k_nodes": 5000})

    assert response.status_code == 200
    assert cast(DummyRAG, api_module.rag).ner_graph_top_ks[-1] == 300


def test_stream_query_error_event_carries_context_overflow_code(
    monkeypatch: pytest.MonkeyPatch, client: TestClient
) -> None:
    """A context-window overflow tags the generic stream error with its code.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
        client: The TestClient instance.
    """
    original_run_query = type(api_module.rag).run_query

    def _exploding_run_query(self: Any, *a: Any, **kw: Any) -> Any:
        raise ValueError("The query and retrieved context exceed the configured context window (4096 tokens).")

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

    assert '"code": "context_overflow"' in text
    assert "context window" not in text.replace('"code": "context_overflow"', "")


def test_stream_query_error_event_carries_generation_failed_code(
    monkeypatch: pytest.MonkeyPatch, client: TestClient
) -> None:
    """A generic stream failure tags the error event with the generation code.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
        client: The TestClient instance.
    """
    original_run_query = type(api_module.rag).run_query

    def _exploding_run_query(self: Any, *a: Any, **kw: Any) -> Any:
        raise RuntimeError("boom-generic")

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

    assert '"code": "generation_failed"' in text
    assert "boom-generic" not in text


def test_stream_query_reports_a_dead_embedding_endpoint_distinctly(
    monkeypatch: pytest.MonkeyPatch, client: TestClient
) -> None:
    """An unusable embedding endpoint gets its own code, not the generic one.

    ``generation_failed`` reads as "the model failed" and sends an
    operator to the chat model's logs. When retrieval cannot embed the
    query, the fault is upstream and entirely different — a distinct code
    is what lets the SPA say so.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
        client: The TestClient instance.
    """
    from docint.utils.openai_cfg import EmbeddingEndpointError

    original_run_query = type(api_module.rag).run_query

    def _dead_embedding_endpoint(self: Any, *a: Any, **kw: Any) -> Any:
        """Fail the way an unusable dense endpoint does.

        Raises:
            EmbeddingEndpointError: Always.
        """
        raise EmbeddingEndpointError(
            "Dense embedding failed against http://embed-only:8000 (model=BAAI/bge-m3): "
            "Error code: 404. EMBED_API_BASE must end in /v1."
        )

    monkeypatch.setattr(type(api_module.rag), "run_query", _dead_embedding_endpoint)
    try:
        with client.stream(
            "POST",
            "/stream_query",
            json={"question": "hello", "retrieval_mode": "stateless"},
        ) as resp:
            text = "".join(chunk.decode() for chunk in resp.iter_raw())
    finally:
        monkeypatch.setattr(type(api_module.rag), "run_query", original_run_query)

    assert '"code": "embedding_unavailable"' in text
    # The diagnosis belongs in the logs: it carries the internal address.
    assert "embed-only" not in text
    assert "EMBED_API_BASE" not in text


def test_query_accepts_a_multi_field_date_filter(client: TestClient) -> None:
    """A rule ORing both timestamp keys must pass wire validation."""
    response = client.post(
        "/query",
        json={
            "question": "anything",
            "metadata_filters": [
                {
                    "fields": [
                        "reference_metadata.timestamp",
                        "reference_metadata.posting_timestamp",
                    ],
                    "operator": "date_on_or_after",
                    "value": "2026-01-01",
                }
            ],
        },
    )

    assert response.status_code != 422


def test_query_rejects_a_filter_naming_no_field(client: TestClient) -> None:
    """A rule with neither ``field`` nor ``fields`` cannot be honoured."""
    response = client.post(
        "/query",
        json={
            "question": "anything",
            "metadata_filters": [{"operator": "eq", "value": "x"}],
        },
    )

    assert response.status_code == 422


def test_query_rejects_a_non_numeric_range_bound(client: TestClient) -> None:
    """Qdrant has no string range, so such a rule can only be refused.

    Accepting it would compile to nothing on every path and run the query
    unfiltered — the caller asked to narrow and would silently get everything.
    """
    response = client.post(
        "/query",
        json={
            "question": "anything",
            "metadata_filters": [{"field": "section_path", "operator": "gte", "value": "chapter-two"}],
        },
    )

    assert response.status_code == 422


def test_query_accepts_a_numeric_range_bound_sent_as_a_string(client: TestClient) -> None:
    """A text input has no way to send a JSON number; "3" must still work."""
    response = client.post(
        "/query",
        json={
            "question": "anything",
            "metadata_filters": [{"field": "page_number", "operator": "gte", "value": "3"}],
        },
    )

    assert response.status_code != 422


def test_search_returns_hits_for_the_scoped_collection(client: TestClient) -> None:
    """The endpoint returns the RAG layer's hits under the owner's collection."""
    response = client.post("/search", json={"question": "berlin konferenz"})

    assert response.status_code == 200
    body = response.json()
    assert body["status"] in {"ok", "partial", "not_indexed"}
    assert isinstance(body["hits"], list)


def test_search_silently_drops_a_short_keyword_when_others_remain(client: TestClient) -> None:
    """A short word like 'a' is unindexable but valid inside a phrase.

    It is dropped from the Qdrant pre-filter; the phrase post-filter still
    checks the full query text. The request must not be rejected.
    """
    response = client.post("/search", json={"question": "berlin a"})

    assert response.status_code != 422


def test_search_rejects_a_query_of_only_short_keywords(client: TestClient) -> None:
    """When every keyword is too short for the index, the result is empty."""
    response = client.post("/search", json={"question": "a b"})

    assert response.status_code == 422


def test_search_requires_a_query(client: TestClient) -> None:
    """A keyword-less search must not become a scan of the whole collection."""
    response = client.post("/search", json={"question": "   "})

    assert response.status_code == 422


def test_search_aggregate_returns_groups_for_the_scoped_collection(client: TestClient) -> None:
    """The endpoint returns the RAG layer's groups under the owner's collection."""
    response = client.post("/search/aggregate", json={"question": "election", "group_by": "author"})
    assert response.status_code == 200
    body = response.json()
    assert body["group_by"] == "author"
    assert body["groups"][0] == {"value": "acme_news", "count": 2, "samples": []}
    assert body["unassigned"] == 0
    assert body["limit"] == 100


def test_search_aggregate_accepts_a_blank_query(client: TestClient) -> None:
    """Grouping the whole collection is a legitimate ask (a facet, not a scan)."""
    response = client.post("/search/aggregate", json={"question": "  ", "group_by": "network"})
    assert response.status_code == 200


def test_search_aggregate_rejects_a_keyword_below_the_index_minimum(client: TestClient) -> None:
    """An unindexable keyword can never match, so it must be refused."""
    response = client.post("/search/aggregate", json={"question": "election a", "group_by": "author"})
    assert response.status_code == 422


def test_search_aggregate_rejects_an_unknown_group_field(client: TestClient) -> None:
    """Faceting is a closed whitelist — an unlisted field is refused, not passed through."""
    response = client.post("/search/aggregate", json={"question": "election", "group_by": "reference_metadata.author"})
    assert response.status_code == 422


def test_search_aggregate_forwards_sizing(client: TestClient) -> None:
    """The group and sample limits reach the RAG layer unchanged.

    The effective ``limit`` also comes back in the response, so the frontend
    can compare it against ``groups.length`` instead of assuming the default.
    """
    response = client.post(
        "/search/aggregate",
        json={"question": "election", "group_by": "author", "limit_groups": 7, "samples_per_group": 3},
    )
    last_aggregate = cast(DummyRAG, api_module.rag).last_aggregate
    assert last_aggregate["limit_groups"] == 7
    assert last_aggregate["samples_per_group"] == 3
    assert response.json()["limit"] == 7


def test_scope_can_be_set_and_read_back(client: TestClient) -> None:
    """A scope survives the round trip so a reload restores it."""
    response = client.put("/sessions/s1/scope", json={"chunk_ids": ["c1", "c2"]})

    assert response.status_code == 200
    assert response.json()["chunk_ids"] == ["c1", "c2"]


def test_scope_that_exceeds_the_context_budget_is_refused(client: TestClient) -> None:
    """Refusing is the only honest option — truncating hides lost evidence.

    Scoped answering splices the chunks straight into the prompt, so an
    oversize selection cannot be honoured. Silently dropping some would produce
    an answer that looks complete and is not.
    """
    rag = cast(DummyRAG, api_module.rag)
    rag.scope_fits = False
    try:
        response = client.put("/sessions/s1/scope", json={"chunk_ids": ["c1"]})
    finally:
        rag.scope_fits = True

    assert response.status_code == 422


def test_scope_can_be_cleared(client: TestClient) -> None:
    """Clearing returns the session to normal retrieval."""
    client.put("/sessions/s1/scope", json={"chunk_ids": ["c1"]})

    response = client.delete("/sessions/s1/scope")

    assert response.status_code == 200
    assert response.json()["chunk_ids"] == []


def test_scope_on_an_unowned_session_is_not_found(client: TestClient) -> None:
    """A session that is missing or another owner's must look identical."""
    sessions = cast(DummyRAG, api_module.rag).sessions
    sessions.scope_owned = False
    try:
        response = client.put("/sessions/other/scope", json={"chunk_ids": ["c1"]})
    finally:
        sessions.scope_owned = True

    assert response.status_code == 404


def test_chunk_endpoint_returns_the_full_text(client: TestClient) -> None:
    """Expanding a hit needs the whole chunk, which search does not carry.

    ``preview`` is capped, and returning full text for every hit would inflate
    each search by an order of magnitude for something most hits never need —
    so it is fetched on demand.
    """
    response = client.get("/search/chunk", params={"id": "c1"})

    assert response.status_code == 200
    body = response.json()
    assert body["id"] == "c1"
    assert body["text"] == "the whole chunk text"


def test_chunk_endpoint_404s_for_an_unknown_chunk(client: TestClient) -> None:
    """A chunk that is gone must not read as an empty one."""
    response = client.get("/search/chunk", params={"id": "gone"})

    assert response.status_code == 404
