"""FastAPI app exposing chat, ingestion, collection, and citation endpoints."""

import asyncio
import functools
import io
import json
import zipfile
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Literal, cast

from anyio import to_thread
from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    UploadFile,
)
from fastapi.responses import FileResponse, Response, StreamingResponse
from loguru import logger
from pydantic import BaseModel, Field
from qdrant_client import models
from starlette.middleware.cors import CORSMiddleware

from docint import __version__
from docint.agents import (
    AgentOrchestrator,
    ClarificationConfig,
    ClarificationPolicy,
    ContextualUnderstandingAgent,
    RAGRetrievalAgent,
    ResultValidationResponseAgent,
    RetrievalResult,
    SimpleClarificationAgent,
    SimpleUnderstandingAgent,
    Turn,
)
from docint.agents.history import build_prior_turn
from docint.cli import ingest as ingest_module
from docint.core.auth.principal import resolve_principal
from docint.core.rag import RAG, EmptyIngestionError
from docint.core.retrieval_filters import build_metadata_filters, build_qdrant_filter
from docint.core.state.session_manager import SessionCollectionMismatchError
from docint.utils.cursor import InvalidCursorError
from docint.utils.env_cfg import (
    load_frontend_env,
    load_host_env,
    load_path_env,
    load_response_validation_env,
)
from docint.utils.hashing import compute_file_hash
from docint.utils.logger_cfg import init_logger
from docint.utils.translate_client import translate

# Names re-exported for test monkey-patching. pyrefly treats these as
# private re-exports without an explicit ``__all__``.
__all__ = [
    "RAG",
    "ClarificationConfig",
    "ClarificationPolicy",
    "EmptyIngestionError",
    "asyncio",
    "ingest_module",
]

init_logger()

# CORS allowlist for the Vite dev server during local development.
allowed_origins = load_host_env().cors_allowed_origins.split(",")

app = FastAPI(title="Document Intelligence")
app.add_middleware(
    middleware_class=cast(Any, CORSMiddleware),
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

rag = RAG(qdrant_collection="")
SIMULATED_STREAM_TOKEN_DELAY_SECONDS = 0.03
# Interval (seconds) between checks for client disconnect while the SSE
# ingestion stream is otherwise idle. Shorter values notice disconnects
# faster at a tiny CPU cost; longer values are cheaper. 1 s is a
# compromise. Tests may monkeypatch this to a smaller value.
INGEST_DISCONNECT_POLL_INTERVAL_S = 1.0
# Interval (seconds) between client-disconnect checks while a blocking sync
# generator (chat/summary streaming) is being drained on a worker thread.
# Mirrors INGEST_DISCONNECT_POLL_INTERVAL_S; tests may monkeypatch it small.
STREAM_DISCONNECT_POLL_INTERVAL_S = 1.0

# Agent components (kept lightweight; swap with richer agents as needed)
_understanding_agent = SimpleUnderstandingAgent()
_clarification_agent = SimpleClarificationAgent()
_clarification_policy = ClarificationPolicy(ClarificationConfig())


def _select_understanding_agent() -> SimpleUnderstandingAgent | ContextualUnderstandingAgent:
    """Return the history-aware contextual understanding agent when an LLM is configured.

    Shared by ``_build_orchestrator`` (non-streaming ``/agent/chat``) and
    ``agent_chat_stream`` so both paths run identical, history-aware intent
    analysis and query rewriting. Falls back to the keyword-based simple agent
    when no LLM is configured.

    Returns:
        ContextualUnderstandingAgent bound to ``rag.text_model`` when available,
        otherwise the module-level simple agent.
    """
    if getattr(rag, "text_model_id", None):
        try:
            return ContextualUnderstandingAgent(llm=rag.text_model)
        except Exception as e:
            logger.warning("Failed to init ContextualUnderstandingAgent: {}", e)
    return _understanding_agent


def _build_orchestrator() -> AgentOrchestrator:
    """Construct an orchestrator bound to the current RAG instance.

    Returns:
        AgentOrchestrator: The constructed agent orchestrator.
    """
    retrieval_agent = RAGRetrievalAgent(rag)
    understanding = _select_understanding_agent()
    validation_cfg = load_response_validation_env()
    validation_llm = rag.text_model if isinstance(understanding, ContextualUnderstandingAgent) else None

    return AgentOrchestrator(
        understanding=understanding,
        clarifier=_clarification_agent,
        retriever=retrieval_agent,
        responder=ResultValidationResponseAgent(
            enabled=validation_cfg.enabled,
            llm=validation_llm,
        ),
        policy=_clarification_policy,
    )


# --- Helper Functions ---


def _resolve_data_dir() -> Path:
    """Return the configured data directory for ingestion.

    Returns:
        Path: The path to the data directory.
    """
    return load_path_env().data


def _require_active_collection() -> str:
    """Return the active collection name, asserting it still exists in Qdrant.

    Guards against two desync modes between the API singleton and Qdrant:

    * The singleton has no active collection (typical first-request state) —
      returns HTTP 400 so the UI can prompt the user to select one.
    * The singleton's active collection has been deleted out-of-band (e.g.,
      Qdrant volume reset, or a stale ``rag.qdrant_collection`` from before
      ``delete_collection`` started clearing the singleton) — returns HTTP
      404 with a clear message instead of letting the next query leak
      Qdrant's raw "Collection X doesn't exist" 404 to the user.

    Returns:
        str: The active collection name (already validated).

    Raises:
        HTTPException: 400 if no collection is selected, 404 if the active
            collection no longer exists in Qdrant.
    """
    name = rag.qdrant_collection
    if not name:
        raise HTTPException(status_code=400, detail="No collection selected")
    if name not in rag.list_collections():
        logger.warning(
            "Active collection '{}' is missing from Qdrant; resetting singleton.",
            name,
        )
        rag.qdrant_collection = ""
        rag.index = None
        rag.query_engine = None
        raise HTTPException(
            status_code=404,
            detail=(f"Collection '{name}' no longer exists. Please select another collection."),
        )
    return name


def _require_owned_collection(logical_name: str, principal: str) -> str:
    """Resolve a caller-owned logical collection to its physical Qdrant name.

    The single ownership gate for collection-scoped endpoints. It mirrors
    :func:`_get_owned_report`: a collection the caller does not own (or that
    does not exist) is indistinguishable from "not found" (HTTP 404), so one
    user's collection names never leak to another.

    Args:
        logical_name (str): The user-visible collection name from the request.
        principal (str): The resolved calling principal.

    Returns:
        str: The physical (owner-namespaced) Qdrant collection name to use.

    Raises:
        HTTPException: 400 if the name is blank; 404 if the caller does not own it.
    """
    name = (logical_name or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="Collection name required")
    physical = rag.ensure_collection_owner_manager().resolve(principal, name)
    if physical is None:
        raise HTTPException(status_code=404, detail=f"Collection '{name}' not found")
    return physical


def _resolve_request_collection(collection: str | None, principal: str) -> str:
    """Resolve a collection-scoped request to its physical Qdrant name.

    The single resolver for the read/query and analysis/export endpoints. When
    the caller supplies an explicit logical ``collection`` it is owner-gated via
    :func:`_require_owned_collection` (404 when not owned or missing) and its
    physical name returned. When omitted, it falls back to the process-default
    active collection (validated by :func:`_require_active_collection`).

    Clients should pass ``collection`` explicitly — it is the only owner-gated,
    concurrency-safe path. The fallback exists for single-collection CLI-style
    use and pre-multi-tenant clients; it reads the process default and is not
    owner-scoped, so it returns nothing useful once ``/collections/select`` no
    longer mutates global state (real multi-user deployments always pass it).

    Args:
        collection (str | None): The caller's logical collection name, if any.
        principal (str): The resolved calling principal.

    Returns:
        str: The physical (owner-namespaced) Qdrant collection name.

    Raises:
        HTTPException: 400 when neither a collection nor a default is available;
            404 when the caller does not own the named collection.
    """
    if collection:
        return _require_owned_collection(collection, principal)
    return _require_active_collection()


@contextmanager
def _scoped_collection(collection: str | None, principal: str) -> Iterator[str]:
    """Resolve + owner-gate a request collection and bind it for the engine.

    Combines :func:`_resolve_request_collection` with
    :meth:`docint.core.rag.RAG.collection_scope` so every ``rag`` call inside
    the block (and any anyio worker thread it spawns) reads the request's own
    physical collection rather than a shared global. Use this for synchronous,
    non-streaming endpoints; streaming endpoints must open the scope *inside*
    their event generator so it stays active while the body is consumed.

    Args:
        collection (str | None): The caller's logical collection name, if any.
        principal (str): The resolved calling principal.

    Yields:
        str: The resolved physical collection name.
    """
    physical = _resolve_request_collection(collection, principal)
    with rag.collection_scope(physical):
        yield physical


def _resolve_qdrant_src_dir() -> Path:
    """Return the configured Qdrant sources directory (separate from collections).

    Returns:
        Path: The path to the Qdrant sources directory.

    Raises:
        RuntimeError: If the Qdrant sources directory is not configured.
    """
    path_config = load_path_env()
    if path_config.qdrant_sources is None:
        raise RuntimeError("Qdrant sources directory is not configured")
    return path_config.qdrant_sources


def _safe_relative_dest(batch_dir: Path, raw_name: str) -> Path:
    """Resolve an uploaded file's relative path safely under ``batch_dir``.

    Preserves subdirectories from a browser folder upload (the
    ``webkitRelativePath`` sent as the multipart filename) while neutralizing
    path traversal: backslashes are normalized to ``/`` and empty, ``.`` and
    ``..`` segments are dropped, so the result can never escape ``batch_dir``.

    Args:
        batch_dir (Path): The collection's upload directory (containment root).
        raw_name (str): Client-supplied name, possibly a relative path.

    Returns:
        Path: A path strictly inside ``batch_dir``.
    """
    raw = (raw_name or "upload").replace("\\", "/")
    parts = [segment for segment in raw.split("/") if segment not in ("", ".", "..")]
    if not parts:
        parts = ["upload"]
    return batch_dir.joinpath(*parts)


def _resolve_source_file_path(
    collection: str,
    file_hash: str,
    *,
    filename_hint: str | None = None,
) -> Path | None:
    """Resolve a ``(collection, file_hash)`` pair to an on-disk source file.

    Mirrors the lookup chain used by :func:`preview_source`: scroll Qdrant
    for a matching payload to recover the original ``file_path``, then fall
    back to the data directory and the ``qdrant-sources`` mount under
    ``<collection>/<filename>``. ``filename_hint`` lets callers (e.g. the
    session ZIP endpoint) skip the Qdrant scroll when the citation row
    already carries the filename.
    """
    file_path_str: str | None = None
    try:
        points, _ = rag.qdrant_client.scroll(
            collection_name=collection,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="file_hash",
                        match=models.MatchValue(value=file_hash),
                    )
                ]
            ),
            limit=1,
            with_payload=True,
        )
        if not points:
            points, _ = rag.qdrant_client.scroll(
                collection_name=collection,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="metadata.file_hash",
                            match=models.MatchValue(value=file_hash),
                        )
                    ]
                ),
                limit=1,
                with_payload=True,
            )
        if points:
            payload = points[0].payload or {}
            file_path_str = (
                payload.get("file_path")
                or payload.get("path")
                or (payload.get("metadata") or {}).get("file_path")
                or (payload.get("origin") or {}).get("file_path")
            )
    except Exception as exc:
        logger.warning("Failed to resolve source file for {}/{}: {}", collection, file_hash, exc)

    candidates: list[Path] = []
    filename: str | None
    if file_path_str:
        candidates.append(Path(file_path_str))
        filename = Path(file_path_str).name
    else:
        filename = filename_hint

    if filename:
        try:
            candidates.append(_resolve_data_dir() / filename)
        except Exception:
            pass
        try:
            candidates.append(_resolve_qdrant_src_dir() / collection / filename)
        except Exception:
            pass

    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def _format_sse(event: str, data: dict[str, Any]) -> str:
    """Return a serialized Server-Sent Event payload.

    Args:
        event (str): The event type.
        data (dict[str, Any]): The event data.

    Returns:
        str: The formatted SSE string.
    """
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _validation_payload(
    *,
    question: str,
    answer: str | None,
    sources: list[dict[str, Any]],
    summary_diagnostics: dict[str, Any] | None = None,
    retrieval_query: str | None = None,
    rewritten_query: str | None = None,
    intent: str | None = None,
    tool_used: str | None = None,
) -> dict[str, bool | str | None]:
    """Validate a response against retrieved sources and return metadata.

    Args:
        question (str): The user query or summarize prompt.
        answer (str | None): The generated answer text.
        sources (list[dict[str, Any]]): Retrieved source payloads.
        summary_diagnostics (dict[str, Any] | None): Optional summary coverage diagnostics.
        retrieval_query (str | None): Query actually used for retrieval (after any rewrite/expansion).
        rewritten_query (str | None): Rewritten query from the understanding agent, if any.
        intent (str | None): Detected intent label, if any.
        tool_used (str | None): Retrieval tool that produced the sources, if any.

    Returns:
        dict[str, bool | str | None]: Validation metadata dictionary suitable for API responses.
    """
    validation_cfg = load_response_validation_env()
    validation_llm = None
    if getattr(rag, "text_model_id", None):
        try:
            validation_llm = rag.text_model
        except Exception as exc:
            logger.warning("Failed to initialize validation LLM: {}", exc)

    validator = ResultValidationResponseAgent(
        enabled=validation_cfg.enabled,
        llm=validation_llm,
    )
    retrieval = RetrievalResult(
        answer=answer,
        sources=sources,
        summary_diagnostics=summary_diagnostics,
        retrieval_query=retrieval_query,
        rewritten_query=rewritten_query,
        intent=intent,
        tool_used=tool_used,
    )
    validated = validator.finalize(retrieval, Turn(user_input=question))
    return {
        "validation_checked": validated.validation_checked,
        "validation_mismatch": validated.validation_mismatch,
        "validation_reason": validated.validation_reason,
    }


def _iter_text_tokens(text: str) -> list[str]:
    """Split text into whitespace-preserving token chunks for SSE streaming.

    Args:
        text (str): The text to chunk.

    Returns:
        list[str]: Token chunks suitable for incremental UI rendering.
    """
    if not text:
        return []
    return [chunk for chunk in text.split(" ") if chunk] if " " in text else [text]


async def _stream_simulated_text(answer_text: str) -> AsyncIterator[str]:
    """Yield SSE token events for already-generated answers with visible pacing.

    Args:
        answer_text (str): Full answer text that must be replayed as a token stream.

    Yields:
        SSE ``data:`` lines for each token-sized chunk.
    """
    for token in _iter_text_tokens(answer_text):
        yield f"data: {json.dumps({'token': token + ' '})}\n\n"
        await asyncio.sleep(SIMULATED_STREAM_TOKEN_DELAY_SECONDS)


async def _aiter_sync_gen(
    gen_factory: Callable[[], Iterator[Any]],
    request: Request | None = None,
) -> AsyncIterator[Any]:
    """Drive a blocking sync generator on a worker thread, yielding its items.

    The generator is built and fully iterated inside ``to_thread.run_sync`` so
    neither construction nor any ``next()`` (query rewrite, embedding, Qdrant
    search, rerank, LLM streaming) runs on the asyncio event loop — that keeps
    the loop free to serve concurrent requests. Items cross back to the loop
    via a thread-safe queue: ``None`` signals normal completion and an
    ``Exception`` instance is re-raised on the loop. Mirrors the thread-bridge
    used by ``/ingest/upload``.

    Args:
        gen_factory (Callable[[], Iterator[Any]]): Zero-arg callable returning
            the blocking generator. A factory (not the generator itself) is
            used so construction also happens off the loop.
        request (Request | None): Optional request polled for client
            disconnect; when disconnected the worker-awaiter is cancelled and
            iteration stops. The worker thread cannot be force-killed, so it
            runs to completion and its remaining output is discarded.

    Yields:
        Any: Each item produced by the generator, in order.
    """
    queue: asyncio.Queue[Any] = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def _safe_put(item: Any) -> None:
        """Enqueue an item from the worker thread, tolerating a closed loop."""
        try:
            loop.call_soon_threadsafe(queue.put_nowait, item)
        except Exception as exc:
            # The loop may be gone after a client disconnect/teardown; log
            # rather than letting the worker-thread exception vanish.
            logger.warning("Could not enqueue stream item (loop unavailable): {}", exc)

    def _pump() -> None:
        """Iterate the blocking generator, forwarding items then a sentinel."""
        try:
            for item in gen_factory():
                _safe_put(item)
            _safe_put(None)
        except Exception as exc:  # surface to the loop, then stop
            _safe_put(exc)

    task = asyncio.create_task(to_thread.run_sync(_pump))
    try:
        while True:
            try:
                item = await asyncio.wait_for(
                    queue.get(),
                    timeout=STREAM_DISCONNECT_POLL_INTERVAL_S,
                )
            except TimeoutError:
                if request is not None and await request.is_disconnected():
                    return
                continue
            if item is None:
                break
            if isinstance(item, Exception):
                raise item
            yield item
    finally:
        if not task.done():
            task.cancel()


# --- Pydantic models for request and response payloads ---


class SelectCollectionIn(BaseModel):
    """Request payload selecting the active Qdrant collection for a session."""

    name: str


class SelectCollectionOut(BaseModel):
    """Response confirming the active collection selection."""

    ok: bool
    name: str


class MetadataFilterIn(BaseModel):
    """Single metadata filter applied to retrieval queries."""

    field: str
    operator: Literal[
        "eq",
        "neq",
        "gt",
        "gte",
        "lt",
        "lte",
        "in",
        "contains",
        "mime_match",
        "date_after",
        "date_on_or_after",
        "date_before",
        "date_on_or_before",
    ]
    value: str | int | float | bool | None = None
    values: list[str | int | float | bool] = Field(default_factory=list)


class QueryIn(BaseModel):
    """Request payload for a single RAG query."""

    question: str
    session_id: str | None = None
    # Caller's *logical* collection name. When provided it is owner-gated and
    # resolved to the per-request physical collection, so concurrent queries on
    # different collections never interfere. When omitted, the server falls back
    # to its process-default active collection (legacy single-collection use).
    collection: str | None = None
    metadata_filters: list[MetadataFilterIn] = Field(default_factory=list)
    retrieval_mode: Literal["session", "stateless"] = "session"
    query_mode: Literal["answer", "entity_occurrence", "entity_occurrence_multi"] = "answer"


class QueryOut(BaseModel):
    """Grounded answer plus retrieval provenance for a RAG query."""

    answer: str
    sources: list[dict[str, Any]] = []
    session_id: str
    graph_debug: dict[str, Any] | None = None
    retrieval_query: str | None = None
    coverage_unit: str | None = None
    retrieval_mode: str | None = None
    entity_match_candidates: list[dict[str, Any]] = []
    entity_match_groups: list[dict[str, Any]] = []
    validation_checked: bool | None = None
    validation_mismatch: bool | None = None
    validation_reason: str | None = None


class SummaryDiagnosticsOut(BaseModel):
    """Diagnostics describing coverage and sampling for a summary response."""

    total_documents: int
    covered_documents: int
    coverage_ratio: float
    uncovered_documents: list[str] = []
    coverage_target: float
    coverage_unit: str | None = None
    candidate_count: int | None = None
    deduped_count: int | None = None
    sampled_count: int | None = None


class SummarizeOut(BaseModel):
    """Response payload for a collection-level summary request."""

    summary: str
    sources: list[dict[str, Any]] = []
    summary_diagnostics: SummaryDiagnosticsOut | None = None
    validation_checked: bool | None = None
    validation_mismatch: bool | None = None
    validation_reason: str | None = None


class IngestIn(BaseModel):
    """Request payload triggering ingestion into a named collection."""

    collection: str
    hybrid: bool | None = True


class IngestOut(BaseModel):
    """Response confirming ingestion and reporting its configuration."""

    ok: bool
    collection: str
    data_dir: str
    hybrid: bool
    empty: bool = False


class SessionListOut(BaseModel):
    """List of sessions visible to the caller."""

    sessions: list[dict[str, Any]]


class SessionHistoryOut(BaseModel):
    """Ordered history of messages for a single session."""

    messages: list[dict[str, Any]]


class NERStatsOut(BaseModel):
    """Aggregate statistics over extracted entities and relations."""

    totals: dict[str, int]
    top_entities: list[dict[str, Any]] = []
    entity_types: list[dict[str, Any]] = []
    top_relations: list[dict[str, Any]] = []
    documents: list[dict[str, Any]] = []


class NERSearchOut(BaseModel):
    """Matching entities returned from a NER search query."""

    results: list[dict[str, Any]] = []


class NERGraphOut(BaseModel):
    """Derived entity graph (nodes + edges) for interactive exploration."""

    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    meta: dict[str, int] = {}


class FrontendConfigOut(BaseModel):
    """Deploy-time frontend configuration served to the SPA."""

    graph_top_k: int
    graph_max_top_k: int
    collection_timeout: int


class FileTypeCount(BaseModel):
    """One file-type tally in a collection's document summary."""

    label: str
    count: int


class DocumentsSummaryOut(BaseModel):
    """Collection-wide document aggregates for the Inspector KPI strip."""

    document_count: int
    node_count: int
    file_types: list[FileTypeCount]
    entity_types: list[str]


class VersionOut(BaseModel):
    """App release version."""

    version: str


class AgentChatIn(BaseModel):
    """Request payload for a single agent chat turn."""

    message: str
    session_id: str | None = None
    # Caller's *logical* collection name; owner-gated and resolved to the
    # per-request physical collection. Falls back to the process default when
    # omitted (legacy single-collection use).
    collection: str | None = None


class AgentChatOut(BaseModel):
    """Response payload for an agent chat turn; either a clarification or answer."""

    status: Literal["clarification", "answer"]
    message: str | None = None
    answer: str | None = None
    sources: list[dict[str, Any]] = []
    session_id: str | None = None
    reason: str | None = None
    intent: str | None = None
    confidence: float | None = None
    tool_used: str | None = None
    latency_ms: float | None = None
    validation_checked: bool | None = None
    validation_mismatch: bool | None = None
    validation_reason: str | None = None


class ReportCreateIn(BaseModel):
    """Request payload creating a new (empty) report."""

    title: str
    collection_name: str | None = None
    operator: str | None = None
    reference_number: str | None = None
    session_id: str | None = None


class ReportUpdateIn(BaseModel):
    """Request payload updating a report's title or case metadata."""

    title: str | None = None
    operator: str | None = None
    reference_number: str | None = None
    show_toc: bool | None = None
    show_collection_overview: bool | None = None


class ReportItemIn(BaseModel):
    """Request payload adding one snapshotted artifact to a report."""

    artifact_type: str
    dedupe_key: str
    snapshot: dict[str, Any]
    note: str | None = None


class TranslateIn(BaseModel):
    """Request payload for on-demand snippet translation."""

    text: str


class ReportItemNoteIn(BaseModel):
    """Request payload setting or clearing an item note."""

    note: str | None = None


class ReportReorderIn(BaseModel):
    """Request payload reordering a report's items."""

    item_ids: list[int]


class ReportListOut(BaseModel):
    """List of reports visible to the caller."""

    reports: list[dict[str, Any]]


# --- API Endpoints ---


@app.get("/config", response_model=FrontendConfigOut, tags=["Meta"])
def get_frontend_config() -> dict[str, int]:
    """Return deploy-time frontend configuration for the SPA.

    Served without a principal dependency so the SPA can read it on first load,
    before any collection or session exists. Values are read from environment
    variables on each call (see :func:`docint.utils.env_cfg.load_frontend_env`).

    Returns:
        dict[str, int]: ``graph_top_k``, ``graph_max_top_k`` and
        ``collection_timeout``.
    """
    cfg = load_frontend_env()
    return {
        "graph_top_k": cfg.graph_top_k,
        "graph_max_top_k": cfg.graph_max_top_k,
        "collection_timeout": cfg.collection_timeout,
    }


@app.get("/version", response_model=VersionOut, tags=["Meta"])
def get_version() -> VersionOut:
    """Return the running app version (unauthenticated, no principal)."""
    return VersionOut(version=__version__)


@app.get("/collections/list", response_model=list[str], tags=["Collections"])
def collections_list(principal: str = Depends(resolve_principal)) -> list[str]:
    """List the calling principal's collections (logical names).

    Collections are owner-scoped: a caller only sees the collections they
    ingested themselves. Names are the user-visible logical names, not the
    owner-namespaced physical Qdrant names.

    Args:
        principal (str): The resolved request principal.

    Returns:
        list[str]: The caller's collection names, sorted.

    Raises:
        HTTPException: If an error occurs while listing collections.
    """
    try:
        return rag.ensure_collection_owner_manager().list_for(principal)
    except Exception as e:
        logger.error("HTTPException: Error listing collections: {}", e)
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/collections/select", response_model=SelectCollectionOut, tags=["Collections"])
def collections_select(
    payload: SelectCollectionIn, principal: str = Depends(resolve_principal)
) -> dict[str, bool | str]:
    """Validate that the caller owns a collection (non-mutating).

    This endpoint no longer changes any server-side state: selection is purely a
    client concern (the SPA keeps the chosen collection locally and sends it on
    each request via the ``collection`` field). It exists only as an ownership
    check — 200 with the name when the caller owns it, 404 otherwise — so a UI
    can confirm a selection without leaking another user's collections. Making
    it stateless is what allows concurrent users on different collections to
    stop clobbering each other (the WS2 fix).

    Args:
        payload (SelectCollectionIn): The payload containing the collection name.
        principal (str): The resolved request principal.

    Returns:
        dict[str, bool | str]: ``{"ok": True, "name": <logical>}`` when owned.

    Raises:
        HTTPException: 400 if the collection name is missing, 404 if the caller
            does not own it.
    """
    name = payload.name.strip()
    _require_owned_collection(name, principal)
    return {"ok": True, "name": name}


@app.delete("/collections/{name}", tags=["Collections"])
def collections_delete(name: str, principal: str = Depends(resolve_principal)) -> dict[str, bool]:
    """Delete a collection the caller owns.

    Deleting a collection the caller does not own (or one that does not exist)
    is a 404, so a user can never delete another user's data. The Qdrant
    collection is dropped first; only then is the ownership mapping removed, so
    a failed Qdrant delete leaves ownership intact for retry.

    Args:
        name (str): The user-visible collection name to delete.
        principal (str): The resolved request principal.

    Returns:
        dict[str, bool]: A dictionary indicating success.

    Raises:
        HTTPException: 404 if the caller does not own it; 500 on backend failure.
    """
    physical = _require_owned_collection(name, principal)
    try:
        rag.delete_collection(physical)
        rag.ensure_collection_owner_manager().delete(principal, name)
        return {"ok": True}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("HTTPException: Error deleting collection: {}", e)
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/query", response_model=QueryOut, tags=["Query"])
def query(payload: QueryIn, request: Request) -> dict[str, Any]:
    """Handle a query request.

    Args:
        payload (QueryIn): The query payload containing the question and session ID.
        request (Request): The incoming request used to resolve the calling principal
            for session-backed chats.

    Returns:
        QueryOut: The query response containing the answer, sources, and session ID.

    Raises:
        HTTPException: If an error occurs while processing the query.
    """
    try:
        principal = resolve_principal(request)
        physical = _resolve_request_collection(payload.collection, principal)

        metadata_filters = build_metadata_filters(payload.metadata_filters)
        vector_store_kwargs = {}
        qdrant_filter = build_qdrant_filter(payload.metadata_filters)
        if qdrant_filter is not None:
            vector_store_kwargs["qdrant_filters"] = qdrant_filter

        with rag.collection_scope(physical):
            if payload.query_mode in {"entity_occurrence", "entity_occurrence_multi"}:
                if payload.query_mode == "entity_occurrence_multi":
                    data = rag.run_multi_entity_occurrence_query(
                        payload.question,
                        qdrant_filter=qdrant_filter,
                    )
                else:
                    data = rag.run_entity_occurrence_query(
                        payload.question,
                        qdrant_filter=qdrant_filter,
                    )
                session_id = payload.session_id or "stateless"
            else:
                if getattr(rag, "query_engine", None) is None:
                    if getattr(rag, "index", None) is None:
                        rag.create_index()
                    rag.create_query_engine()

                if payload.retrieval_mode == "stateless":
                    retrieval_query = payload.question
                    graph_debug: dict[str, Any] | None = None
                    expand_with_debug = getattr(rag, "expand_query_with_graph_with_debug", None)
                    if callable(expand_with_debug):
                        try:
                            expanded, debug_payload = cast("tuple[Any, Any]", expand_with_debug(retrieval_query))
                            retrieval_query = str(expanded)
                            if isinstance(debug_payload, dict):
                                graph_debug = debug_payload
                        except Exception as exc:
                            logger.warning(
                                "Graph debug expansion failed for stateless query: {}",
                                exc,
                            )

                    data = rag.run_query(
                        retrieval_query,
                        metadata_filters=metadata_filters,
                        metadata_filter_rules=payload.metadata_filters,
                        vector_store_kwargs=vector_store_kwargs or None,
                    )
                    if graph_debug is not None:
                        data["graph_debug"] = graph_debug
                    session_id = payload.session_id or "stateless"
                else:
                    session_id = rag.start_session(
                        payload.session_id,
                        owner=principal,
                    )
                    data = rag.chat(
                        payload.question,
                        session_id=session_id,
                        owner=principal,
                        metadata_filters=metadata_filters,
                        metadata_filters_active=(metadata_filters is not None or bool(vector_store_kwargs)),
                        metadata_filter_rules=payload.metadata_filters,
                        vector_store_kwargs=vector_store_kwargs or None,
                    )

        answer = str(data.get("response") or data.get("answer") or "") if isinstance(data, dict) else ""
        sources: list[dict[str, Any]] = data.get("sources", []) if isinstance(data, dict) else []
        graph_debug = (
            data.get("graph_debug") if isinstance(data, dict) and isinstance(data.get("graph_debug"), dict) else None
        )
        retrieval_query_value: str | None = (
            str(data.get("retrieval_query") or "")
            if isinstance(data, dict) and data.get("retrieval_query") is not None
            else None
        )
        coverage_unit = (
            str(data.get("coverage_unit") or "")
            if isinstance(data, dict) and data.get("coverage_unit") is not None
            else None
        )
        retrieval_mode = (
            str(data.get("retrieval_mode") or "")
            if isinstance(data, dict) and data.get("retrieval_mode") is not None
            else None
        )
        entity_match_candidates: list[Any] = (
            data.get("entity_match_candidates", [])
            if isinstance(data, dict) and isinstance(data.get("entity_match_candidates"), list)
            else []
        )
        entity_match_groups: list[Any] = (
            data.get("entity_match_groups", [])
            if isinstance(data, dict) and isinstance(data.get("entity_match_groups"), list)
            else []
        )

        summary_diagnostics_query = (
            data.get("summary_diagnostics")
            if isinstance(data, dict) and isinstance(data.get("summary_diagnostics"), dict)
            else None
        )
        # `retrieval_mode` here is the session-routing mode
        # ("session"/"stateless"), not the retrieval tool, so it is not
        # forwarded as `tool_used`. The orchestrator path populates
        # `tool_used` directly on the RetrievalResult instead.
        validation = _validation_payload(
            question=payload.question,
            answer=answer,
            sources=sources,
            summary_diagnostics=summary_diagnostics_query,
            retrieval_query=retrieval_query_value,
        )
        return {
            "answer": answer,
            "sources": sources,
            "session_id": session_id,
            "graph_debug": graph_debug,
            "retrieval_query": retrieval_query_value,
            "coverage_unit": coverage_unit,
            "retrieval_mode": retrieval_mode,
            "entity_match_candidates": entity_match_candidates,
            "entity_match_groups": entity_match_groups,
            **validation,
        }
    except HTTPException:
        raise
    except SessionCollectionMismatchError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:
        logger.error("Unexpected error processing query: {}", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/stream_query", tags=["Query"])
async def stream_query(payload: QueryIn, request: Request) -> StreamingResponse:
    """Handle a streaming query request.

    Args:
        payload (QueryIn): The query payload containing the question and session ID.
        request (Request): The incoming request used to resolve the calling principal
            for session-backed chats.

    Returns:
        StreamingResponse: A streaming response that yields SSE events during the query.

    Raises:
        HTTPException: If an error occurs while processing the streaming query.
    """
    principal = resolve_principal(request)
    physical = _resolve_request_collection(payload.collection, principal)

    metadata_filters = build_metadata_filters(payload.metadata_filters)
    vector_store_kwargs = {}
    qdrant_filter = build_qdrant_filter(payload.metadata_filters)
    if qdrant_filter is not None:
        vector_store_kwargs["qdrant_filters"] = qdrant_filter

    session_owner: str | None = None
    if (
        payload.query_mode not in {"entity_occurrence", "entity_occurrence_multi"}
        and payload.retrieval_mode != "stateless"
    ):
        session_owner = principal
        # Up-front collection-pin check so a mismatch is a clean 409 rather than
        # an in-stream SSE error: resuming an owned session against a different
        # collection must be refused before any retrieval runs.
        if payload.session_id:
            pinned = rag.ensure_session_manager().get_session_collection(payload.session_id, session_owner)
            if pinned is not None and pinned != physical:
                raise HTTPException(
                    status_code=409,
                    detail=f"Session '{payload.session_id}' is pinned to a different collection.",
                )

    async def _stream_body() -> AsyncIterator[str]:
        """Generate SSE events for the streaming query.

        Runs inside the request's :meth:`RAG.collection_scope` (opened by the
        ``event_generator`` wrapper) so every retrieval/generation call resolves
        the caller's own physical collection.

        Returns:
            AsyncIterator[str]: An asynchronous iterator yielding SSE events.

        Yields:
            Iterator[AsyncIterator[str]]: An asynchronous iterator yielding SSE events.
        """
        try:
            full_answer = ""
            final_payload: dict[str, Any] | None = None
            if payload.query_mode in {"entity_occurrence", "entity_occurrence_multi"}:
                if payload.query_mode == "entity_occurrence_multi":
                    occurrence_data = await to_thread.run_sync(
                        functools.partial(
                            rag.run_multi_entity_occurrence_query,
                            payload.question,
                            qdrant_filter=qdrant_filter,
                        )
                    )
                else:
                    occurrence_data = await to_thread.run_sync(
                        functools.partial(
                            rag.run_entity_occurrence_query,
                            payload.question,
                            qdrant_filter=qdrant_filter,
                        )
                    )
                answer_text = str(occurrence_data.get("response") or occurrence_data.get("answer") or "")
                async for event in _stream_simulated_text(answer_text):
                    event_payload = json.loads(event[6:].strip())
                    token = str(event_payload.get("token") or "")
                    full_answer += token
                    yield event

                final_payload = {
                    "answer": occurrence_data.get("response"),
                    "sources": occurrence_data.get("sources") or [],
                    "session_id": payload.session_id or "stateless",
                    "reasoning": occurrence_data.get("reasoning"),
                    "retrieval_query": occurrence_data.get("retrieval_query"),
                    "coverage_unit": occurrence_data.get("coverage_unit"),
                    "retrieval_mode": occurrence_data.get("retrieval_mode"),
                    "entity_match_candidates": occurrence_data.get("entity_match_candidates") or [],
                    "entity_match_groups": occurrence_data.get("entity_match_groups") or [],
                }
            elif payload.retrieval_mode == "stateless":
                retrieval_query = payload.question
                graph_debug: dict[str, Any] | None = None
                expand_with_debug = getattr(rag, "expand_query_with_graph_with_debug", None)
                if callable(expand_with_debug):
                    try:
                        expanded, debug_payload = cast(
                            "tuple[Any, Any]",
                            await to_thread.run_sync(expand_with_debug, retrieval_query),
                        )
                        retrieval_query = str(expanded)
                        if isinstance(debug_payload, dict):
                            graph_debug = debug_payload
                    except Exception as exc:
                        logger.warning(
                            "Graph debug expansion failed for stateless stream query: {}",
                            exc,
                        )

                stateless_data = await rag.run_query_async(
                    retrieval_query,
                    metadata_filters=metadata_filters,
                    metadata_filter_rules=payload.metadata_filters,
                    vector_store_kwargs=vector_store_kwargs or None,
                )
                if graph_debug is not None:
                    stateless_data["graph_debug"] = graph_debug

                answer_text = str(stateless_data.get("response") or stateless_data.get("answer") or "")
                async for event in _stream_simulated_text(answer_text):
                    event_payload = json.loads(event[6:].strip())
                    token = str(event_payload.get("token") or "")
                    full_answer += token
                    yield event

                final_payload = {
                    "response": answer_text,
                    "sources": stateless_data.get("sources") or [],
                    "session_id": payload.session_id or "stateless",
                    "reasoning": stateless_data.get("reasoning"),
                    "graph_debug": stateless_data.get("graph_debug"),
                }
            else:

                def _make_chat_stream() -> Iterator[Any]:
                    """Build the blocking chat stream off the event loop.

                    Session start, history load, and the retrieval/LLM stream
                    are all synchronous and so run on the worker thread driven
                    by ``_aiter_sync_gen``.

                    Returns:
                        Iterator[Any]: The sync chat-chunk generator.
                    """
                    # The React chat UI calls /stream_query, so this is where
                    # generation-time history is wired: bind the prior
                    # user/assistant exchange (owner-scoped) onto the synthesis
                    # templates while keeping this endpoint's own internal
                    # retrieval rewrite (``skip_query_rewrite=False``).
                    session_id = rag.start_session(payload.session_id, owner=session_owner)
                    prior_turn = (
                        build_prior_turn(rag.sessions.get_session_history(session_id, owner=session_owner))
                        if rag.sessions is not None
                        else None
                    )
                    return cast(
                        "Iterator[Any]",
                        rag.stream_chat(
                            payload.question,
                            session_id=session_id,
                            owner=session_owner,
                            metadata_filters=metadata_filters,
                            metadata_filters_active=(metadata_filters is not None or bool(vector_store_kwargs)),
                            metadata_filter_rules=payload.metadata_filters,
                            vector_store_kwargs=vector_store_kwargs or None,
                            prior_turn=prior_turn,
                            skip_query_rewrite=False,
                        ),
                    )

                async for chunk in _aiter_sync_gen(_make_chat_stream, request):
                    if isinstance(chunk, str):
                        full_answer += chunk
                        yield f"data: {json.dumps({'token': chunk})}\n\n"
                    elif isinstance(chunk, dict):
                        final_payload = chunk

            payload_out = dict(final_payload or {})
            answer = str(payload_out.get("response") or payload_out.get("answer") or "")
            if not answer:
                answer = full_answer
            sources = payload_out.get("sources")
            if not isinstance(sources, list):
                sources = cast(list[dict[str, Any]], [])
            stream_summary_diagnostics = (
                payload_out.get("summary_diagnostics")
                if isinstance(payload_out.get("summary_diagnostics"), dict)
                else None
            )
            stream_retrieval_query = (
                str(payload_out.get("retrieval_query")) if payload_out.get("retrieval_query") else None
            )
            # `retrieval_mode` here is the session-routing mode, not the
            # retrieval tool, so it is not forwarded as `tool_used`.
            validation = _validation_payload(
                question=payload.question,
                answer=answer,
                sources=sources,
                summary_diagnostics=stream_summary_diagnostics,
                retrieval_query=stream_retrieval_query,
            )
            payload_out.update(validation)
            # Persist validation onto the row stream_chat already wrote so
            # restored sessions see the same banner state as fresh turns.
            # turn_idx is set only by the session-mode branch in
            # session_manager.stream_chat; the stateless / entity branches
            # don't persist a turn at all and so won't carry it.
            turn_idx = payload_out.pop("turn_idx", None)
            stream_session_id = payload_out.get("session_id")
            if (
                isinstance(turn_idx, int)
                and isinstance(stream_session_id, str)
                and stream_session_id
                and rag.sessions is not None
            ):
                try:
                    rag.sessions.update_turn_validation(
                        session_id=stream_session_id,
                        turn_idx=turn_idx,
                        validation_checked=cast("bool | None", validation.get("validation_checked")),
                        validation_mismatch=cast("bool | None", validation.get("validation_mismatch")),
                        validation_reason=cast("str | None", validation.get("validation_reason")),
                    )
                except Exception as exc:
                    logger.warning(
                        "Failed to persist validation for session={} idx={}: {}",
                        stream_session_id,
                        turn_idx,
                        exc,
                    )
            if payload_out:
                yield f"data: {json.dumps(payload_out)}\n\n"
        except ValueError as exc:
            msg = str(exc)
            if "context window" in msg.lower() or "context size" in msg.lower():
                logger.warning("Context window overflow during SSE generation: {}", exc)
                yield f"data: {json.dumps({'error': msg})}\n\n"
            else:
                logger.exception("Stream error during SSE generation")
                yield f"data: {json.dumps({'error': 'Internal server error'})}\n\n"
        except Exception:
            logger.exception("Stream error during SSE generation")
            yield f"data: {json.dumps({'error': 'Internal server error'})}\n\n"

    async def event_generator() -> AsyncIterator[str]:
        """Bind the request's physical collection, then stream the body.

        The collection scope is opened here rather than in the handler so it
        stays active while Starlette consumes the generator and copies into the
        anyio worker threads spawned for retrieval/generation. Warming the index
        also happens inside the scope so the correct collection is materialized.

        Yields:
            str: SSE event lines from the scoped stream body.
        """
        with rag.collection_scope(physical):
            if (
                payload.query_mode not in {"entity_occurrence", "entity_occurrence_multi"}
                and getattr(rag, "index", None) is None
            ):
                await to_thread.run_sync(rag.create_index)
            async for chunk in _stream_body():
                yield chunk

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/summarize", response_model=SummarizeOut, tags=["Query"])
def summarize(
    refresh: bool = Query(False),
    collection: str | None = None,
    principal: str = Depends(resolve_principal),
) -> dict[str, Any]:
    """Generate a summary for the caller's collection.

    Args:
        refresh (bool): If ``True``, bypass cached collection summaries.
        collection (str | None): Caller's logical collection; owner-gated and
            scoped per request, falling back to the process default when omitted.
        principal (str): The resolved request principal.

    Returns:
        dict[str, list[dict] | str]: A dictionary containing the summary and sources.

    Raises:
        HTTPException: 400/404 from collection resolution; 500 on generation error.
    """
    with _scoped_collection(collection, principal):
        try:
            data = rag.summarize_collection(refresh=refresh)
            summary = str(data.get("response") or data.get("answer") or "") if isinstance(data, dict) else ""
            sources: list[dict[str, Any]] = data.get("sources", []) if isinstance(data, dict) else []
            summary_diagnostics = data.get("summary_diagnostics") if isinstance(data, dict) else None

            validation = _validation_payload(
                question=rag.summarize_prompt,
                answer=summary,
                sources=sources,
                summary_diagnostics=summary_diagnostics,
            )
            return {
                "summary": summary,
                "sources": sources,
                "summary_diagnostics": summary_diagnostics,
                **validation,
            }
        except HTTPException as e:
            logger.error("HTTPException: Error generating summary: {}", e)
            raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/summarize/stream", tags=["Query"])
async def summarize_stream(
    request: Request,
    refresh: bool = Query(False),
    collection: str | None = None,
    principal: str = Depends(resolve_principal),
) -> StreamingResponse:
    """Generate a streaming summary for the caller's collection.

    Args:
        request (Request): The incoming request, used to detect client
            disconnects while the blocking summary stream is drained.
        refresh (bool): If ``True``, bypass cached collection summaries.
        collection (str | None): Caller's logical collection; owner-gated and
            scoped per request, falling back to the process default when omitted.
        principal (str): The resolved request principal.

    Returns:
        StreamingResponse: A streaming response that yields SSE events during summarization.

    Raises:
        HTTPException: 400/404 from collection resolution.
    """
    physical = _resolve_request_collection(collection, principal)

    async def _summary_body() -> AsyncIterator[str]:
        """Generate SSE events for the streaming summary (inside the scope).

        Yields:
            AsyncIterator[str]: An asynchronous iterator yielding SSE events.
        """
        try:
            full_summary = ""
            final_payload: dict[str, Any] | None = None
            async for chunk in _aiter_sync_gen(lambda: rag.stream_summarize_collection(refresh=refresh), request):
                if isinstance(chunk, str):
                    full_summary += chunk
                    yield f"data: {json.dumps({'token': chunk})}\n\n"
                elif isinstance(chunk, dict):
                    final_payload = chunk

            payload_out = dict(final_payload or {})
            summary = str(payload_out.get("response") or payload_out.get("answer") or "")
            if not summary:
                summary = full_summary
            sources = payload_out.get("sources")
            if not isinstance(sources, list):
                sources = cast(list[dict[str, Any]], [])
            summary_diagnostics = payload_out.get("summary_diagnostics")
            if not isinstance(summary_diagnostics, dict):
                summary_diagnostics = None
            validation = _validation_payload(
                question=rag.summarize_prompt,
                answer=summary,
                sources=sources,
                summary_diagnostics=summary_diagnostics,
            )
            payload_out.update(validation)
            if payload_out:
                yield f"data: {json.dumps(payload_out)}\n\n"
        except Exception as e:
            logger.error("Stream error: {}", e)
            yield f"data: {json.dumps({'error': 'An internal error occurred during streaming.'})}\n\n"

    async def event_generator() -> AsyncIterator[str]:
        """Bind the request's physical collection, then stream the summary body.

        Yields:
            str: SSE event lines from the scoped summary body.
        """
        with rag.collection_scope(physical):
            async for chunk in _summary_body():
                yield chunk

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/collections/ner", tags=["Query"], deprecated=True)
def get_collection_ner(
    refresh: bool = False,
    collection: str | None = None,
    principal: str = Depends(resolve_principal),
) -> dict[str, list[dict[str, Any]]]:
    """Get all NER data (entities and relations) for the caller's collection.

    Deprecated: scrolls the entire collection in one response and is the
    pre-pagination path. Prefer ``GET /collections/ner/sources`` (paginated,
    optionally server-filtered by entity) and ``GET /collections/ner/stats``
    for the entity dropdown. Retained to keep external consumers working
    until they migrate.

    Args:
        refresh (bool): If ``True``, bypass in-memory cache and re-fetch from storage.
        collection (str | None): Caller's logical collection; owner-gated and
            scoped per request, falling back to the process default when omitted.
        principal (str): The resolved request principal.

    Returns:
        dict[str, list[dict]]: A dictionary containing the list of NER sources.

    Raises:
        HTTPException: 400/404 from collection resolution; 500 on error.
    """
    try:
        with _scoped_collection(collection, principal):
            sources = rag.get_collection_ner(refresh=refresh)
            return {"sources": sources}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error fetching collection NER: {}", e)
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/collections/hate-speech", tags=["Query"])
def get_collection_hate_speech(
    cursor: str | None = None,
    limit: int = Query(default=0, ge=0, le=500),
    category: str | None = None,
    min_confidence: str | None = None,
    collection: str | None = None,
    principal: str = Depends(resolve_principal),
) -> dict[str, Any]:
    """Return flagged hate-speech chunks for the caller's collection.

    The endpoint operates in two modes:

    * **Legacy (default)**: ``cursor`` omitted and ``limit=0`` — returns the
      full list under ``{"results": [...]}``, matching the original shape.
    * **Paginated**: any of ``cursor`` / ``limit`` / ``category`` /
      ``min_confidence`` supplied — returns ``{"items": [...], "next_cursor":
      ...}`` and uses the in-memory hate-speech cache for slicing.

    Args:
        cursor (str | None): Opaque cursor token from a previous paginated call.
        limit (int): Page size (1-500). ``0`` selects legacy mode.
        category (str | None): Optional case-insensitive category filter.
        min_confidence (str | None): Optional confidence floor (``low`` <
            ``medium`` < ``high``).
        collection (str | None): Caller's logical collection; owner-gated and
            scoped per request, falling back to the process default when omitted.
        principal (str): The resolved request principal.

    Returns:
        dict[str, Any]: Either the legacy ``{"results": ...}`` payload or a
        paginated ``{"items": ..., "next_cursor": ...}`` envelope.
    """
    paginated = cursor is not None or limit > 0 or category is not None or min_confidence is not None
    try:
        with _scoped_collection(collection, principal):
            if not paginated:
                return {"results": rag.get_collection_hate_speech()}
            items, next_cursor = rag.iter_hate_speech(
                cursor=cursor,
                limit=limit or 50,
                category=category,
                min_confidence=min_confidence,
            )
            return {"items": items, "next_cursor": next_cursor}
    except HTTPException:
        raise
    except InvalidCursorError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.error("Error fetching collection hate-speech results: {}", e)
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/collections/ner/sources", tags=["Query"])
def get_collection_ner_sources(
    cursor: str | None = None,
    limit: int = Query(default=50, ge=1, le=500),
    entity_key: str | None = None,
    entity_text: str | None = None,
    entity_type: str | None = None,
    entity_merge_mode: Literal["orthographic", "exact", "resolved"] = Query(default="orthographic"),
    collection: str | None = None,
    principal: str = Depends(resolve_principal),
) -> dict[str, Any]:
    """Return one page of NER-bearing source rows for the caller's collection.

    Always paginated — there is no full-list mode. When an entity filter is
    supplied the matcher mirrors the SPA's ``sourceContainsEntity`` (same
    exact-text and compact-lookup rules) so results align with the UI's
    client-side filter prior to pagination. ``entity_merge_mode="resolved"``
    expands the filter to the canonical entity's sibling aliases so the
    drill-down reflects the merged mention count.

    Args:
        cursor (str | None): Opaque cursor token from a previous call.
        limit (int): Records per page (1-500).
        entity_key (str | None): ``"<text>::<type>"`` shorthand (matches the
            SPA's ``Analysis.tsx`` ``keyOf``).
        entity_text (str | None): Explicit entity surface form.
        entity_type (str | None): Explicit entity type/label.
        entity_merge_mode (Literal): Clustering mode; ``"resolved"`` includes
            sibling aliases of the canonical entity.
        collection (str | None): Caller's logical collection; owner-gated and
            scoped per request, falling back to the process default when omitted.
        principal (str): The resolved request principal.

    Returns:
        dict[str, Any]: ``{"items": [...], "next_cursor": ...}``.
    """
    try:
        with _scoped_collection(collection, principal):
            items, next_cursor = rag.iter_collection_ner_sources(
                cursor=cursor,
                limit=limit,
                entity_key=entity_key,
                entity_text=entity_text,
                entity_type=entity_type,
                entity_merge_mode=entity_merge_mode,
            )
            return {"items": items, "next_cursor": next_cursor}
    except HTTPException:
        raise
    except InvalidCursorError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.error("Error fetching collection NER sources: {}", e)
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/collections/ner/warm", tags=["Query"])
async def warm_collection_ner(
    collection: str | None = None,
    principal: str = Depends(resolve_principal),
) -> dict[str, Any]:
    """Pre-warm the NER aggregate cache for the caller's collection.

    Runs :meth:`docint.core.rag.RAG._get_collection_ner_aggregate` on a
    worker thread so the first ``/collections/ner/stats`` call after a
    collection switch doesn't pay the full Qdrant scroll cost on a user
    interaction. Safe to call concurrently — the underlying cache uses
    a per-collection key and tolerates repeat-loads. The collection scope is
    open across the ``to_thread`` hop, so the worker warms the correct cache.

    Args:
        collection (str | None): Caller's logical collection; owner-gated and
            scoped per request, falling back to the process default when omitted.
        principal (str): The resolved request principal.

    Returns:
        dict[str, Any]: ``{"ok": True}`` once warming completes.
    """
    try:
        with _scoped_collection(collection, principal):
            await to_thread.run_sync(rag._get_collection_ner_aggregate)  # pyrefly: ignore[bad-argument-type]  # anyio run_sync over-strict on bound method with keyword-only args
        return {"ok": True}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error warming collection NER aggregate: {}", e)
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/collections/ner/stats", response_model=NERStatsOut, tags=["Query"])
def get_collection_ner_stats(
    top_k: int = 15,
    min_mentions: int = 2,
    entity_type: str | None = None,
    include_relations: bool = True,
    entity_merge_mode: Literal["orthographic", "exact", "resolved"] = Query(default="orthographic"),
    collection: str | None = None,
    principal: str = Depends(resolve_principal),
) -> dict[str, Any]:
    """Get collection-wide NER statistics for the caller's collection.

    Args:
        top_k (int): Maximum number of top entities/relations to include.
        min_mentions (int): Minimum mention count for ranked outputs.
        entity_type (str | None): Optional case-insensitive entity type filter.
        include_relations (bool): Whether relation aggregates are included.
        entity_merge_mode (Literal["orthographic", "exact", "resolved"]): Entity clustering mode used for
            derived views ("resolved" groups by durable canonical entity id).
        collection (str | None): Caller's logical collection; owner-gated and
            scoped per request, falling back to the process default when omitted.
        principal (str): The resolved request principal.

    Returns:
        dict[str, Any]: A dashboard-friendly NER stats payload.

    Raises:
        HTTPException: 400/404 from collection resolution; 500 on error.
    """
    try:
        with _scoped_collection(collection, principal):
            return rag.get_collection_ner_stats(
                top_k=top_k,
                min_mentions=min_mentions,
                entity_type=entity_type,
                include_relations=include_relations,
                entity_merge_mode=entity_merge_mode,
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error fetching collection NER stats: {}", e)
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/collections/ner/search", response_model=NERSearchOut, tags=["Query"])
def search_collection_ner_entities(
    q: str = "",
    entity_type: str | None = None,
    limit: int = 100,
    entity_merge_mode: Literal["orthographic", "exact", "resolved"] = Query(default="orthographic"),
    collection: str | None = None,
    principal: str = Depends(resolve_principal),
) -> dict[str, list[dict[str, Any]]]:
    """Search entities across the caller's collection.

    Args:
        q (str): Substring query applied to entity text.
        entity_type (str | None): Optional case-insensitive type filter.
        limit (int): Maximum number of rows to return.
        entity_merge_mode (Literal["orthographic", "exact", "resolved"]): Entity clustering mode used for
            derived views ("resolved" groups by durable canonical entity id).
        collection (str | None): Caller's logical collection; owner-gated and
            scoped per request, falling back to the process default when omitted.
        principal (str): The resolved request principal.

    Returns:
        dict[str, list[dict]]: Dictionary containing matched entities.

    Raises:
        HTTPException: 400/404 from collection resolution; 500 on error.
    """
    try:
        with _scoped_collection(collection, principal):
            return {
                "results": rag.search_collection_ner_entities(
                    q=q,
                    entity_type=entity_type,
                    limit=limit,
                    entity_merge_mode=entity_merge_mode,
                )
            }
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error searching collection entities: {}", e)
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/collections/ner/graph", response_model=NERGraphOut, tags=["Query"])
def get_collection_ner_graph(
    top_k_nodes: int | None = Query(default=None, ge=1),
    min_edge_weight: int = Query(default=1, ge=1),
    entity_merge_mode: Literal["orthographic", "exact", "resolved"] = Query(default="orthographic"),
    collection: str | None = None,
    principal: str = Depends(resolve_principal),
) -> dict[str, Any]:
    """Return a derived entity graph for the caller's collection.

    Wraps :meth:`docint.core.rag.RAG.get_collection_ner_graph`, exposing the
    same node/edge payload the GraphRAG expansion uses so the SPA can render an
    interactive, zoomable entity graph. Nodes are the top ``top_k_nodes``
    entities by mention count; edges combine extracted relations with
    co-occurrence links. Node ids are cluster keys — clients map a node back to
    an entity for drill-down via its ``text``/``type`` fields.

    Args:
        top_k_nodes (int | None): Maximum number of highest-mention entity
            nodes. Defaults to ``NER_GRAPH_TOP_K`` (80) when omitted and is
            clamped to ``[1, NER_GRAPH_MAX_TOP_K]`` (default ceiling 500).
        min_edge_weight (int): Minimum edge weight to include.
        entity_merge_mode (Literal["orthographic", "exact", "resolved"]): Entity
            clustering mode used for derived views ("resolved" groups by durable
            canonical entity id).
        collection (str | None): Caller's logical collection; owner-gated and
            scoped per request, falling back to the process default when omitted.
        principal (str): The resolved request principal.

    Returns:
        dict[str, Any]: Graph payload containing ``nodes``, ``edges`` and ``meta``.

    Raises:
        HTTPException: 400/404 from collection resolution; 500 on error.
    """
    cfg = load_frontend_env()
    requested = cfg.graph_top_k if top_k_nodes is None else top_k_nodes
    effective_top_k = min(max(1, requested), cfg.graph_max_top_k)
    try:
        with _scoped_collection(collection, principal):
            return rag.get_collection_ner_graph(
                top_k_nodes=effective_top_k,
                min_edge_weight=min_edge_weight,
                entity_merge_mode=entity_merge_mode,
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error building collection NER graph: {}", e)
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/collections/entities/resolve", tags=["Query"])
def resolve_collection_entities(
    collection: str | None = None,
    principal: str = Depends(resolve_principal),
) -> dict[str, int]:
    """Resolve the caller's collection's entities into durable canonicals.

    Runs the batch resolution pipeline (name embeddings + conservative LLM
    tie-break) that merges semantically-equivalent named entities into the
    hidden ``{collection}_entities`` store, so the ``entity_merge_mode=
    "resolved"`` views group them. Idempotent — already-resolved surfaces are
    skipped.

    Args:
        collection (str | None): Caller's logical collection; owner-gated and
            scoped per request, falling back to the process default when omitted.
        principal (str): The resolved request principal.

    Returns:
        dict[str, int]: Resolution summary counts (``processed``, ``minted``,
        ``attached``, ``skipped``, ``entities_touched``).

    Raises:
        HTTPException: 400/404 from collection resolution; 500 on error.
    """
    try:
        with _scoped_collection(collection, principal):
            summary = rag.resolve_entities()
            return {
                "processed": summary.processed,
                "minted": summary.minted,
                "attached": summary.attached,
                "skipped": summary.skipped,
                "entities_touched": summary.entities_touched,
            }
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error resolving collection entities: {}", e)
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/collections/documents", tags=["Query"])
def get_collection_documents(
    cursor: str | None = None,
    limit: int = Query(default=0, ge=0, le=500),
    collection: str | None = None,
    principal: str = Depends(resolve_principal),
) -> dict[str, Any]:
    """Return documents in the caller's collection.

    The endpoint operates in two modes for backward compatibility:

    * **Legacy (default)**: ``cursor`` omitted and ``limit=0`` — returns the
      full list under ``{"documents": [...]}``, matching the original shape.
    * **Paginated**: any of ``cursor`` / ``limit`` supplied — returns
      ``{"items": [...], "next_cursor": ...}`` and uses the in-memory
      document cache for slicing.

    Args:
        cursor (str | None): Opaque cursor token from a previous paginated call.
        limit (int): Page size (1-500). ``0`` selects legacy mode.
        collection (str | None): Caller's logical collection; owner-gated and
            scoped per request, falling back to the process default when omitted.
        principal (str): The resolved request principal.

    Returns:
        dict[str, Any]: Either the legacy ``{"documents": ...}`` payload or a
        paginated ``{"items": ..., "next_cursor": ...}`` envelope.
    """
    paginated = cursor is not None or limit > 0
    try:
        with _scoped_collection(collection, principal):
            if not paginated:
                return {"documents": rag.list_documents()}
            items, next_cursor = rag.iter_documents(cursor=cursor, limit=limit or 50)
            return {"items": items, "next_cursor": next_cursor}
    except HTTPException:
        raise
    except InvalidCursorError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.error("Error fetching collection documents: {}", e)
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/collections/documents/count", tags=["Query"])
def get_collection_documents_count(
    collection: str | None = None,
    principal: str = Depends(resolve_principal),
) -> dict[str, int]:
    """Return the number of unique documents in the caller's collection.

    Backed by the same per-collection cache as ``/collections/documents``
    pagination, so the first call after a collection switch pays the
    Qdrant scroll once and the dashboard KPI then reads from cache.

    Args:
        collection (str | None): Caller's logical collection; owner-gated and
            scoped per request, falling back to the process default when omitted.
        principal (str): The resolved request principal.
    """
    try:
        with _scoped_collection(collection, principal):
            return {"count": rag.get_document_count()}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error fetching collection document count: {}", e)
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/collections/documents/summary", response_model=DocumentsSummaryOut, tags=["Query"])
def get_collection_documents_summary(
    collection: str | None = None,
    principal: str = Depends(resolve_principal),
) -> dict[str, Any]:
    """Return collection-wide document aggregates for the Inspector's KPI strip.

    Unlike the paginated ``/collections/documents`` list, this reports the
    document/node totals and the file-type / entity-type breakdown over the
    *entire* collection, so the Inspector's summary cards stay accurate no matter
    how many pages the user has scrolled in (the paginated rows previously
    undercounted file types on large collections). Backed by the same
    per-collection cache as the count/list endpoints.

    Args:
        collection (str | None): Caller's logical collection; owner-gated and
            scoped per request, falling back to the process default when omitted.
        principal (str): The resolved request principal.
    """
    try:
        with _scoped_collection(collection, principal):
            return rag.get_document_summary()
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error fetching collection document summary: {}", e)
        raise HTTPException(status_code=500, detail=str(e)) from e


def _csv_attachment_headers(stem: str) -> dict[str, str]:
    """Build streaming CSV response headers, including RFC 6266 filename."""
    from urllib.parse import quote

    safe_stem = stem.replace('"', "_")
    ascii_only = "".join(ch if ord(ch) < 128 else "_" for ch in safe_stem)
    filename = f"{ascii_only}.csv"
    star = quote(f"{safe_stem}.csv", safe="")
    return {
        "Content-Disposition": f"attachment; filename=\"{filename}\"; filename*=UTF-8''{star}",
        "X-Accel-Buffering": "no",
        "Cache-Control": "no-store",
    }


def _download_headers(stem: str, ext: str, *, inline: bool = False) -> dict[str, str]:
    """Build Content-Disposition headers (RFC 6266) for a report download.

    Args:
        stem (str): Filename stem (without extension). Non-ASCII is preserved in
            the ``filename*`` form and transliterated to ``_`` in the ASCII
            ``filename`` fallback.
        ext (str): File extension without the leading dot.
        inline (bool): Serve inline (e.g. the HTML view) instead of as an
            attachment download.

    Returns:
        dict[str, str]: Response headers.
    """
    from urllib.parse import quote

    safe_stem = (stem or "report").replace('"', "_")
    ascii_only = "".join(ch if ord(ch) < 128 else "_" for ch in safe_stem)
    disposition = "inline" if inline else "attachment"
    star = quote(f"{safe_stem}.{ext}", safe="")
    return {
        "Content-Disposition": f"{disposition}; filename=\"{ascii_only}.{ext}\"; filename*=UTF-8''{star}",
        "Cache-Control": "no-store",
    }


def _report_stem(report: dict[str, Any]) -> str:
    """Build a download filename stem from a report dict."""
    return f"report-{report.get('id')}-{report.get('title') or 'report'}"


def _get_owned_report(report_id: int, principal: str) -> dict[str, Any]:
    """Fetch a report owned by ``principal`` or raise 404.

    Args:
        report_id (int): The report id.
        principal (str): The resolved request principal.

    Returns:
        dict[str, Any]: The report, including its ordered items.

    Raises:
        HTTPException: 404 when the report is missing or owned by another principal.
    """
    report = rag.ensure_report_manager().get_report(report_id, principal)
    if report is None:
        raise HTTPException(status_code=404, detail="Report not found.")
    return report


def _capture_collection_overview(report_id: int, collection: str, principal: str) -> dict[str, Any] | None:
    """Build and persist a report's frozen document-overview snapshot.

    Reads the full document list under the caller's scoped collection and stores
    the aggregated manifest on the report. Raises on failure — callers decide
    whether to swallow it (create: fail-soft) or surface it (refresh: 502).

    Args:
        report_id (int): The report id.
        collection (str): The report's logical collection.
        principal (str): The resolved request principal (owner).

    Returns:
        dict | None: The updated report, or ``None`` when the report is not owned.
    """
    from datetime import UTC, datetime

    from docint.core.collection_overview import build_collection_overview

    with _scoped_collection(collection, principal):
        documents = rag.list_documents()
    overview = build_collection_overview(documents, collection, datetime.now(UTC))
    return rag.ensure_report_manager().set_collection_overview_snapshot(report_id, principal, overview)


@app.get("/collections/{name}/export/documents.csv", tags=["Query"])
def export_documents_csv(name: str, principal: str = Depends(resolve_principal)) -> StreamingResponse:
    """Stream the documents table as CSV.

    The endpoint reads from :meth:`docint.core.rag.RAG.list_documents` (cached
    after the first call) and emits one row per document. Output matches the
    CLI's ``query --documents`` schema column-for-column. The path ``name`` is
    the caller's logical collection: it is owner-gated (404 if not owned) and
    resolved to its physical collection for the duration of the read, so exports
    are stateless and isolated per user. Rows are materialized within the scope;
    the response then streams the in-memory list.
    """
    from docint.utils.csv_stream import DOCUMENT_COLUMNS, document_row, stream_csv

    with _scoped_collection(name, principal):
        docs = rag.list_documents()

    def row_iter() -> Iterator[dict[str, Any]]:
        for doc in docs:
            yield document_row(doc)

    return StreamingResponse(
        stream_csv(row_iter(), DOCUMENT_COLUMNS),
        media_type="text/csv; charset=utf-8",
        headers=_csv_attachment_headers(f"{name}-documents"),
    )


@app.get("/collections/{name}/export/entities.csv", tags=["Query"])
def export_entities_csv(
    name: str,
    top_k: int = Query(default=50, ge=1, le=100_000),
    min_mentions: int = Query(default=1, ge=1),
    entity_type: str | None = None,
    entity_merge_mode: Literal["orthographic", "exact", "resolved"] = Query(default="orthographic"),
    principal: str = Depends(resolve_principal),
) -> StreamingResponse:
    """Stream the top entities by mention frequency as CSV.

    Mirrors the CLI's ``query --entities`` export (``rank,entity,type,mentions``).
    Defaults match the CLI's ``DEFAULT_ENTITY_LIMIT`` so the two paths produce
    identical output for the same collection. ``entity_merge_mode="resolved"``
    streams the durable canonical entities (same as the Analysis/Dashboard
    resolved view); it falls back to orthographic on collections that have not
    been resolved. The path ``name`` is owner-gated and resolved to its physical
    collection for the read.
    """
    from docint.utils.csv_stream import ENTITY_STATS_COLUMNS, entity_stats_row, stream_csv

    with _scoped_collection(name, principal):
        stats = rag.get_collection_ner_stats(
            top_k=top_k,
            min_mentions=min_mentions,
            entity_type=entity_type,
            include_relations=False,
            entity_merge_mode=entity_merge_mode,
        )
    entities = list(stats.get("top_entities") or [])

    def row_iter() -> Iterator[dict[str, Any]]:
        for idx, entity in enumerate(entities, start=1):
            yield entity_stats_row(entity, rank=idx)

    return StreamingResponse(
        stream_csv(row_iter(), ENTITY_STATS_COLUMNS),
        media_type="text/csv; charset=utf-8",
        headers=_csv_attachment_headers(f"{name}-entities"),
    )


@app.get("/collections/{name}/export/ner-sources.csv", tags=["Query"])
def export_ner_sources_csv(
    name: str,
    entity_key: str | None = None,
    entity_text: str | None = None,
    entity_type: str | None = None,
    entity_merge_mode: Literal["orthographic", "exact", "resolved"] = Query(default="orthographic"),
    principal: str = Depends(resolve_principal),
) -> StreamingResponse:
    """Stream entity findings (per-source rows) as CSV.

    Output schema matches ``entityFindingsToCsv`` in
    ``frontend/src/lib/exports.ts``. Filtering uses the same matcher as the
    paginated ``/collections/ner/sources`` endpoint, so the export reflects
    exactly what the SPA's entity inspector shows. ``entity_merge_mode=
    "resolved"`` includes the canonical entity's sibling aliases. The path
    ``name`` is owner-gated and resolved to its physical collection; all pages
    are materialized within that scope before the response streams them (the
    request scope cannot remain bound across the post-return streaming hops).
    """
    from docint.utils.csv_stream import NER_SOURCE_COLUMNS, ner_source_row, stream_csv

    if entity_key and not (entity_text or entity_type):
        if "::" in entity_key:
            entity_text, entity_type = entity_key.split("::", 1)
        else:
            entity_text = entity_key

    label_type = entity_type or "Unlabeled"
    entity_label = f"{entity_text} [{label_type}]" if entity_text else ""

    rows: list[dict[str, Any]] = []
    with _scoped_collection(name, principal):
        cursor: str | None = None
        while True:
            page, cursor = rag.iter_collection_ner_sources(
                cursor=cursor,
                limit=500,
                entity_text=entity_text,
                entity_type=entity_type,
                entity_merge_mode=entity_merge_mode,
            )
            for source in page:
                rows.append(ner_source_row(source, entity_label=entity_label))
            if cursor is None:
                break

    return StreamingResponse(
        stream_csv(iter(rows), NER_SOURCE_COLUMNS),
        media_type="text/csv; charset=utf-8",
        headers=_csv_attachment_headers(f"{name}-ner-sources"),
    )


@app.get("/collections/{name}/export/hate-speech.csv", tags=["Query"])
def export_hate_speech_csv(
    name: str,
    category: str | None = None,
    min_confidence: str | None = None,
    principal: str = Depends(resolve_principal),
) -> StreamingResponse:
    """Stream the hate-speech findings table as CSV.

    Output schema matches ``hateSpeechToCsv`` in
    ``frontend/src/lib/exports.ts``. Filtering uses the same logic as the
    paginated ``/collections/hate-speech`` endpoint. The path ``name`` is
    owner-gated and resolved to its physical collection for the read.
    """
    from docint.core.rag import _filter_hate_speech
    from docint.utils.csv_stream import HATE_SPEECH_COLUMNS, hate_speech_row, stream_csv

    with _scoped_collection(name, principal):
        findings = _filter_hate_speech(
            rag.get_collection_hate_speech(),
            category=category,
            min_confidence=min_confidence,
        )

    def row_iter() -> Iterator[dict[str, Any]]:
        for finding in findings:
            yield hate_speech_row(finding)

    return StreamingResponse(
        stream_csv(row_iter(), HATE_SPEECH_COLUMNS),
        media_type="text/csv; charset=utf-8",
        headers=_csv_attachment_headers(f"{name}-hate-speech"),
    )


@app.get("/sessions/list", response_model=SessionListOut, tags=["Sessions"])
def list_sessions(
    principal: str = Depends(resolve_principal),
) -> dict[str, list[dict[str, Any]]]:
    """List the calling principal's chat sessions.

    Args:
        principal (str): The resolved request principal.

    Returns:
        dict[str, list[dict[str, Any]]]: A dictionary containing the list of sessions.

    Raises:
        HTTPException: If an error occurs while listing sessions.
    """
    try:
        sessions = rag.ensure_session_manager().list_sessions(principal)
        return {"sessions": sessions}
    except Exception as e:
        logger.error("Error listing sessions: {}", e)
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get(
    "/sessions/{session_id}/history",
    response_model=SessionHistoryOut,
    tags=["Sessions"],
)
def get_session_history(
    session_id: str, principal: str = Depends(resolve_principal)
) -> dict[str, list[dict[str, Any]]]:
    """Get history for a session owned by the calling principal.

    A session that does not exist or is owned by another principal is
    reported as 404 (no existence leak).

    Args:
        session_id (str): The ID of the session.
        principal (str): The resolved request principal.

    Returns:
        dict[str, list[dict[str, Any]]]: A dictionary containing the session messages.

    Raises:
        HTTPException: 404 when the session is not found for this
            principal; 500 on unexpected errors.
    """
    try:
        messages = rag.ensure_session_manager().get_session_history(session_id, principal)
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error fetching history: {}", e)
        raise HTTPException(status_code=500, detail=str(e)) from e
    # NOTE: empty also covers "owned but zero turns" (brand-new session),
    # which collapses to 404 here; acceptable for Plan 1 (see Plan 2).
    if not messages:
        raise HTTPException(status_code=404, detail="Session not found.")
    return {"messages": messages}


def _collect_session_source_files(session_id: str, principal: str) -> list[tuple[str, Path]]:
    """Return the unique source files referenced by a session's citations.

    Each entry is ``(filename_in_zip, path_on_disk)``. Files that can't be
    resolved on disk are skipped — the ZIP is best-effort and surfaces only
    files the backend can still serve. Collection is resolved from each
    source's ``collection`` field when present and falls back to the session's
    own *pinned* collection (owner-scoped), not any process-global active
    collection, so the bundle stays correct under concurrent multi-tenant use.

    Args:
        session_id (str): The session whose citations should be packaged.
        principal (str): The resolved request principal; sessions owned by
            another principal yield an empty result (404 at the endpoint).

    Returns:
        list[tuple[str, Path]]: Pairs ready for :meth:`zipfile.ZipFile.write`.
    """
    sm = rag.ensure_session_manager()
    messages = sm.get_session_history(session_id, principal)
    session_collection = sm.get_session_collection(session_id, principal)
    selected: dict[str, tuple[str, Path]] = {}
    used_arcnames: set[str] = set()
    for message in messages:
        for source in message.get("sources") or []:
            if not isinstance(source, dict):
                continue
            file_hash = source.get("file_hash")
            if not file_hash or file_hash in selected:
                continue
            collection = str(source.get("collection") or session_collection or "")
            if not collection:
                continue
            filename = str(source.get("filename") or "")
            path = _resolve_source_file_path(
                collection,
                str(file_hash),
                filename_hint=filename or None,
            )
            if path is None:
                continue

            arcname = filename or path.name
            base = arcname
            counter = 1
            while arcname in used_arcnames:
                stem, dot, ext = base.partition(".")
                arcname = f"{stem}_{counter}{dot}{ext}" if dot else f"{base}_{counter}"
                counter += 1
            used_arcnames.add(arcname)
            selected[str(file_hash)] = (arcname, path)
    return list(selected.values())


@app.get("/sessions/{session_id}/sources.zip", tags=["Sessions"])
def export_session_sources_zip(session_id: str, principal: str = Depends(resolve_principal)) -> StreamingResponse:
    """Stream a ZIP bundle of every source file cited in a session.

    Resolves each citation's ``file_hash`` to an on-disk file using the same
    lookup chain as ``/sources/preview``, deduplicates by hash, and writes the
    files into an in-memory ZIP (typical sessions cite tens of files, not
    thousands). Sources whose underlying file can't be found are skipped
    rather than failing the whole download. Sessions owned by another
    principal collapse to 404 — they look identical to "no sources".

    Args:
        session_id (str): The session ID to package.
        principal (str): The resolved request principal.

    Returns:
        StreamingResponse: ``application/zip`` payload with an
        ``attachment; filename="session-<id>-sources.zip"`` header.

    Raises:
        HTTPException: 404 if the session has no resolvable sources or is
            owned by another principal.
    """
    try:
        files = _collect_session_source_files(session_id, principal)
    except Exception as e:
        logger.error("Error assembling session sources for {}: {}", session_id, e)
        raise HTTPException(status_code=500, detail=str(e)) from e

    if not files:
        raise HTTPException(status_code=404, detail="No source files found for this session")

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for arcname, path in files:
            try:
                zf.write(path, arcname=arcname)
            except OSError as exc:
                logger.warning("Skipping unreadable source {}: {}", path, exc)
    buffer.seek(0)

    def iter_chunks(chunk_size: int = 64 * 1024) -> Iterator[bytes]:
        while True:
            chunk = buffer.read(chunk_size)
            if not chunk:
                break
            yield chunk

    headers = {
        "Content-Disposition": f'attachment; filename="session-{session_id}-sources.zip"',
        "X-Accel-Buffering": "no",
        "Cache-Control": "no-store",
    }
    return StreamingResponse(iter_chunks(), media_type="application/zip", headers=headers)


@app.delete("/sessions/{session_id}", tags=["Sessions"])
def delete_session(session_id: str, principal: str = Depends(resolve_principal)) -> dict[str, bool]:
    """Delete a session owned by the calling principal.

    A session that does not exist or is owned by another principal is
    reported as 404 (no existence leak).

    Args:
        session_id (str): The ID of the session to delete.
        principal (str): The resolved request principal.

    Returns:
        dict[str, bool]: A dictionary indicating whether the deletion
            was successful.

    Raises:
        HTTPException: 404 when the session is not found for this
            principal; 500 on unexpected errors.
    """
    try:
        success = rag.ensure_session_manager().delete_session(session_id, principal)
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error deleting session: {}", e)
        raise HTTPException(status_code=500, detail=str(e)) from e
    if not success:
        raise HTTPException(status_code=404, detail="Session not found.")
    return {"ok": success}


@app.post("/reports", tags=["Reports"])
def create_report(payload: ReportCreateIn, principal: str = Depends(resolve_principal)) -> dict[str, Any]:
    """Create a new, empty report owned by the calling principal.

    Args:
        payload (ReportCreateIn): Title and optional collection/session scope.
        principal (str): The resolved request principal.

    Returns:
        dict[str, Any]: The created report.
    """
    try:
        report = rag.ensure_report_manager().create_report(
            title=payload.title,
            owner=principal,
            collection_name=payload.collection_name,
            operator=payload.operator,
            reference_number=payload.reference_number,
            session_id=payload.session_id,
        )
    except Exception as e:
        logger.error("Error creating report: {}", e)
        raise HTTPException(status_code=500, detail=str(e)) from e

    # Default-on document overview: capture once at create. Fail-soft — a Qdrant
    # hiccup must not fail report creation; the snapshot stays null until a
    # successful refresh.
    if payload.collection_name:
        try:
            enriched = _capture_collection_overview(report["id"], payload.collection_name, principal)
            if enriched is not None:
                report = enriched
        except Exception as e:
            logger.warning("Collection-overview capture failed for report {}: {}", report["id"], e)
    return report


@app.get("/reports", response_model=ReportListOut, tags=["Reports"])
def list_reports(
    collection: str | None = None, principal: str = Depends(resolve_principal)
) -> dict[str, list[dict[str, Any]]]:
    """List the caller's reports, optionally filtered by collection.

    Args:
        collection (str | None): Optional collection filter.
        principal (str): The resolved request principal.

    Returns:
        dict[str, list[dict[str, Any]]]: The caller's report summaries.
    """
    try:
        return {"reports": rag.ensure_report_manager().list_reports(principal, collection)}
    except Exception as e:
        logger.error("Error listing reports: {}", e)
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/reports/{report_id}", tags=["Reports"])
def get_report(report_id: int, principal: str = Depends(resolve_principal)) -> dict[str, Any]:
    """Return a report (with items) owned by the calling principal.

    Args:
        report_id (int): The report id.
        principal (str): The resolved request principal.

    Returns:
        dict[str, Any]: The report and its ordered items.

    Raises:
        HTTPException: 404 when the report is missing or not owned.
    """
    return _get_owned_report(report_id, principal)


@app.patch("/reports/{report_id}", tags=["Reports"])
def update_report(
    report_id: int, payload: ReportUpdateIn, principal: str = Depends(resolve_principal)
) -> dict[str, Any]:
    """Update a report (title, case metadata, or contents toggle) owned by the caller.

    Args:
        report_id (int): The report id.
        payload (ReportUpdateIn): Fields to update; only non-null fields apply.
        principal (str): The resolved request principal.

    Returns:
        dict[str, Any]: The updated report.

    Raises:
        HTTPException: 404 when the report is missing or not owned.
    """
    report = rag.ensure_report_manager().update_report(
        report_id,
        principal,
        title=payload.title,
        operator=payload.operator,
        reference_number=payload.reference_number,
        show_toc=payload.show_toc,
        show_collection_overview=payload.show_collection_overview,
    )
    if report is None:
        raise HTTPException(status_code=404, detail="Report not found.")
    return report


@app.post("/reports/{report_id}/collection-overview/refresh", tags=["Reports"])
def refresh_report_collection_overview(report_id: int, principal: str = Depends(resolve_principal)) -> dict[str, Any]:
    """Recapture a report's document-overview snapshot from its collection.

    Point-in-time refresh: rebuilds the frozen manifest from the collection's
    *current* documents.

    Args:
        report_id (int): The report id.
        principal (str): The resolved request principal (owner).

    Returns:
        dict[str, Any]: The report with its refreshed ``collection_overview`` snapshot.

    Raises:
        HTTPException: 404 when the report is missing or owned by another
            principal (or its collection is no longer owned); 400 when the report
            has no collection; 502 when the manifest build fails.
    """
    report = _get_owned_report(report_id, principal)
    collection = report.get("collection_name")
    if not collection:
        raise HTTPException(status_code=400, detail="Report has no collection to summarize.")
    try:
        updated = _capture_collection_overview(report_id, collection, principal)
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Collection-overview refresh failed for report {}: {}", report_id, e)
        raise HTTPException(status_code=502, detail="Failed to build the document overview.") from e
    return updated if updated is not None else report


@app.delete("/reports/{report_id}", tags=["Reports"])
def delete_report(report_id: int, principal: str = Depends(resolve_principal)) -> dict[str, bool]:
    """Delete a report (and its items) owned by the calling principal.

    Args:
        report_id (int): The report id.
        principal (str): The resolved request principal.

    Returns:
        dict[str, bool]: ``{"ok": True}`` on success.

    Raises:
        HTTPException: 404 when the report is missing or not owned.
    """
    if not rag.ensure_report_manager().delete_report(report_id, principal):
        raise HTTPException(status_code=404, detail="Report not found.")
    return {"ok": True}


@app.post("/reports/{report_id}/items", tags=["Reports"])
def add_report_item(
    report_id: int, payload: ReportItemIn, principal: str = Depends(resolve_principal)
) -> dict[str, Any]:
    """Add a snapshotted artifact to a report (idempotent by dedupe key).

    Args:
        report_id (int): The report id.
        payload (ReportItemIn): Artifact type, dedupe key, snapshot, optional note.
        principal (str): The resolved request principal.

    Returns:
        dict[str, Any]: The added (or pre-existing) item.

    Raises:
        HTTPException: 404 when the report is missing or not owned.
    """
    item = rag.ensure_report_manager().add_item(
        report_id,
        principal,
        artifact_type=payload.artifact_type,
        dedupe_key=payload.dedupe_key,
        snapshot=payload.snapshot,
        note=payload.note,
    )
    if item is None:
        raise HTTPException(status_code=404, detail="Report not found.")
    return item


@app.patch("/reports/{report_id}/items/{item_id}", tags=["Reports"])
def annotate_report_item(
    report_id: int, item_id: int, payload: ReportItemNoteIn, principal: str = Depends(resolve_principal)
) -> dict[str, Any]:
    """Set or clear the note on a report item.

    Args:
        report_id (int): The report id.
        item_id (int): The item id.
        payload (ReportItemNoteIn): The new note (``None`` clears it).
        principal (str): The resolved request principal.

    Returns:
        dict[str, Any]: The updated item.

    Raises:
        HTTPException: 404 when the report/item is missing or not owned.
    """
    item = rag.ensure_report_manager().annotate_item(report_id, principal, item_id, note=payload.note)
    if item is None:
        raise HTTPException(status_code=404, detail="Report or item not found.")
    return item


@app.delete("/reports/{report_id}/items/{item_id}", tags=["Reports"])
def remove_report_item(report_id: int, item_id: int, principal: str = Depends(resolve_principal)) -> dict[str, bool]:
    """Remove a single item from a report owned by the calling principal.

    Args:
        report_id (int): The report id.
        item_id (int): The item id.
        principal (str): The resolved request principal.

    Returns:
        dict[str, bool]: ``{"ok": True}`` on success.

    Raises:
        HTTPException: 404 when the report/item is missing or not owned.
    """
    if not rag.ensure_report_manager().remove_item(report_id, principal, item_id):
        raise HTTPException(status_code=404, detail="Report or item not found.")
    return {"ok": True}


@app.post("/reports/{report_id}/items/reorder", tags=["Reports"])
def reorder_report_items(
    report_id: int, payload: ReportReorderIn, principal: str = Depends(resolve_principal)
) -> dict[str, Any]:
    """Reorder a report's items to match the supplied id order.

    Args:
        report_id (int): The report id.
        payload (ReportReorderIn): Desired item id order.
        principal (str): The resolved request principal.

    Returns:
        dict[str, Any]: The reordered report.

    Raises:
        HTTPException: 404 when the report is missing or not owned.
    """
    report = rag.ensure_report_manager().reorder_items(report_id, principal, payload.item_ids)
    if report is None:
        raise HTTPException(status_code=404, detail="Report not found.")
    return report


@app.get("/reports/{report_id}/export.md", tags=["Reports"])
def export_report_markdown(report_id: int, principal: str = Depends(resolve_principal)) -> Response:
    """Export a report as a single Markdown document (attachment download)."""
    from docint.core.state.report_render import render_markdown

    report = _get_owned_report(report_id, principal)
    return Response(
        content=render_markdown(report),
        media_type="text/markdown; charset=utf-8",
        headers=_download_headers(_report_stem(report), "md"),
    )


@app.get("/reports/{report_id}/export.html", tags=["Reports"])
def export_report_html(report_id: int, principal: str = Depends(resolve_principal)) -> Response:
    """Export a report as a self-contained HTML document (served inline)."""
    from docint.core.state.report_render import render_html

    report = _get_owned_report(report_id, principal)
    return Response(
        content=render_html(report),
        media_type="text/html; charset=utf-8",
        headers=_download_headers(_report_stem(report), "html", inline=True),
    )


@app.get("/reports/{report_id}/export.json", tags=["Reports"])
def export_report_json(report_id: int, principal: str = Depends(resolve_principal)) -> Response:
    """Export the full report (with snapshots) as JSON (attachment download)."""
    from docint.core.state.report_render import render_json

    report = _get_owned_report(report_id, principal)
    return Response(
        content=render_json(report),
        media_type="application/json",
        headers=_download_headers(_report_stem(report), "json"),
    )


@app.get("/reports/{report_id}/export.zip", tags=["Reports"])
def export_report_zip(report_id: int, principal: str = Depends(resolve_principal)) -> Response:
    """Export a report as a ZIP bundle of per-type CSVs (attachment download)."""
    from docint.core.state.report_render import report_csv_bundle

    report = _get_owned_report(report_id, principal)
    return Response(
        content=report_csv_bundle(report),
        media_type="application/zip",
        headers=_download_headers(_report_stem(report), "zip"),
    )


@app.get("/reports/{report_id}/export.pdf", tags=["Reports"])
def export_report_pdf(report_id: int, principal: str = Depends(resolve_principal)) -> Response:
    """Export a report as a real paginated PDF rendered by WeasyPrint.

    Returns 503 if the PDF engine (WeasyPrint + native libs) is unavailable,
    leaving the other export formats unaffected.
    """
    from docint.core.state.report_render import PdfEngineUnavailableError, render_pdf

    report = _get_owned_report(report_id, principal)
    try:
        pdf_bytes = render_pdf(report)
    except PdfEngineUnavailableError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers=_download_headers(_report_stem(report), "pdf"),
    )


@app.post("/agent/chat", response_model=AgentChatOut, tags=["Agent"])
def agent_chat(payload: AgentChatIn, request: Request) -> AgentChatOut:
    """Agentic chat endpoint: understand → maybe clarify → retrieve/respond.

    Args:
        payload (AgentChatIn): Message, optional session id, and optional
            logical collection (owner-gated; falls back to the process default).
        request (Request): The incoming request used to resolve the calling principal.

    Returns:
        AgentChatOut: Clarification prompt or answer with sources.

    Raises:
        HTTPException: 400/404 from collection resolution; 409 when the session
            is pinned to a different collection.
    """
    owner = resolve_principal(request)
    physical = _resolve_request_collection(payload.collection, owner)

    # Scope every retrieval/generation call inside the turn to the resolved
    # physical collection (per-request ContextVar), and thread the session id
    # explicitly so the turn persists under the right conversation.
    try:
        with rag.collection_scope(physical):
            session_id = rag.start_session(payload.session_id, owner=owner)
            ctx = rag.sessions.get_agent_context(session_id) if rag.sessions else None
            if ctx and rag.sessions:
                ctx.history = rag.sessions.get_session_history(session_id, owner=owner)

            turn = Turn(user_input=payload.message, session_id=session_id)
            orchestrator = _build_orchestrator()
            result = orchestrator.handle_turn(turn, context=ctx)
    except SessionCollectionMismatchError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    if result.clarification is not None and result.clarification.needed:
        if ctx:
            ctx.clarifications += 1
        return AgentChatOut(
            status="clarification",
            message=result.clarification.message,
            reason=result.clarification.reason,
            session_id=session_id,
            intent=result.analysis.intent if result.analysis else None,
            confidence=result.analysis.confidence if result.analysis else None,
        )

    retrieval = result.retrieval
    if retrieval is None:
        raise HTTPException(status_code=500, detail="No retrieval result available")

    return AgentChatOut(
        status="answer",
        answer=retrieval.answer,
        sources=retrieval.sources,
        session_id=retrieval.session_id or session_id,
        intent=retrieval.intent,
        confidence=retrieval.confidence,
        tool_used=retrieval.tool_used,
        latency_ms=retrieval.latency_ms,
        validation_checked=retrieval.validation_checked,
        validation_mismatch=retrieval.validation_mismatch,
        validation_reason=retrieval.validation_reason,
    )


@app.post("/ingest", response_model=IngestOut, tags=["Ingestion"])
def ingest(payload: IngestIn, request: Request) -> dict[str, bool | str]:
    """Trigger ingestion for the caller's collection using the configured data directory.

    The ``collection`` in the payload is the caller's *logical* name. Ingestion
    registers ownership (the first ingester owns it) and resolves it to an
    owner-namespaced physical Qdrant collection, so two users can ingest the
    same logical name without colliding.

    Args:
        payload (IngestIn): The ingestion payload containing the collection name and hybrid flag.
        request (Request): The incoming request used to resolve the calling principal.

    Returns:
        dict[str, bool | str]: A dictionary with keys ``ok``, ``collection``, ``data_dir``,
            ``hybrid``, and ``empty``. The ``empty`` field is ``True`` if ingestion produced
            no documents; ``False`` otherwise. Soft-empty outcomes (where the file set contained
            no parseable content) return HTTP 200 with ``empty=true`` instead of an error.

    Raises:
        HTTPException: 400 if the collection name is missing or the data
            directory does not exist; 500 for any unexpected backend error.
    """
    name = payload.collection.strip()
    if not name:
        logger.error("HTTPException: Collection name required")
        raise HTTPException(status_code=400, detail="Collection name required")

    principal = resolve_principal(request)
    physical = rag.ensure_collection_owner_manager().register(principal, name)

    data_dir = _resolve_data_dir()
    if not data_dir.is_dir():
        logger.error("HTTPException: Data directory does not exist: {}", data_dir)
        raise HTTPException(
            status_code=400,
            detail=f"Data directory does not exist: {data_dir}",
        )

    try:
        ingest_module.ingest_docs(
            physical,
            data_dir,
            hybrid=payload.hybrid if payload.hybrid is not None else True,
        )
    except EmptyIngestionError as exc:
        logger.warning(
            "Ingestion produced no content for '{}'; returning empty response.",
            exc.collection_name,
        )
        return {
            "ok": True,
            "collection": name,
            "data_dir": str(data_dir),
            "hybrid": payload.hybrid if payload.hybrid is not None else True,
            "empty": True,
        }
    except Exception as exc:
        logger.error("Unexpected error during ingestion of '{}': {}", name, exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "ok": True,
        "collection": name,
        "data_dir": str(data_dir),
        "hybrid": payload.hybrid if payload.hybrid is not None else True,
    }


@app.post("/agent/chat/stream", tags=["Agent"])
async def agent_chat_stream(payload: AgentChatIn, request: Request) -> StreamingResponse:
    """Streaming variant of agent chat with token events and final metadata.

    Args:
        payload (AgentChatIn): Message, optional session id, and optional logical
            collection (owner-gated; falls back to the process default).
        request (Request): The incoming request used to resolve the calling principal.

    Returns:
        StreamingResponse: SSE stream with clarification or answer tokens and metadata.

    Raises:
        HTTPException: 400/404 from collection resolution; 409 when the session
            is pinned to a different collection.
    """
    owner = resolve_principal(request)
    physical = _resolve_request_collection(payload.collection, owner)
    # Up-front collection-pin check so a mismatch is a clean 409 rather than an
    # in-stream error event.
    if payload.session_id:
        pinned = rag.ensure_session_manager().get_session_collection(payload.session_id, owner)
        if pinned is not None and pinned != physical:
            raise HTTPException(
                status_code=409,
                detail=f"Session '{payload.session_id}' is pinned to a different collection.",
            )

    async def event_generator() -> AsyncIterator[str]:
        """Generate SSE events for the agent chat stream.

        The request's physical collection is bound for the whole generator (so
        it propagates into the anyio worker threads spawned for analysis and
        generation), and the resolved session id is threaded explicitly into
        ``stream_chat`` so the turn persists under the right conversation.

        Yields:
            AsyncIterator[str]: An asynchronous iterator yielding SSE events.
        """
        with rag.collection_scope(physical):

            def _prepare() -> tuple[str, Any, Any, Any]:
                """Run the blocking session/understanding pre-amble off the loop.

                Session start, history load, and intent analysis are synchronous
                (and may issue LLM calls), so they run on a worker thread.

                Returns:
                    tuple[str, Any, Any, Any]: ``(session_id, ctx, analysis,
                    clarification_decision)``.
                """
                session_id = rag.start_session(payload.session_id, owner=owner)
                ctx = rag.sessions.get_agent_context(session_id) if rag.sessions else None
                if ctx and rag.sessions:
                    ctx.history = rag.sessions.get_session_history(session_id, owner=owner)
                turn = Turn(user_input=payload.message, session_id=session_id)
                analysis = _select_understanding_agent().analyze(turn, context=ctx)
                clarification_decision = _clarification_policy.evaluate(
                    analysis, clarifications_so_far=ctx.clarifications if ctx else 0
                )
                return session_id, ctx, analysis, clarification_decision

            session_id, ctx, analysis, clarification_decision = await to_thread.run_sync(_prepare)

            if clarification_decision.needed:
                if ctx:
                    ctx.clarifications += 1
                payload_out = {
                    "status": "clarification",
                    "message": clarification_decision.message,
                    "reason": clarification_decision.reason,
                    "intent": analysis.intent,
                    "confidence": analysis.confidence,
                    "session_id": session_id,
                }
                yield _format_sse("clarification", payload_out)
                return

            # Stream via RAG chat (history-aware: rewritten query + prior turn)
            query_text = analysis.rewritten_query or payload.message

            def _make_agent_stream() -> Iterator[Any]:
                """Build the blocking agent chat stream off the event loop.

                Returns:
                    Iterator[Any]: The sync chat-chunk generator.
                """
                prior_turn = build_prior_turn(ctx.history) if ctx else None
                return cast(
                    "Iterator[Any]",
                    rag.stream_chat(query_text, session_id=session_id, owner=owner, prior_turn=prior_turn),
                )

            # Tokens
            async for chunk in _aiter_sync_gen(_make_agent_stream, request):
                if isinstance(chunk, str):
                    yield _format_sse("token", {"token": chunk})
                elif isinstance(chunk, dict):
                    meta = {
                        "status": "answer",
                        "sources": chunk.get("sources", []),
                        "session_id": chunk.get("session_id", session_id),
                        "intent": analysis.intent,
                        "confidence": analysis.confidence,
                        "tool_used": "rag_chat",
                    }
                    yield _format_sse("done", meta)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/ingest/upload", tags=["Ingestion"])
async def ingest_upload(
    request: Request,
    collection: str = Form(...),
    files: list[UploadFile] = File(...),  # noqa: B008 — FastAPI dependency marker
    hybrid: bool | None = Form(True),
) -> StreamingResponse:
    """Upload files for ingestion and stream progress as SSE events.

    Args:
        request (Request): The incoming request, used to detect client
            disconnect so the awaiter can be cancelled promptly.
        collection (str): The name of the collection to ingest into.
        files (list[UploadFile]): The list of files to upload.
        hybrid (bool | None): Whether to enable hybrid search (default: True).

    Returns:
        StreamingResponse: A streaming response that yields SSE events during ingestion.

    Raises:
        HTTPException: If the collection name is missing or no files are provided.
        HTTPException: If an error occurs during file upload.
    """
    name = collection.strip()
    if not name:
        logger.error("HTTPException: Collection name required for upload")
        raise HTTPException(status_code=400, detail="Collection name required")
    if not files:
        logger.error("HTTPException: At least one file is required for upload")
        raise HTTPException(status_code=400, detail="At least one file is required")

    # Ownership: the first uploader owns the logical name; resolve it to an
    # owner-namespaced physical collection so two users uploading the same
    # logical name keep separate Qdrant collections and source-file stores.
    principal = resolve_principal(request)
    physical = rag.ensure_collection_owner_manager().register(principal, name)

    # We use a persistent directory for uploads to support previewing files later.
    # The files are ingested into Qdrant and kept in the collection directory.

    async def event_stream() -> AsyncIterator[str]:
        """Stream SSE events during the ingestion process.

        Returns:
            AsyncIterator[str]: A stream of Server-Sent Events (SSE) as strings.

        Yields:
            Iterator[AsyncIterator[str]]: A stream of SSE events during the ingestion process.
        """
        # Use the dedicated sources directory (sibling to Qdrant collections) to store uploaded files
        qdrant_src_dir = _resolve_qdrant_src_dir()
        batch_dir = qdrant_src_dir / physical
        batch_dir.mkdir(parents=True, exist_ok=True)

        yield _format_sse(
            "start",
            {
                "collection": name,
                "target_dir": str(batch_dir),
                "files": [f.filename for f in files],
            },
        )

        for upload in files:
            dest = _safe_relative_dest(batch_dir, upload.filename or "upload")
            dest.parent.mkdir(parents=True, exist_ok=True)
            filename = str(dest.relative_to(batch_dir))
            bytes_written = 0
            try:
                with dest.open("wb") as buffer:
                    while True:
                        chunk = await upload.read(1024 * 1024)
                        if not chunk:
                            break
                        buffer.write(chunk)
                        bytes_written += len(chunk)
                        yield _format_sse(
                            "upload_progress",
                            {"filename": filename, "bytes_written": bytes_written},
                        )

                # We calculate hash but don't store the file index anymore
                file_hash = compute_file_hash(dest)
                yield _format_sse(
                    "file_saved",
                    {
                        "filename": filename,
                        "file_hash": file_hash,
                        "path": str(dest),
                    },
                )
            except Exception as exc:  # pragma: no cover - streamed errors are logged
                logger.error("Error saving uploaded file {}: {}", filename, exc)
                yield _format_sse(
                    "error",
                    {"message": f"Failed to save {filename}"},
                )
                return

        yield _format_sse("ingestion_started", {"collection": name})
        try:
            queue: asyncio.Queue[str | None | Exception] = asyncio.Queue()
            loop: asyncio.AbstractEventLoop = asyncio.get_running_loop()

            def _safe_put(item: str | None | Exception) -> None:
                """Enqueue a message, tolerating a closed loop after disconnect.

                If the event loop has already been torn down (e.g., the
                coroutine was cancelled and the loop is gone by the time
                the worker thread finishes), we log rather than silently
                dropping the message.

                Args:
                    item (str | None | Exception): The message or sentinel
                        to enqueue. ``None`` signals normal completion;
                        ``Exception`` signals failure.
                """
                try:
                    loop.call_soon_threadsafe(queue.put_nowait, item)
                except Exception as exc:
                    # CPython raises RuntimeError("Event loop is closed");
                    # uvloop or an OS-level SIGPIPE/EBADF during teardown
                    # can surface OSError / BrokenPipeError. Catch broadly
                    # so an exception from the worker thread still lands
                    # in the log rather than vanishing after disconnect.
                    logger.warning(
                        "Could not enqueue ingest message for collection '{}' (loop unavailable after disconnect): {}",
                        name,
                        exc,
                    )

            def progress_callback(msg: str) -> None:
                """Callback function to report progress during ingestion.

                Args:
                    msg (str): Progress message to be reported.
                """
                _safe_put(msg)

            async def run_ingestion() -> None:
                """Run the ingestion process in a separate thread.

                Logs cancellation and worker-thread exceptions explicitly
                so a disconnected client does not cause silent data loss.
                The underlying worker thread cannot be killed (Python has
                no safe thread-kill primitive), so on cancellation we
                only release the awaiting coroutine; the thread runs to
                completion and its output is discarded.
                """
                try:
                    await to_thread.run_sync(
                        ingest_module.ingest_docs,
                        physical,
                        batch_dir,
                        hybrid if hybrid is not None else True,
                        progress_callback,
                    )
                    _safe_put(None)
                except asyncio.CancelledError:
                    logger.warning(
                        "Ingestion task for collection '{}' cancelled; the "
                        "worker thread will continue until ingest_docs "
                        "returns and its output will be discarded.",
                        name,
                    )
                    raise
                except Exception as e:
                    logger.error("Ingestion of collection '{}' failed: {}", name, e)
                    _safe_put(e)

            ingestion_task = asyncio.create_task(run_ingestion())

            # Poll periodically so client disconnects are noticed even when
            # the worker is quiet (embedding batches, etc.). The interval is
            # a module-level constant so tests can override it.
            while True:
                try:
                    msg = await asyncio.wait_for(
                        queue.get(),
                        timeout=INGEST_DISCONNECT_POLL_INTERVAL_S,
                    )
                except TimeoutError:
                    if await request.is_disconnected():
                        ingestion_task.cancel()
                        logger.warning(
                            "Client disconnected during /ingest/upload for "
                            "collection '{}'; cancelled the awaiter. Worker "
                            "thread continues to completion.",
                            name,
                        )
                        return
                    continue
                if msg is None:
                    break
                if isinstance(msg, EmptyIngestionError):
                    yield _format_sse(
                        "warning",
                        {
                            "message": str(msg),
                            "collection": msg.collection_name,
                        },
                    )
                    yield _format_sse(
                        "ingestion_complete",
                        {
                            "collection": name,
                            "data_dir": str(batch_dir),
                            "empty": True,
                        },
                    )
                    return
                if isinstance(msg, Exception):
                    raise msg
                event_name = (
                    "warning"
                    if isinstance(msg, str) and msg.strip().lower().startswith("warning:")
                    else "ingestion_progress"
                )
                yield _format_sse(event_name, {"message": msg})

            yield _format_sse(
                "ingestion_complete",
                {"collection": name, "data_dir": str(batch_dir)},
            )
        except Exception as exc:  # pragma: no cover - streamed errors are logged
            logger.error("Error during streamed ingestion: {}", exc)
            # Include the exception class + message so the user can tell at a
            # glance whether the failure was, e.g., an embedding-endpoint
            # outage versus a Qdrant connectivity issue versus a parse error.
            yield _format_sse(
                "error",
                {
                    "message": f"Ingestion failed: {type(exc).__name__}: {exc}",
                },
            )

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/sources/preview", tags=["Sources"])
def preview_source(collection: str, file_hash: str, principal: str = Depends(resolve_principal)) -> FileResponse:
    """Serve a previously ingested source file the caller owns.

    ``collection`` is the caller's *logical* name; it is owner-gated and
    resolved to its owner-namespaced physical collection before the source
    store is touched (404 when the caller does not own it). This both prevents
    one user from previewing another's files and makes previews resolve under
    the correct physical path for namespaced users.

    Args:
        collection (str): The caller's logical collection name.
        file_hash (str): The hash of the file to preview.
        principal (str): The resolved request principal.

    Returns:
        FileResponse: A response containing the requested file.

    Raises:
        HTTPException: 404 when the caller does not own the collection or the
            file cannot be found.
    """
    physical = _require_owned_collection(collection, principal)
    path = _resolve_source_file_path(physical, file_hash)
    if path is None:
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path)


@app.post("/translate", tags=["Translate"])
def translate_text(payload: TranslateIn, principal: str = Depends(resolve_principal)) -> dict[str, Any]:
    """Translate a client-supplied snippet into the operator's locale.

    Authenticated for consistency, but not collection-scoped: it translates text
    the caller already holds, so there is nothing to leak and no store re-fetch.
    Fail-soft — a transport error returns ``ok: false`` with the original shape.

    Args:
        payload (TranslateIn): The snippet to translate.
        principal (str): The resolved request principal.

    Returns:
        dict[str, Any]: ``{ok, translation, model, target_lang, error}``.
    """
    result = translate(payload.text)
    return {
        "ok": result.ok,
        "translation": result.translation,
        "model": result.model,
        "target_lang": result.target_lang,
        "error": result.error,
    }
