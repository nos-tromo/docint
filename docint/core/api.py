"""FastAPI app exposing chat, ingestion, collection, and citation endpoints."""

import asyncio
import io
import json
import time
import zipfile
from collections.abc import AsyncIterator, Callable, Iterator, Sequence
from contextlib import asynccontextmanager, contextmanager
from dataclasses import asdict
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
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from loguru import logger
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel, Field, model_validator
from qdrant_client import models
from starlette.middleware.cors import CORSMiddleware

from docint import __version__
from docint.agents import (
    AgentOrchestrator,
    ClarificationConfig,
    ClarificationPolicy,
    ContextualUnderstandingAgent,
    QueryReformulationAgent,
    RAGRetrievalAgent,
    ResultValidationResponseAgent,
    RetrievalResult,
    SimpleClarificationAgent,
    SimpleUnderstandingAgent,
    Turn,
    is_weak_answer,
)
from docint.agents.history import build_prior_turn
from docint.cli import ingest as ingest_module
from docint.core.auth.principal import Principal, resolve_principal
from docint.core.errors import install_error_handlers
from docint.core.ingest.ingestion_pipeline import NoSupportedFilesError
from docint.core.jobs import IngestJobManager, IngestJobState, JobStatus, PushEvent
from docint.core.rag import RAG, EmptyIngestionError, IngestStats
from docint.core.retrieval_filters import (
    build_metadata_filters,
    build_qdrant_filter,
    normalize_numeric_bound,
)
from docint.core.search.fulltext import parse_keywords
from docint.core.state.session_manager import SessionCollectionMismatchError
from docint.utils.cursor import InvalidCursorError
from docint.utils.duration import format_elapsed
from docint.utils.env_cfg import (
    load_corrective_retry_env,
    load_frontend_env,
    load_hate_speech_env,
    load_host_env,
    load_image_ingestion_config,
    load_language_env,
    load_metrics_env,
    load_ner_env,
    load_path_env,
    load_resolution_env,
    load_response_validation_env,
    load_summary_env,
    resolve_enable_hybrid,
    set_offline_env,
)
from docint.utils.hashing import compute_file_hash
from docint.utils.logfmt import format_bytes
from docint.utils.logger_cfg import init_logger
from docint.utils.openai_cfg import EmbeddingEndpointError
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
# Announce the offline mode *after* the sink exists. Importing ``env_cfg``
# already applied the vars silently; logging it there would print on
# loguru's default handler, in a different format from every line after it.
set_offline_env()

# CORS allowlist for the Vite dev server during local development.
allowed_origins = load_host_env().cors_allowed_origins.split(",")


@asynccontextmanager
async def _lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Probe Qdrant on startup; close ingest-job subscriber streams on shutdown.

    The session store initializes (and migrates) eagerly here, and a failed
    migration is fatal: unlike Qdrant, an unwritable or stale sessions DB
    never self-heals, and serving anyway means every conversations query
    500s on the columns the skipped migration didn't add (this is exactly
    what a root-owned sessions volume did on a hardened host — ADR 0001).

    Qdrant is contacted lazily, so without the startup probe a mis-wired
    deployment (backend not on data-net, data-plane stack down) surfaces
    only at the first ingest or query. The probe logs a loud, actionable
    error but never blocks startup — Qdrant may come up after the backend,
    and the SQLite-backed endpoints work without it.

    ``GET /ingest/jobs/events`` never terminates on its own — each connection
    idles on a ping loop until the client disconnects. Without this, uvicorn's
    graceful shutdown (which waits on in-flight responses) would stall on
    every still-attached tab until the forced-exit timeout. ``job_manager`` is
    resolved as a module global at call time (defined further down this
    file), not at import time — the lifespan only ever runs long after the
    module has finished loading.

    Args:
        _app (FastAPI): The FastAPI application (unused; required by the
            lifespan protocol).

    Yields:
        None: Control while the application serves requests.
    """

    def _init_session_store() -> None:
        rag.ensure_session_manager().init_session_store_if_needed()

    await to_thread.run_sync(_init_session_store)
    await to_thread.run_sync(rag.probe_qdrant)
    await to_thread.run_sync(rag.probe_rerank_endpoint)
    await to_thread.run_sync(rag.reconcile_quantization)
    yield
    await job_manager.stop()


app = FastAPI(title="Document Intelligence", lifespan=_lifespan)
app.add_middleware(
    middleware_class=cast(Any, CORSMiddleware),
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
install_error_handlers(app)

# Prometheus metrics for the obs-plane scrape target. Aggregate request
# counters/histograms only (method, path template, status code, latency) —
# no document content, collection names, or user identifiers are recorded.
# Served without a principal dependency, like /version and /config, so
# obs-plane can scrape it unauthenticated.
if load_metrics_env().enabled:
    Instrumentator().instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)

rag = RAG(qdrant_collection="")
SIMULATED_STREAM_TOKEN_DELAY_SECONDS = 0.03
# Interval (seconds) between client-disconnect checks while a blocking sync
# generator (chat/summary streaming) is being drained on a worker thread.
# Tests may monkeypatch this to a smaller value.
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
    # The retry rides on the same LLM as validation: without a mismatch verdict
    # there is nothing to retry on, so gating them together keeps the two from
    # drifting into a state where one is armed and the other cannot fire.
    reformulator = (
        QueryReformulationAgent(llm=validation_llm)
        if load_corrective_retry_env().enabled and validation_llm is not None
        else None
    )

    return AgentOrchestrator(
        understanding=understanding,
        clarifier=_clarification_agent,
        retriever=retrieval_agent,
        responder=ResultValidationResponseAgent(
            enabled=validation_cfg.enabled,
            llm=validation_llm,
        ),
        policy=_clarification_policy,
        reformulator=reformulator,
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


def _require_owned_collection(logical_name: str, principal: Principal) -> str:
    """Resolve a caller-owned logical collection to its physical Qdrant name.

    The single ownership gate for collection-scoped endpoints. It mirrors
    :func:`_get_owned_report`: a collection the caller does not own (or that
    does not exist) is indistinguishable from "not found" (HTTP 404), so one
    user's collection names never leak to another. Admins may resolve
    another owner's collection via the request's ``owner`` query param
    (carried on ``Principal.requested_owner``); for everyone else
    ``effective_owner == name`` and behavior is exactly as before.

    Args:
        logical_name (str): The user-visible collection name from the request.
        principal (Principal): The resolved calling principal.

    Returns:
        str: The physical (owner-namespaced) Qdrant collection name to use.

    Raises:
        HTTPException: 400 if the name is blank; 404 if the caller does not own it.
    """
    name = (logical_name or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="Collection name required")
    physical = rag.ensure_collection_owner_manager().resolve(principal.effective_owner, name)
    if physical is None:
        raise HTTPException(status_code=404, detail=f"Collection '{name}' not found")
    return physical


def _resolve_request_collection(collection: str | None, principal: Principal) -> str:
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
        principal (Principal): The resolved calling principal.

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
def _scoped_collection(collection: str | None, principal: Principal) -> Iterator[str]:
    """Resolve + owner-gate a request collection and bind it for the engine.

    Combines :func:`_resolve_request_collection` with
    :meth:`docint.core.rag.RAG.collection_scope` so every ``rag`` call inside
    the block (and any anyio worker thread it spawns) reads the request's own
    physical collection rather than a shared global. Use this for synchronous,
    non-streaming endpoints; streaming endpoints must open the scope *inside*
    their event generator so it stays active while the body is consumed.

    Args:
        collection (str | None): The caller's logical collection name, if any.
        principal (Principal): The resolved calling principal.

    Yields:
        str: The resolved physical collection name.
    """
    physical = _resolve_request_collection(collection, principal)
    with rag.collection_scope(physical):
        yield physical


def _validate_requested_scope(
    requested: Sequence[str], physical: str, session_id: str | None, owner: str | None
) -> None:
    """Refuse a request-carried scope that cannot fit the chat context budget.

    Mirrors ``PUT /sessions/{id}/scope``: scoped answering splices the chosen
    chunks straight into the prompt, so an oversize selection is refused rather
    than truncated — a silently shortened evidence set produces an answer that
    looks complete and is not. Callers run this *before* opening a stream, so
    the refusal is a plain 422 instead of an in-stream error indistinguishable
    from a generation failure.

    A selection identical to the one already pinned was measured when it was
    pinned, so it is not measured again — that keeps the ordinary turn free of
    an extra Qdrant round trip.

    Args:
        requested (Sequence[str]): Chunk ids carried on the request.
        physical (str): Resolved physical collection to measure against.
        session_id (str | None): The session whose stored scope to compare to.
        owner (str | None): The principal that must own that session.

    Raises:
        HTTPException: 422 when the selection cannot fit the context budget.
    """
    if not requested:
        return
    stored: list[str] = rag.ensure_session_manager().get_scope(session_id, owner) if session_id else []
    if list(requested) == list(stored):
        return
    with rag.collection_scope(physical):
        measured = rag.measure_scope(list(requested))
    if not measured["fits"]:
        raise HTTPException(status_code=422, detail="Invalid request.")


def _apply_turn_scope(session_id: str, owner: str | None, requested: Sequence[str] | None) -> list[str]:
    """Return the chunk ids this turn answers from, pinning a new selection.

    Called once the session id exists (``start_session`` is what mints it), so
    a scope that arrived with the very first turn is both used for that turn
    and stored for the next one.

    Args:
        session_id (str): The resolved session for this turn.
        owner (str | None): The principal that owns it.
        requested (Sequence[str] | None): Chunk ids carried on the request, if
            any. ``None`` falls back to the session's stored scope.

    Returns:
        list[str]: The effective scope; empty means ordinary retrieval.
    """
    sessions = rag.ensure_session_manager()
    if requested is None:
        return list(sessions.get_scope(session_id, owner))
    ids = [str(entry) for entry in requested]
    if ids != list(sessions.get_scope(session_id, owner)):
        sessions.set_scope(session_id, owner, ids)
    return ids


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


def _reformulated_query(question: str, failed_query: str | None, reason: str | None) -> str | None:
    """Rewrite a retrieval query whose answer response validation rejected.

    Built fresh per call like the validator in :func:`_validation_payload`: the
    agent holds no state, so there is nothing to cache and nothing to leak
    between requests. Blocking — call it off the event loop.

    Args:
        question (str): The user's original question.
        failed_query (str | None): The retrieval query that produced the
            rejected answer.
        reason (str | None): The validator's reason for rejecting the answer.

    Returns:
        str | None: A fresh retrieval query, or ``None`` when none could be
            produced — no chat model configured, or the model declined. The
            caller treats ``None`` as "skip the retry".
    """
    if not getattr(rag, "text_model_id", None):
        return None
    try:
        llm = rag.text_model
    except Exception as exc:
        logger.warning("Failed to initialize reformulation LLM: {}", exc)
        return None

    return QueryReformulationAgent(llm=llm).reformulate(
        user_query=question,
        failed_query=failed_query,
        validation_reason=reason,
    )


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


class AdminOwnerCollections(BaseModel):
    """One foreign owner's logical collection names (admin listing)."""

    owner: str
    collections: list[str]


class AdminCollectionsOut(BaseModel):
    """Admin-shaped /collections/list response: own plus per-owner groups."""

    mine: list[str]
    others: list[AdminOwnerCollections]


class MetadataFilterIn(BaseModel):
    """Single metadata filter applied to retrieval queries.

    A filter targets either one ``field`` or several ``fields``. When several
    are given the rule matches if **any** of them matches, which is how a date
    bound covers both ``reference_metadata.timestamp`` (chunks and transcript
    segments) and ``reference_metadata.posting_timestamp`` (media artifacts
    linked to a posting) in one rule.
    """

    field: str = ""
    fields: list[str] = Field(default_factory=list)
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

    @model_validator(mode="after")
    def _require_a_target_field(self) -> "MetadataFilterIn":
        """Reject a rule that names no metadata key at all.

        Returns:
            MetadataFilterIn: The validated model.

        Raises:
            ValueError: When neither ``field`` nor ``fields`` is populated.
        """
        if not self.field.strip() and not [entry for entry in self.fields if entry.strip()]:
            raise ValueError("a metadata filter must name 'field' or 'fields'")
        return self

    @model_validator(mode="after")
    def _require_a_numeric_range_bound(self) -> "MetadataFilterIn":
        """Reject a range comparison whose bound is not a number.

        Qdrant's ``models.Range`` bounds are floats and there is no string
        equivalent, so such a rule compiles to nothing on every path and the
        query would run unfiltered — returning strictly more than the caller
        asked for. Refusing it is the only honest option. Numeric strings are
        accepted: an HTML text input cannot send a JSON number.

        Returns:
            MetadataFilterIn: The validated model.

        Raises:
            ValueError: When a range operator carries a non-numeric bound.
        """
        if self.operator in {"gt", "gte", "lt", "lte"} and normalize_numeric_bound(self.value) is None:
            raise ValueError(f"operator '{self.operator}' needs a numeric value")
        return self


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
    # Hand-picked chunk ids this turn must answer from. Present ⇒ it *is* the
    # scope for this turn and is pinned to the session; absent ⇒ the session's
    # stored scope still applies (``DELETE /sessions/{id}/scope`` clears it).
    # The scope travels with the request because the session row is minted by
    # the first turn, so a client with a selection has nowhere to write it
    # beforehand — installing it afterwards left that first answer unscoped
    # while the UI already claimed it was scoped.
    scope_chunk_ids: list[str] | None = None


class ScopeIn(BaseModel):
    """Request payload for pinning a search scope to a session."""

    chunk_ids: list[str] = Field(default_factory=list)


class ScopeOut(BaseModel):
    """A session's scope plus what it costs against the chat budget."""

    chunk_ids: list[str] = []
    est_tokens: int = 0
    usable_tokens: int = 0
    missing: int = 0


class SearchIn(BaseModel):
    """Request payload for a full-text keyword search."""

    question: str
    collection: str | None = None
    metadata_filters: list[MetadataFilterIn] = Field(default_factory=list)
    limit: int = Field(default=50, ge=1, le=500)
    cursor: str | None = None


class SearchOut(BaseModel):
    """Keyword-search hits plus the collection's search-index state."""

    #: ``ok`` — every point in the collection is indexed. ``partial`` — some
    #: are not, so the hit list is incomplete (a backfill is running or was
    #: interrupted); ``index_status.missing`` says how many. ``not_indexed`` —
    #: none are, so ``make search-index`` has never run here.
    status: Literal["ok", "partial", "not_indexed"]
    hits: list[dict[str, Any]] = []
    total: int | None = 0
    next_cursor: str | None = None
    index_status: dict[str, Any] = {}


class QueryOut(BaseModel):
    """Grounded answer plus retrieval provenance for a RAG query."""

    answer: str
    sources: list[dict[str, Any]] = []
    session_id: str
    graph_debug: dict[str, Any] | None = None
    retrieval_query: str | None = None
    coverage_unit: str | None = None
    retrieval_mode: str | None = None
    #: How many hand-picked chunks a ``retrieval_mode="scoped"`` turn answered
    #: from; absent on an ordinary retrieval.
    scoped_chunk_count: int | None = None
    #: Whether the sources went through the reranker: ``{"applied": bool,
    #: "error": str | None}``. ``applied=False`` means the reranker was
    #: unreachable and the sources are in raw retrieval order — a degraded
    #: turn that must not pass as a normal one. ``None`` when no reranker
    #: was in the loop (scoped turn, no sources).
    rerank: dict[str, Any] | None = None
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
    # True when the tree summarizer's SUMMARY_MAX_LLM_CALLS budget cut the
    # build short, so the summary reflects only part of the collection. The
    # SPA surfaces it as an explicit notice; `None` on payloads cached before
    # the flag existed.
    partial: bool | None = None


class SummarizeOut(BaseModel):
    """Response payload for a collection-level summary request.

    Documents the ``200`` (cache-hit) shape of ``POST /summarize`` and the
    identical ``200`` shape of ``GET /summarize`` -- both build their body with
    ``_cached_summary_payload``. ``job_id`` is unused in either; it exists only
    so this model also documents the POST's ``202`` (queued-build) shape. The endpoint declares
    ``response_model=None`` and returns explicit ``JSONResponse``s (its status
    code varies by outcome), so this class is documentation/typing only.
    """

    summary: str
    sources: list[dict[str, Any]] = []
    summary_diagnostics: SummaryDiagnosticsOut | None = None
    validation_checked: bool | None = None
    validation_mismatch: bool | None = None
    validation_reason: str | None = None
    job_id: str | None = None


class IngestDefaultsOut(BaseModel):
    """Deployment-default enrichment toggles for the ingest UI."""

    ner: bool
    hate_speech: bool


class IngestIn(BaseModel):
    """Request payload triggering ingestion into a named collection."""

    collection: str
    hybrid: bool | None = None
    ner: bool | None = None
    hate_speech: bool | None = None


class IngestFinalizeIn(IngestIn):
    """Finalize payload: an ingest request plus how long its upload took.

    A run starts when the client begins uploading, not when the job is
    created, so the client reports the leg the server never saw. It sends an
    *elapsed duration*, never a timestamp: no client clock is trusted, and the
    value is clamped server-side (:func:`docint.core.jobs._clamp_lead`) because
    it bounds a log line. The field is on this subclass rather than
    :class:`IngestIn` so the legacy synchronous ``POST /ingest``, which has no
    upload leg, does not advertise it.
    """

    upload_elapsed_ms: float | None = Field(default=None, ge=0)


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
    max_upload_bytes: int
    language: str


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


class HealthOut(BaseModel):
    """Dependency status report (currently: Qdrant reachability)."""

    status: str
    qdrant: bool


class WhoamiOut(BaseModel):
    """The resolved calling identity, for the SPA header."""

    username: str
    # Decorative only — the edge gateway's Authelia displayname, injected
    # alongside (not instead of) X-Auth-User. Never part of principal
    # resolution/identity; None when the gateway isn't in front (dev).
    display_name: str | None = None


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
    retried: bool | None = None
    retry_query: str | None = None


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
def get_frontend_config() -> dict[str, int | str]:
    """Return deploy-time frontend configuration for the SPA.

    Served without a principal dependency so the SPA can read it on first load,
    before any collection or session exists. Values are read from environment
    variables on each call (see :func:`docint.utils.env_cfg.load_frontend_env`
    and :func:`docint.utils.env_cfg.load_language_env`).

    Returns:
        dict[str, int | str]: ``graph_top_k``, ``graph_max_top_k``,
        ``collection_timeout``, ``max_upload_bytes`` (the per-request upload
        ceiling nginx enforces, which the SPA uses to size its upload batches),
        and ``language`` (the active ``RESPONSE_LANGUAGE`` locale, ``"en"`` or
        ``"de"``).
    """
    cfg = load_frontend_env()
    return {
        "graph_top_k": cfg.graph_top_k,
        "graph_max_top_k": cfg.graph_max_top_k,
        "collection_timeout": cfg.collection_timeout,
        "max_upload_bytes": cfg.max_upload_bytes,
        "language": load_language_env().code,
    }


@app.get("/config/ingest-defaults", response_model=IngestDefaultsOut, tags=["Meta"])
def get_ingest_defaults() -> dict[str, bool]:
    """Return the deployment's default enrichment toggles for the ingest UI.

    Served without a principal dependency (like ``/config``) so the SPA can
    seed its checkboxes; the values mirror ``NER_ENABLED`` and
    ``ENABLE_HATE_SPEECH_DETECTION``.

    Returns:
        dict[str, bool]: ``ner`` and ``hate_speech`` deployment defaults.
    """
    return {
        "ner": load_ner_env().enabled,
        "hate_speech": load_hate_speech_env().enabled,
    }


@app.get("/version", response_model=VersionOut, tags=["Meta"])
def get_version() -> VersionOut:
    """Return the running app version (unauthenticated, no principal)."""
    return VersionOut(version=__version__)


@app.get("/health", response_model=HealthOut, tags=["Meta"])
async def get_health() -> HealthOut:
    """Report dependency status, re-running the Qdrant probe on demand.

    Unauthenticated like ``/version`` so `make health` and the container
    tooling can call it without a principal. Always HTTP 200 — the Docker
    healthcheck watches ``/version`` (backend liveness), while this endpoint
    reports whether the vector store is usable *right now*, not just at the
    startup probe.

    Returns:
        HealthOut: ``status="ok"`` when Qdrant answered its readiness
            endpoint, ``status="degraded"`` otherwise.
    """
    qdrant_ok = await to_thread.run_sync(rag.probe_qdrant)
    return HealthOut(status="ok" if qdrant_ok else "degraded", qdrant=qdrant_ok)


@app.get("/whoami", response_model=WhoamiOut, tags=["Meta"])
def get_whoami(
    request: Request,
    principal: Principal = Depends(resolve_principal),  # noqa: B008 — FastAPI dependency marker
) -> WhoamiOut:
    """Return the resolved calling identity, for the SPA's AppHeader.

    Principal-gated like every collection-scoped endpoint (401 without a
    trusted header or a configured dev default identity) — unlike ``/config``
    and ``/version``, which are deliberately unauthenticated.

    ``display_name`` is read straight off the ``X-Auth-Name`` request header
    (Authelia's displayname, injected by the edge gateway) and is purely
    decorative — it plays no part in identity/principal resolution, unlike
    ``username``. ``None`` when the header is absent (dev without the
    gateway in front).

    Args:
        request (Request): The incoming request, for the decorative
            ``X-Auth-Name`` header.
        principal (Principal): The resolved request principal.

    Returns:
        WhoamiOut: The caller's resolved principal name plus, if present,
            the gateway's decorative display name.
    """
    return WhoamiOut(username=principal.name, display_name=request.headers.get("X-Auth-Name"))


@app.get("/collections/list", response_model=list[str] | AdminCollectionsOut, tags=["Collections"])
def collections_list(
    all: bool = False,
    principal: Principal = Depends(resolve_principal),  # noqa: B008 — FastAPI dependency marker
) -> list[str] | AdminCollectionsOut:
    """List collections: the caller's own, or (admins, with ``?all=true``) everyone's.

    Collections are owner-scoped: a caller only sees the collections they
    ingested themselves. Names are the user-visible logical names, not the
    owner-namespaced physical Qdrant names. Without ``all`` the response is
    the caller's logical names exactly as before. With ``all=true`` an admin
    additionally receives every other owner's collections grouped per owner;
    for non-admins the flag is silently ignored (plain list) so collection
    existence never leaks.

    Args:
        all (bool): When true and the caller is an admin, also return every
            other owner's collections grouped by owner. Ignored otherwise.
        principal (Principal): The resolved request principal.

    Returns:
        list[str] | AdminCollectionsOut: The caller's collection names,
            sorted; or, for an admin with ``all=true``, ``mine`` (the
            caller's own names) plus ``others`` (every other owner's names,
            grouped and sorted by owner).

    Raises:
        HTTPException: If an error occurs while listing collections.
    """
    try:
        mgr = rag.ensure_collection_owner_manager()
        mine = mgr.list_for(principal.name)
        if not (all and principal.is_admin):
            return mine
        others: dict[str, list[str]] = {}
        for owner, logical in mgr.list_all():
            if owner is None or owner == principal.name:
                # None-owner legacy rows are unreachable in production
                # (no default identity => 401 before any resolve).
                continue
            others.setdefault(owner, []).append(logical)
        return AdminCollectionsOut(
            mine=mine,
            others=[AdminOwnerCollections(owner=o, collections=c) for o, c in others.items()],
        )
    except Exception as e:
        logger.opt(exception=e).error("Error listing collections")
        raise HTTPException(status_code=500, detail="Request failed.") from e


@app.post("/collections/select", response_model=SelectCollectionOut, tags=["Collections"])
def collections_select(
    payload: SelectCollectionIn,
    principal: Principal = Depends(resolve_principal),  # noqa: B008 — FastAPI dependency marker
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
        principal (Principal): The resolved request principal.

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
def collections_delete(name: str, principal: Principal = Depends(resolve_principal)) -> dict[str, bool]:  # noqa: B008 — FastAPI dependency marker
    """Delete a collection resolved under the caller's effective owner.

    Deleting a collection the caller's effective owner does not own (or one
    that does not exist) is a 404, so a user can never delete another user's
    data outside their effective-owner scope — including an admin's own
    namespace when a foreign ``owner`` is not explicitly requested. The Qdrant
    collection is dropped first; only then is the ownership mapping removed, so
    a failed Qdrant delete leaves ownership intact for retry. The collection's
    chat sessions are cascade-deleted after the Qdrant collection is dropped and
    before the ownership mapping is removed.

    Args:
        name (str): The user-visible collection name to delete.
        principal (Principal): The resolved request principal.

    Returns:
        dict[str, bool]: A dictionary indicating success.

    Raises:
        HTTPException: 404 if the caller does not own it; 500 on backend failure.
    """
    physical = _require_owned_collection(name, principal)
    try:
        rag.delete_collection(physical)
        deleted_sessions = rag.ensure_session_manager().delete_sessions_for_collection(physical)
        if deleted_sessions:
            logger.info("Deleted {} chat session(s) pinned to collection '{}'.", deleted_sessions, name)
        rag.ensure_collection_owner_manager().delete(principal.effective_owner, name)
        return {"ok": True}
    except HTTPException:
        raise
    except Exception as e:
        logger.opt(exception=e).error("Error deleting collection")
        raise HTTPException(status_code=500, detail="Request failed.") from e


@app.put("/sessions/{session_id}/scope", response_model=ScopeOut, tags=["Sessions"])
def set_session_scope(
    session_id: str,
    payload: ScopeIn,
    collection: str | None = None,
    principal: Principal = Depends(resolve_principal),  # noqa: B008 — FastAPI dependency marker
) -> ScopeOut:
    """Restrict a session's answers to a hand-picked set of chunks.

    Refuses a selection larger than the chat context window rather than
    truncating it: scoped answering splices the chunks straight into the
    prompt, and silently dropping part of an investigator's evidence would
    produce an answer that looks complete and is not.

    Args:
        session_id (str): The session to scope.
        payload (ScopeIn): The chunk ids to answer from.
        collection (str | None): Caller's logical collection, owner-gated.
        principal (Principal): The resolved request principal.

    Returns:
        ScopeOut: The stored scope and its measured cost.

    Raises:
        HTTPException: 404 when the session is missing or not owned, 422 when
            the selection cannot fit the context budget, 500 on failure.
    """
    try:
        with _scoped_collection(collection, principal):
            measured = rag.measure_scope(payload.chunk_ids)
            if payload.chunk_ids and not measured["fits"]:
                raise HTTPException(status_code=422, detail="Invalid request.")
            stored = rag.ensure_session_manager().set_scope(
                session_id,
                principal.effective_owner,
                payload.chunk_ids,
            )
        if not stored:
            raise HTTPException(status_code=404, detail="Not found.")
        return ScopeOut(
            chunk_ids=list(payload.chunk_ids),
            est_tokens=int(measured["est_tokens"]),
            usable_tokens=int(measured["usable_tokens"]),
            missing=int(measured["missing"]),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.opt(exception=e).error("Error setting session scope")
        raise HTTPException(status_code=500, detail="Request failed.") from e


@app.delete("/sessions/{session_id}/scope", response_model=ScopeOut, tags=["Sessions"])
def clear_session_scope(
    session_id: str,
    principal: Principal = Depends(resolve_principal),  # noqa: B008 — FastAPI dependency marker
) -> ScopeOut:
    """Return a session to normal retrieval.

    Args:
        session_id (str): The session to unscope.
        principal (Principal): The resolved request principal.

    Returns:
        ScopeOut: An empty scope.

    Raises:
        HTTPException: 404 when the session is missing or not owned.
    """
    if not rag.ensure_session_manager().clear_scope(session_id, principal.effective_owner):
        raise HTTPException(status_code=404, detail="Not found.")
    return ScopeOut()


class ChunkOut(BaseModel):
    """One chunk's full text, for expanding a search hit."""

    id: str
    text: str


@app.get("/search/chunk", response_model=ChunkOut, tags=["Query"])
def get_search_chunk(
    id: str,
    collection: str | None = None,
    principal: Principal = Depends(resolve_principal),  # noqa: B008 — FastAPI dependency marker
) -> ChunkOut:
    """Return one chunk's full text.

    Search hits carry a capped preview; this backs expanding a single hit
    without inflating every search response with text most hits never need.

    Args:
        id (str): Qdrant point id from a search hit.
        collection (str | None): Caller's logical collection, owner-gated.
        principal (Principal): The resolved request principal.

    Returns:
        ChunkOut: The chunk id and its full text.

    Raises:
        HTTPException: 404 when the chunk is gone or carries no text — a
            re-ingested collection mints new ids, and an empty string would
            read as an empty chunk rather than a missing one. 500 on failure.
    """
    try:
        with _scoped_collection(collection, principal):
            text = rag.get_chunk_text(id)
    except HTTPException:
        raise
    except Exception as e:
        logger.opt(exception=e).error("Error fetching chunk text")
        raise HTTPException(status_code=500, detail="Request failed.") from e
    if text is None:
        raise HTTPException(status_code=404, detail="Not found.")
    return ChunkOut(id=id, text=text)


@app.post("/search", response_model=SearchOut, tags=["Query"])
def search_collection(
    payload: SearchIn,
    principal: Principal = Depends(resolve_principal),  # noqa: B008 — FastAPI dependency marker
) -> SearchOut:
    """Return chunks containing every keyword in the query.

    Pure local lookup — one native Qdrant scroll, no embedding call and no
    inference. Keywords are ANDed and order-independent; matching is
    case-insensitive and prefix-based, so the head of a compound finds the
    compound.

    A collection that has never been backfilled returns ``status:
    "not_indexed"`` with no hits, so an empty ``hits`` list under ``status:
    "ok"`` means "no matches" and nothing else.

    Args:
        payload (SearchIn): Query, optional collection, filters and paging.
        principal (Principal): The resolved request principal.

    Returns:
        SearchOut: Hits, exact total, next cursor and index state.

    Raises:
        HTTPException: 422 for an unusable query, 400/404 from collection
            resolution, 500 on unexpected failure.
    """
    # Validate at the boundary rather than deep in the RAG layer: an unusable
    # query should be refused before a collection is even resolved. An empty
    # keyword list would otherwise reach the search as "match everything".
    if not parse_keywords(payload.question):
        raise HTTPException(status_code=422, detail="Invalid request.")

    try:
        with _scoped_collection(payload.collection, principal):
            data = rag.search_fulltext(
                payload.question,
                base_filter=build_qdrant_filter(payload.metadata_filters),
                limit=payload.limit,
                cursor=payload.cursor,
            )
        return SearchOut(**data)
    except HTTPException:
        raise
    except Exception as e:
        logger.opt(exception=e).error("Error running collection search")
        raise HTTPException(status_code=500, detail="Request failed.") from e


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

        _validate_requested_scope(
            payload.scope_chunk_ids or [],
            physical,
            payload.session_id,
            principal.effective_owner,
        )

        with rag.collection_scope(physical):
            if getattr(rag, "query_engine", None) is None:
                if getattr(rag, "index", None) is None:
                    rag.create_index()
                rag.create_query_engine()

            if payload.retrieval_mode == "stateless":
                # Stateless mode has no session to pin a scope to, so the
                # request is the only place one can come from.
                stateless_scope = [str(entry) for entry in payload.scope_chunk_ids or []]
                retrieval_query = payload.question
                graph_debug: dict[str, Any] | None = None
                expand_with_debug = getattr(rag, "expand_query_with_graph_with_debug", None)
                if callable(expand_with_debug) and not stateless_scope:
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
                    scoped_node_ids=stateless_scope or None,
                )
                if graph_debug is not None:
                    data["graph_debug"] = graph_debug
                session_id = payload.session_id or "stateless"
            else:
                session_id = rag.start_session(
                    payload.session_id,
                    owner=principal.effective_owner,
                )
                # A session pinned to hand-picked chunks answers only from
                # them — no vector retrieval at all. Resolved here, once the
                # id exists, so a scope carried by the very first turn is used
                # for that turn and pinned for the next.
                scoped_node_ids = _apply_turn_scope(
                    session_id,
                    principal.effective_owner,
                    payload.scope_chunk_ids,
                )
                data = rag.chat(
                    payload.question,
                    session_id=session_id,
                    owner=principal.effective_owner,
                    metadata_filters=metadata_filters,
                    metadata_filters_active=(metadata_filters is not None or bool(vector_store_kwargs)),
                    metadata_filter_rules=payload.metadata_filters,
                    vector_store_kwargs=vector_store_kwargs or None,
                    scoped_node_ids=scoped_node_ids or None,
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
            "scoped_chunk_count": (
                data.get("scoped_chunk_count") if isinstance(data, dict) and retrieval_mode == "scoped" else None
            ),
            "rerank": data.get("rerank") if isinstance(data, dict) else None,
            **validation,
        }
    except HTTPException:
        raise
    except SessionCollectionMismatchError as exc:
        raise HTTPException(status_code=409, detail="Session is pinned to a different collection.") from exc
    except Exception as exc:
        logger.opt(exception=exc).error("Unexpected error processing query")
        raise HTTPException(status_code=500, detail="Request failed.") from exc


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
    if payload.retrieval_mode != "stateless":
        session_owner = principal.effective_owner
        # Up-front collection-pin check so a mismatch is a clean 409 rather than
        # an in-stream SSE error: resuming an owned session against a different
        # collection must be refused before any retrieval runs.
        if payload.session_id:
            pinned = rag.ensure_session_manager().get_session_collection(payload.session_id, session_owner)
            if pinned is not None and pinned != physical:
                raise HTTPException(
                    status_code=409,
                    detail="Session is pinned to a different collection.",
                )

    # Same reasoning as the pin check above: an unaffordable scope must be a
    # clean 422 before the SSE body opens, not an in-stream error.
    _validate_requested_scope(
        payload.scope_chunk_ids or [],
        physical,
        payload.session_id,
        session_owner,
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
            if payload.retrieval_mode == "stateless":
                # Stateless mode has no session to pin a scope to, so the
                # request is the only place one can come from.
                stateless_scope = [str(entry) for entry in payload.scope_chunk_ids or []]
                retrieval_query = payload.question
                graph_debug: dict[str, Any] | None = None
                expand_with_debug = getattr(rag, "expand_query_with_graph_with_debug", None)
                if callable(expand_with_debug) and not stateless_scope:
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
                    scoped_node_ids=stateless_scope or None,
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
                # Resolved once, on the first pass, and reused by the corrective
                # retry. The prior turn especially: recomputing it after the
                # first pass has persisted would bind the answer this turn is
                # replacing as that same turn's own context.
                stream_state: dict[str, Any] = {}

                def _make_chat_stream(
                    question: str | None = None,
                    *,
                    replace_turn_idx: int | None = None,
                ) -> Iterator[Any]:
                    """Build the blocking chat stream off the event loop.

                    Session start, history load, and the retrieval/LLM stream
                    are all synchronous and so run on the worker thread driven
                    by ``_aiter_sync_gen``.

                    Args:
                        question (str | None): Query to answer. ``None`` (the
                            first pass) uses the user's question and keeps this
                            endpoint's internal retrieval rewrite; a value (the
                            corrective retry) is already a retrieval query, so
                            rewriting it again would undo the correction.
                        replace_turn_idx (int | None): Overwrite this persisted
                            turn instead of appending one.

                    Returns:
                        Iterator[Any]: The sync chat-chunk generator.
                    """
                    # The React chat UI calls /stream_query, so this is where
                    # generation-time history is wired: bind the prior
                    # user/assistant exchange (owner-scoped) onto the synthesis
                    # templates while keeping this endpoint's own internal
                    # retrieval rewrite (``skip_query_rewrite=False``).
                    if "session_id" not in stream_state:
                        resolved_id = rag.start_session(payload.session_id, owner=session_owner)
                        stream_state["session_id"] = resolved_id
                        stream_state["prior_turn"] = (
                            build_prior_turn(rag.sessions.get_session_history(resolved_id, owner=session_owner))
                            if rag.sessions is not None
                            else None
                        )
                        # A session pinned to hand-picked chunks answers only
                        # from them — no vector retrieval at all. Resolved here,
                        # once the id exists, so a scope carried by the very
                        # first turn is used for that turn and pinned for the
                        # next.
                        stream_state["scoped_node_ids"] = (
                            _apply_turn_scope(
                                resolved_id,
                                session_owner,
                                payload.scope_chunk_ids,
                            )
                            or None
                        )
                    return cast(
                        "Iterator[Any]",
                        rag.stream_chat(
                            payload.question if question is None else question,
                            session_id=stream_state["session_id"],
                            owner=session_owner,
                            metadata_filters=metadata_filters,
                            metadata_filters_active=(metadata_filters is not None or bool(vector_store_kwargs)),
                            metadata_filter_rules=payload.metadata_filters,
                            vector_store_kwargs=vector_store_kwargs or None,
                            prior_turn=stream_state["prior_turn"],
                            skip_query_rewrite=question is not None,
                            scoped_node_ids=stream_state["scoped_node_ids"],
                            replace_turn_idx=replace_turn_idx,
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
            # session_manager.stream_chat; the stateless branch doesn't
            # persist a turn at all and so won't carry it.
            turn_idx = payload_out.pop("turn_idx", None)

            # Corrective retry: an answer the validator rejected *and* that is
            # itself weak gets one more attempt with a reformulated query. The
            # gate is deliberately narrow — a mismatched but substantive answer
            # is still worth showing, a scoped turn runs no retrieval for a new
            # query to change, and without a persisted turn there is nothing to
            # overwrite, which would leave the user asking twice.
            # Session mode only, and named as such rather than inferred from the
            # absent turn_idx: the retry re-enters ``_make_chat_stream``, which
            # only exists on that branch.
            retry_query: str | None = None
            if (
                payload.retrieval_mode != "stateless"
                and isinstance(turn_idx, int)
                and validation.get("validation_mismatch") is True
                and is_weak_answer(answer)
                and payload_out.get("retrieval_mode") != "scoped"
                and load_corrective_retry_env().enabled
            ):
                try:
                    retry_query = await to_thread.run_sync(
                        _reformulated_query,
                        payload.question,
                        stream_retrieval_query,
                        cast("str | None", validation.get("validation_reason")),
                    )
                except Exception as exc:
                    logger.opt(exception=exc).warning("Corrective retry reformulation failed")

            if retry_query:
                # Announced before the second stream opens: the SPA discards
                # the rejected answer on this frame, so a silent swap is
                # impossible even if everything after it fails.
                yield f"data: {json.dumps({'retry': {'query': retry_query}})}\n\n"

                def _make_retry_stream() -> Iterator[Any]:
                    """Build the retry's chat stream, overwriting the first turn.

                    Returns:
                        Iterator[Any]: The sync chat-chunk generator.
                    """
                    return _make_chat_stream(retry_query, replace_turn_idx=cast(int, turn_idx))

                retry_payload: dict[str, Any] | None = None
                retry_answer = ""
                retry_ok = True
                try:
                    async for chunk in _aiter_sync_gen(_make_retry_stream, request):
                        if isinstance(chunk, str):
                            retry_answer += chunk
                            yield f"data: {json.dumps({'token': chunk})}\n\n"
                        elif isinstance(chunk, dict):
                            retry_payload = chunk
                except Exception as exc:
                    # Never turn a delivered answer into an error frame: the
                    # first attempt's envelope still describes what was
                    # persisted, so fall back to reporting that.
                    logger.opt(exception=exc).warning("Corrective retry failed; keeping the first answer")
                    retry_ok = False

                if retry_ok:
                    payload_out = dict(retry_payload or {})
                    payload_out.pop("turn_idx", None)
                    retry_final = str(payload_out.get("response") or payload_out.get("answer") or "") or retry_answer
                    retry_sources = payload_out.get("sources")
                    if not isinstance(retry_sources, list):
                        retry_sources = cast(list[dict[str, Any]], [])
                    # Validated against the user's original question, never the
                    # reformulation: the reformulation is a retrieval tactic,
                    # and grading the answer against it would let a drifting
                    # rewrite mark its own homework.
                    validation = _validation_payload(
                        question=payload.question,
                        answer=retry_final,
                        sources=retry_sources,
                        summary_diagnostics=(
                            payload_out.get("summary_diagnostics")
                            if isinstance(payload_out.get("summary_diagnostics"), dict)
                            else None
                        ),
                        retrieval_query=str(payload_out.get("retrieval_query") or retry_query),
                    )
                    payload_out.update(validation)
                    payload_out["retried"] = True
                    payload_out["retry_query"] = retry_query

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
                        retried=cast("bool | None", payload_out.get("retried")),
                        retry_query=cast("str | None", payload_out.get("retry_query")),
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
        except EmbeddingEndpointError:
            # Retrieval could not embed the query, so no generation was ever
            # attempted. Reported apart from `generation_failed`, which reads
            # as "the chat model failed" and points at the wrong service; the
            # exception's own message names the endpoint and the env var, and
            # stays in the logs where an internal address belongs.
            logger.exception("Dense embedding endpoint unusable during SSE generation")
            yield f"data: {json.dumps({'error': 'Internal server error', 'code': 'embedding_unavailable'})}\n\n"
        except ValueError as exc:
            msg = str(exc)
            if "context window" in msg.lower() or "context size" in msg.lower():
                logger.warning("Context window overflow during SSE generation: {}", msg)
                yield f"data: {json.dumps({'error': 'Internal server error', 'code': 'context_overflow'})}\n\n"
            else:
                logger.exception("Stream error during SSE generation")
                yield f"data: {json.dumps({'error': 'Internal server error', 'code': 'generation_failed'})}\n\n"
        except Exception:
            logger.exception("Stream error during SSE generation")
            yield f"data: {json.dumps({'error': 'Internal server error', 'code': 'generation_failed'})}\n\n"

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
            if getattr(rag, "index", None) is None:
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


@app.get("/collections/ner", tags=["Query"], deprecated=True)
def get_collection_ner(
    refresh: bool = False,
    collection: str | None = None,
    principal: Principal = Depends(resolve_principal),  # noqa: B008 — FastAPI dependency marker
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
        principal (Principal): The resolved request principal.

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
        logger.opt(exception=e).error("Error fetching collection NER")
        raise HTTPException(status_code=500, detail="Request failed.") from e


@app.get("/collections/hate-speech", tags=["Query"])
def get_collection_hate_speech(
    cursor: str | None = None,
    limit: int = Query(default=0, ge=0, le=500),
    category: str | None = None,
    min_confidence: str | None = None,
    collection: str | None = None,
    principal: Principal = Depends(resolve_principal),  # noqa: B008 — FastAPI dependency marker
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
        principal (Principal): The resolved request principal.

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
        raise HTTPException(status_code=400, detail="Invalid request.") from e
    except Exception as e:
        logger.opt(exception=e).error("Error fetching collection hate-speech results")
        raise HTTPException(status_code=500, detail="Request failed.") from e


@app.get("/collections/ner/sources", tags=["Query"])
def get_collection_ner_sources(
    cursor: str | None = None,
    limit: int = Query(default=50, ge=1, le=500),
    entity_key: str | None = None,
    entity_text: str | None = None,
    entity_type: str | None = None,
    entity_merge_mode: Literal["orthographic", "exact", "resolved"] = Query(default="orthographic"),
    collection: str | None = None,
    principal: Principal = Depends(resolve_principal),  # noqa: B008 — FastAPI dependency marker
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
        principal (Principal): The resolved request principal.

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
        raise HTTPException(status_code=400, detail="Invalid request.") from e
    except Exception as e:
        logger.opt(exception=e).error("Error fetching collection NER sources")
        raise HTTPException(status_code=500, detail="Request failed.") from e


@app.post("/collections/ner/warm", tags=["Query"])
async def warm_collection_ner(
    collection: str | None = None,
    principal: Principal = Depends(resolve_principal),  # noqa: B008 — FastAPI dependency marker
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
        principal (Principal): The resolved request principal.

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
        logger.opt(exception=e).error("Error warming collection NER aggregate")
        raise HTTPException(status_code=500, detail="Request failed.") from e


@app.get("/collections/ner/stats", response_model=NERStatsOut, tags=["Query"])
def get_collection_ner_stats(
    top_k: int = 15,
    min_mentions: int = 2,
    entity_type: str | None = None,
    include_relations: bool = True,
    entity_merge_mode: Literal["orthographic", "exact", "resolved"] = Query(default="orthographic"),
    collection: str | None = None,
    principal: Principal = Depends(resolve_principal),  # noqa: B008 — FastAPI dependency marker
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
        principal (Principal): The resolved request principal.

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
        logger.opt(exception=e).error("Error fetching collection NER stats")
        raise HTTPException(status_code=500, detail="Request failed.") from e


@app.get("/collections/ner/search", response_model=NERSearchOut, tags=["Query"])
def search_collection_ner_entities(
    q: str = "",
    entity_type: str | None = None,
    limit: int = 100,
    entity_merge_mode: Literal["orthographic", "exact", "resolved"] = Query(default="orthographic"),
    collection: str | None = None,
    principal: Principal = Depends(resolve_principal),  # noqa: B008 — FastAPI dependency marker
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
        principal (Principal): The resolved request principal.

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
        logger.opt(exception=e).error("Error searching collection entities")
        raise HTTPException(status_code=500, detail="Request failed.") from e


@app.get("/collections/ner/graph", response_model=NERGraphOut, tags=["Query"])
def get_collection_ner_graph(
    top_k_nodes: int | None = Query(default=None, ge=1),
    min_edge_weight: int = Query(default=1, ge=1),
    entity_merge_mode: Literal["orthographic", "exact", "resolved"] = Query(default="orthographic"),
    collection: str | None = None,
    principal: Principal = Depends(resolve_principal),  # noqa: B008 — FastAPI dependency marker
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
        principal (Principal): The resolved request principal.

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
        logger.opt(exception=e).error("Error building collection NER graph")
        raise HTTPException(status_code=500, detail="Request failed.") from e


@app.post("/collections/entities/resolve", tags=["Query"])
def resolve_collection_entities(
    collection: str | None = None,
    principal: Principal = Depends(resolve_principal),  # noqa: B008 — FastAPI dependency marker
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
        principal (Principal): The resolved request principal.

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
        logger.opt(exception=e).error("Error resolving collection entities")
        raise HTTPException(status_code=500, detail="Request failed.") from e


@app.get("/collections/documents", tags=["Query"])
def get_collection_documents(
    cursor: str | None = None,
    limit: int = Query(default=0, ge=0, le=500),
    collection: str | None = None,
    principal: Principal = Depends(resolve_principal),  # noqa: B008 — FastAPI dependency marker
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
        principal (Principal): The resolved request principal.

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
        raise HTTPException(status_code=400, detail="Invalid request.") from e
    except Exception as e:
        logger.opt(exception=e).error("Error fetching collection documents")
        raise HTTPException(status_code=500, detail="Request failed.") from e


@app.get("/collections/documents/count", tags=["Query"])
def get_collection_documents_count(
    collection: str | None = None,
    principal: Principal = Depends(resolve_principal),  # noqa: B008 — FastAPI dependency marker
) -> dict[str, int]:
    """Return the number of unique documents in the caller's collection.

    Backed by the same per-collection cache as ``/collections/documents``
    pagination, so the first call after a collection switch pays the
    Qdrant scroll once and the dashboard KPI then reads from cache.

    Args:
        collection (str | None): Caller's logical collection; owner-gated and
            scoped per request, falling back to the process default when omitted.
        principal (Principal): The resolved request principal.
    """
    try:
        with _scoped_collection(collection, principal):
            return {"count": rag.get_document_count()}
    except HTTPException:
        raise
    except Exception as e:
        logger.opt(exception=e).error("Error fetching collection document count")
        raise HTTPException(status_code=500, detail="Request failed.") from e


@app.get("/collections/documents/summary", response_model=DocumentsSummaryOut, tags=["Query"])
def get_collection_documents_summary(
    collection: str | None = None,
    principal: Principal = Depends(resolve_principal),  # noqa: B008 — FastAPI dependency marker
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
        principal (Principal): The resolved request principal.
    """
    try:
        with _scoped_collection(collection, principal):
            return rag.get_document_summary()
    except HTTPException:
        raise
    except Exception as e:
        logger.opt(exception=e).error("Error fetching collection document summary")
        raise HTTPException(status_code=500, detail="Request failed.") from e


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
        principal (str): The resolved request principal name.

    Returns:
        dict[str, Any]: The report, including its ordered items.

    Raises:
        HTTPException: 404 when the report is missing or owned by another principal.
    """
    report = rag.ensure_report_manager().get_report(report_id, principal)
    if report is None:
        raise HTTPException(status_code=404, detail="Report not found.")
    return report


def _thumbnail_from_point(payload: dict[str, Any]) -> dict[str, Any] | None:
    """Shape a companion point's stored thumbnail into the snapshot's frozen form.

    Args:
        payload (dict[str, Any]): The ``_images`` point payload.

    Returns:
        dict[str, Any] | None: The ``thumbnail`` object to freeze, or ``None``
        when the point predates thumbnails.
    """
    b64 = payload.get("thumbnail_b64")
    if not b64 or not isinstance(b64, str):
        return None
    mime = str(payload.get("thumbnail_mime") or "image/jpeg")
    source_type = str(payload.get("source_type") or "")
    return {
        "data_uri": f"data:{mime};base64,{b64}",
        "width": payload.get("width"),
        "height": payload.get("height"),
        "kind": "video_keyframe" if "keyframe" in source_type else "image",
    }


def _enrich_snapshot_thumbnails(
    snapshot: dict[str, Any],
    artifact_type: str,
    report: dict[str, Any],
    principal: Principal,
) -> dict[str, Any]:
    """Freeze visual evidence into a report snapshot at add-time.

    Sources and findings that carry image identity (``image_id`` from the
    ``_images`` companion) gain a ``thumbnail`` object holding a data URI, so
    the stored snapshot stays self-contained: the SPA and every export render
    it with no Qdrant access, and the report survives re-ingestion or
    collection deletion like the rest of the frozen snapshot.

    The companion collection is derived from the *report's own* collection via
    the caller's ownership mapping — a snapshot's ``image_collection`` is
    treated as a cross-check, never as an address, so a crafted snapshot
    cannot make the server read an arbitrary collection.

    Fail-soft by contract: any failure (companion missing, Qdrant down, point
    without a thumbnail) returns the snapshot untouched — a text-only item,
    never a refused add.

    Args:
        snapshot (dict[str, Any]): The caller-supplied snapshot (mutated copy semantics: edited in place).
        artifact_type (str): The report item's artifact type.
        report (dict[str, Any]): The owned report the item is being added to.
        principal (Principal): The resolved request principal.

    Returns:
        dict[str, Any]: The snapshot, enriched where possible.
    """
    try:
        containers: list[dict[str, Any]] = []
        if artifact_type == "chat_answer":
            containers = [s for s in (snapshot.get("sources") or []) if isinstance(s, dict) and s.get("image_id")]
        elif artifact_type in ("entity_finding", "hate_speech_finding") and snapshot.get("image_id"):
            containers = [snapshot]
        if not containers:
            return snapshot

        logical = str(report.get("collection_name") or "").strip()
        if not logical:
            return snapshot
        try:
            physical = _require_owned_collection(logical, principal)
        except HTTPException:
            return snapshot
        template = (load_image_ingestion_config().collection_name or "").strip() or "{collection}_images"
        companion = template.format(collection=physical) if "{collection}" in template else template

        containers = [c for c in containers if not c.get("image_collection") or c.get("image_collection") == companion]
        image_ids = sorted({str(c["image_id"]) for c in containers})
        client = getattr(rag, "qdrant_client", None)
        if not image_ids or client is None:
            return snapshot

        points, _ = client.scroll(
            collection_name=companion,
            scroll_filter=models.Filter(
                must=[models.FieldCondition(key="image_id", match=models.MatchAny(any=image_ids))]
            ),
            limit=len(image_ids),
            with_payload=["image_id", "thumbnail_b64", "thumbnail_mime", "width", "height", "source_type"],
            with_vectors=False,
        )
        thumbnails: dict[str, dict[str, Any]] = {}
        for point in points:
            payload = getattr(point, "payload", None) or {}
            thumb = _thumbnail_from_point(payload)
            if thumb is not None:
                thumbnails[str(payload.get("image_id"))] = thumb
        for container in containers:
            thumb = thumbnails.get(str(container.get("image_id")))
            if thumb is not None:
                container["thumbnail"] = thumb
    except Exception as exc:
        logger.warning("Snapshot thumbnail enrichment skipped: {}", exc)
    return snapshot


def _capture_collection_overview(report_id: int, collection: str, principal: Principal) -> dict[str, Any] | None:
    """Build and persist a report's frozen document-overview snapshot.

    Reads the full document list under the caller's scoped collection and stores
    the aggregated manifest on the report. Raises on failure — callers decide
    whether to swallow it (create: fail-soft) or surface it (refresh: 502).

    Args:
        report_id (int): The report id.
        collection (str): The report's logical collection.
        principal (Principal): The resolved request principal (owner).

    Returns:
        dict | None: The updated report, or ``None`` when the report is not owned.
    """
    from datetime import UTC, datetime

    from docint.core.collection_overview import build_collection_overview

    with _scoped_collection(collection, principal):
        documents = rag.list_documents()
    overview = build_collection_overview(documents, collection, datetime.now(UTC))
    return rag.ensure_report_manager().set_collection_overview_snapshot(report_id, principal.effective_owner, overview)


@app.get("/collections/{name}/export/documents.csv", tags=["Query"])
def export_documents_csv(name: str, principal: Principal = Depends(resolve_principal)) -> StreamingResponse:  # noqa: B008 — FastAPI dependency marker
    """Stream the documents table as CSV.

    The endpoint reads from :meth:`docint.core.rag.RAG.list_documents` (cached
    after the first call) and emits one row per document. Output matches the
    CLI's ``query --documents`` schema column-for-column. The path ``name`` is
    the caller's logical collection: it is resolved to its physical collection
    under the caller's effective owner (404 if not owned there), so exports
    are stateless and isolated per effective owner — including an admin
    exporting a foreign owner's collection via the ``owner`` query param.
    Rows are materialized within the scope; the response then streams the
    in-memory list.
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
    principal: Principal = Depends(resolve_principal),  # noqa: B008 — FastAPI dependency marker
) -> StreamingResponse:
    """Stream the top entities by mention frequency as CSV.

    Mirrors the CLI's ``query --entities`` export (``rank,entity,type,mentions``).
    Defaults match the CLI's ``DEFAULT_ENTITY_LIMIT`` so the two paths produce
    identical output for the same collection. ``entity_merge_mode="resolved"``
    streams the durable canonical entities (same as the Analysis/Dashboard
    resolved view); it falls back to orthographic on collections that have not
    been resolved. The path ``name`` is resolved to its physical collection
    under the caller's effective owner for the read.
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
    principal: Principal = Depends(resolve_principal),  # noqa: B008 — FastAPI dependency marker
) -> StreamingResponse:
    """Stream entity findings (per-source rows) as CSV.

    Output schema matches ``entityFindingsToCsv`` in
    ``frontend/src/lib/exports.ts``. Filtering uses the same matcher as the
    paginated ``/collections/ner/sources`` endpoint, so the export reflects
    exactly what the SPA's entity inspector shows. ``entity_merge_mode=
    "resolved"`` includes the canonical entity's sibling aliases. The path
    ``name`` is resolved to its physical collection under the caller's
    effective owner; all pages are materialized within that scope before the
    response streams them (the request scope cannot remain bound across the
    post-return streaming hops).
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
    principal: Principal = Depends(resolve_principal),  # noqa: B008 — FastAPI dependency marker
) -> StreamingResponse:
    """Stream the hate-speech findings table as CSV.

    Output schema matches ``hateSpeechToCsv`` in
    ``frontend/src/lib/exports.ts``. Filtering uses the same logic as the
    paginated ``/collections/hate-speech`` endpoint. The path ``name`` is
    resolved to its physical collection under the caller's effective owner
    for the read.
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
    collection: str | None = None,
    principal: Principal = Depends(resolve_principal),  # noqa: B008 — FastAPI dependency marker
) -> dict[str, list[dict[str, Any]]]:
    """List the calling principal's chat sessions, optionally scoped to a collection.

    When ``collection`` (a *logical* name) is supplied it is resolved to its
    physical Qdrant name under the caller's effective owner (an admin's
    ``owner`` query param, or the caller themself) and the listing is
    restricted to sessions pinned to it and owned by that same effective
    owner — an admin browsing a foreign collection sees the owner's
    sessions there, exactly as the owner would. A collection that does not resolve
    (not owned by the effective owner, or no longer exists) yields an empty
    list rather than a 404 — a stale client selection must not break the
    sidebar. When ``collection`` is omitted, every session the caller owns
    is returned (backward compatible).

    Args:
        collection (str | None): Optional logical collection name to scope to.
        principal (Principal): The resolved request principal.

    Returns:
        dict[str, list[dict[str, Any]]]: A dictionary containing the list of sessions.

    Raises:
        HTTPException: 500 if an error occurs while listing sessions.
    """
    try:
        sm = rag.ensure_session_manager()
        if collection is not None:
            physical = rag.ensure_collection_owner_manager().resolve(principal.effective_owner, collection)
            if physical is None:
                return {"sessions": []}
            sessions = sm.list_sessions(principal.effective_owner, collection=physical)
        else:
            sessions = sm.list_sessions(principal.effective_owner)
        return {"sessions": sessions}
    except Exception as e:
        logger.opt(exception=e).error("Error listing sessions")
        raise HTTPException(status_code=500, detail="Request failed.") from e


@app.get(
    "/sessions/{session_id}/history",
    response_model=SessionHistoryOut,
    tags=["Sessions"],
)
def get_session_history(
    session_id: str,
    principal: Principal = Depends(resolve_principal),  # noqa: B008 — FastAPI dependency marker
) -> dict[str, list[dict[str, Any]]]:
    """Get history for a session owned by the calling principal.

    A session that does not exist or is owned by another principal is
    reported as 404 (no existence leak).

    Args:
        session_id (str): The ID of the session.
        principal (Principal): The resolved request principal.

    Returns:
        dict[str, list[dict[str, Any]]]: A dictionary containing the session messages.

    Raises:
        HTTPException: 404 when the session is not found for this
            principal; 500 on unexpected errors.
    """
    try:
        messages = rag.ensure_session_manager().get_session_history(session_id, principal.effective_owner)
    except HTTPException:
        raise
    except Exception as e:
        logger.opt(exception=e).error("Error fetching history")
        raise HTTPException(status_code=500, detail="Request failed.") from e
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
def export_session_sources_zip(session_id: str, principal: Principal = Depends(resolve_principal)) -> StreamingResponse:  # noqa: B008 — FastAPI dependency marker
    """Stream a ZIP bundle of every source file cited in a session.

    Resolves each citation's ``file_hash`` to an on-disk file using the same
    lookup chain as ``/sources/preview``, deduplicates by hash, and writes the
    files into an in-memory ZIP (typical sessions cite tens of files, not
    thousands). Sources whose underlying file can't be found are skipped
    rather than failing the whole download. Sessions owned by another
    principal collapse to 404 — they look identical to "no sources".

    Args:
        session_id (str): The session ID to package.
        principal (Principal): The resolved request principal.

    Returns:
        StreamingResponse: ``application/zip`` payload with an
        ``attachment; filename="session-<id>-sources.zip"`` header.

    Raises:
        HTTPException: 404 if the session has no resolvable sources or is
            owned by another principal.
    """
    try:
        files = _collect_session_source_files(session_id, principal.effective_owner)
    except Exception as e:
        logger.opt(exception=e).error(f"Error assembling session sources for {session_id}")
        raise HTTPException(status_code=500, detail="Request failed.") from e

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
def delete_session(session_id: str, principal: Principal = Depends(resolve_principal)) -> dict[str, bool]:  # noqa: B008 — FastAPI dependency marker
    """Delete a session owned by the calling principal.

    A session that does not exist or is owned by another principal is
    reported as 404 (no existence leak).

    Args:
        session_id (str): The ID of the session to delete.
        principal (Principal): The resolved request principal.

    Returns:
        dict[str, bool]: A dictionary indicating whether the deletion
            was successful.

    Raises:
        HTTPException: 404 when the session is not found for this
            principal; 500 on unexpected errors.
    """
    try:
        success = rag.ensure_session_manager().delete_session(session_id, principal.effective_owner)
    except HTTPException:
        raise
    except Exception as e:
        logger.opt(exception=e).error("Error deleting session")
        raise HTTPException(status_code=500, detail="Request failed.") from e
    if not success:
        raise HTTPException(status_code=404, detail="Session not found.")
    return {"ok": success}


@app.post("/reports", tags=["Reports"])
def create_report(payload: ReportCreateIn, principal: Principal = Depends(resolve_principal)) -> dict[str, Any]:  # noqa: B008 — FastAPI dependency marker
    """Create a new, empty report owned by the calling principal.

    Args:
        payload (ReportCreateIn): Title and optional collection/session scope.
        principal (Principal): The resolved request principal.

    Returns:
        dict[str, Any]: The created report.
    """
    try:
        report = rag.ensure_report_manager().create_report(
            title=payload.title,
            owner=principal.effective_owner,
            collection_name=payload.collection_name,
            operator=payload.operator,
            reference_number=payload.reference_number,
            session_id=payload.session_id,
        )
    except Exception as e:
        logger.opt(exception=e).error("Error creating report")
        raise HTTPException(status_code=500, detail="Request failed.") from e

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
    collection: str | None = None,
    principal: Principal = Depends(resolve_principal),  # noqa: B008 — FastAPI dependency marker
) -> dict[str, list[dict[str, Any]]]:
    """List the caller's reports, optionally filtered by collection.

    Args:
        collection (str | None): Optional collection filter.
        principal (Principal): The resolved request principal.

    Returns:
        dict[str, list[dict[str, Any]]]: The caller's report summaries.
    """
    try:
        return {"reports": rag.ensure_report_manager().list_reports(principal.effective_owner, collection)}
    except Exception as e:
        logger.opt(exception=e).error("Error listing reports")
        raise HTTPException(status_code=500, detail="Request failed.") from e


@app.get("/reports/{report_id}", tags=["Reports"])
def get_report(report_id: int, principal: Principal = Depends(resolve_principal)) -> dict[str, Any]:  # noqa: B008 — FastAPI dependency marker
    """Return a report (with items) owned by the calling principal.

    Args:
        report_id (int): The report id.
        principal (Principal): The resolved request principal.

    Returns:
        dict[str, Any]: The report and its ordered items.

    Raises:
        HTTPException: 404 when the report is missing or not owned.
    """
    return _get_owned_report(report_id, principal.effective_owner)


@app.patch("/reports/{report_id}", tags=["Reports"])
def update_report(
    report_id: int,
    payload: ReportUpdateIn,
    principal: Principal = Depends(resolve_principal),  # noqa: B008 — FastAPI dependency marker
) -> dict[str, Any]:
    """Update a report (title, case metadata, or contents toggle) owned by the caller.

    Args:
        report_id (int): The report id.
        payload (ReportUpdateIn): Fields to update; only non-null fields apply.
        principal (Principal): The resolved request principal.

    Returns:
        dict[str, Any]: The updated report.

    Raises:
        HTTPException: 404 when the report is missing or not owned.
    """
    report = rag.ensure_report_manager().update_report(
        report_id,
        principal.effective_owner,
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
def refresh_report_collection_overview(
    report_id: int,
    principal: Principal = Depends(resolve_principal),  # noqa: B008 — FastAPI dependency marker
) -> dict[str, Any]:
    """Recapture a report's document-overview snapshot from its collection.

    Point-in-time refresh: rebuilds the frozen manifest from the collection's
    *current* documents.

    Args:
        report_id (int): The report id.
        principal (Principal): The resolved request principal (owner).

    Returns:
        dict[str, Any]: The report with its refreshed ``collection_overview`` snapshot.

    Raises:
        HTTPException: 404 when the report is missing or owned by another
            principal (or its collection is no longer owned); 400 when the report
            has no collection; 502 when the manifest build fails.
    """
    report = _get_owned_report(report_id, principal.effective_owner)
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
def delete_report(report_id: int, principal: Principal = Depends(resolve_principal)) -> dict[str, bool]:  # noqa: B008 — FastAPI dependency marker
    """Delete a report (and its items) owned by the calling principal.

    Args:
        report_id (int): The report id.
        principal (Principal): The resolved request principal.

    Returns:
        dict[str, bool]: ``{"ok": True}`` on success.

    Raises:
        HTTPException: 404 when the report is missing or not owned.
    """
    if not rag.ensure_report_manager().delete_report(report_id, principal.effective_owner):
        raise HTTPException(status_code=404, detail="Report not found.")
    return {"ok": True}


@app.post("/reports/{report_id}/items", tags=["Reports"])
def add_report_item(
    report_id: int,
    payload: ReportItemIn,
    principal: Principal = Depends(resolve_principal),  # noqa: B008 — FastAPI dependency marker
) -> dict[str, Any]:
    """Add a snapshotted artifact to a report (idempotent by dedupe key).

    Args:
        report_id (int): The report id.
        payload (ReportItemIn): Artifact type, dedupe key, snapshot, optional note.
        principal (Principal): The resolved request principal.

    Returns:
        dict[str, Any]: The added (or pre-existing) item.

    Raises:
        HTTPException: 404 when the report is missing or not owned.
    """
    report = _get_owned_report(report_id, principal.effective_owner)
    snapshot = _enrich_snapshot_thumbnails(payload.snapshot, payload.artifact_type, report, principal)
    item = rag.ensure_report_manager().add_item(
        report_id,
        principal.effective_owner,
        artifact_type=payload.artifact_type,
        dedupe_key=payload.dedupe_key,
        snapshot=snapshot,
        note=payload.note,
    )
    if item is None:
        raise HTTPException(status_code=404, detail="Report not found.")
    return item


@app.patch("/reports/{report_id}/items/{item_id}", tags=["Reports"])
def annotate_report_item(
    report_id: int,
    item_id: int,
    payload: ReportItemNoteIn,
    principal: Principal = Depends(resolve_principal),  # noqa: B008 — FastAPI dependency marker
) -> dict[str, Any]:
    """Set or clear the note on a report item.

    Args:
        report_id (int): The report id.
        item_id (int): The item id.
        payload (ReportItemNoteIn): The new note (``None`` clears it).
        principal (Principal): The resolved request principal.

    Returns:
        dict[str, Any]: The updated item.

    Raises:
        HTTPException: 404 when the report/item is missing or not owned.
    """
    item = rag.ensure_report_manager().annotate_item(report_id, principal.effective_owner, item_id, note=payload.note)
    if item is None:
        raise HTTPException(status_code=404, detail="Report or item not found.")
    return item


@app.delete("/reports/{report_id}/items/{item_id}", tags=["Reports"])
def remove_report_item(
    report_id: int,
    item_id: int,
    principal: Principal = Depends(resolve_principal),  # noqa: B008 — FastAPI dependency marker
) -> dict[str, bool]:
    """Remove a single item from a report owned by the calling principal.

    Args:
        report_id (int): The report id.
        item_id (int): The item id.
        principal (Principal): The resolved request principal.

    Returns:
        dict[str, bool]: ``{"ok": True}`` on success.

    Raises:
        HTTPException: 404 when the report/item is missing or not owned.
    """
    if not rag.ensure_report_manager().remove_item(report_id, principal.effective_owner, item_id):
        raise HTTPException(status_code=404, detail="Report or item not found.")
    return {"ok": True}


@app.post("/reports/{report_id}/items/reorder", tags=["Reports"])
def reorder_report_items(
    report_id: int,
    payload: ReportReorderIn,
    principal: Principal = Depends(resolve_principal),  # noqa: B008 — FastAPI dependency marker
) -> dict[str, Any]:
    """Reorder a report's items to match the supplied id order.

    Args:
        report_id (int): The report id.
        payload (ReportReorderIn): Desired item id order.
        principal (Principal): The resolved request principal.

    Returns:
        dict[str, Any]: The reordered report.

    Raises:
        HTTPException: 404 when the report is missing or not owned.
    """
    report = rag.ensure_report_manager().reorder_items(report_id, principal.effective_owner, payload.item_ids)
    if report is None:
        raise HTTPException(status_code=404, detail="Report not found.")
    return report


@app.get("/reports/{report_id}/export.md", tags=["Reports"])
def export_report_markdown(report_id: int, principal: Principal = Depends(resolve_principal)) -> Response:  # noqa: B008 — FastAPI dependency marker
    """Export a report as a single Markdown document (attachment download)."""
    from docint.core.state.report_render import render_markdown

    report = _get_owned_report(report_id, principal.effective_owner)
    return Response(
        content=render_markdown(report),
        media_type="text/markdown; charset=utf-8",
        headers=_download_headers(_report_stem(report), "md"),
    )


@app.get("/reports/{report_id}/export.html", tags=["Reports"])
def export_report_html(report_id: int, principal: Principal = Depends(resolve_principal)) -> Response:  # noqa: B008 — FastAPI dependency marker
    """Export a report as a self-contained HTML document (served inline)."""
    from docint.core.state.report_render import render_html

    report = _get_owned_report(report_id, principal.effective_owner)
    return Response(
        content=render_html(report),
        media_type="text/html; charset=utf-8",
        headers=_download_headers(_report_stem(report), "html", inline=True),
    )


@app.get("/reports/{report_id}/export.json", tags=["Reports"])
def export_report_json(report_id: int, principal: Principal = Depends(resolve_principal)) -> Response:  # noqa: B008 — FastAPI dependency marker
    """Export the full report (with snapshots) as JSON (attachment download)."""
    from docint.core.state.report_render import render_json

    report = _get_owned_report(report_id, principal.effective_owner)
    return Response(
        content=render_json(report),
        media_type="application/json",
        headers=_download_headers(_report_stem(report), "json"),
    )


@app.get("/reports/{report_id}/export.zip", tags=["Reports"])
def export_report_zip(report_id: int, principal: Principal = Depends(resolve_principal)) -> Response:  # noqa: B008 — FastAPI dependency marker
    """Export a report as a ZIP bundle of per-type CSVs (attachment download)."""
    from docint.core.state.report_render import report_csv_bundle

    report = _get_owned_report(report_id, principal.effective_owner)
    return Response(
        content=report_csv_bundle(report),
        media_type="application/zip",
        headers=_download_headers(_report_stem(report), "zip"),
    )


@app.get("/reports/{report_id}/export.pdf", tags=["Reports"])
def export_report_pdf(report_id: int, principal: Principal = Depends(resolve_principal)) -> Response:  # noqa: B008 — FastAPI dependency marker
    """Export a report as a real paginated PDF rendered by WeasyPrint.

    Returns 503 if the PDF engine (WeasyPrint + native libs) is unavailable,
    leaving the other export formats unaffected.
    """
    from docint.core.state.report_render import PdfEngineUnavailableError, render_pdf

    report = _get_owned_report(report_id, principal.effective_owner)
    try:
        pdf_bytes = render_pdf(report)
    except PdfEngineUnavailableError as e:
        logger.opt(exception=e).error("PDF export engine unavailable")
        raise HTTPException(status_code=503, detail="PDF export is not available.") from e
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
            session_id = rag.start_session(payload.session_id, owner=owner.name)
            ctx = rag.sessions.get_agent_context(session_id) if rag.sessions else None
            if ctx and rag.sessions:
                ctx.history = rag.sessions.get_session_history(session_id, owner=owner.name)

            turn = Turn(user_input=payload.message, session_id=session_id)
            orchestrator = _build_orchestrator()
            result = orchestrator.handle_turn(turn, context=ctx)
    except SessionCollectionMismatchError as exc:
        raise HTTPException(status_code=409, detail="Session is pinned to a different collection.") from exc

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
        retried=retrieval.retried,
        retry_query=retrieval.retry_query,
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
    physical = rag.ensure_collection_owner_manager().register(principal.effective_owner, name)

    data_dir = _resolve_data_dir()
    if not data_dir.is_dir():
        logger.error("HTTPException: Data directory does not exist: {}", data_dir)
        raise HTTPException(
            status_code=400,
            detail="Server storage is not available.",
        )

    # ``payload.hybrid`` is None unless the caller explicitly opted in or out;
    # None must reach ``ingest_docs`` unchanged so the RAG engine's own
    # ``resolve_enable_hybrid()`` default decides, rather than always forcing
    # hybrid on. Calling the same resolver here reports what that resolution
    # actually produced, for callers that read the response's ``hybrid`` field.
    resolved_hybrid = payload.hybrid if payload.hybrid is not None else resolve_enable_hybrid()
    # Timed here rather than inside ``ingest_docs`` so the duration covers the
    # whole request, model loading included — this endpoint is synchronous, so
    # the request *is* the run (unlike the job API, which times its own).
    started_ticks = time.monotonic()
    try:
        ingest_module.ingest_docs(
            physical,
            data_dir,
            hybrid=payload.hybrid,
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
            "hybrid": resolved_hybrid,
            "empty": True,
        }
    except NoSupportedFilesError:
        logger.warning("No ingestable files for collection '{}'; returning empty response.", name)
        return {
            "ok": True,
            "collection": name,
            "data_dir": str(data_dir),
            "hybrid": resolved_hybrid,
            "empty": True,
        }
    except Exception as exc:
        logger.opt(exception=exc).error(f"Unexpected error during ingestion of '{name}'")
        raise HTTPException(status_code=500, detail="Request failed.") from exc

    logger.info("Ingestion complete in {}.", format_elapsed(time.monotonic() - started_ticks))
    return {
        "ok": True,
        "collection": name,
        "data_dir": str(data_dir),
        "hybrid": resolved_hybrid,
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
        pinned = rag.ensure_session_manager().get_session_collection(payload.session_id, owner.name)
        if pinned is not None and pinned != physical:
            raise HTTPException(
                status_code=409,
                detail="Session is pinned to a different collection.",
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
                session_id = rag.start_session(payload.session_id, owner=owner.name)
                ctx = rag.sessions.get_agent_context(session_id) if rag.sessions else None
                if ctx and rag.sessions:
                    ctx.history = rag.sessions.get_session_history(session_id, owner=owner.name)
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
                    rag.stream_chat(query_text, session_id=session_id, owner=owner.name, prior_turn=prior_turn),
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


def _auto_resolve_requested(ner: bool | None) -> bool:
    """Whether resolution should follow this ingest run automatically.

    Resolution is part of entity extraction: it runs whenever the run's
    effective NER is on (per-request override, else ``NER_ENABLED``), unless
    the operator disabled it via ``RES_AUTO_RESOLVE``.

    Args:
        ner (bool | None): The per-request NER override, if any.

    Returns:
        bool: ``True`` when resolution should run after the ingest.
    """
    ner_effective = load_ner_env().enabled if ner is None else ner
    return ner_effective and load_resolution_env().auto_resolve


def _run_ingest_job(state: IngestJobState, push: PushEvent) -> dict[str, Any]:
    """Execute one ingest job: pipeline, then optional entity resolution.

    Injected into :class:`IngestJobManager` so ``core/jobs.py`` stays free of
    docint domain imports. Runs on a worker thread; ``push`` is thread-safe.

    Resolution runs *here*, inside the job, rather than in a request handler —
    which is the fix for it being silently skipped whenever the client had
    disconnected.

    Args:
        state (IngestJobState): The job being executed.
        push (PushEvent): Thread-safe event publisher.

    Returns:
        dict[str, Any]: ``{"empty": bool, "resolution": dict | None}``.

    Raises:
        RuntimeError: If ``state.batch_dir`` is ``None``. ``batch_dir`` is
            optional on ``IngestJobState`` only because ``kind="summary"``
            jobs omit it; this runner is only ever registered for
            ``kind="ingest"`` jobs, which always populate it, so this should
            never actually trigger.
    """
    if state.batch_dir is None:
        raise RuntimeError("Ingest jobs always carry a batch_dir.")
    if not state.batch_dir.is_dir():
        # Nothing was staged (e.g. every upload batch failed, or finalize was
        # called with nothing ever uploaded). Report a soft empty completion
        # instead of letting the reader fail on a missing directory.
        push("warning", {"message": f"No staged files found for '{state.logical_name}'."})
        return {"empty": True, "resolution": None}

    empty = False
    stats: IngestStats | None = None
    try:
        stats = ingest_module.ingest_docs(
            state.physical,
            state.batch_dir,
            state.hybrid,
            lambda msg: push(
                "warning" if msg.strip().lower().startswith("warning:") else "ingestion_progress",
                {"message": msg},
            ),
            ner=state.ner,
            hate_speech=state.hate_speech,
        )
    except EmptyIngestionError:
        # Never forward str(exc)/exc.collection_name here: both carry the
        # physical, owner-namespaced collection name (see EmptyIngestionError's
        # construction sites in rag.py), and this message is caller-facing —
        # snapshot() promises only the logical name is ever echoed.
        push(
            "warning",
            {
                "message": f"No content was ingested into '{state.logical_name}'.",
                "collection": state.logical_name,
            },
        )
        return {"empty": True, "resolution": None}
    except NoSupportedFilesError:
        # The staged directory held nothing ingestable (e.g. only audio/video
        # with Nextext unconfigured, which the pre-passes cannot claim). A
        # soft empty ingest, not a hard failure.
        logger.warning("No ingestable files for collection '{}'; completing as empty.", state.logical_name)
        push("warning", {"message": f"No ingestable files found for '{state.logical_name}'."})
        return {"empty": True, "resolution": None}

    resolution: dict[str, Any] | None = None
    if state.resolve:
        push("ingestion_progress", {"message": "Resolving entities..."})
        try:
            with rag.collection_scope(state.physical):
                summary = rag.resolve_entities()
        except Exception:
            logger.exception("Auto-resolution after ingest failed for '{}'", state.logical_name)
            push("warning", {"message": "Entity resolution failed."})
        else:
            resolution = {
                "processed": summary.processed,
                "minted": summary.minted,
                "attached": summary.attached,
                "skipped": summary.skipped,
                "entities_touched": summary.entities_touched,
            }

    if load_summary_env().on_ingest:
        push("ingestion_progress", {"message": "Building collection summary..."})
        try:
            with rag.collection_scope(state.physical):
                rag.build_tree_summary(
                    progress=lambda mapped, total: push(
                        "ingestion_progress",
                        {"message": f"Summarizing {mapped}/{total}", "mapped": mapped, "total_units": total},
                    )
                )
        except Exception:
            logger.exception("Summary stage after ingest failed for '{}'", state.logical_name)
            push("warning", {"message": "Collection summary generation failed."})

    # ``stats`` rides the result dict as a plain mapping of ints so
    # ``core/jobs.py`` can render it on the run-summary line without
    # importing a docint domain type — the property that keeps the job
    # registry testable without Qdrant or models.
    return {
        "empty": empty,
        "resolution": resolution,
        "stats": asdict(stats) if stats is not None else None,
    }


def _run_summary_job(state: IngestJobState, push: PushEvent) -> dict[str, Any]:
    """Execute one summary-rebuild job: tree summary under the job's collection scope.

    Injected via :func:`_run_job` so ``core/jobs.py`` stays free of docint
    domain imports. Runs on a worker thread; ``push`` is thread-safe.

    Args:
        state (IngestJobState): The job being executed.
        push (PushEvent): Thread-safe event publisher.

    Returns:
        dict[str, Any]: ``{"empty": bool, "resolution": None}``.
    """

    def _progress(mapped: int, total: int) -> None:
        push(
            "summary_progress",
            {"message": f"Summarizing {mapped}/{total}", "mapped": mapped, "total_units": total},
        )

    with rag.collection_scope(state.physical):
        payload = rag.build_tree_summary(progress=_progress)
    empty = not str((payload or {}).get("response") or "").strip()
    return {"empty": empty, "resolution": None}


def _run_job(state: IngestJobState, push: PushEvent) -> dict[str, Any]:
    """Dispatch a job to its kind-specific runner.

    Args:
        state (IngestJobState): The job being executed.
        push (PushEvent): Thread-safe event publisher.

    Returns:
        dict[str, Any]: The kind-specific runner's result.
    """
    if state.kind == "summary":
        return _run_summary_job(state, push)
    return _run_ingest_job(state, push)


job_manager = IngestJobManager(runner=_run_job)


def get_job_manager() -> IngestJobManager:
    """Return the ingest job manager the request handlers should use.

    The application's own manager is a module-level instance built once at
    import. Routing every handler through this dependency lets a test override
    it with ``app.dependency_overrides`` and supply a manager of its own,
    instead of mutating the shared instance's private state between tests.

    Returns:
        IngestJobManager: The manager backing the ``/ingest/jobs*`` endpoints.
    """
    return job_manager


def _cached_summary_payload(physical: str) -> dict[str, Any] | None:
    """Build the API payload for a collection's cached tree summary, or ``None``.

    The single source of the ``200`` body that both ``GET /summarize`` and the
    cache-hit branch of ``POST /summarize`` return, so the two can never drift
    into describing the same cached summary differently.

    ``rag.cached_collection_summary()`` is a pure read -- a KV lookup plus a
    revision/fingerprint compare. The ``_validation_payload`` merge below is
    not: with ``RESPONSE_VALIDATION_ENABLED`` on (the default) it runs the
    validation agent, which may call the text model. Callers must therefore
    treat this whole function as blocking IO -- it is why the GET handler is a
    plain ``def`` (FastAPI runs it in a worker thread) and why the POST hops
    through ``to_thread.run_sync``.

    A cached-but-blank summary is deliberately returned as a payload rather
    than folded into ``None``: reporting it as a miss would make the POST queue
    a rebuild for every collection that legitimately summarizes to nothing, and
    the SPA's bounded re-attach would then surface that benign case as a
    failure. Callers decide what an empty summary means.

    Args:
        physical (str): The resolved, owner-namespaced Qdrant collection.

    Returns:
        dict[str, Any] | None: The ``SummarizeOut`` body, or ``None`` when
        nothing is cached for this collection.
    """
    with rag.collection_scope(physical):
        data = rag.cached_collection_summary()
    if not isinstance(data, dict):
        return None
    summary = str(data.get("response") or "")
    sources = data.get("sources") if isinstance(data.get("sources"), list) else []
    summary_diagnostics = data.get("summary_diagnostics") if isinstance(data.get("summary_diagnostics"), dict) else None
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


@app.post("/summarize", response_model=None, tags=["Query"])
async def summarize(
    refresh: bool = Query(False),
    collection: str | None = None,
    principal: Principal = Depends(resolve_principal),  # noqa: B008 — FastAPI dependency marker
    jobs: IngestJobManager = Depends(get_job_manager),  # noqa: B008 — FastAPI dependency marker
) -> Response:
    """Serve the cached collection summary, or queue a rebuild job.

    A cache hit answers 200 with the stored payload. A miss -- or an explicit
    ``refresh=true`` -- queues a ``kind="summary"`` background job and answers
    202 with its ``job_id``; progress arrives on ``GET /ingest/jobs/events``
    (``summary_started`` / ``summary_progress`` / ``summary_completed``). A
    second call while a build is in flight answers 409 carrying the in-flight
    ``job_id``.

    Queuing requires an explicit ``collection``: ``_resolve_request_collection``
    falls back to the process-default active collection
    (``rag.qdrant_collection``) when ``collection`` is omitted, and that value
    may be a physical, owner-namespaced Qdrant name rather than the caller's
    logical one. A physical name must never be echoed into a job's
    ``logical_name`` snapshot or back to a client, and there is no reverse
    (physical -> logical) lookup available here to recover a clean logical
    name from the fallback -- so the queue path requires the caller to pass
    ``collection`` explicitly and 400s otherwise. The cache-read path has no
    such requirement: it only ever reads, never creates a job, so the
    process-default fallback is safe to use there.

    Args:
        refresh (bool): Force a rebuild even when a cached summary exists.
        collection (str | None): Caller's logical collection; owner-gated and
            scoped per request. Required (400 if omitted) whenever a build is
            queued; optional for a cache read, which falls back to the
            process-default active collection.
        principal (Principal): The resolved request principal.
        jobs (IngestJobManager): The shared job registry.

    Returns:
        Response: 200 ``SummarizeOut`` JSON on a cache hit, 202 ``{"job_id"}``
        when a build was queued.

    Raises:
        HTTPException: 400/404 from collection resolution (also 400 when
            ``collection`` is omitted but a build must be queued); 409 when a
            summary job is already in flight for this collection.
    """
    physical = _resolve_request_collection(collection, principal)
    if not refresh:
        payload = await to_thread.run_sync(_cached_summary_payload, physical)
        if payload is not None:
            return JSONResponse(payload)

    logical_name = (collection or "").strip()
    if not logical_name:
        raise HTTPException(
            status_code=400,
            detail="Collection name required to queue a summary build.",
        )
    state, created = await jobs.create_if_idle(
        owner=principal.effective_owner,
        logical_name=logical_name,
        physical=physical,
        kind="summary",
    )
    if not created:
        raise HTTPException(
            status_code=409,
            detail={"message": "Summary generation already in progress.", "job_id": state.job_id},
        )
    return JSONResponse({"job_id": state.job_id}, status_code=202)


@app.get("/summarize", response_model=None, tags=["Query"])
def summarize_cached(
    collection: str | None = None,
    principal: Principal = Depends(resolve_principal),  # noqa: B008 -- FastAPI dependency marker
) -> Response:
    """Serve the cached collection summary, or report that there is none.

    The read-only half of ``POST /summarize``: a hit answers 200 with the
    identical body (:func:`_cached_summary_payload` builds both), a miss
    answers 204, and nothing is ever queued.

    The distinction is carried by the HTTP method rather than by a flag on the
    POST because the SPA fires this automatically whenever the Summary tab is
    opened. A build is a minutes-long job of up to ``SUMMARY_MAX_LLM_CALLS``
    model calls, so a caller who forgot to pass some ``queue=false`` would
    start one by merely looking at a collection; a handler with no queue branch
    in it cannot.

    204 rather than 404 on a miss: ``_resolve_request_collection`` already
    spends 404 on "collection not owned / not found", and a client must be able
    to tell "you may not read this" from "there is nothing here yet".

    Declared ``def`` rather than ``async def`` on purpose -- building the
    payload is blocking IO (see :func:`_cached_summary_payload`), so FastAPI
    must run it in a worker thread instead of on the event loop.

    Args:
        collection (str | None): The caller's logical collection name.
            Optional: this path never creates a job, so the process-default
            fallback in ``_resolve_request_collection`` is safe here, unlike
            the POST's queue path which 400s without it.
        principal (Principal): The resolved request principal.

    Returns:
        Response: 200 ``SummarizeOut`` JSON on a hit; 204 with no body on a miss.

    Raises:
        HTTPException: 400/404 from collection resolution.
    """
    payload = _cached_summary_payload(_resolve_request_collection(collection, principal))
    if payload is None:
        return Response(status_code=204)
    return JSONResponse(payload)


@app.post("/ingest/upload", tags=["Ingestion"])
async def ingest_upload(
    request: Request,
    collection: str = Form(...),
    files: list[UploadFile] = File(...),  # noqa: B008 — FastAPI dependency marker
) -> StreamingResponse:
    """Upload and stage files for a collection, streaming save progress as SSE events.

    Saves the file(s) to the collection's batch directory but does NOT ingest
    them — the caller runs one ingestion pass afterwards via
    ``/ingest/finalize``, which queues a server-owned job and takes the run's
    ``hybrid``/``ner``/``hate_speech`` options (this endpoint has none — it
    only writes bytes to disk). The SPA uploads a large selection as several
    batches so ingestion happens once over the whole staged directory instead
    of once per batch (re-initializing the pipeline's models per batch and
    hard-failing on any batch that happened to hold only reader-unsupported
    files, e.g. audio/video → ``NoSupportedFilesError``).

    Args:
        request (Request): The incoming request, used to resolve the principal.
        collection (str): The name of the collection to ingest into.
        files (list[UploadFile]): The list of files to upload.

    Returns:
        StreamingResponse: A streaming response that yields SSE events while
        the file(s) are saved to disk.

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
    physical = rag.ensure_collection_owner_manager().register(principal.effective_owner, name)

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

        staged_bytes = 0
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

                staged_bytes += bytes_written
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
            except Exception as exc:
                logger.opt(exception=exc).error("Error saving uploaded file {}", filename)
                # `filename` travels as a structured field (client-supplied
                # name, echoed) so the SPA can validate it against its own
                # upload list before display; the message itself stays static.
                yield _format_sse(
                    "error",
                    {"message": "Failed to save file.", "code": "save_failed", "filename": filename},
                )
                return

        # An upload that is never finalized creates no job, so without this
        # line it leaves no trace in the log at all. Per-file detail belongs
        # to the run-start banner, which knows the job id; this is only the
        # batch landing on disk.
        logger.info(
            "Upload batch staged | collection={!r} files={} bytes={}",
            name,
            len(files),
            format_bytes(staged_bytes),
        )
        yield _format_sse(
            "upload_complete",
            {"collection": name, "files_saved": len(files)},
        )

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/ingest/finalize", status_code=202, tags=["Ingestion"])
async def ingest_finalize(
    payload: IngestFinalizeIn,
    request: Request,
    jobs: IngestJobManager = Depends(get_job_manager),  # noqa: B008 — FastAPI dependency marker
) -> dict[str, str]:
    """Queue an ingest job over a collection's already-staged upload batches.

    The SPA uploads a large selection as several batches to ``/ingest/upload``
    (each saved but not ingested), then calls this once. Ingestion runs as a
    server-owned job: progress is consumed from ``GET /ingest/jobs/events``, so
    a browser reload no longer severs the run's only view.

    Args:
        payload (IngestFinalizeIn): Collection (logical name), run options,
            and the client's upload elapsed time, which anchors the run's
            duration at the moment the user started rather than here.
        request (Request): The incoming request, for principal resolution.
        jobs (IngestJobManager): The ingest job registry.

    Returns:
        dict[str, str]: ``{"job_id": ...}``.

    Raises:
        HTTPException: 400 when the collection name is blank; 404 when the
            caller does not own it; 409 (detail carries the existing
            ``job_id``) when that collection already has a job in flight.
    """
    principal = resolve_principal(request)
    name = payload.collection.strip()
    if not name:
        logger.error("HTTPException: Collection name required for finalize")
        raise HTTPException(status_code=400, detail="Collection name required")
    physical = rag.ensure_collection_owner_manager().register(principal.effective_owner, name)

    # create_if_idle() checks for an in-flight job and creates one only if
    # idle, atomically under one lock. A separate active_for() check followed
    # by a separate create() call would be a TOCTOU: two interleaved finalize
    # requests could both observe no in-flight job and both create one.
    state, created = await jobs.create_if_idle(
        owner=principal.effective_owner,
        logical_name=name,
        physical=physical,
        batch_dir=_resolve_qdrant_src_dir() / physical,
        hybrid=payload.hybrid,
        ner=payload.ner,
        hate_speech=payload.hate_speech,
        resolve=_auto_resolve_requested(payload.ner),
        upload_lead_s=(payload.upload_elapsed_ms or 0.0) / 1000.0,
    )
    if not created:
        raise HTTPException(
            status_code=409,
            detail={"message": "Ingestion already in progress.", "job_id": state.job_id},
        )
    return {"job_id": state.job_id}


@app.get("/ingest/jobs", tags=["Ingestion"])
async def list_ingest_jobs(
    request: Request,
    jobs: IngestJobManager = Depends(get_job_manager),  # noqa: B008 — FastAPI dependency marker
) -> dict[str, list[dict[str, Any]]]:
    """List the caller's ingest jobs, newest first.

    Not called by the SPA — reload re-discovery goes through the persisted
    ``activeJobId`` plus the SSE replay instead (see ``GET /ingest/jobs/events``).
    Available for other/future clients that want to enumerate a caller's jobs.

    Args:
        request (Request): The incoming request, for principal resolution.
        jobs (IngestJobManager): The ingest job registry.

    Returns:
        dict[str, list[dict[str, Any]]]: ``{"jobs": [snapshot, ...]}``.
    """
    principal = resolve_principal(request)
    states = await jobs.list_for_owner(principal.effective_owner)
    return {"jobs": [s.snapshot() for s in states]}


# NB: declared BEFORE /ingest/jobs/{job_id}. FastAPI matches routes in
# declaration order, so the reverse order parses "events" as a job id.
@app.get("/ingest/jobs/events", tags=["Ingestion"])
async def ingest_job_events(
    request: Request,
    jobs: IngestJobManager = Depends(get_job_manager),  # noqa: B008 — FastAPI dependency marker
) -> StreamingResponse:
    """Stream SSE events for every job the caller owns, over one connection.

    Replays each job's collapsed history on connect, so a client that
    reconnects mid-run resumes the live view instead of waiting for the next
    frame.

    Args:
        request (Request): The incoming request, for principal resolution.
        jobs (IngestJobManager): The ingest job registry.

    Returns:
        StreamingResponse: ``text/event-stream`` of tagged job frames.
    """
    principal = resolve_principal(request)
    return StreamingResponse(
        jobs.subscribe_owner(principal.effective_owner),
        media_type="text/event-stream",
    )


@app.get("/ingest/jobs/{job_id}", tags=["Ingestion"])
async def get_ingest_job(
    job_id: str,
    request: Request,
    jobs: IngestJobManager = Depends(get_job_manager),  # noqa: B008 — FastAPI dependency marker
) -> dict[str, Any]:
    """Return a point-in-time snapshot of one owned job.

    Args:
        job_id (str): Job identifier.
        request (Request): The incoming request, for principal resolution.
        jobs (IngestJobManager): The ingest job registry.

    Returns:
        dict[str, Any]: The job snapshot.

    Raises:
        HTTPException: 404 when unknown or owned by someone else — the two are
            deliberately indistinguishable so existence never leaks.
    """
    principal = resolve_principal(request)
    state = await jobs.get(job_id, principal.effective_owner)
    if state is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return state.snapshot()


@app.delete("/ingest/jobs/{job_id}", tags=["Ingestion"])
async def delete_ingest_job(
    job_id: str,
    request: Request,
    jobs: IngestJobManager = Depends(get_job_manager),  # noqa: B008 — FastAPI dependency marker
) -> dict[str, bool]:
    """Dismiss a finished job from the registry.

    Args:
        job_id (str): Job identifier.
        request (Request): The incoming request, for principal resolution.
        jobs (IngestJobManager): The ingest job registry.

    Returns:
        dict[str, bool]: ``{"ok": True}``.

    Raises:
        HTTPException: 404 when unknown or cross-owner; 409 while the job is
            still running — the worker thread cannot be killed, so a dismissed
            running job would keep writing unobserved.
    """
    principal = resolve_principal(request)
    state = await jobs.get(job_id, principal.effective_owner)
    if state is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if state.status in (JobStatus.QUEUED, JobStatus.RUNNING):
        raise HTTPException(status_code=409, detail="Job is still running")
    await jobs.remove(job_id, principal.effective_owner)
    return {"ok": True}


@app.get("/sources/preview", tags=["Sources"])
def preview_source(collection: str, file_hash: str, principal: Principal = Depends(resolve_principal)) -> FileResponse:  # noqa: B008 — FastAPI dependency marker
    """Serve a previously ingested source file resolved under the caller's effective owner.

    ``collection`` is the caller's *logical* name; it is resolved to its
    owner-namespaced physical collection under ``principal.effective_owner``
    before the source store is touched (404 when that owner does not own it).
    This both prevents previewing a file outside the caller's effective-owner
    scope and makes previews resolve under the correct physical path for
    namespaced users — including an admin previewing a foreign owner's file
    via the ``owner`` query param.

    Args:
        collection (str): The caller's logical collection name.
        file_hash (str): The hash of the file to preview.
        principal (Principal): The resolved request principal.

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
    # Browsers refuse to *display* text/markdown and text/csv — they hand both
    # to the download manager, in a tab and in the preview dialog's iframe
    # alike. This endpoint exists to show a source, so serve those as plain
    # text; downloads go through the session-ZIP endpoint.
    if path.suffix.lower() in {".md", ".csv"}:
        return FileResponse(path, media_type="text/plain; charset=utf-8")
    return FileResponse(path)


@app.post("/translate", tags=["Translate"])
def translate_text(payload: TranslateIn, principal: Principal = Depends(resolve_principal)) -> dict[str, Any]:  # noqa: B008 — FastAPI dependency marker
    """Translate a client-supplied snippet into the operator's locale.

    Authenticated for consistency, but not collection-scoped: it translates text
    the caller already holds, so there is nothing to leak and no store re-fetch.
    Fail-soft — a transport error returns ``ok: false`` with the original shape.

    Args:
        payload (TranslateIn): The snippet to translate.
        principal (Principal): The resolved request principal.

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
