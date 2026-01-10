import asyncio
import json
from pathlib import Path
from typing import Any, AsyncIterator, Literal, cast

import anyio
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from loguru import logger
from pydantic import BaseModel
from qdrant_client import models
from starlette.middleware.cors import CORSMiddleware

from docint.cli import ingest as ingest_module
from docint.agents import (
    AgentOrchestrator,
    ClarificationConfig,
    ClarificationPolicy,
    RAGRetrievalAgent,
    SimpleClarificationAgent,
    SimpleUnderstandingAgent,
    Turn,
)
from docint.core.rag import RAG
from docint.utils.env_cfg import load_host_env, load_path_env
from docint.utils.hashing import compute_file_hash

# Load allowed origins from environment or default to Streamlit's default ports
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

# Agent components (kept lightweight; swap with richer agents as needed)
_understanding_agent = SimpleUnderstandingAgent()
_clarification_agent = SimpleClarificationAgent()
_clarification_policy = ClarificationPolicy(ClarificationConfig())


def _build_orchestrator() -> AgentOrchestrator:
    """
    Construct an orchestrator bound to the current RAG instance.

    Returns:
        AgentOrchestrator: The constructed agent orchestrator.
    """
    retrieval_agent = RAGRetrievalAgent(rag)
    return AgentOrchestrator(
        understanding=_understanding_agent,
        clarifier=_clarification_agent,
        retriever=retrieval_agent,
        policy=_clarification_policy,
    )


# --- Helper Functions ---


def _resolve_data_dir() -> Path:
    """
    Return the configured data directory for ingestion.

    Returns:
        Path: The path to the data directory.
    """

    return load_path_env().data


def _resolve_qdrant_col_dir() -> Path:
    """
    Return the configured Qdrant collections directory.

    Returns:
        Path: The path to the Qdrant collections directory.
    """
    return load_path_env().qdrant_collections


def _resolve_qdrant_src_dir() -> Path:
    """
    Return the configured Qdrant sources directory (separate from collections).

    Returns:
        Path: The path to the Qdrant sources directory.

    Raises:
        RuntimeError: If the Qdrant sources directory is not configured.
    """
    path_config = load_path_env()
    if path_config.qdrant_sources is None:
        raise RuntimeError("Qdrant sources directory is not configured")
    return path_config.qdrant_sources


def _format_sse(event: str, data: dict[str, Any]) -> str:
    """
    Return a serialized Server-Sent Event payload.

    Args:
        event (str): The event type.
        data (dict[str, Any]): The event data.

    Returns:
        str: The formatted SSE string.
    """
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


# --- Pydantic models for request and response payloads ---


class SelectCollectionIn(BaseModel):
    name: str


class SelectCollectionOut(BaseModel):
    ok: bool
    name: str


class QueryIn(BaseModel):
    question: str
    session_id: str | None = None


class QueryOut(BaseModel):
    answer: str
    sources: list[dict] = []
    session_id: str


class SummarizeOut(BaseModel):
    summary: str
    sources: list[dict] = []


class IngestIn(BaseModel):
    collection: str
    hybrid: bool | None = True


class IngestOut(BaseModel):
    ok: bool
    collection: str
    data_dir: str
    hybrid: bool


class SessionListOut(BaseModel):
    sessions: list[dict]


class SessionHistoryOut(BaseModel):
    messages: list[dict]


class AgentChatIn(BaseModel):
    message: str
    session_id: str | None = None


class AgentChatOut(BaseModel):
    status: Literal["clarification", "answer"]
    message: str | None = None
    answer: str | None = None
    sources: list[dict] = []
    session_id: str | None = None
    reason: str | None = None
    intent: str | None = None
    confidence: float | None = None
    tool_used: str | None = None
    latency_ms: float | None = None


# --- API Endpoints ---


@app.get("/collections/list", response_model=list[str], tags=["Collections"])
def collections_list() -> list[str]:
    """
    List existing collections.

    Returns:
        list[str]: A list of collection names.

    Raises:
        HTTPException: If an error occurs while listing collections.
    """
    try:
        return rag.list_collections()
    except Exception as e:
        logger.error("HTTPException: Error listing collections: {}", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/collections/select", response_model=SelectCollectionOut, tags=["Collections"]
)
def collections_select(payload: SelectCollectionIn) -> dict[str, bool | str]:
    """
    Select a collection to use for queries.

    Args:
        payload (SelectCollectionIn): The payload containing the collection name.

    Returns:
        dict[str, bool | str]: A dictionary indicating success and the selected collection name.

    Raises:
        HTTPException: If the collection name is missing or an error occurs while selecting the collection.
        HTTPException: If an error occurs while selecting the collection.
    """
    try:
        name = payload.name.strip()
        if not name:
            logger.error("HTTPException: Collection name required")
            raise HTTPException(status_code=400, detail="Collection name required")
        rag.select_collection(name)

        if getattr(rag, "index", None) is None:
            try:
                rag.create_index()
                rag.create_query_engine()
            except Exception:
                pass

        # Pre-warm IE cache if enabled
        if getattr(rag, "enable_ie", False):
            try:
                rag.get_collection_ie(refresh=True)
            except Exception:
                logger.warning("Could not pre-warm IE cache for collection '{}'.", name)

        return {"ok": True, "name": name}
    except HTTPException as e:
        logger.error("HTTPException: Error selecting collection: {}", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query", response_model=QueryOut, tags=["Query"])
def query(payload: QueryIn) -> dict[str, list[dict] | str]:
    """
    Handle a query request.

    Args:
        payload (QueryIn): The query payload containing the question and session ID.

    Returns:
        QueryOut: The query response containing the answer, sources, and session ID.

    Raises:
        HTTPException: If an error occurs while processing the query.
    """
    try:
        if not rag.qdrant_collection:
            logger.error("HTTPException: No collection selected")
            raise HTTPException(status_code=400, detail="No collection selected")

        if getattr(rag, "query_engine", None) is None:
            if getattr(rag, "index", None) is None:
                rag.create_index()
            rag.create_query_engine()

        session_id = rag.start_session(payload.session_id)
        data = rag.chat(payload.question)

        answer = (
            str(data.get("response") or data.get("answer") or "")
            if isinstance(data, dict)
            else ""
        )
        sources: list[dict] = data.get("sources", []) if isinstance(data, dict) else []

        return {"answer": answer, "sources": sources, "session_id": session_id}
    except HTTPException as e:
        logger.error("HTTPException: Error processing query: {}", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/stream_query", tags=["Query"])
async def stream_query(payload: QueryIn) -> StreamingResponse:
    """
    Handle a streaming query request.

    Args:
        payload (QueryIn): The query payload containing the question and session ID.

    Returns:
        StreamingResponse: A streaming response that yields SSE events during the query.

    Raises:
        HTTPException: If an error occurs while processing the streaming query.
    """
    if not rag.qdrant_collection:
        raise HTTPException(status_code=400, detail="No collection selected")

    # Ensure index exists
    if getattr(rag, "index", None) is None:
        rag.create_index()

    rag.start_session(payload.session_id)

    async def event_generator() -> AsyncIterator[str]:
        """
        Generate SSE events for the streaming query.

        Returns:
            AsyncIterator[str]: An asynchronous iterator yielding SSE events.

        Yields:
            Iterator[AsyncIterator[str]]: An asynchronous iterator yielding SSE events.
        """
        try:
            # Iterate over the sync generator
            for chunk in rag.stream_chat(payload.question):
                if isinstance(chunk, str):
                    yield f"data: {json.dumps({'token': chunk})}\n\n"
                elif isinstance(chunk, dict):
                    yield f"data: {json.dumps(chunk)}\n\n"
        except Exception as e:
            logger.error(f"Stream error: {e}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/summarize", response_model=SummarizeOut, tags=["Query"])
def summarize() -> dict[str, list[dict] | str]:
    """
    Generate a summary for the currently selected collection.

    Returns:
        dict[str, list[dict] | str]: A dictionary containing the summary and sources.

    Raises:
        HTTPException: If an error occurs while generating the summary.
    """

    try:
        if not rag.qdrant_collection:
            logger.error("HTTPException: No collection selected")
            raise HTTPException(status_code=400, detail="No collection selected")

        if getattr(rag, "query_engine", None) is None:
            if getattr(rag, "index", None) is None:
                rag.create_index()
            rag.create_query_engine()

        data = rag.summarize_collection()
        summary = (
            str(data.get("response") or data.get("answer") or "")
            if isinstance(data, dict)
            else ""
        )
        sources: list[dict] = data.get("sources", []) if isinstance(data, dict) else []

        return {"summary": summary, "sources": sources}
    except HTTPException as e:
        logger.error("HTTPException: Error generating summary: {}", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/summarize/stream", tags=["Query"])
async def summarize_stream() -> StreamingResponse:
    """
    Generate a streaming summary for the currently selected collection.

    Returns:
        StreamingResponse: A streaming response that yields SSE events during summarization.

    Raises:
        HTTPException: If an error occurs while generating the summary.
    """
    if not rag.qdrant_collection:
        raise HTTPException(status_code=400, detail="No collection selected")

    async def event_generator() -> AsyncIterator[str]:
        """
        Generate SSE events for the streaming summary.

        Yields:
            AsyncIterator[str]: An asynchronous iterator yielding SSE events.
        """
        try:
            for chunk in rag.stream_summarize_collection():
                if isinstance(chunk, str):
                    yield f"data: {json.dumps({'token': chunk})}\n\n"
                elif isinstance(chunk, dict):
                    yield f"data: {json.dumps(chunk)}\n\n"
        except Exception as e:
            logger.error(f"Stream error: {e}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/collections/ie", tags=["Query"])
def get_collection_ie() -> dict[str, list[dict]]:
    """
    Get all IE data (entities and relations) for the currently selected collection.

    Returns:
        dict[str, list[dict]]: A dictionary containing the list of IE sources.

    Raises:
        HTTPException: If no collection is selected or an error occurs.
    """
    if not rag.qdrant_collection:
        raise HTTPException(status_code=400, detail="No collection selected")
    try:
        sources = rag.get_collection_ie()
        return {"sources": sources}
    except Exception as e:
        logger.error(f"Error fetching collection IE: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/collections/documents", tags=["Query"])
def get_collection_documents() -> dict[str, list[dict]]:
    """
    Get list of documents in the currently selected collection.

    Returns:
        dict[str, list[dict]]: A dictionary containing the list of documents.

    Raises:
        HTTPException: If no collection is selected or an error occurs.
    """
    if not rag.qdrant_collection:
        raise HTTPException(status_code=400, detail="No collection selected")
    try:
        docs = rag.list_documents()
        return {"documents": docs}
    except Exception as e:
        logger.error(f"Error fetching collection documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/sessions/list", response_model=SessionListOut, tags=["Sessions"])
def list_sessions() -> dict[str, list[dict]]:
    """
    List all available chat sessions.

    Returns:
        dict[str, list[dict]]: A dictionary containing the list of sessions.

    Raises:
        ValueError: If the session manager is not initialized.
        HTTPException: If an error occurs while listing sessions.
    """
    try:
        if rag.sessions is None:
            rag.start_session()  # Initialize session manager if needed

        if rag.sessions is None:
            raise ValueError("Session manager not initialized")

        sessions = rag.sessions.list_sessions()
        return {"sessions": sessions}
    except Exception as e:
        logger.error(f"Error listing sessions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get(
    "/sessions/{session_id}/history",
    response_model=SessionHistoryOut,
    tags=["Sessions"],
)
def get_session_history(session_id: str) -> dict[str, list[dict]]:
    """
    Get history for a specific session.

    Args:
        session_id (str): The ID of the session.

    Returns:
        dict[str, list[dict]]: A dictionary containing the session messages.

    Raises:
        ValueError: If the session manager is not initialized.
        HTTPException: If an error occurs while fetching session history.
    """
    try:
        if rag.sessions is None:
            rag.start_session()

        if rag.sessions is None:
            raise ValueError("Session manager not initialized")

        messages = rag.sessions.get_session_history(session_id)
        return {"messages": messages}
    except Exception as e:
        logger.error(f"Error fetching history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/sessions/{session_id}", tags=["Sessions"])
def delete_session(session_id: str) -> dict[str, bool]:
    """
    Delete a session.

    Args:
        session_id (str): The ID of the session to delete.

    Returns:
        dict[str, bool]: A dictionary indicating whether the deletion was successful.

    Raises:
        ValueError: If the session manager is not initialized.
        HTTPException: If an error occurs while deleting the session.
    """
    try:
        if rag.sessions is None:
            rag.start_session()

        if rag.sessions is None:
            raise ValueError("Session manager not initialized")

        success = rag.sessions.delete_session(session_id)
        return {"ok": success}
    except Exception as e:
        logger.error(f"Error deleting session: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/agent/chat", response_model=AgentChatOut, tags=["Agent"])
def agent_chat(payload: AgentChatIn) -> AgentChatOut:
    """
    Agentic chat endpoint: understand → maybe clarify → retrieve/respond.

    Args:
        payload (AgentChatIn): Message and optional session id.

    Returns:
        AgentChatOut: Clarification prompt or answer with sources.
    """
    if not rag.qdrant_collection:
        raise HTTPException(status_code=400, detail="No collection selected")

    # Ensure a session is active and get per-session agent context
    session_id = rag.start_session(payload.session_id)
    ctx = rag.sessions.get_agent_context(session_id) if rag.sessions else None

    turn = Turn(user_input=payload.message, session_id=session_id)
    orchestrator = _build_orchestrator()

    result = orchestrator.handle_turn(turn, context=ctx)

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
    )


@app.post("/ingest", response_model=IngestOut, tags=["Ingestion"])
def ingest(payload: IngestIn) -> dict[str, bool | str]:
    """
    Trigger ingestion for the requested collection using the configured data directory.

    Args:
        payload (IngestIn): The ingestion payload containing the collection name and hybrid flag.

    Returns:
        dict[str, bool | str]: A dictionary indicating success, collection name, data directory, and hybrid flag.

    Raises:
        HTTPException: If the collection name is missing, data directory does not exist, or an error occurs during ingestion.
    """

    try:
        name = payload.collection.strip()
        if not name:
            logger.error("HTTPException: Collection name required")
            raise HTTPException(status_code=400, detail="Collection name required")

        data_dir = _resolve_data_dir()
        if not data_dir.is_dir():
            logger.error("HTTPException: Data directory does not exist: {}", data_dir)
            raise HTTPException(
                status_code=400,
                detail=f"Data directory does not exist: {data_dir}",
            )

        ingest_module.ingest_docs(
            name,
            data_dir,
            hybrid=payload.hybrid if payload.hybrid is not None else True,
        )

        # After ingestion, prepare the in-memory RAG instance for immediate querying.
        rag.select_collection(name)
        try:
            if getattr(rag, "index", None) is None:
                rag.create_index()
            rag.create_query_engine()

            # Pre-warm IE cache if enabled
            if getattr(rag, "enable_ie", False):
                try:
                    rag.get_collection_ie(refresh=True)
                except Exception:
                    logger.warning(
                        "Exception: Failed to pre-warm IE cache for collection: {}",
                        name,
                    )
        except Exception:
            # If eager preparation fails, queries will lazily prepare the engine.
            logger.warning(
                "Exception: Failed to create query engine for collection: {}", name
            )
            pass

        return {
            "ok": True,
            "collection": name,
            "data_dir": str(data_dir),
            "hybrid": payload.hybrid if payload.hybrid is not None else True,
        }
    except HTTPException as e:
        logger.error("HTTPException: Error during ingestion: {}", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/agent/chat/stream", tags=["Agent"])
async def agent_chat_stream(payload: AgentChatIn) -> StreamingResponse:
    """
    Streaming variant of agent chat with token events and final metadata.

    Args:
        payload (AgentChatIn): Message and optional session id.

    Returns:
        StreamingResponse: SSE stream with clarification or answer tokens and metadata.
    """

    async def event_generator() -> AsyncIterator[str]:
        """
        Generate SSE events for the agent chat stream.

        Yields:
            AsyncIterator[str]: An asynchronous iterator yielding SSE events.
        """
        if not rag.qdrant_collection:
            yield _format_sse("error", {"detail": "No collection selected"})
            return

        session_id = rag.start_session(payload.session_id)
        ctx = rag.sessions.get_agent_context(session_id) if rag.sessions else None
        turn = Turn(user_input=payload.message, session_id=session_id)

        analysis = _understanding_agent.analyze(turn)
        clarification_decision = _clarification_policy.evaluate(
            analysis, clarifications_so_far=ctx.clarifications if ctx else 0
        )

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

        # Stream via RAG chat
        stream = rag.stream_chat(turn.user_input)

        # Tokens
        for chunk in stream:
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
    collection: str = Form(...),
    files: list[UploadFile] = File(...),
    hybrid: bool | None = Form(True),
) -> StreamingResponse:
    """
    Upload files for ingestion and stream progress as SSE events.

    Args:
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

    # We use a persistent directory for uploads to support previewing files later.
    # The files are ingested into Qdrant and kept in the collection directory.

    async def event_stream() -> AsyncIterator[str]:
        """
        Stream SSE events during the ingestion process.

        Returns:
            AsyncIterator[str]: A stream of Server-Sent Events (SSE) as strings.

        Yields:
            Iterator[AsyncIterator[str]]: A stream of SSE events during the ingestion process.
        """
        # Use the dedicated sources directory (sibling to Qdrant collections) to store uploaded files
        qdrant_src_dir = _resolve_qdrant_src_dir()
        batch_dir = qdrant_src_dir / name
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
            filename = Path(upload.filename or "upload").name
            dest = batch_dir / filename
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
                    {"message": f"Failed to save {filename}: {exc}"},
                )
                return

        yield _format_sse("ingestion_started", {"collection": name})
        try:
            queue: asyncio.Queue[str | None | Exception] = asyncio.Queue()
            loop: asyncio.AbstractEventLoop = asyncio.get_event_loop()

            def progress_callback(msg: str) -> None:
                """
                Callback function to report progress during ingestion.

                Args:
                    msg (str): Progress message to be reported.
                """
                loop.call_soon_threadsafe(queue.put_nowait, msg)

            async def run_ingestion() -> None:
                """
                Run the ingestion process in a separate thread.
                """
                try:
                    await anyio.to_thread.run_sync(
                        ingest_module.ingest_docs,
                        name,
                        batch_dir,
                        hybrid if hybrid is not None else True,
                        progress_callback,
                    )
                    loop.call_soon_threadsafe(queue.put_nowait, None)
                except Exception as e:
                    loop.call_soon_threadsafe(queue.put_nowait, e)

            asyncio.create_task(run_ingestion())

            while True:
                msg = await queue.get()
                if msg is None:
                    break
                if isinstance(msg, Exception):
                    raise msg
                yield _format_sse("ingestion_progress", {"message": msg})

            rag.select_collection(name)
            try:
                if getattr(rag, "index", None) is None:
                    rag.create_index()
                rag.create_query_engine()
            except Exception:
                logger.warning(
                    "Exception: Failed to create query engine for collection: {}",
                    name,
                )
            yield _format_sse(
                "ingestion_complete",
                {"collection": name, "data_dir": str(batch_dir)},
            )
        except Exception as exc:  # pragma: no cover - streamed errors are logged
            logger.error("Error during streamed ingestion: {}", exc)
            yield _format_sse(
                "error",
                {"message": f"Ingestion failed: {exc}"},
            )

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/sources/preview", tags=["Sources"])
def preview_source(collection: str, file_hash: str) -> FileResponse:
    """
    Serve a previously ingested source file for preview purposes.

    Args:
        collection (str): The name of the collection.
        file_hash (str): The hash of the file to preview.

    Returns:
        FileResponse: A response containing the requested file.

    Raises:
        HTTPException: If an error occurs while retrieving the source preview.
    """
    file_path_str = None
    qdrant_col_dir = _resolve_qdrant_col_dir()
    qdrant_src_dir = _resolve_qdrant_src_dir()

    # 1. Try to resolve filename via Qdrant
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
            # Fallback: try checking 'metadata.file_hash' just in case
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
    except Exception as e:
        logger.warning("Failed to query Qdrant for file preview: {}", e)

    # 2. If we found a path from Qdrant, try to use it
    if file_path_str:
        path = Path(file_path_str)
        if path.exists() and path.is_file():
            return FileResponse(path)

        # Fallback: Check if the file exists in the default data directory
        filename = path.name
        data_dir = _resolve_data_dir()
        alt_path = data_dir / filename

        if alt_path.exists() and alt_path.is_file():
            logger.info("Found file at alternative path: {}", alt_path)
            return FileResponse(alt_path)

        # Check sources root first, then legacy collection path
        src_path = qdrant_src_dir / collection / filename
        if src_path.exists() and src_path.is_file():
            logger.info("Found file at sources path: {}", src_path)
            return FileResponse(src_path)

        col_path = qdrant_col_dir / collection / "sources" / filename
        if col_path.exists() and col_path.is_file():
            logger.info("Found file at legacy collection path: {}", col_path)
            return FileResponse(col_path)

    # 3. Fallback: Scan the sources directory for a matching hash
    # This handles cases where Qdrant is down or the file path in Qdrant is invalid
    try:
        sources_dir = qdrant_src_dir / collection
        legacy_sources_dir = qdrant_col_dir / collection / "sources"

        for candidate_dir in (sources_dir, legacy_sources_dir):
            if not candidate_dir.exists():
                continue
            logger.info("Scanning sources directory for file hash: {}", file_hash)
            for file_path in candidate_dir.iterdir():
                if file_path.is_file() and not file_path.name.startswith("."):
                    try:
                        if compute_file_hash(file_path) == file_hash:
                            logger.info("Found file via hash scan: {}", file_path)
                            return FileResponse(file_path)
                    except Exception:
                        continue
    except Exception as e:
        logger.error("Error scanning sources directory: {}", e)

    raise HTTPException(status_code=404, detail="File not found")
