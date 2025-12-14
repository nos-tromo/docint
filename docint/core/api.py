import json
from pathlib import Path
from typing import Any, AsyncIterator, cast

import anyio
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from loguru import logger
from pydantic import BaseModel, Field
from qdrant_client import models
from starlette.middleware.cors import CORSMiddleware

from docint.cli import ingest as ingest_module
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

QDRANT_COL_DIR = load_path_env().qdrant_collections
rag = RAG(qdrant_collection="")


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
    table_row_limit: int | None = Field(default=None, gt=0)
    table_row_filter: str | None = None


class IngestOut(BaseModel):
    ok: bool
    collection: str
    data_dir: str
    hybrid: bool


class SessionListOut(BaseModel):
    sessions: list[dict]


class SessionHistoryOut(BaseModel):
    messages: list[dict]


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

    async def event_generator():
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


def _resolve_data_dir() -> Path:
    """
    Return the configured data directory for ingestion.

    Returns:
        Path: The path to the data directory.
    """

    return load_path_env().data


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
            table_row_limit=payload.table_row_limit,
            table_row_filter=payload.table_row_filter,
        )

        # After ingestion, prepare the in-memory RAG instance for immediate querying.
        rag.select_collection(name)
        try:
            if getattr(rag, "index", None) is None:
                rag.create_index()
            rag.create_query_engine()
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


@app.post("/ingest/upload", tags=["Ingestion"])
async def ingest_upload(
    collection: str = Form(...),
    files: list[UploadFile] = File(...),
    hybrid: bool | None = Form(True),
    table_row_limit: int | None = Form(default=None, gt=0),
    table_row_filter: str | None = Form(default=None),
) -> StreamingResponse:
    """
    Upload files for ingestion and stream progress as SSE events.

    Args:
        collection (str): The name of the collection to ingest into.
        files (list[UploadFile]): The list of files to upload.
        hybrid (bool | None): Whether to enable hybrid search (default: True).
        table_row_limit (int | None): Optional limit applied to tabular rows.
        table_row_filter (str | None): Optional pandas-compatible query string to filter rows.

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
        # Use the Qdrant collections directory to store source files
        # This keeps vectors and source data in the same volume
        qdrant_col_dir = QDRANT_COL_DIR
        # We use a 'sources' subdirectory to avoid conflicting with Qdrant's internal files
        batch_dir = qdrant_col_dir / name / "sources"
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
            await anyio.to_thread.run_sync(
                ingest_module.ingest_docs,
                name,
                batch_dir,
                hybrid if hybrid is not None else True,
                table_row_limit,
                table_row_filter,
            )
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
    qdrant_col_dir = QDRANT_COL_DIR

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

        # Check qdrant_collections/collection/sources/filename
        col_path = qdrant_col_dir / collection / "sources" / filename
        if col_path.exists() and col_path.is_file():
            logger.info("Found file at collection path: {}", col_path)
            return FileResponse(col_path)

    # 3. Fallback: Scan the sources directory for a matching hash
    # This handles cases where Qdrant is down or the file path in Qdrant is invalid
    try:
        sources_dir = qdrant_col_dir / collection / "sources"

        if sources_dir.exists():
            logger.info("Scanning sources directory for file hash: {}", file_hash)
            for file_path in sources_dir.iterdir():
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
