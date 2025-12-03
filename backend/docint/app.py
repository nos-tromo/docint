import json
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncIterator

import anyio
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from loguru import logger
from pydantic import BaseModel, Field

from docint.cli import ingest as ingest_module
from docint.core.rag import RAG
from docint.utils.hashing import compute_file_hash
from docint.utils.logging_cfg import setup_logging

# --- Application Setup ---

setup_logging()

app = FastAPI(title="Document Intelligence")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "*",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

rag = RAG(qdrant_collection="")
SUMMARY_PROMPT = (
    "Provide a concise overview of the active collection. Highlight the main "
    "topics, document types, and notable findings. Limit the response to 8 sentences."
)


def _format_sse(event: str, data: dict[str, Any]) -> str:
    """Return a serialized Server-Sent Event payload."""

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


@app.post("/summarize", response_model=SummarizeOut, tags=["Query"])
def summarize() -> dict[str, list[dict] | str]:
    """Generate a summary for the currently selected collection."""

    try:
        if not rag.qdrant_collection:
            logger.error("HTTPException: No collection selected")
            raise HTTPException(status_code=400, detail="No collection selected")

        if getattr(rag, "query_engine", None) is None:
            if getattr(rag, "index", None) is None:
                rag.create_index()
            rag.create_query_engine()

        data = rag.summarize_collection(SUMMARY_PROMPT)
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


def _resolve_data_dir() -> Path:
    """
    Return the configured data directory for ingestion.

    Returns:
        Path: The path to the data directory.
    """

    if ingest_module.DATA_PATH:
        return Path(ingest_module.DATA_PATH)
    return Path.home() / "docint" / "data"


def _collection_root(collection: str) -> Path:
    base = _resolve_data_dir()
    return base / "collections" / collection


def _record_file_index(
    collection: str, file_hash: str, path: Path, content_type: str | None
) -> None:
    """Persist a simple hash → path mapping for later previews."""

    collection_root = _collection_root(collection)
    collection_root.mkdir(parents=True, exist_ok=True)
    index_path = collection_root / "file_index.json"
    existing: dict[str, dict[str, str]] = {}
    if index_path.exists():
        try:
            existing = json.loads(index_path.read_text())
        except Exception:
            logger.warning("Unable to read file index for collection {}", collection)
    existing[file_hash] = {
        "path": str(path),
        "content_type": content_type or "",
    }
    index_path.write_text(json.dumps(existing, indent=2))


def _lookup_file_by_hash(collection: str, file_hash: str) -> Path | None:
    index_path = _collection_root(collection) / "file_index.json"
    if not index_path.exists():
        return _search_collection_for_hash(collection, file_hash)
    try:
        data = json.loads(index_path.read_text())
    except Exception:
        return _search_collection_for_hash(collection, file_hash)
    entry = data.get(file_hash)
    if not isinstance(entry, dict):
        return _search_collection_for_hash(collection, file_hash)
    path = Path(entry.get("path", "")) if entry.get("path") else None
    if path is not None and path.exists():
        return path
    return _search_collection_for_hash(collection, file_hash)


def _search_collection_for_hash(collection: str, file_hash: str) -> Path | None:
    root = _collection_root(collection)
    if not root.exists():
        return None
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        try:
            current_hash = compute_file_hash(path)
        except Exception:
            continue
        if current_hash == file_hash:
            _record_file_index(collection, file_hash, path, None)
            return path
    return None


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
    """Upload files for ingestion and stream progress as SSE events."""

    name = collection.strip()
    if not name:
        logger.error("HTTPException: Collection name required for upload")
        raise HTTPException(status_code=400, detail="Collection name required")
    if not files:
        logger.error("HTTPException: At least one file is required for upload")
        raise HTTPException(status_code=400, detail="At least one file is required")

    data_dir = _resolve_data_dir()
    if not data_dir.is_dir():
        logger.error("HTTPException: Data directory does not exist: {}", data_dir)
        raise HTTPException(status_code=400, detail="Data directory does not exist")

    upload_root = _collection_root(name) / "uploads"
    batch_dir = upload_root / datetime.utcnow().strftime("%Y%m%d%H%M%S")
    batch_dir.mkdir(parents=True, exist_ok=True)

    async def event_stream() -> AsyncIterator[str]:
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
                file_hash = compute_file_hash(dest)
                _record_file_index(name, file_hash, dest, upload.content_type)
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
                    "Exception: Failed to create query engine for collection: {}", name
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
    """Serve a previously ingested source file for preview purposes."""

    file_path = _lookup_file_by_hash(collection.strip(), file_hash.strip())
    if file_path is None or not file_path.exists():
        logger.error(
            "HTTPException: Unable to locate file for collection={} hash={}",
            collection,
            file_hash,
        )
        raise HTTPException(status_code=404, detail="Source file not found")

    base_dir = _resolve_data_dir().resolve()
    resolved = file_path.resolve()
    try:
        resolved.relative_to(base_dir)
    except ValueError:
        logger.error("HTTPException: File outside data directory: {}", resolved)
        raise HTTPException(status_code=400, detail="Invalid file path")

    return FileResponse(resolved)
