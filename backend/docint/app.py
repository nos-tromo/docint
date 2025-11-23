from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from pydantic import BaseModel

from docint.core import ingest as ingest_module
from docint.core.rag import RAG
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
    table_row_limit: int | None = None
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
