from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from docint.core import ingest as ingest_module
from docint.core.rag import RAG
from docint.utils.logging_cfg import setup_logging

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


class SelectCollectionIn(BaseModel):
    name: str


class QueryIn(BaseModel):
    question: str
    session_id: str | None = None


class QueryOut(BaseModel):
    answer: str
    sources: list[dict] = []
    session_id: str


class IngestIn(BaseModel):
    collection: str
    hybrid: bool | None = True


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
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/collections/select", tags=["Collections"])
def collections_select(payload: SelectCollectionIn) -> dict[str, str | bool]:
    """
    Select a collection to use for queries.

    Args:
        payload (SelectCollectionIn): The payload containing the collection name.

    Returns:
        dict[str, str | bool]: A dictionary indicating success and the selected collection name.

    Raises:
        HTTPException: If the collection name is missing or an error occurs while selecting the collection.
        HTTPException: If an error occurs while selecting the collection.
    """    
    try:
        name = payload.name.strip()
        if not name:
            raise HTTPException(status_code=400, detail="Collection name required")
        rag.select_collection(name)

        # Try to prepare the index lazily. Failures here shouldn't block
        # selecting a collection, so swallow exceptions and allow the frontend
        # to continue. The query endpoint will build the engine on demand if it
        # is still missing.
        if getattr(rag, "index", None) is None:
            try:
                rag.create_index()
                rag.create_query_engine()
            except Exception:
                pass

        return {"ok": True, "name": name}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query", response_model=QueryOut, tags=["Query"])
def query(payload: QueryIn):
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
            raise HTTPException(status_code=400, detail="No collection selected")

        # Ensure the query engine exists before handling questions. If it's not
        # ready yet (e.g. because collection selection failed to build it), try
        # building it now.
        if getattr(rag, "query_engine", None) is None:
            if getattr(rag, "index", None) is None:
                rag.create_index()
            rag.create_query_engine()

        session_id = rag.start_session(payload.session_id)
        data = rag.chat(payload.question)

        answer = str(data.get("response") or data.get("answer") or "")
        sources = data.get("sources") or []

        return {"answer": answer, "sources": sources, "session_id": session_id}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _resolve_data_dir() -> Path:
    """Return the configured data directory for ingestion."""

    if ingest_module.DATA_PATH:
        return Path(ingest_module.DATA_PATH)
    return Path.home() / "docint" / "data"


@app.post("/ingest", tags=["Ingestion"])
def ingest(payload: IngestIn) -> dict[str, object]:
    """Trigger ingestion for the requested collection using the configured data directory."""

    try:
        name = payload.collection.strip()
        if not name:
            raise HTTPException(status_code=400, detail="Collection name required")

        data_dir = _resolve_data_dir()
        if not data_dir.is_dir():
            raise HTTPException(
                status_code=400,
                detail=f"Data directory does not exist: {data_dir}",
            )

        ingest_module.ingest_docs(name, data_dir, hybrid=payload.hybrid if payload.hybrid is not None else True)

        # After ingestion, prepare the in-memory RAG instance for immediate querying.
        rag.select_collection(name)
        try:
            if getattr(rag, "index", None) is None:
                rag.create_index()
            rag.create_query_engine()
        except Exception:
            # If eager preparation fails, queries will lazily prepare the engine.
            pass

        return {
            "ok": True,
            "collection": name,
            "data_dir": str(data_dir),
            "hybrid": payload.hybrid if payload.hybrid is not None else True,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
