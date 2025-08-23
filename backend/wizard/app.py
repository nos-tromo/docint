from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from wizard.modules.rag import RAG

app = FastAPI(title="Wizard API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "*",
    ],  # narrow later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

rag = RAG()


class SelectCollectionIn(BaseModel):
    name: str


class QueryIn(BaseModel):
    question: str


class QueryOut(BaseModel):
    answer: str
    sources: list[dict] = []


@app.get("/collections/list", response_model=list[str])
def collections_list():
    try:
        return rag.list_collections()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/collections/select")
def collections_select(payload: SelectCollectionIn):
    try:
        name = payload.name.strip() or "default"
        rag.select_or_create_collection(name)
        # attach index / lazy build query engine
        if getattr(rag, "index", None) is None:
            rag._create_empty_index()
        rag._create_query_engine()
        return {"ok": True, "name": name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query", response_model=QueryOut)
def query(payload: QueryIn):
    try:
        res = (
            rag.query_engine.query(payload.question)
            if getattr(rag, "query_engine", None)
            else rag.chat(payload.question)
        )
        # normalize
        answer = ""
        sources: list[dict] = []
        if hasattr(res, "response"):  # LlamaIndex Response/AgentChatResponse
            answer = str(getattr(res, "response"))
            # try to surface sources if present
            sn = getattr(res, "source_nodes", None)
            if sn:
                for n in sn:
                    meta = (
                        getattr(n, "metadata", {})
                        or getattr(getattr(n, "node", None), "metadata", {})
                        or {}
                    )
                    text = (
                        getattr(n, "text", "")
                        or getattr(getattr(n, "node", None), "text", "")
                        or ""
                    )
                    sources.append(
                        {
                            "filename": meta.get("file_name")
                            or meta.get("file_path")
                            or "unknown",
                            "filetype": meta.get("file_type") or "",
                            "source": meta.get("source") or "",
                            "page": meta.get("page"),
                            "row": meta.get("row"),
                            "text": text,
                        }
                    )
        elif isinstance(res, dict):
            answer = str(res.get("response") or res.get("answer") or "")
            sources = res.get("sources") or []
        elif isinstance(res, tuple):
            answer = str(res[0]) if len(res) else ""
            sources = res[1] if len(res) > 1 and isinstance(res[1], list) else []
        else:
            answer = str(res)

        return {"answer": answer, "sources": sources}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
