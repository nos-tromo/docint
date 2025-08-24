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
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

rag = RAG()


class SelectCollectionIn(BaseModel):
    name: str


class QueryIn(BaseModel):
    question: str
    session_id: str | None = None


class QueryOut(BaseModel):
    answer: str
    sources: list[dict] = []
    session_id: str


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
            rag.create_index()
        rag.create_query_engine()
        return {"ok": True, "name": name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query", response_model=QueryOut)
def query(payload: QueryIn):
    try:
        session_id = rag.start_session(payload.session_id)
        data, _ = rag.chat(payload.question)

        answer = str(data.get("response") or data.get("answer") or "")
        sources = data.get("sources") or []

        return {"answer": answer, "sources": sources, "session_id": session_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
