# Architecture

Docint is a modular RAG stack. This document describes the runtime
components, how they are wired together, and how a user request flows from
the Streamlit UI down to Qdrant and back.

## Component map

```
+------------------+       HTTP/SSE         +---------------------+
|   Streamlit UI   | <--------------------> |   FastAPI backend   |
|  (docint/ui/)    |                        |  (docint/core/api)  |
+------------------+                        +----------+----------+
                                                       |
                                                       v
                                           +-----------+-----------+
                                           |   AgentOrchestrator   |
                                           |  (docint/agents/)     |
                                           +-----------+-----------+
                                                       |
                          +----------------+-----------+-----------+
                          |                |                       |
                          v                v                       v
                 +----------------+  +------------+        +-----------------+
                 | Understanding  |  | Clarifier  |        |   Retrieval     |
                 |    agent       |  |  agent     |        |     agent       |
                 +----------------+  +------------+        +--------+--------+
                                                                    |
                                                                    v
                                                        +-----------+-----------+
                                                        |        RAG            |
                                                        |  (docint/core/rag.py) |
                                                        +---+---+---+-----------+
                                                            |   |   |
                                                            v   v   v
                                                          Qdrant | SQLite | Filesystem
                                                        (vectors)| (state)| (sources)
```

The same RAG engine is used for both ingestion (write path) and retrieval
(read path).

## Key directories

| Path | Purpose |
|---|---|
| `docint/core/api.py` | FastAPI app, request/response models, streaming handlers |
| `docint/core/rag.py` | RAG engine: ingest, retrieve, rerank, chat/stream |
| `docint/core/ingest/` | Ingestion pipeline and shared image service |
| `docint/core/readers/` | File-type-specific readers (PDF, images, tables, JSON / Nextext transcripts) |
| `docint/core/storage/` | Qdrant-backed docstore, hierarchical node storage, source staging |
| `docint/core/state/` | Conversation sessions and citation tracking (SQLAlchemy) |
| `docint/core/ner.py` | Named-entity extraction and graph building |
| `docint/agents/` | Agent orchestrator, understanding, clarification, retrieval, generation |
| `docint/cli/` | CLI entry points (`ingest`, `query`, `eval`) |
| `docint/ui/` | Streamlit page modules |
| `docint/utils/env_cfg.py` | Centralised environment-variable configuration |

## Request flow: answering a user query

The diagram below expands what happens when the UI calls `POST /query` or
`POST /agent/chat`.

1. **FastAPI entry** — `docint/core/api.py` validates the payload
   (`QueryIn` / `AgentChatIn`). For `/query`, it routes directly to
   `RAG.run_query()` or `RAG.chat()`; for `/agent/chat` it calls the
   orchestrator.
2. **AgentOrchestrator.handle_turn()** — `docint/agents/orchestrator.py:47`
   runs the four-step pipeline:
   1. **UnderstandingAgent** (`docint/agents/understanding.py`). Produces an
      `IntentAnalysis` with `intent`, `confidence`, and extracted entities.
      Two implementations ship — a heuristic `SimpleUnderstandingAgent` and
      an LLM-backed `ContextualUnderstandingAgent` (auto-selected when a
      text LLM is available).
   2. **ClarificationPolicy** (`docint/agents/policies.py`) decides whether
      to ask the user for clarification — based on confidence, missing
      entities, and a per-session clarification budget.
   3. **RAGRetrievalAgent** (`docint/agents/retrieval.py`) routes by
      intent: `ner` / `extract` calls `RAG.get_collection_ner()`; the
      default path calls `RAG.chat()` or `RAG.run_query()` depending on
      whether the caller is stateful or stateless.
   4. **ResultValidationResponseAgent**
      (`docint/agents/generation.py:50`) — optional; re-checks answer
      groundedness against the returned sources and sets
      `validation_mismatch` when the LLM disagrees with the answer.
3. **RAG layer** — `docint/core/rag.py`:
   - Builds the Qdrant query with optional graph expansion
     (`expand_query_with_graph_with_debug`), metadata filters
     (`docint/core/retrieval_filters.py`) and reranker weights.
   - Runs dense + sparse retrieval, applies rerank (LLM or
     `FlagEmbeddingReranker`), and postprocessors for parent-context
     expansion and social/source diversity.
   - Calls the response synthesiser to produce a final answer string.
4. **Session persistence** — for non-stateless queries,
   `SessionManager.chat()` (`docint/core/state/session_manager.py`)
   condenses the user message against the rolling summary, stores a
   `Turn` (`docint/core/state/turn.py`) and its `Citation` rows
   (`docint/core/state/citation.py`) in SQLite.
5. **Response envelope** — the backend reassembles a `QueryOut` /
   `AgentChatOut` payload and returns JSON. Streaming variants replay the
   already-generated answer as SSE tokens via `_stream_simulated_text`
   (`docint/core/api.py:189`).

## Request flow: ingesting documents

1. **Client call** — `POST /ingest` or `POST /ingest/upload`
   (`docint/core/api.py:1126`, `1259`) or the `ingest` CLI
   (`docint/cli/ingest.py`).
2. **RAG.ingest_docs()** (`docint/core/rag.py`) takes over:
   - Stages source files into the Qdrant sources directory
     (`docint/core/storage/sources.py`).
   - Builds the `DocumentIngestionPipeline`
     (`docint/core/ingest/ingestion_pipeline.py`).
3. **Ingestion pipeline** — iterates files, dispatching to the matching
   reader in `docint/core/readers/`:
   - PDFs go through the page-level pipeline
     (`docint/core/readers/documents/`) — triage → layout → OCR →
     extraction → chunking.
   - Images go through CLIP + vision tagging
     (`docint/core/readers/images.py`,
     `docint/core/ingest/images_service.py`).
   - CSV / Parquet / Excel are handled by
     `docint/core/readers/tables.py`.
   - JSON / JSONL / NDJSON by `docint/core/readers/json.py`, which also
     detects Nextext transcripts; each transcript segment becomes one
     retrievable node (same one-to-one pattern as specialized table schemas)
     with timing and speaker metadata in `reference_metadata`.
4. **Hierarchical chunking** — `HierarchicalNodeParser`
   (`docint/core/storage/hierarchical.py`) produces coarse parent chunks
   and fine child chunks, linked by `node_id` metadata. Child chunks can
   later be expanded to their parent context at retrieval time.
5. **NER & hate-speech detection** — chunk-level GLiNER extraction runs in
   parallel workers; entities and hate-speech flags are attached as
   metadata on the resulting nodes.
6. **Persistence** —
   - Chunks are embedded (dense + optional sparse) and upserted into
     Qdrant.
   - Serialised nodes are persisted in a Qdrant-backed KV docstore
     (`docint/core/storage/docstore.py`) with retry/backoff on transient
     Qdrant failures.
   - A file-hash ledger skips re-ingesting unchanged files.

## Stateless vs. session-aware retrieval

- **Stateless** — `retrieval_mode="stateless"` on `/query`. No session is
  created, no history is used. Useful for one-shot questions and API
  integrations.
- **Session** — `retrieval_mode="session"` (the default). Each turn is
  condensed against the rolling summary and prior turns, and persisted as
  a `Turn` row plus `Citation` rows tied to a `Conversation`.

Both modes ultimately call `RAG.run_query()`; the difference lives in
`SessionManager.chat()` which decides how to condense the question.

## Streaming

Three endpoints stream responses back to the client:

- `POST /stream_query` — streams the answer to a single query.
- `POST /summarize/stream` — streams a collection summary.
- `POST /agent/chat/stream` — streams the orchestrator's final answer.

All three use `_stream_simulated_text()` (`docint/core/api.py:189`) to
replay the complete answer as SSE tokens with a fixed token delay
(`SIMULATED_STREAM_TOKEN_DELAY_SECONDS`), so the client sees a token-level
drip feed.

Ingestion via `POST /ingest/upload` (`docint/core/api.py:1259`) is also a
streaming endpoint, but the stream carries **progress events** (upload,
processing, done) rather than generated tokens.

## Configuration surface

All configuration is centralised in `docint/utils/env_cfg.py` as frozen
dataclasses with paired `load_*_env()` factories. New modules that need
environment access must import from there instead of calling
`os.getenv()` directly — see [configuration.md](configuration.md) for the
full list of dataclasses and variables.

## Further reading

- [ingestion.md](ingestion.md) — the write path in depth.
- [retrieval-and-agents.md](retrieval-and-agents.md) — the read path in
  depth.
- [api-reference.md](api-reference.md) — every HTTP endpoint.
