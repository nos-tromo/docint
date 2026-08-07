# Architecture

Docint is a modular RAG stack. This document describes the runtime
components, how they are wired together, and how a user request flows from
the React SPA down to Qdrant and back.

## Component map

```
+------------------+       HTTP/SSE         +---------------------+
|    React SPA     | <--------------------> |   FastAPI backend   |
|   (frontend/)    |                        |  (docint/core/api)  |
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
| `docint/core/ingest/` | Ingestion pipeline, shared image service, and media transcription (social + standalone) |
| `docint/core/readers/` | File-type-specific readers (PDF, images, tables, JSON / Nextext transcripts) |
| `docint/core/storage/` | Qdrant-backed docstore, hierarchical node storage, source staging |
| `docint/core/state/` | Conversation sessions and citation tracking (SQLAlchemy) |
| `docint/core/ner.py` | Named-entity extraction and graph building |
| `docint/agents/` | Agent orchestrator, understanding, clarification, retrieval, generation |
| `docint/cli/` | CLI entry points (`serve`, `ingest`, `query`, `eval`, `resolve`, `verify`) |
| `frontend/` | React SPA (Vite + TypeScript); see [ui-guide.md](ui-guide.md) |
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

1. **Client call** — the SPA stages files with `POST /ingest/upload` and
   then queues one server-owned job with `POST /ingest/finalize`
   (`docint/core/api.py`, registry in `docint/core/jobs.py`); `POST /ingest`
   and the `ingest` CLI (`docint/cli/ingest.py`) ingest `DATA_PATH`
   directly. See `docs/ingestion.md` for the job lifecycle.
2. **RAG.ingest_docs()** (`docint/core/rag.py`) takes over:
   - Stages source files into the Qdrant sources directory
     (`docint/core/storage/sources.py`).
   - Builds the `DocumentIngestionPipeline`
     (`docint/core/ingest/ingestion_pipeline.py`).
3. **Ingestion pipeline** — before the per-file reader dispatch below, two
   pre-passes sweep the whole batch tree for audio/video:
   `docint/core/ingest/social_linker.py` (media linked via a social export's
   `postings.csv` / `media.csv` manifest) runs first, then
   `docint/core/ingest/standalone_media.py` picks up any other audio/video
   file the linker did not already claim. Both route through the shared
   `docint/core/ingest/media_transcribe.py` engine (`MediaTranscriber`) — a
   remote Nextext call producing transcript segments (text nodes) and, for
   video, keyframes (CLIP image points) — the social linker now delegates
   its per-file Nextext routing to this shared engine rather than calling
   Nextext directly. Every remaining file then dispatches to the matching
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
- `POST /agent/chat/stream` — streams the orchestrator's final answer.

`POST /summarize` is not a streaming endpoint: it answers `200` from cache
or `202` with a `job_id` and lets the caller follow the build on
`GET /ingest/jobs/events` (see below) — the same job-plus-SSE shape as
ingestion, not token streaming.

All three use `_stream_simulated_text()` (`docint/core/api.py:189`) to
replay the complete answer as SSE tokens with a fixed token delay
(`SIMULATED_STREAM_TOKEN_DELAY_SECONDS`), so the client sees a token-level
drip feed.

Ingestion streams are also SSE, but carry **progress events** rather than
generated tokens: `POST /ingest/upload` streams save progress for the bytes
it is staging, and `GET /ingest/jobs/events` carries the run itself — one
owner-multiplexed connection, replayed on connect so a reload re-attaches
mid-run.

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
