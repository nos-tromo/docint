# API reference

The FastAPI app exposes a REST + SSE surface defined in
`docint/core/api.py`. This document lists every route, groups them by the
tag used in the app, and documents the request and response models.

All request and response bodies are JSON. Pydantic models referenced in
this doc are declared at the top of `docint/core/api.py:208` and onward.

## Base URL & CORS

- The backend binds to port `8000` by default.
- `CORS_ALLOWED_ORIGINS` (default `http://localhost:8501,http://127.0.0.1:8501`)
  controls allowed origins. The CORS middleware accepts `*` for
  methods and headers.
- See [configuration.md](configuration.md#host-endpoints--hostconfig) for
  the full list of host env vars.

## Route map

| Method | Path | Tag | Purpose |
|---|---|---|---|
| `GET`  | `/collections/list` | `Collections` | List all Qdrant collections. |
| `POST` | `/collections/select` | `Collections` | Activate a collection, pre-warms the NER cache. |
| `DELETE` | `/collections/{name}` | `Collections` | Delete a collection. |
| `POST` | `/query` | `Query` | Stateless or session-aware query, non-streaming. |
| `POST` | `/stream_query` | `Query` | Streaming variant of `/query` (SSE tokens). |
| `POST` | `/summarize` | `Query` | Collection-level summary, non-streaming. |
| `POST` | `/summarize/stream` | `Query` | Streaming summary (SSE tokens). |
| `GET`  | `/collections/ner` | `Query` | Full NER dump for the active collection. |
| `GET`  | `/collections/ner/stats` | `Query` | Aggregated NER statistics. |
| `GET`  | `/collections/ner/search` | `Query` | Search for entities by name/pattern. |
| `GET`  | `/collections/hate-speech` | `Query` | Hate-speech findings for the active collection. |
| `GET`  | `/collections/documents` | `Query` | List documents in a collection. |
| `GET`  | `/sessions/list` | `Sessions` | List stored sessions. |
| `GET`  | `/sessions/{session_id}` | `Sessions` | Return conversation history for a session. |
| `DELETE` | `/sessions/{session_id}` | `Sessions` | Delete a session. |
| `POST` | `/agent/chat` | `Agent` | Run the agent orchestrator for one turn (non-streaming). |
| `POST` | `/agent/chat/stream` | `Agent` | Streaming orchestrator variant (SSE tokens). |
| `POST` | `/ingest` | `Ingestion` | Trigger ingestion of the configured `DATA_PATH` into a collection. |
| `POST` | `/ingest/upload` | `Ingestion` | Upload + ingest a single file with streaming progress events. |
| `GET`  | `/sources/preview` | `Sources` | Return a preview of a source file staged under `QDRANT_SRC_DIR`. |

## Collections

### `GET /collections/list`

Returns the list of Qdrant collections as `list[str]`.

### `POST /collections/select`

Request (`SelectCollectionIn`):

```json
{ "name": "demo" }
```

Response (`SelectCollectionOut`):

```json
{ "ok": true, "name": "demo" }
```

Side effects: calls `RAG.select_collection()`, builds the index and query
engine if needed, and pre-warms the NER cache when `enable_ner` is on.

### `DELETE /collections/{name}`

Deletes the named Qdrant collection. Returns `{ "ok": true }` on success.

## Query

### `POST /query`

Runs a single question against the active collection. Source:
`docint/core/api.py:427`.

Request (`QueryIn`):

```json
{
  "question": "What is in this document?",
  "session_id": null,
  "metadata_filters": [],
  "retrieval_mode": "session",
  "query_mode": "answer"
}
```

- `retrieval_mode` — `session` (default) or `stateless`. `session` walks
  through `SessionManager.chat()` and persists a `Turn`; `stateless` calls
  `RAG.run_query()` directly.
- `query_mode` — `answer` (default), `entity_occurrence`, or
  `entity_occurrence_multi`. The entity modes route through
  `RAG.run_entity_occurrence_query()` / `run_multi_entity_occurrence_query()`.
- `metadata_filters` — list of `MetadataFilterIn` objects with
  `{field, operator, value, values}`. Supported operators: `eq`, `neq`,
  `gt`, `gte`, `lt`, `lte`, `in`, `contains`, `mime_match`, `date_after`,
  `date_on_or_after`, `date_before`, `date_on_or_before`.

Response (`QueryOut`):

```json
{
  "answer": "...",
  "sources": [{"text": "...", "metadata": {...}, "score": 0.81}],
  "session_id": "...",
  "graph_debug": null,
  "retrieval_query": "...",
  "coverage_unit": null,
  "retrieval_mode": "session",
  "entity_match_candidates": [],
  "entity_match_groups": [],
  "validation_checked": true,
  "validation_mismatch": false,
  "validation_reason": null
}
```

### `POST /stream_query`

Same inputs as `/query`, streamed as SSE events with token-level output.
The first event carries the `session_id`, subsequent events carry
`{"type": "token", "value": "..."}`, and a final event carries the
complete payload.

### `POST /summarize`

Generates a collection-level summary. Response (`SummarizeOut`):

```json
{
  "summary": "...",
  "sources": [{...}],
  "summary_diagnostics": {
    "total_documents": 100,
    "covered_documents": 72,
    "coverage_ratio": 0.72,
    "uncovered_documents": ["..."],
    "coverage_target": 0.7,
    "coverage_unit": "document",
    "candidate_count": 400,
    "deduped_count": 150,
    "sampled_count": 30
  },
  "validation_checked": true,
  "validation_mismatch": false,
  "validation_reason": null
}
```

`SummaryConfig` (see [configuration.md](configuration.md#summarisation--summaryconfig))
controls `coverage_target`, `max_docs`, and source caps.

### `POST /summarize/stream`

Streaming variant of `/summarize`.

### `GET /collections/ner`

Returns the full cached NER result for the active collection.

### `GET /collections/ner/stats`

Response (`NERStatsOut`):

```json
{
  "totals": {"entities": 1234, "relations": 56, "documents": 78},
  "top_entities": [{...}],
  "entity_types": [{...}],
  "top_relations": [{...}],
  "documents": [{...}]
}
```

### `GET /collections/ner/search`

Accepts a query string and returns matching entity records as
`NERSearchOut` (`{"results": [...]}`).

### `GET /collections/hate-speech`

Returns the list of chunks flagged by hate-speech detection as
`HateSpeechOut`.

### `GET /collections/documents`

Lists the documents currently stored in the active collection.

## Sessions

### `GET /sessions/list`

Returns `SessionListOut` — `{"sessions": [...]}`. Each entry is a row
from `conversations` via `SessionManager.list_sessions()`.

### `GET /sessions/{session_id}`

Returns `SessionHistoryOut` — `{"messages": [...]}` where each message
comes from `Turn` / `Citation`.

### `DELETE /sessions/{session_id}`

Deletes the session and its turns/citations.

## Agent

### `POST /agent/chat`

Runs the orchestrator for one turn. Source: `docint/core/api.py:1070`.

Request (`AgentChatIn`):

```json
{ "message": "Find every mention of Acme Corp.", "session_id": null }
```

Response (`AgentChatOut`):

```json
{
  "status": "answer",
  "message": null,
  "answer": "...",
  "sources": [{...}],
  "session_id": "...",
  "reason": null,
  "intent": "qa",
  "confidence": 0.82,
  "tool_used": "RAG.chat",
  "latency_ms": 412.5,
  "validation_checked": true,
  "validation_mismatch": false,
  "validation_reason": null
}
```

`status` is either `"answer"` (retrieval completed) or `"clarification"`
(the clarification policy decided more information is needed — in which
case `message` holds the question and `reason` the trigger).

### `POST /agent/chat/stream`

Streaming variant that replays the orchestrator's final answer as SSE
tokens.

## Ingestion

### `POST /ingest`

Starts an ingestion job over the configured `DATA_PATH`. Source:
`docint/core/api.py:1126`.

Request (`IngestIn`):

```json
{ "collection": "demo", "hybrid": true }
```

Response (`IngestOut`):

```json
{ "ok": true, "collection": "demo", "data_dir": "/var/lib/docint/data", "hybrid": true }
```

The endpoint returns immediately — follow ingestion progress via logs or
`/ingest/upload` for file-by-file feedback.

### `POST /ingest/upload`

Streaming multipart upload endpoint. Accepts a single file plus a
`collection` form field, streams back progress events:

```
event: progress
data: {"stage": "upload", "message": "…"}

event: progress
data: {"stage": "processing", "message": "…"}

event: done
data: {"ok": true, "stats": {...}}
```

See `docint/ui/ingest.py` for the reference client that consumes this
stream.

## Sources

### `GET /sources/preview`

Returns a preview (or download) of a source file staged under
`QDRANT_SRC_DIR`. Takes query parameters for the source identifier /
path. Used by the UI Inspector page to render citations.

## Request-model reference

All Pydantic models used by the routes live at the top of
`docint/core/api.py`:

- `SelectCollectionIn` / `SelectCollectionOut` (`api.py:208`)
- `MetadataFilterIn` (`api.py:217`)
- `QueryIn` / `QueryOut` (`api.py:238`, `248`)
- `SummaryDiagnosticsOut` / `SummarizeOut` (`api.py:263`, `275`)
- `IngestIn` / `IngestOut` (`api.py:284`, `289`)
- `SessionListOut` / `SessionHistoryOut` (`api.py:296`, `300`)
- `NERStatsOut` / `NERSearchOut` / `HateSpeechOut` (`api.py:304`, `312`, `316`)
- `AgentChatIn` / `AgentChatOut` (`api.py:320`, `325`)

## Streaming semantics

`_stream_simulated_text()` at `docint/core/api.py:189` is the shared
helper behind all token-level streaming. It:

1. Runs the non-streaming handler.
2. Splits the final answer into tokens.
3. Yields each token as an SSE event with a fixed delay
   (`SIMULATED_STREAM_TOKEN_DELAY_SECONDS`, `0.03` s).
4. Yields a final event with the complete `QueryOut` / `AgentChatOut`
   payload, so the client can update citations and metadata in one atomic
   step after the token stream ends.

`POST /ingest/upload` uses a different pattern: it wraps the real
`ingest` pipeline with a callback that yields **progress events** rather
than generated tokens.
