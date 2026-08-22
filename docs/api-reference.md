# API reference

The FastAPI app exposes a REST + SSE surface defined in
`docint/core/api.py`. This document lists every route, groups them by the
tag used in the app, and documents the request and response models.

All request and response bodies are JSON. Pydantic models referenced in
this doc are declared at the top of `docint/core/api.py:208` and onward.

## Base URL & CORS

- The backend binds to port `8000` by default.
- `CORS_ALLOWED_ORIGINS` (default `http://localhost:5173,http://127.0.0.1:5173`)
  controls allowed origins. The CORS middleware accepts `*` for
  methods and headers.
- See [configuration.md](configuration.md#host-endpoints--hostconfig) for
  the full list of host env vars.

## Route map

| Method | Path | Tag | Purpose |
|---|---|---|---|
| `GET`  | `/config` | `Meta` | Deploy-time frontend config (graph node default + ceiling, collection timeout). |
| `GET`  | `/collections/list` | `Collections` | List all Qdrant collections. |
| `POST` | `/collections/select` | `Collections` | Activate a collection, pre-warms the NER cache. |
| `DELETE` | `/collections/{name}` | `Collections` | Delete a collection. |
| `POST` | `/query` | `Query` | Stateless or session-aware query, non-streaming. |
| `POST` | `/stream_query` | `Query` | Streaming variant of `/query` (SSE tokens). |
| `POST` | `/search` | `Query` | Full-text keyword search over chunk text (no embeddings, no inference). |
| `POST` | `/summarize` | `Query` | Collection-level (tree/map-reduce) summary: `200` from cache, `202` queues a job. |
| `GET`  | `/summarize` | `Query` | Read-only: the cached summary (`200`) or `204` when there is none. Never queues. |
| `GET`  | `/collections/ner` | `Query` | Full NER dump for the active collection. |
| `GET`  | `/collections/ner/stats` | `Query` | Aggregated NER statistics. |
| `GET`  | `/collections/ner/search` | `Query` | Search for entities by name/pattern. |
| `GET`  | `/collections/ner/graph` | `Query` | Derived entity graph (nodes + edges) for the active collection. |
| `GET`  | `/collections/hate-speech` | `Query` | Hate-speech findings for the active collection. |
| `GET`  | `/collections/documents` | `Query` | List documents in a collection. |
| `GET`  | `/collections/documents/summary` | `Query` | Collection-wide document aggregates (document/node totals + file-type / entity-type breakdown) for the Inspector KPI strip. |
| `GET`  | `/sessions/list` | `Sessions` | List stored sessions. |
| `GET`  | `/sessions/{session_id}` | `Sessions` | Return conversation history for a session. |
| `DELETE` | `/sessions/{session_id}` | `Sessions` | Delete a session. |
| `PUT` | `/sessions/{session_id}/scope` | `Sessions` | Restrict the session's answers to hand-picked chunks. |
| `DELETE` | `/sessions/{session_id}/scope` | `Sessions` | Return the session to normal retrieval. |
| `POST` | `/agent/chat` | `Agent` | Run the agent orchestrator for one turn (non-streaming). |
| `POST` | `/agent/chat/stream` | `Agent` | Streaming orchestrator variant (SSE tokens). |
| `POST` | `/ingest/upload` | `Ingestion` | Stage files into a collection's batch directory (upload only, no ingestion). |
| `POST` | `/ingest/finalize` | `Ingestion` | Queue one ingest job over the staged batches; `202 {job_id}`. |
| `GET`  | `/ingest/jobs/events` | `Ingestion` | Owner-multiplexed SSE stream of job events, with collapsed replay on connect. |
| `GET`  | `/ingest/jobs` | `Ingestion` | List the caller's jobs, newest first. |
| `GET`  | `/ingest/jobs/{job_id}` | `Ingestion` | Snapshot of one owned job. |
| `DELETE` | `/ingest/jobs/{job_id}` | `Ingestion` | Dismiss a finished job (409 while running). |
| `POST` | `/ingest` | `Ingestion` | Ingest the configured `DATA_PATH` directly (CLI/batch path). |
| `GET`  | `/sources/preview` | `Sources` | Return a preview of a source file staged under `QDRANT_SRC_DIR`. |

## Meta

### `GET /config`

Deploy-time frontend configuration for the SPA, read once on load and served
without a principal. Returns `graph_top_k`, `graph_max_top_k`, and
`collection_timeout` (`FrontendConfigOut`); the first two come from
`NER_GRAPH_TOP_K` / `NER_GRAPH_MAX_TOP_K` (see Configuration).

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
  "retrieval_mode": "session"
}
```

- `retrieval_mode` — `session` (default) or `stateless`. `session` walks
  through `SessionManager.chat()` and persists a `Turn`; `stateless` calls
  `RAG.run_query()` directly.
- `metadata_filters` — list of `MetadataFilterIn` objects with
  `{field, fields, operator, value, values}`. Supported operators: `eq`, `neq`,
  `gt`, `gte`, `lt`, `lte`, `in`, `contains`, `mime_match`, `date_after`,
  `date_on_or_after`, `date_before`, `date_on_or_before`.

  A filter targets either a single `field` or several `fields`; when several
  are given the rule matches if any of them matches. The SPA uses this to apply
  one date bound to both `reference_metadata.timestamp` and
  `reference_metadata.posting_timestamp`. A rule naming neither is rejected
  with 422.

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

### `POST /search`

Full-text keyword search over the chunk text of the caller's collection. Pure
local lookup — one native Qdrant scroll, **no embedding call and no
inference** — which is what makes it fast enough to drive a search box.

Request:

```json
{
  "question": "berlin konferenz",
  "collection": "<logical name>",
  "metadata_filters": [],
  "limit": 50,
  "cursor": null
}
```

- `question` — whitespace-separated keywords. **All** must match the same
  chunk, in any order.
- `metadata_filters` — the same `MetadataFilterIn` shape as `/query`, ANDed
  with the keyword conditions so filters constrain the search.
- `field` — which payload field the keywords match: `text` (default, the
  chunk body), `author`, `network` or `uuid`. `422` on any other value.
  There is no `file_name` field: filter by filename with `metadata_filters`.

  One option can cover several payload keys, and the query must be satisfied
  by **one** of them: `author` searches `reference_metadata.author`,
  `vanity`, `posting_author` and `posting_vanity` (the last two are what an
  image or transcript inherits from its parent posting), plus the numeric
  `author_id` / `posting_author_id`.

  Names match case-insensitively on word prefixes, with a multi-word query
  required to occur as a contiguous phrase in a single key. **Ids match
  exactly**, tried in both numeric and string form, because they are stored
  as numbers and a full-text matcher cannot touch a number; a query
  containing whitespace is never treated as an id.

  `uuid` is exact-match only. It searches the posting's own
  `reference_metadata.uuid` and the `posting_uuid` every derived image,
  keyframe and transcript segment carries, so one paste returns the post and
  all of its artifacts. The pasted form and its dash-normalised twin are both
  tried (exports store it undashed). A multi-word query matches nothing.

  A field whose keys are not all indexed correctly answers
  `status: "not_indexed"` — run `make search-index` once. The `_images`
  companion is searched only for `text`, `author` and `uuid`.
- `limit` — hits per page, `1..500` (default `50`).
- `cursor` — opaque page cursor from a previous response.

Response:

```json
{
  "status": "ok",
  "hits": [
    {
      "id": "<qdrant point id>",
      "chunk_id": "...",
      "filename": "...",
      "page": 3,
      "row": null,
      "preview": "...",
      "kind": "text",
      "entity_types": ["LOC"],
      "est_tokens": 412,
      "truncated": true
    }
  ],
  "total": 14,
  "next_cursor": null,
  "index_status": {
    "indexed": true,
    "total": 724,
    "with_search_text": 724,
    "missing": 0,
    "complete": true
  }
}
```

`preview` is capped at 600 characters; `truncated` says whether there is more.
Fetch the whole chunk on demand with `GET /search/chunk?id=<point id>` — it
returns `{id, text}`, or `404` when the point is gone (re-ingestion mints new
ids, and an empty body would read as an empty chunk rather than a missing one).

Search covers two lanes: the collection's document chunks and its
`{collection}_images` companion, whose points carry an image's caption and tags.
`kind` is `"text"` or `"image"` — an image hit's body is a caption, not document
prose. The lanes run in sequence and a page fills across the boundary, so a
short final page of text hits never strands the image hits behind it. `total`
covers both.

Matching semantics:

- **Case-insensitive**, including non-ASCII text. This depends on the
  `search_text` payload index (`PREFIX` tokenizer, `lowercase=true`) — an
  un-indexed `MatchText` only case-folds ASCII, so German title-case tokens
  would not match their lowercase form.
- **Prefix**, not substring: `Partei` finds `Parteitag`, but `tag` does not.
  German compounds are head-final, so the discriminating fragment is normally
  the start of the word.
- Coarse parent chunks are excluded, so a hierarchical collection does not
  return both a parent and its child for one logical hit.

Status and errors:

- `status: "ok"` — every point in the collection is indexed, so an empty `hits`
  list means "no matches" and nothing else.
- `status: "partial"` — some points are not indexed, so **the hit list is
  incomplete**: a backfill is running or was interrupted.
  `index_status.missing` says how many chunks are still unsearchable. This is a
  distinct status rather than a nested field precisely so a client cannot miss
  it by ignoring `index_status` — a search that silently under-returns is the
  worst failure mode for an investigative tool.
- `status: "not_indexed"` — no point carries the field; the collection has
  never been backfilled. `hits` is empty. Run
  `make search-index COLLECTION=<name>` once.
- `422` — the query is blank, or a keyword is shorter than 2 characters. Such a
  keyword cannot be indexed and would contribute a condition that never
  matches, silently reducing the whole search to zero hits.
- `404` — the collection is not owned by the caller.

### `GET /search/export.csv`

Streams **every** matching chunk as CSV — the exhaustive, chunk-level
counterpart to `/search` above, and what an investigator downloads to work
with a result set outside the app. One lane, mirroring `/search` and
exported exactly as it returns: this export must never show more, or fewer,
rows than the panel does for the same query.

Query parameters:

- `collection` — caller's logical collection; falls back to the process
  default when omitted.
- `question` — whitespace-separated keywords. **Optional**: a blank
  `question` exports the whole filtered collection rather than being
  refused — the panel itself never issues a blank query, but a full dump is
  a legitimate export.
- `field` — as above (default `text`).
- `metadata_filters` — JSON-encoded array, same shape as `/query`.
- `session_id` — a session whose stored chat scope counts as "marked",
  unioned with `marked_ids`.
- `marked_ids` — comma-separated Qdrant point ids to mark, for a selection
  made before a session exists.

Mirrors `/search`: the same AND-of-keywords prefilter, the same phrase
post-filter on a multi-word query (a chunk containing both words far apart
does not count), and the same `{collection}_images` companion scroll for a
field an image point can carry — an image hit the panel shows is a row here
too, with `kind=image`.

Columns (`SEARCH_EXPORT_COLUMNS`): `marked`, `kind`, `source`, `page`, `row`,
`chunk_id`, `chunk_text`, `network`, `author`, `author_id`, `vanity`, `url`,
`timestamp`, `posting_network`, `posting_author`, `posting_author_id`,
`posting_vanity`, `posting_timestamp`, `posting_url`, `posting_text`,
`type`, `uuid`, `posting_uuid`, `posting_id`, `media_id`, `speaker`,
`language`, `detected_language`, `source_file`. Unlike a citation card's
truncated preview, `chunk_text` is always the chunk's **full** text. `kind`
is `text` or `image`, mirroring `/search`. `marked` is `true` when the row's
point id is in `session_id`'s stored scope or in `marked_ids`, so a prior
hand-picked selection can be recovered from a re-export.

Rows are sorted by `source`, then `page`, then `row`.

Status and errors:

- `200` — a `text/csv; charset=utf-8` streamed attachment (a UTF-8 BOM, then
  the header row, then one row per matching chunk). An empty collection
  streams a header-only CSV rather than erroring — there is nothing to
  index, so it is not a missing backfill.
- `409` — a keyword search (`question` non-blank) runs over a collection
  that is unindexed or only **partially** indexed for `search_text`
  (`make search-index` still running or interrupted), or, when `field` is
  not `text`, whose field index is missing outright. Unlike `/search`,
  which can carry `status: "partial"` beside a banner, a downloaded CSV has
  no channel of its own to report incompleteness, so this is refused
  outright rather than risk a truncated file being filed as the complete
  evidence. Also `409` when the matching set exceeds the row cap
  (`MAX_EXPORT_ROWS`, 50,000 chunks) — the same reasoning: a silently
  truncated file is indistinguishable from a complete one once downloaded,
  so an oversize export is refused rather than cut short. Narrow the query
  or add metadata filters and try again.
- `422` — `field` is not one of the whitelisted values, a non-blank
  `question` parses to no usable keyword (every word shorter than the index
  minimum), or `metadata_filters` is not valid JSON.
- `404` — the collection is not owned by the caller.

### `POST /summarize`

Serves the collection's cached tree summary, or queues a rebuild. Query
parameters: `refresh` (bool, default `false`) forces a rebuild even when a
cached summary exists; `collection` is the caller's logical collection name
— optional on a cache read (falls back to the process-default active
collection) but **required** (`400` otherwise) whenever a build must be
queued, since a job snapshot must never leak the owner-namespaced physical
Qdrant name back to a client.

| Status | Meaning |
|---|---|
| `200` | Cache hit — `SummarizeOut` body below. |
| `202` | Cache miss, or `refresh=true` — a `kind="summary"` job was queued; `{"job_id": "…"}`. Progress arrives on `GET /ingest/jobs/events` (`summary_started` / `summary_progress` / `summary_completed`), the same owner-multiplexed stream ingest jobs use. |
| `400` | `collection` omitted while queuing a build. |
| `404` | Caller does not own the collection. |
| `409` | A summary build is already in flight for this collection — `detail` carries `{"message", "job_id"}` of the running job. |

A collection that has never been summarized, or whose last automatic build
failed, has no degraded fallback: this endpoint answers `202` and builds one in
the background. The SPA therefore does not open the Summary tab against it —
it probes with `GET /summarize` first and reaches this route only when an
operator presses Create or Refresh.

### `GET /summarize`

The read-only half. Returns the cached summary — the identical body the POST's
cache hit returns, built by the same `_cached_summary_payload` — or reports that
there is none. **It never queues a build.** Query parameter: `collection`, the
caller's logical name, optional here (this path creates no job, so the
process-default fallback is safe) unlike on the POST.

| Status | Meaning |
|---|---|
| `200` | Cache hit — the `SummarizeOut` body below. |
| `204` | Nothing cached for this collection. No body, no job queued. |
| `404` | Caller does not own the collection. |

The split is by HTTP method rather than a flag on the POST because the SPA
fires this whenever the Summary tab opens. A build is a minutes-long job of up
to `SUMMARY_MAX_LLM_CALLS` model calls, so a caller who forgot to pass some
`queue=false` would start one by merely looking; a handler with no queue branch
cannot. `204` rather than `404` on a miss keeps "there is nothing here yet"
distinguishable from "you may not read this", which `404` already means.

`200` response (`SummarizeOut`):

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
    "coverage_unit": "documents",
    "candidate_count": 100,
    "deduped_count": 72,
    "sampled_count": 24,
    "partial": false,
    "llm_calls": 143
  },
  "validation_checked": true,
  "validation_mismatch": false,
  "validation_reason": null
}
```

`total_documents`/`covered_documents`/`candidate_count`/`deduped_count` are
now unit counts, not document-sample counts — `coverage_unit` is
`"documents"`, `"posts"`, or `"units"` depending on what the collection's map
units turned out to be. `partial` is `true` when `SUMMARY_MAX_LLM_CALLS` cut
the build short — skipping units, truncating one unit's windows, or stopping
a reduce-fold tier early — so the summary does not reflect the whole
collection. A `200` may therefore carry `partial: true`: every build that
*completes* is cached, including a capped one and one over an empty
collection, and the SPA's coverage banner renders an explicit "incomplete
summary" notice for it. Only a build that fails mid-way caches nothing (it
fails its job instead). Caching a partial build is deliberate: `/summarize`
answers `200` solely from the cache, so a build that is never cached is never
served — the client's post-completion refetch would miss, silently queue
another full build, and report a failure, forever for an empty collection.
`SummaryConfig` (see
[configuration.md](configuration.md#summarisation--summaryconfig)) controls
these knobs. `partial` is declared on the `SummaryDiagnosticsOut` Pydantic
model; `llm_calls` is present on the wire but not declared (the route is
`response_model=None`, so nothing strips it).

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

### `GET /collections/ner/graph`

Returns a derived entity graph (`NERGraphOut`) for the active collection,
powering the Analysis NER **Graph** view. Nodes are the top `top_k_nodes`
(default 80, max 500) entities by mention count; edges combine extracted
relations with co-occurrence links above `min_edge_weight`. `entity_merge_mode`
matches the other NER views. Node `id` is a cluster key — clients map a node
back to an entity for drill-down via its `text`/`type` fields.

```json
{
  "nodes": [{"id": "acme::org", "text": "Acme", "type": "ORG", "mentions": 9}],
  "edges": [{"source": "acme::org", "target": "rivertown::loc", "label": "located_in", "kind": "relation", "weight": 3}],
  "meta": {"node_count": 80, "edge_count": 142}
}
```

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

### `PUT /sessions/{session_id}/scope`

Restrict a session's answers to a hand-picked set of chunks, selected from the
search panel. Request `{"chunk_ids": ["<point id>", ...]}`; response carries the
stored ids plus `est_tokens`, `usable_tokens` and `missing`.

Scoped answering splices the chosen chunks straight into the prompt, so the
selection is bounded by the chat context window rather than by a top-k. A
selection that does not fit is **refused with 422**, never truncated: silently
dropping part of an investigator's evidence would produce an answer that looks
complete and is not.

The scope is stored on the conversation, so it survives a reload and reopening
the session — like the pinned collection. Owner-gated: a session that is missing
or belongs to another principal returns `404` either way.

While a scope is active, `/query` and `/stream_query` report
`retrieval_mode: "scoped"` and `scoped_chunk_count`, and answer **only** from
those chunks — no vector query, no rerank. Re-ingestion mints new point ids, so
a scope can outlive its chunks; the count of ids Qdrant no longer has comes back
as `missing`.

### `DELETE /sessions/{session_id}/scope`

Return the session to normal retrieval. Returns an empty scope. `404` when the
session is missing or not owned.

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

The SPA's ingest flow is two-phase: stage bytes with `POST /ingest/upload`
(once per batch), then queue **one** server-owned job over the whole staged
directory with `POST /ingest/finalize`. Progress is consumed from the
owner-multiplexed SSE stream, not from the request that started the run — so
navigating away or reloading no longer severs the only view of a run that
keeps going regardless.

Jobs live in memory. They survive a browser reload (the client re-discovers
them by owner) but **not** a backend restart; the staged files remain on disk
and hash dedup makes a re-run cheap. All job endpoints are owner-scoped: a
job belonging to another principal 404s rather than 403s, so its existence
never leaks.

### `POST /ingest/upload`

Streaming multipart upload. Accepts one or more files plus a `collection`
form field, saves them into that collection's batch directory, and streams
save progress as SSE. It does **not** ingest — the caller runs one ingestion
pass afterwards via `/ingest/finalize`.

Splitting a large selection across several upload batches means ingestion
happens once over the whole staged directory, instead of once per batch
(which would re-initialise the pipeline's models per batch and hard-fail on
any batch that happened to hold only reader-unsupported files).

### `POST /ingest/finalize`

Queues an ingest job over a collection's already-staged batches. Returns
`202` immediately:

```json
{ "job_id": "…" }
```

Request (`IngestIn`) carries the logical collection name and the run's
enrichment options (`hybrid`, `ner`, `hate_speech`). Entity resolution runs
as a stage *inside* the job, so it no longer depends on a client staying
attached.

| Status | Meaning |
|---|---|
| `202` | Job queued; `job_id` returned. |
| `400` | Blank collection name. |
| `404` | Caller does not own the collection. |
| `409` | That collection already has a job in flight — `detail` carries the in-flight `job_id`. |

The `409` is load-bearing, not a convenience: overlapping runs over one
collection can double-write, because file hashes are only recorded after a
run's final node batch.

### `GET /ingest/jobs/events`

One SSE connection carrying every job the caller owns, of either `kind`,
each frame tagged with its `job_id`. On connect the stream replays a
**collapsed** history per job — the kind's `*_started` event, then the
retained warnings, then the latest `*_progress`, then the terminal frame —
so a browser that reloads mid-run re-attaches and resumes the live view.
Event names are kind-specific: `ingestion_started` / `ingestion_progress` /
`ingestion_complete` for `kind="ingest"`, `summary_started` /
`summary_progress` (carrying `mapped`/`total_units`) / `summary_completed`
for `kind="summary"`. A client must filter frames by `job_id` (and, if it
cares, `kind`) since both kinds share the one connection.

Progress is collapsed to the newest frame because a long run emits thousands
of them and only the newest describes the current state. Warnings are
retained individually, since each carries unique information.

Terminal frames are `ingestion_complete`/`summary_completed` or `error`. The
`error` frame carries a machine-readable `code` (`ingestion_failed` or
`summary_failed`) alongside static display copy — exception text never
reaches a client, as it can carry connection strings or file paths.

### `GET /ingest/jobs`

Lists the caller's jobs of either `kind`, newest first. Not used by the SPA
(reload re-discovery goes through the persisted job id plus the SSE replay);
available for other clients that want to enumerate a caller's runs.

### `GET /ingest/jobs/{job_id}`

Point-in-time snapshot of one owned job: `kind` (`"ingest"` or `"summary"`),
status, latest message, error, and timestamps. The owner-namespaced physical
collection name is deliberately excluded — callers only ever see their own
logical name.

A `404` here is how a client detects an **interrupted** run: the backend
restarted while the job was in flight.

### `DELETE /ingest/jobs/{job_id}`

Dismisses a finished job from the registry. Refuses with `409` while the job
is still queued or running: the worker thread cannot be killed, so a
dismissed running job would keep writing unobserved.

### `POST /ingest`

Ingests the configured `DATA_PATH` directly, without the upload/finalize
staging dance. This is the CLI and batch path — the SPA does not use it.

Request (`IngestIn`):

```json
{ "collection": "demo", "hybrid": true }
```

Response (`IngestOut`):

```json
{ "ok": true, "collection": "demo", "data_dir": "…", "hybrid": true }
```

Returns when the run completes; follow progress via logs.

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
- `SummaryDiagnosticsOut` / `SummarizeOut` (`api.py:692`, `706`)
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

The ingestion endpoints use a different pattern: they carry **progress
events** rather than generated tokens. `POST /ingest/upload` streams the
save progress of the bytes it is staging, while `GET /ingest/jobs/events`
streams the run itself — one connection per caller covering every job they
own, replayed on connect rather than tied to the request that started the
run.
