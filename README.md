# Document Intelligence

Document Intelligence is a document RAG stack for ingestion, retrieval,
and chat. It ships with:

- a FastAPI backend
- a React SPA served by an nginx sidecar that reverse-proxies API
  requests to the backend on the internal Docker network
- pluggable inference via any OpenAI-compatible API or an external routed vLLM service

## What You Need

- Docker for the containerized setup
- Python 3.11 and `uv` for local development
- an inference backend: any OpenAI-compatible endpoint configured via `.env`
  - vLLM is deployed separately and consumed via one routed base URL
  - local development needs an OpenAI-compatible endpoint you manage yourself
  - on a non-CUDA dev host, `vllm-service` also ships standalone CPU-only
    profiles for the services Docint calls remotely: `gliner-only`
    (NER), `rerank-only`, `clip-only`, and `embed-only` (dense embedding
    plus sparse weights and tokenization, from one bge-m3 instance).
    Point `NER_API_BASE` / `RERANK_API_BASE` / `CLIP_API_BASE` /
    `SPARSE_API_BASE` / `EMBED_API_BASE` at each dedicated container
    instead of the full router — see `docs/configuration.md` for the
    per-service defaults. `EMBED_API_BASE` is the one that does not take
    the bare host: the OpenAI SDK appends `/embeddings` to it, so it must
    end in `/v1` (`http://embed-only:8000/v1`) while `SPARSE_API_BASE`
    stays `http://embed-only:8000` — the same container, addressed two
    ways. Getting it wrong 404s every embedding call; docint reports that
    as an `EmbeddingEndpointError` naming the endpoint and this knob, and
    an ingest run fails on its pre-flight probe before staging any file.
    Setting both `EMBED_API_BASE` and
    `SPARSE_API_BASE` to the same `embed-only` container loads bge-m3
    once instead of twice, and it replaces Ollama's bge-m3 for dense
    embeddings too — Ollama then serves chat only. The `embed-only`
    shape requires a `vllm-service` release that ships it; until that
    release is available, run the full stack (or another of the three
    `*-only` profiles) instead.

> **Re-ingest note:** the sparse model changed from BM42
> (`Qdrant/all_miniLM_L6_v2_with_attentions`) to `BAAI/bge-m3` for
> non-vLLM providers, and dense embeddings on the `embed-only` shape now
> come from that same bge-m3 instance (fp32 transformers) instead of
> Ollama's quantised GGUF build. Dev collections created before this
> change need **bge-m3 vectors, which a plain re-ingest will not give
> them** — for two reasons: file-hash dedup skips any source file
> already recorded as ingested, and swapping models mid-collection would
> leave old and new points computed by different models side by side
> with no way to tell them apart. The dense dimension itself doesn't
> change — bge-m3 is 1024-wide both as Ollama's GGUF build and as fp32
> transformers — so checking the vector width is not a valid way to
> confirm this migration is unnecessary; the drift is in the numeric
> values (quantised vs. fp32) and, for sparse, the switch away from BM42
> entirely. The fix is the same for both changes — **delete the
> collection and ingest it again from scratch** covers dense and sparse
> in one migration, not two. Production (vLLM) collections already used
> bge-m3 for both and are unaffected.

## Quick Start With Docker

1. Create the shared env file:

   ```bash
   cp .env.example .env
   ```

2. Create the shared external networks and volumes once (idempotent):

   ```bash
   make network    # inference-net + data-net
   make volumes    # external Docker volumes
   ```

3. Build and start the stack:

   ```bash
   make build
   make up-dev
   ```

   `make up-dev` layers `docker/compose.override.yaml` so host ports are
   published for local development; `make up` runs the base
   `docker/compose.yaml` alone (production shape, no host ports). Both are
   detached and never build (`up -d --no-build`) — run `make build` first
   (as above), or use `make dev` to build then `up-dev`.

4. Open the app:

   - App: <http://localhost:8080> (override with `DOCINT_HOST_PORT` in `.env`)

   The backend is reachable only via the nginx sidecar — it is no longer
   published on the host. Use `docker compose --env-file .env -f
   docker/compose.yaml exec backend …` to interact with it directly.

   Qdrant is **not** served by this stack — it is provided by the sibling
   `data-plane` project. Start it once with `cd ../data-plane && make up`
   (or `make up-dev` to also publish Qdrant on `localhost:6333`).

### Docker Notes

- docint ships a single CPU-only image. All ML inference (chat,
  embeddings, rerank, NER, CLIP) is delegated to the external
  [vllm-service](https://github.com/nos-tromo/vllm-service) stack.
- Set `INFERENCE_PROVIDER` and `OPENAI_API_BASE` in `.env`.
- The `openai` provider requires `OPENAI_API_KEY` in `.env`.
- The `vllm` provider requires `OPENAI_API_BASE` in `.env`.
  Deploy the standalone vLLM app first, then start Docint. You can use [vllm-service](https://github.com/nos-tromo/vllm-service) to serve text, vision, embedding, reranking and audio endpoints.
  Docint expects the vLLM router to expose one OpenAI-compatible base URL that
  ends in `/v1`, plus the vLLM sparse routes at `/pooling` and `/tokenize`.
- For co-deployed stacks on one server, create one shared external Docker
  network, attach both compose projects to it, and set
  `OPENAI_API_BASE=http://vllm-router:4000/v1`.
- First startup may take a while because model assets are downloaded into the
  shared cache volumes.
- If you use an outbound proxy, put the proxy variables in `.env` so Compose,
  image builds, and containers use the same values.
- Large uploads are handled by client-side batching plus a single ingest job.
   The Ingest view uploads a file selection as several *staged* `/ingest/upload`
   batches (each call only saves files — it never ingests) that stay under the
   frontend nginx proxy's per-request cap (`DOCINT_CLIENT_MAX_BODY_SIZE`,
   default `1g`, advertised to the SPA via `GET /config` as
   `max_upload_bytes`), then calls `POST /ingest/finalize` once to queue an
   ingest job (`202 {job_id}`) over the whole staged directory. So the total
   upload size is no longer bounded by that cap; ingestion sees the complete
   selection at once (a batch that happens to hold only media never trips a
   "No files found" error, and the models load once instead of per batch); and
   a failed upload batch is reported without blocking the rest — finalize
   still runs, and already-saved files are skipped on retry (ingestion is
   idempotent by file hash). Raise `DOCINT_CLIENT_MAX_BODY_SIZE` in `.env` only
   if a *single* file exceeds the default `1g` (both the frontend and backend
   services read the same value).
- **Ingest jobs.** Ingestion runs as a server-owned job (`docint/core/jobs.py`),
   not on the request that started it — so a browser reload no longer discards
   progress or silently skips entity resolution. Endpoints: `POST
   /ingest/finalize` queues a job (`202 {job_id}`; `409` with the existing
   `job_id` if that collection already has a job in flight); `GET /ingest/jobs`
   lists the caller's jobs, newest first; `GET /ingest/jobs/events` is an SSE
   stream of every owned job's events, multiplexed over one connection and
   replaying each job's collapsed history on connect so a reconnecting client
   resumes the live view; `GET /ingest/jobs/{job_id}` returns a point-in-time
   snapshot; `DELETE /ingest/jobs/{job_id}` dismisses a finished job (`409`
   while still queued/running). Jobs are held in memory, bounded by
   `DOCINT_INGEST_CONCURRENCY` (default `1`, serial) — set higher only if the
   inference backend can absorb concurrent embedding/NER/hate-speech traffic
   from more than one ingest at a time. Jobs survive a browser reload but not a
   backend restart; the staged files remain on disk either way, so a re-run is
   cheap (hash-deduped).
- **A run has exactly one duration.** The server measures it once — from the
   moment the user started, upload leg included, through the queue wait and
   every stage of the job — and both the backend log's run summary
   (`Ingest job completed | … duration=00:19 duration_ms=19004 …`) and the
   ingest card's timer show that same value. The card gets it from
   `duration_ms` on the terminal SSE frame, and a reattached
   client from the job snapshot's `duration_ms` / `run_started_at`
   (`started_at` still means "a worker slot was acquired"). The log prints
   both forms for the same reason — a readable one and the exact integer the
   card renders, so the two can be compared rather than trusted. Because the
   upload happens before the job exists, the SPA reports how long it took as
   `upload_elapsed_ms` on `POST /ingest/finalize` — an elapsed duration, never
   a timestamp, so no client clock is trusted, and it is clamped server-side.
   Deriving a second duration on the client is what previously let one run
   report two numbers a second apart.
- **Reading the backend log.** `docker logs -f docint-backend-1` narrates a
   whole ingest: a start banner naming every staged file with its size and
   type, per-file progress, a throttled heartbeat through the long
   enrich/embed/persist stages, the collection-summary build, and the run
   summary above. Every line of a run carries its `job_id`, so
   `docker logs docint-backend-1 | grep <job_id>` isolates one run. Chat
   turns get one `Turn complete | …` line each (retrieval mode, source and
   image counts, rerank state, duration). Queries, answers and document text
   are never logged — only their shapes. Details and knobs:
   [docs/ingestion.md](docs/ingestion.md#observability) and
   [docs/configuration.md](docs/configuration.md#logging--loggingconfig).
- Session persistence uses one SQLite file path. Set `SESSIONS_DB_PATH` for
  the normal case or `SESSION_STORE` if you want to supply a full SQLAlchemy
  database URL.
- **Multi-user / data isolation.** Collections, chat sessions, and reports are
   owner-scoped: each user sees and operates only on the collections they
   ingested and the reports they created. Collection names are per-user — two
   users can each have a `my_collection` without collision (the physical Qdrant
   collection is namespaced per owner; the bare name is shown in the UI). Every
   collection-scoped request carries its collection explicitly and is owner-gated
   (cross-owner access is a 404), and the active collection is resolved
   per-request, so concurrent users on different collections never interfere.
   Chat sessions are further scoped to the collection they were created in: the
   sidebar lists only the active collection's chats, resuming a chat requires its
   collection to be active, and **deleting a collection also deletes its chat
   sessions** (along with the collection's documents and companion data).
- **Identity.** The React SPA does not add an authenticated-user header itself.
   For single-user Docker or local setups, set `DOCINT_DEFAULT_IDENTITY` in
   `.env` so every request shares one owner. To run multi-user, put docint behind
   a trusted proxy that authenticates each user and injects a user header, and
   set `DOCINT_AUTH_HEADER` to that header name (default `X-Auth-User`); a request
   with no header falls back to `DOCINT_DEFAULT_IDENTITY` (or 401 if neither is
   set). Legacy sessions and collections with no owner are backfilled to
   `DOCINT_DEFAULT_IDENTITY` when the backend initializes.

### Shared Docker Volumes

The compose file uses external volumes so model artifacts and backend
state survive container recreation — and so `docker compose down -v`
cannot destroy staged sources or the session database:

- `docling-cache`
- `huggingface-cache`
- `ollama-cache`
- `sessions-storage`
- `source-preview-cache`

`make volumes` creates them directly with `docker volume create`.

## Local Development

Use this when you want to run the Python services directly instead of through
Docker.

1. Copy the local env file:

   ```bash
   cp .env.example .env
   ```

2. Ensure the required services exist:

   - Qdrant at `http://localhost:6333` — provided by the sibling
     `data-plane` project (`cd ../data-plane && make up-dev`)
   - an OpenAI-compatible inference endpoint, such as an external vLLM service

3. Install dependencies:

   ```bash
   uv sync
   ```

4. Optional: pre-download local model assets:

   ```bash
   uv run load-models
   ```

5. Start the backend:

   ```bash
   uv run uvicorn docint.core.api:app --reload
   ```

6. Start the frontend in another terminal (optional — for live development):

   ```bash
   cd frontend
   pnpm install
   pnpm dev      # → http://localhost:5173 (proxies /api to :8000)
   ```

## Common Commands

Ingest data:

```bash
uv run ingest --help
```

Query data:

```bash
uv run query --help
```

Resolve entities — merge duplicate and semantically-similar named entities
(e.g. `USA`/`United States`) for a collection into durable canonical records.
Re-runnable and idempotent; results surface in the NER views under
`entity_merge_mode=resolved`. Tuned by `RES_EMBED_THRESHOLD` (0.86),
`RES_LLM_TIEBREAK` (true), `RES_CASE_NORMALIZE` (true), `RES_VECTOR_K` (5):

Runs in a one-off `backend` container (production is Docker-only), so it
reaches the `qdrant` / `vllm-router` network aliases — bring up data-plane and
vllm-service first.

```bash
make resolve                    # prompts for the collection name
make resolve COLLECTION=mydocs  # non-interactive
# or over HTTP, on the selected collection:
# curl -X POST http://localhost:8000/collections/entities/resolve
```

Verify that a collection's Qdrant vector store and SQLite KV docstore
are in sync — reports drift (KV-only orphans, Qdrant-only orphans,
broken hierarchical parents) and optionally repairs KV-only orphans:

```bash
uv run verify --collection my_collection
uv run verify --collection my_collection --repair
```

Run tests and checks:

```bash
uv run pytest
uv run pre-commit run --all-files
```

Stop the Docker stack:

```bash
make stop
```

## Numbered Chat Citations

Answers refer to their evidence by number ("source 3"), and the chat window's
source cards carry the matching number. The number is assigned server-side
before generation: the last node postprocessor stamps `citation_index` onto
the snippet set the synthesizer renders, so the model reads its number instead
of counting, and the same value rides the source payload out to the SPA.

The list can have gaps: the SPA drops broken-preview duplicates from the card
list, and the surviving cards keep their original numbers rather than closing
the gap, because renumbering would break the link to the answer.

Conversations replayed from the session DB take their numbers from citation
row order, which is the order the generator saw. Answers written before this
feature still contain hand-counted ordinals that may not line up.

## Corrective Retry

When response validation judges an answer both ungrounded and weak — empty,
very short, or a refusal like "Evidence insufficient." — the backend rewrites
the retrieval query from the validator's own reason and answers once more,
then validates that second answer against the original question. One attempt
only.

It happens in the open. The chat stream sends a retry frame *before* the
replacement arrives, so the rejected answer is dropped on screen rather than
swapped out at the end, and the turn keeps an amber notice naming the query
that was used — in the live stream and after a reload. Nothing is appended:
the second attempt overwrites the first, so one question stays one turn.

Turns answering from a hand-picked scope are left alone (they run no retrieval
for a new query to change), and a substantive answer is never discarded just
because the validator flagged it. A triggered retry costs up to three extra
model round-trips, so the turns that fire it are noticeably slower. Turn it off
with `CORRECTIVE_RETRY_ENABLED=false`; see
[Retrieval and agents](docs/retrieval-and-agents.md#corrective-retry).

## Images As Sources

Images are ordinary sources. A stored image — a standalone file, a figure
embedded in a PDF, a video keyframe — is retrieved by CLIP, ranked against the
text chunks by the same reranker on the same scale, shown to the model as part
of the evidence, numbered like any other citation, and quotable in the
collection summary alongside its document's text.

What the model sees of an image is what was stored for it at ingest time: its
caption and tags, and — where a document OCR model is configured — the text
printed *inside* it. A caption says what a picture shows; OCR says what it
says, which is what someone searching for a screenshot's wording actually
typed, so it is stored ahead of the caption and indexed for keyword search
too. No pixels are sent at query time, and no vision call happens on the chat
path.

Settings that shape the lane:

- `IMAGE_RETRIEVE_TOP_K` (default `5`) — how many CLIP candidates enter the
  ranking. They then compete with text chunks for the answer's source slots;
  a query with no relevant imagery spends none of them.
- `IMAGE_OCR_ENABLED` (on when `OCR_MODEL` is set) — read the text inside
  images. `KEYFRAME_OCR_ENABLED` (default off) extends that to video
  keyframes, where usually only slides carry text.
- `IMAGE_RERANK_MIN_SCORE` (default `0.05`) — the reranker score an image
  caption must reach. The floor sits on the reranker, never on raw CLIP
  similarity, which is not comparable across queries: an unrelated query and a
  matching one both land in the same narrow CLIP band. Raise it if unrelated
  images still appear; lower it if relevant ones are missing.

If the rerank endpoint is down, images surface ungated rather than vanishing —
a degraded ranking is more useful than a silently emptied lane.

## Collection Summary

`POST /summarize` answers a collection-level summary from a map-reduce
("tree") summarizer, not by sampling a handful of documents. Every point in
the collection is partitioned into map units — one per document, or one per
coarse author/hour bucket for row-level social content — each unit is
summarized independently (windowed across multiple LLM calls when it is
large, then folded into one unit summary), and the per-unit summaries are
folded hierarchically down to a single synthesis call. Evidence chunks the
map stage points to become citation sources on the answer, the same as any
other retrieval.

Per-unit summaries are cached, keyed by a fingerprint of the unit's own
content, so an incremental re-ingest only re-summarizes units that actually
changed — unchanged documents are served from cache at no LLM cost. Changing
the summary prompts or the `SUMMARY_MAP_WINDOW_TOKENS` /
`SUMMARY_REDUCE_FANIN` knobs (see
[configuration.md](docs/configuration.md#summarisation--summaryconfig))
changes that fingerprint too, so cached summaries are invalidated once, the
next time a summary is built.

Reading and building are separate routes. `GET /summarize` only ever reads: it
answers `200` with the stored summary or `204` when there is none, and queues
nothing. `POST /summarize` is job-backed, mirroring ingestion: a cache hit
answers `200` immediately; a cache miss, or an explicit `refresh=true`, queues a
background build and answers `202 {job_id}`, with progress on the same
owner-multiplexed `GET /ingest/jobs/events` stream ingestion uses
(`summary_started` / `summary_progress` / `summary_completed`). A second call
while a build is already running answers `409` with that build's `job_id`.

The split exists because the SPA fires the read whenever the Summary tab opens.
A build is minutes of map-reduce over the whole collection, so the tab shows
what is already stored and offers a **Create** button when there is nothing —
it never starts a build merely because someone looked. `SUMMARY_ON_INGEST`
(default `true`) triggers a rebuild as the last stage of every ingest job, so a
normally-ingested collection already has one to show.

That post-ingest rebuild is fail-soft: if it raises (an LLM outage, for
example), the ingest job does not fail — it surfaces a `warning` event
("Collection summary generation failed.") on the same job and finishes
normally, with its documents fully ingested and retrievable. The collection
is simply left without a cached summary, so the next Summary view falls into
the `202` build-in-background path described above.

## Server-Side Exports For Large Collections

The React UI streams collection-wide CSVs from the backend so the browser
never accumulates the whole result set in memory. Two paths exist for jobs
that would otherwise tax the SPA:

```bash
# Server-streamed CSV from anywhere with HTTP access to the backend.
# Export endpoints are owner-gated by the collection name in the path — there is
# no "select" step. Behind a trusted proxy, pass the user header
# (e.g. -H 'X-Auth-User: alice'); with DOCINT_DEFAULT_IDENTITY set the header is
# optional. The example assumes the API is reachable on port 8000.
curl -O "http://localhost:8000/collections/my_collection/export/entities.csv"
curl -O "http://localhost:8000/collections/my_collection/export/hate-speech.csv"
curl -O "http://localhost:8000/collections/my_collection/export/documents.csv"

# The entities export honours the same merge modes as the Analysis view —
# pass entity_merge_mode=resolved to stream the durable canonical entities
# (run `make resolve` first; falls back to orthographic if not resolved).
curl -O "http://localhost:8000/collections/my_collection/export/entities.csv?entity_merge_mode=resolved"
```

For batch jobs that take many minutes (or shouldn't hold an HTTP
connection open), the `query` CLI runs inside the backend container and
writes the same CSV files to a mounted volume:

```bash
docker compose --env-file .env -f docker/compose.yaml \
  exec backend query --collection my_collection --all \
  --output /var/lib/docint/sources/my_collection/exports
docker compose cp \
  backend:/var/lib/docint/sources/my_collection/exports ./exports
```

Both paths share the schemas defined in `docint/utils/csv_stream.py`, so
the streaming endpoint and the CLI produce byte-identical CSVs for the
same collection.

## Curated Reports (Report Builder)

The collection-wide exports above are all-or-nothing. For an investigative case
you usually want a *curated* document — only the chat answers, entity findings,
and hate-speech findings that matter, with the duplicate chunks a single entity
drags in collapsed. The **Report Builder** (the **Report** tab in the SPA) is
that workflow:

- An **"+ Report"** control sits on every chat answer, entity finding, and
  hate-speech finding. Clicking it snapshots that one artifact into the active
  report (auto-creating an "Untitled report" the first time). Re-adding the same
  chunk is a no-op — findings are deduped by `chunk_id`.
- The **Report** view lists your reports and, for the active one, shows the
  picked artifacts grouped by type with per-item notes, reordering, and removal.
- Reports are **owner-scoped** and persisted server-side in the same SQLite
  store as chat sessions; each item is **snapshotted at add-time**, so later
  re-ingestion of the collection never changes a finished report.
- **Visual evidence travels with the reference.** When an added artifact points
  at an image — a chat answer citing a figure or photo, a finding on an image
  document, a video keyframe — the server freezes a small **thumbnail** into
  the snapshot (as a self-contained data URI, no live collection needed). The
  Report tab shows it under the item, and the Markdown/HTML/PDF exports render
  it inline beside the text ("Image evidence" / "Video keyframe"). Thumbnails
  are generated at ingestion; collections ingested before this shipped gain
  them on re-ingest of the same files (video keyframes need a fresh transcript
  run — see `docint/core/ingest/media_transcribe.py`).
  A chat answer's images render as a strip of **captioned figures** beneath
  its source list, each captioned with the number the answer cites (`[2]`), so
  a reader can tell which figure the text means; a finding shows its one figure
  inside the finding table.
- **A report belongs to one collection.** Switching the active collection
  releases the active report, so the next "+ Report" click starts one for the
  collection you are actually working in — a report's document overview and its
  evidence always describe the same collection.
- **Findings from pictures show the picture in the Analysis tab too.** An
  entity or hate-speech finding whose chunk was read out of an image (a
  screenshot, a photographed page, a video keyframe) renders the source image
  beside it; clicking it opens the full-size preview. That view is live rather
  than frozen — nothing is being exported there.

Export a finished report in five formats (also available directly over HTTP):

```bash
curl -O "http://localhost:8000/reports/1/export.pdf"   # paginated case-file PDF (WeasyPrint)
curl -O "http://localhost:8000/reports/1/export.md"    # combined Markdown
curl    "http://localhost:8000/reports/1/export.html"  # self-contained HTML (also the PDF source)
curl -O "http://localhost:8000/reports/1/export.json"  # structured selection
curl -O "http://localhost:8000/reports/1/export.zip"   # per-type CSV bundle (reuses csv_stream.py schemas)
```

Every export leads with the **summaries**, then chat answers, entity findings,
and hate-speech findings; entity and hate-speech findings carry their source
**reference metadata** (network, author, timestamp, …) alongside the chunk. The
report name is the single headline and the subheader stays on one line
(collection · creation date · operator).

The PDF is rendered server-side by WeasyPrint into a real paginated document: a
running header carrying the case file (*Aktenzeichen*) in the upper-right
corner, page numbers and an "AI-generated — verify before further processing"
disclaimer in the footer of every page, findings kept whole across page breaks,
and Noto fonts for multi-script text. It needs WeasyPrint's native libraries,
which the backend image installs; if they are absent the `.pdf` route returns
503 while every other format keeps working.

## Social Multimodal Media

Docint can ingest social-media exports that pair text **postings** with linked
**media files** (images, video, audio). The ingestion pipeline reads a
`media.csv` manifest, joins each media file to its parent posting (by `Network
ID`, else `Media ID`, matched against the postings' `Posting ID`), and routes
each artifact to the right backend — images go through
CLIP, video/audio are transcribed by Nextext and keyframe-extracted.

**One flat directory.** Put `postings.csv`, `media.csv`, and every referenced
media file in a **single directory**, and ingest that directory. Media are
resolved by filename *within that one directory* — no subfolders, and no
relative or absolute paths in the manifest (only the basename is used) — so a
file is linked only when it sits directly beside the manifest. Upload it with
the SPA's folder picker, or point `DATA_PATH` at that directory.

**Linker.** During ingestion, `posting_uuid` is written into every artifact
node (image embedding, keyframe, Nextext transcript segment). At retrieval
time, `_attach_posting_group` reads that UUID from `reference_metadata.uuid`
or the top-level `posting_uuid` field and tags each source dict with a
`posting_group` key so the UI can render a post alongside all its media as a
single entity.

**Posting reference metadata.** Every linked artifact also carries the parent
posting's reference fields — `posting_network`, `posting_author`,
`posting_author_id`, `posting_vanity`, `posting_timestamp`, `posting_url`, and
the full `posting_text` — merged *additively* into its `reference_metadata`
(a transcript segment keeps its `network: nextext` / `type: transcript_segment`
identity). These fields are stored in the Qdrant payload at ingest time and
surface everywhere reference metadata renders: chat citations, entity and
hate-speech findings, report exports (MD/HTML/PDF), and the findings CSVs.
Collections ingested before this feature must be **re-ingested** to pick the
fields up — there is no payload migration (cached Nextext transcripts make
this cheap: only embedding is redone, not transcription).

**Nextext transcription.** Video/audio transcription is delegated to an
external Nextext service. Set:

```bash
NEXTEXT_API_BASE=https://<nextext-host>/api/v1   # required to enable social-media ingestion
NEXTEXT_API_KEY=<token>                          # if the endpoint requires auth
NEXTEXT_AUTH_HEADER=X-Auth-User                  # trusted identity header name (must match Nextext's)
NEXTEXT_IDENTITY=docint                          # identity docint sends; empty = send no header
NEXTEXT_TIMEOUT=120                              # per-request HTTP timeout (seconds)
NEXTEXT_POLL_INTERVAL=5                          # polling interval while waiting for job (seconds)
NEXTEXT_POLL_MAX_SECONDS=600                     # hard deadline per transcription job (seconds)
```

The `/api/v1` suffix is required: docint's client calls `{NEXTEXT_API_BASE}/jobs`,
and Nextext mounts its jobs router under `/api/v1` — a base URL without that
suffix will 404 on every request.

Nextext resolves each request's identity from a trusted header (its
`NEXTEXT_AUTH_HEADER`, default `X-Auth-User`) and rejects header-less callers
with 401 unless its own default-identity fallback is configured. docint sends
`NEXTEXT_IDENTITY` (default `docint`) under `NEXTEXT_AUTH_HEADER` (default
`X-Auth-User`) on every request; set `NEXTEXT_IDENTITY` to empty to suppress
the header, e.g. when a gateway in between injects it instead.

When `NEXTEXT_API_BASE` is unset, the Nextext client is disabled and
video/audio files are skipped gracefully — collections with no audio/video are unaffected (loose audio/video in any batch is transcribed when `NEXTEXT_API_BASE` is set — see **Standalone media (audio/video)** below).

**Keyframe sampling.** Keyframes are extracted at a configurable rate, pruned
by cosine similarity before captioning, and ingested alongside the transcript:

```bash
KEYFRAMES_PER_MINUTE=4      # target frame sampling rate (default 4)
KEYFRAMES_MAX=20            # hard ceiling on candidate frames (default 20)
KEYFRAME_DEDUP_COSINE=0.95  # drop frames whose CLIP embedding cosine similarity
                            # to an already-accepted frame exceeds this threshold
```

Transcripts are cached in the per-collection `IngestManifest` by media-file
hash so re-ingestion of unchanged files skips the Nextext round-trip entirely.

## Standalone Media (Audio/Video)

Audio and video do not need a social export at all. Drop loose media files
anywhere in an ingest batch — the SPA's folder upload or `DATA_PATH` — and,
with `NEXTEXT_API_BASE` set, docint transcribes each one automatically and,
for video, extracts keyframes, exactly like the social path above but with
no `postings.csv` / `media.csv` manifest required. This runs as a pipeline
pre-pass right after the social linker, so manifest-linked media is claimed
first and this pass only picks up whatever is left over — a social export
still ingests exactly as documented above, and a batch mixing one with
extra, unreferenced media handles both in the same run. When
`NEXTEXT_API_BASE` is unset, loose audio/video files are skipped with a
one-line warning and the rest of the batch still ingests normally.

The key difference from the social path: there is no posting to stamp, so
every artifact — transcript segments and keyframes — anchors to the media
file's own content hash and filename instead of a `posting_uuid`. Each
retrieves and cites as an independent, normally-ranked source naming the
source clip; unlike social media, there is no `posting_group` cross-modal
clustering.

It reuses the Nextext client and keyframe sampling described above —
`NEXTEXT_API_BASE`, `KEYFRAMES_PER_MINUTE`, `KEYFRAMES_MAX`,
`KEYFRAME_DEDUP_COSINE`, and `NEXTEXT_MAX_CONCURRENCY` (default `4`, caps how
many clips run through Nextext in parallel per batch) all apply unchanged.
One new knob controls which extensions count as media for this pass:

```bash
MEDIA_FILETYPES=.mp4,.mov,.mkv,.webm,.avi,.m4v,.mpg,.mpeg,.mp3,.m4a,.wav,.flac,.aac,.ogg,.opus,.wma
# ^ comma-separated override; defaults to the 16 extensions above
# (DEFAULT_MEDIA_FILETYPES in env_cfg.py). Only the standalone discovery
# pass consults this list — the social linker routes any manifest-resolved,
# non-image file to Nextext regardless of extension.
```

## Localization

The single env var `RESPONSE_LANGUAGE` (values `en` | `de`, default `en`) controls
the *entire* app — both backend and SPA chrome — with one knob:

- **Backend**: prompts, `ui_strings` in reports, and export captions (PDF headers,
  CSV column names). Unknown values silently fall back to `en`.
- **SPA**: The React interface — buttons, labels, navigation, form hints, and
  error messages — flows from a typed locale catalog (`frontend/src/i18n/`)
  with `en` and `de` as canonical languages (316 keys across all screens,
  maintained in parity with each new feature).

### On-Demand Translation of Source Content

Chat source citations, entity findings, and hate-speech findings each show a
hover/focus-revealed **Translate** control. Clicking it fetches an on-demand
machine translation into the operator's active locale (`RESPONSE_LANGUAGE`)
and swaps it in for the original in place — a "Translation" label marks the
swapped view, and a second click ("Show original") brings the original back;
the original is always one click away, never discarded. Long chunks stay
clamped to four lines behind a "Show more" toggle in either view. This is a
display-time overlay only: nothing ingested or stored is ever translated.

Translating a finding before adding it to a report carries that translation
into the report's snapshot, so exports (Markdown, HTML, PDF, CSV, JSON) show
it as an additive labeled block or column next to the original — e.g.
"Machine translation (→ Deutsch)" when the active locale is German.

Translation reuses the same chat model as the rest of docint over the same
router endpoint — there is no dedicated translation runtime and no
`TRANSLATE_API_BASE` to configure. Set `TRANSLATE_MODEL` in `.env` to use a
different model than chat's `TEXT_MODEL`; it defaults to `TEXT_MODEL`.
Airgap-safe: no new container and no new network egress target. A
target-language override (translating into a language other than the active
locale) is not yet supported.

## Standalone vLLM App

The standalone deployment lives in
[vllm-service](https://github.com/nos-tromo/vllm-service/).

For a shared-network deployment on one server:

1. Create the shared external network once:

   ```bash
   docker network create inference-net
   ```

2. Start `vllm-service` on that network.
3. Set `INFERENCE_NET=inference-net` in both projects if you use a different name.
4. Configure Docint with:

   ```bash
   INFERENCE_PROVIDER=vllm
   OPENAI_API_BASE=http://vllm-router:4000/v1
   OPENAI_API_KEY=<token>
   ```

Run that stack separately and configure Docint with:

- `INFERENCE_PROVIDER=vllm`
- `OPENAI_API_BASE=https://<router-host>/v1`
- `OPENAI_API_KEY=<token>`
- `TEXT_MODEL`, `VISION_MODEL`, `EMBED_MODEL`, `SPARSE_MODEL`, and
  `RERANK_MODEL` matching the served model IDs

## Documentation

The [`docs/`](docs/README.md) directory contains the in-repo reference
manual. It complements this README with topic-by-topic deep dives:

- [Getting started](docs/getting-started.md) — install, first ingest,
  first query
- [Architecture](docs/architecture.md) — runtime components and request
  flow
- [Configuration](docs/configuration.md) — every env var grouped by
  dataclass, with defaults
- [API reference](docs/api-reference.md) — every FastAPI route
- [CLI reference](docs/cli-reference.md) — `docint`, `ingest`, `resolve`,
  `query`, `query-eval`, `verify`, `load-models`
- [Ingestion pipeline](docs/ingestion.md) — readers, chunking, NER,
  storage
- [Retrieval and agents](docs/retrieval-and-agents.md) — orchestrator,
  hybrid retrieval, graph-RAG, validation, corrective retry
- [UI guide](docs/ui-guide.md) — React SPA pages and components
- [Deployment](docs/deployment.md) — Docker deployment, volumes,
  co-deployment with vLLM / Ollama, offline image bundles
- [Development](docs/development.md) — dev workflow, pre-commit,
  pytest layout, CI, extension points

## Repository Shape

- `docint/core`: backend, ingestion, retrieval, storage, session state
- `docint/agents`: orchestration and tool-using agent flow
- `frontend/`: React SPA (Vite + TypeScript)
- `docs`: in-repo documentation
- `tests`: unit tests
