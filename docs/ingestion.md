# Ingestion pipeline

Ingestion is the write path of Docint: source files on disk become
embedded, chunked, metadata-rich nodes inside Qdrant. This doc walks
through every stage, from file triage to Qdrant persistence.

## Entry points

| Caller | Target | Notes |
|---|---|---|
| `uv run ingest` | `docint/cli/ingest.py` | Prompts for a collection name, reads files from `DATA_PATH`. |
| `POST /ingest` | `docint/core/api.py` (`ingest`) | Ingests the configured `DATA_PATH` directly. CLI/batch path. |
| `POST /ingest/upload` | `docint/core/api.py` (`ingest_upload`) | Stages files into the collection's batch directory. Upload only — no ingestion. |
| `POST /ingest/finalize` | `docint/core/api.py` (`ingest_finalize`) | Queues one server-owned job over the staged batches. The SPA's path. |
| SPA Ingest page | `frontend/src/routes/Ingest.tsx` | Uploads in batches, then finalizes once; consumes `GET /ingest/jobs/events`. |

All of them end up calling `RAG.ingest_docs()` in `docint/core/rag.py`,
which owns the whole pipeline.

### Server-owned jobs

`/ingest/finalize` does not ingest on the request that called it. It
registers a job in `docint/core/jobs.py` and returns `202 {job_id}`; a
worker thread runs the pipeline while clients consume progress from the
owner-multiplexed SSE stream at `GET /ingest/jobs/events`.

This exists because ingestion used to stream progress on the request that
started it, so any client disconnect — navigation, reload, a closed tab —
severed the only view of a run that kept going regardless. Jobs are held in
memory: they survive a browser reload (the client re-discovers them by
owner) but not a backend restart. The staged files remain on disk either
way, and hash dedup makes a re-run cheap.

Concurrency is bounded by `DOCINT_INGEST_CONCURRENCY` (default `1`, so runs
serialise). A second job for a collection that is already ingesting is
refused with `409` carrying the in-flight `job_id` — overlapping runs can
double-write, because file hashes are only recorded after a run's final node
batch. Entity resolution runs as a stage inside the job, so it no longer
depends on a client staying attached.

## Supported file types

The default list lives in `load_ingestion_env()` in
`docint/utils/env_cfg.py:919`. Summary by category:

- **Documents** — `.pdf`, `.docx`, `.md`, `.txt`
- **Tables** — `.csv`, `.tsv`, `.xls`, `.xlsx`, `.parquet`
- **Structured** — `.json`, `.jsonl`, `.ndjson` (generic payloads and
  Nextext transcripts)
- **Images** — `.png`, `.jpg`, `.jpeg`, `.gif`
- **Media** — `.mp4`, `.mov`, `.mkv`, `.webm`, `.avi`, `.m4v`, `.mpg`,
  `.mpeg`, `.mp3`, `.m4a`, `.wav`, `.flac`, `.aac`, `.ogg`, `.opus`, `.wma`
  (`DEFAULT_MEDIA_FILETYPES`, overridable via `MEDIA_FILETYPES`) — not part
  of `supported_filetypes` and never handed to the generic reader dispatch
  below; a dedicated pre-pass discovers them by extension and routes them
  straight to Nextext (see below).

Only the file types listed above are ingested when uploaded standalone; all
other extensions are silently skipped.

Audio and video need no `postings.csv` / `media.csv` at all: drop loose
media files anywhere in the ingest batch (SPA folder upload or `DATA_PATH`)
and docint forwards each one to a remote
[Nextext](https://github.com/nos-tromo/nextext) service that transcribes it
and, for video, extracts keyframes — the transcript is ingested as text
(one segment per node) and the keyframes as CLIP image points. Every
artifact anchors to the media file's own content hash and filename; there
is no posting to link it to, so transcript segments and keyframes retrieve
and cite as independent, normally-ranked sources naming the source clip.

A social export's `postings.csv` / `media.csv` manifest changes *linking*,
not *whether* transcription happens: media resolved from the manifest is
**additionally** stamped with its parent posting's `posting_uuid` so it
groups with that posting at citation time (see
[Social media exports](#social-media-exports) below), while any other loose
audio/video elsewhere in the batch still goes through the standalone path
above. The tables and the media they reference may sit anywhere in the batch
tree — the default export shape (`./postings.csv`, `./media.csv`,
`./dir/photos/*`, `./dir/videos/*`) is ingested by dropping in the whole
directory. Both require `NEXTEXT_API_BASE`;
when it is unset, audio/video files are skipped with a one-line warning and
the rest of the batch still ingests normally. A pre-made Nextext `.jsonl`
transcript still ingests directly as a structured file if you prefer to
transcribe out of band.

Every other extension is dispatched to the reader that knows how to parse it
(see the next section).

## Social media exports

Docint can ingest social-media exports that pair text **postings** with linked
**media files** (images, video, audio). The ingestion pipeline reads a
`media.csv` manifest, joins each media file to its parent posting (by `Network
ID`, else `Media ID`, matched against the postings' `Posting ID`), and routes
each artifact to the right backend — images go through
CLIP, video/audio are transcribed by Nextext and keyframe-extracted.

**Drop in the whole export directory.** `postings.csv` and `media.csv` may sit
anywhere in the batch, and the media files anywhere beneath it — the default
export shape (`./postings.csv`, `./media.csv`, `./dir/photos/*`,
`./dir/videos/*`) works as-is. Upload the directory with the SPA's folder
picker, or point `DATA_PATH` at it.

Only the **basename** of `Exported media filename` is ever used, looked up
within the batch tree, so a manifest carrying an absolute path or a `../`
traversal cannot reach a file outside the batch. Because the manifest supplies
no directory of its own, the same basename occurring in two subfolders is
*ambiguous*: a copy sitting beside the manifest wins, and otherwise the row is
skipped rather than linked to a guess.

**Albums (multi-item posts).** Some exports carry no media→posting key at all:
`Media ID` and `Network ID` both hold the media's *own* network message id. A
Telegram album is then N consecutive messages recorded as N media rows but a
single posting, filed under the group's **last** message id — so all but one
row names no posting. Rows the manifest cannot join are attached to the first
posting in the same channel whose message number is at or above their own,
**and only when the two timestamps agree** within `SOCIAL_ALBUM_TOLERANCE_S`
(default 5 s). That corroboration is what keeps the inference honest: when the
owning posting is missing from the export, the next one along is hours away and
the row is left unlinked rather than attributed to the wrong post. Exports that
do carry a key are untouched — the inference runs only after the declared key
fails, and needs `Posting ID` to start with the row's own `Author ID`, which a
Meta-style `<postingId>_<accountId>` id does not. Set
`SOCIAL_ALBUM_LINK_ENABLED=false` to switch it off. The counts land in one
ingest log line:

```
Social linker: 352 media linked (94 by manifest key, 258 by album inference, 0 by timestamp, 0 by text match), 0 skipped
(0 with no matching posting, 0 with no local file, 0 with an ambiguous filename)
across 352 manifest rows.
```

**Exports whose postings table is a messages table.** A chat-style export
(X/Twitter and friends) carries its posts in the *messages* schema — `Chat ID` /
`Sender` / `Text` where a postings table has `Posting ID` / `Author` /
`Text Content`. Such a table is accepted in the postings role and renamed before
any rule runs, so all five apply unchanged. A real postings table wins when both
are present.

**How a media row finds its posting.** Five rules, tried in order; the first
that names a known posting wins, and each is consulted only once the ones above
it have failed:

1. **The manifest's declared key** — `Network ID`, else `Media ID`, else
   `Media ID` with a trailing `_<counter>` stripped, matched against
   `Posting ID`. The ordinary path; most exports never leave it.
2. **The posting's network-level id** — some exports mint an internal
   `Posting ID` (a crawler UUID) that the manifest never carries, and name the
   posting by the id its own network uses. That id is read from
   `Network Posting ID`, or from the long numeric id in the permalink when the
   column is empty, as it is for reel-style posts. An id that two postings both
   advertise, or one that is an `Author ID`, is refused rather than resolved to
   a guess.
3. **Album inference** — for exports carrying no key at all; see above.
4. **Timestamp** — the single posting by the same author stamped at the same
   instant. Two such postings, or none, leave the row unlinked. The second case
   is what a partial export looks like, and it must not be papered over with a
   neighbouring post. Switch it off with `SOCIAL_TIMESTAMP_LINK_ENABLED=false`.
5. **Text** — the last resort, for the shape no author-scoped rule reaches: a
   **shared post**, whose manifest names the *original* author while the
   export's row is the sharer's. A row whose text exactly matches that of a
   single posting on the same network attaches to it; equality is exact and
   case-sensitive. Ambiguity and absence both leave the row unlinked, and a
   posting with no text is never indexed — an empty text is shared by every
   media-only post. Switch it off with `SOCIAL_TEXT_LINK_ENABLED=false`.

The ingest log reports the split, so an operator can see at a glance how much of
a run rested on inference rather than on a declared key:

```
Social linker: 2019 media linked (1983 by manifest key, 13 by network id, 0 by album inference, 23 by timestamp, 0 by text match), 101 skipped …
```

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
this cheap: only embedding is redone, not transcription). See
[migrations.md](migrations.md#payload-fields-added-after-a-collection-was-ingested).

**Nextext transcription.** Video/audio transcription is delegated to an
external Nextext service. `NEXTEXT_API_BASE` is required to enable
social-media ingestion, and **must include Nextext's `/api/v1` prefix**:
docint's client calls `{NEXTEXT_API_BASE}/jobs`, and Nextext mounts its jobs
router under `/api/v1`, so a base URL without that suffix 404s on every
request.

Nextext resolves each request's identity from a trusted header (its
`NEXTEXT_AUTH_HEADER`, default `X-Auth-User`) and rejects header-less callers
with 401 unless its own default-identity fallback is configured. docint sends
`NEXTEXT_IDENTITY` (default `docint`) under `NEXTEXT_AUTH_HEADER` (default
`X-Auth-User`) on every request; set `NEXTEXT_IDENTITY` to empty to suppress
the header, e.g. when a gateway in between injects it instead.

When `NEXTEXT_API_BASE` is unset, the Nextext client is disabled and
video/audio files are skipped gracefully — collections with no audio/video are
unaffected (loose audio/video in any batch is transcribed when
`NEXTEXT_API_BASE` is set — see the standalone path under
[Supported file types](#supported-file-types) above).

**Keyframe sampling.** Keyframes are extracted at a configurable rate
(`KEYFRAMES_PER_MINUTE`), capped (`KEYFRAMES_MAX`), pruned by cosine
similarity (`KEYFRAME_DEDUP_COSINE`) before captioning, and ingested alongside
the transcript. Transcripts are cached in the per-collection `IngestManifest`
by media-file hash so re-ingestion of unchanged files skips the Nextext
round-trip entirely.

Sampling is **requested explicitly**, and describing is **declined** — docint
sends `keyframes: true` and `visual_context: false` in each job's options.
Nextext makes both opt-in switches, pointing opposite ways: extraction defaults
to off, while captioning defaults to on. Declining the captions saves a vision
request per sampled frame for prose docint never downloads (it fetches only
`docint.jsonl` and `keyframes.zip`), and it keeps a frames-only job off
Nextext's chat provider entirely, so an unhealthy router cannot fail the job and
take the transcript with it. docint captions the frames itself, into the
structured description + tags its index needs.

**This requires Nextext ≥ v1.9.0.** Nextext's `JobOptions` forbids unknown
fields, so an older build rejects the options with a 422 and the clip is skipped
fail-soft — with no transcript either, not merely without frames. The client
recognises that specific status and names the required version in the warning.
The reverse skew is the quiet one: against a current Nextext, a client that does
not ask for keyframes gets none, plus a 404 on the artifact — which looks
exactly like an audio-only clip. `KEYFRAMES_PER_MINUTE=0` (or `KEYFRAMES_MAX=0`)
remains the way to turn sampling off; Nextext returns no frames for a
non-positive rate.

`keyframes.zip` also carries a `manifest.json` naming each frame's sampling
time (Nextext ≥ v1.11.0). The client pairs it with the frames and stamps
`keyframe_index` and `keyframe_time_sec` on each point, so an extract can say
*when* in the clip a described frame appeared. Without the manifest the frames
are stored untimed rather than guessed. The index is Nextext's own sampling
position, recorded before the near-duplicate prune, so a dropped frame never
renumbers its neighbours.

Every `NEXTEXT_*` and `KEYFRAME*` variable, with its default, is documented in
[configuration.md](configuration.md#nextext-media-processing--nextextconfig).

## Readers

All readers live under `docint/core/readers/`.

### PDFs — `documents/`

The PDF pipeline is page-level and has its own sub-modules:

| File | Responsibility |
|---|---|
| `documents/parse.py` | The docling-parse backbone: opens the PDF once per document and exposes each page's line cells (text, bbox, font name/size) and embedded-image placements; computes reading order with a recursive XY-cut so multi-column pages read column by column. No models, no network. |
| `documents/triage.py` | Classifies pages as text / scanned / mixed from the text-layer coverage against `PIPELINE_TEXT_COVERAGE_THRESHOLD`. |
| `documents/layout.py` | Builds layout blocks from the parsed geometry: `FIGURE` per embedded image, `TITLE`/`HEADER` for short lines set larger or bolder than the body text (they become the chunker's `section_path`), `PAGE_HEADER`/`FOOTER`/`PAGE_NUMBER` for page furniture, `TABLE` for a *"Table N:"* caption or an uncaptioned grid, and per-column/section `TEXT` blocks. |
| `documents/furniture.py` | Finds running heads, footers, page numbers and rotated margin stamps by band position and repetition across pages. Those blocks are kept out of chunk text and page text. |

| `documents/tables.py` | Rebuilds a table's cell grid from cell positions (baselines → rows, whitespace → columns), renders it row-major for the chunk text, and finds tables that carry no caption. |
| `documents/ocr.py` | Text extraction for pages that need OCR: the page's own text layer first (per-line spans), plus the translation of what the OCR engine read back into layout blocks. |
| `documents/extraction.py` | Collects tables (row-major text + cell grid, written as CSV) and images (the embedded image drawn at each `FIGURE` block, via pypdfium2) into the intermediate pipeline model. |
| `documents/chunking.py` | Splits the extracted text into coarse parent chunks and fine child chunks. |
| `documents/artifacts.py` | Persists intermediate artifacts under `PIPELINE_ARTIFACTS_DIR` so reruns are incremental. |
| `documents/orchestrator.py` | Glues the stages above into a single per-document run. |
| `documents/reader.py` | The LlamaIndex-compatible reader class (`CorePDFPipelineReader`) used by the ingestion pipeline. |
| `documents/config.py` | Thin re-export of `load_pipeline_config()` from `env_cfg`. |
| `documents/models.py` | Dataclasses shared by the pipeline stages. |

Two of those stages need pixels read rather than parsed — a page with no text
layer, and a table whose structure the cell positions could not express. Both
call the one OCR engine (`docint/core/ocr/`, see below), so they share its
endpoint and its per-document failure budget.

Tuning lives in [`PipelineConfig`](configuration.md#pipeline--pipelineconfig).
Key knobs: `PIPELINE_TEXT_COVERAGE_THRESHOLD`, `PIPELINE_OCR_ENABLED`,
`PIPELINE_MAX_WORKERS`, `PIPELINE_FORCE_REPROCESS`, `PIPELINE_OCR_*`,
`PIPELINE_TABLE_OCR`.

### Reading pixels — `core/ocr/`

Reading text out of an image is one task, so it has one implementation. A
scanned page, a table's region and an image file all go through
`DocumentOcrEngine`, which owns the client, the rendering, the reachable /
answered-with-an-error distinction and the budget that stops calling a dead
endpoint. What a given model expects and returns is a *family* behind one
interface:

| Family | Models | What comes back |
|---|---|---|
| `dots` | `dots-studio/dots.mocr`, `rednote-hilab/dots.ocr` | Layout JSON: one element per block with a bounding box, a category (title, section header, text, list item, caption, footnote, formula, table, picture, page header/footer) and its text; tables as HTML, expanded into a cell grid. Pages are rendered on the model's own 28-px grid so its internal resize is a no-op and the boxes map straight home. |
| `generic` | anything else, incl. plain recognition models and general vision models | Plain text, as one block spanning the image. Half-resolution retry on failure, higher-resolution retry on an empty answer. |

`OCR_MODEL` picks the family and the endpoint (see
[`OcrClientConfig`](configuration.md#document-ocr--ocrclientconfig)). Unset,
the general vision model reads pages exactly as it did before, and image OCR
stays off.

### Images — `images.py`

`images.py` and `docint/core/ingest/images_service.py` own the image
ingestion path:

- Images are hashed and, if `IMAGE_CACHE_BY_HASH=true`, embeddings are
  looked up before recomputation.
- **Video keyframes reuse a known frame's description.** The same flag covers
  keyframes: a survivor of the near-duplicate prune reaches the vision tagger
  and the OCR engine only when the `_images` companion holds no point for its
  content hash yet. The cosine prune spans a single clip, so without this a
  frame recurring across clips — or across a re-ingest — was described again
  each time. The point is still written on a cache hit, because it carries
  *this* posting's `posting_uuid`; only the model calls are skipped.
- CLIP produces the dense vector, via the remote CLIP service
  (`CLIP_API_BASE`); the model identity is set as `CLIP_MODEL` on the
  vllm-service container, not by docint.
- When `IMAGE_OCR_ENABLED` is on, the OCR engine reads the text printed
  inside the image and stores it as `ocr_text` — ahead of the caption in
  the node text and in the search index, since it is what a reader typed.
  A caption says what a picture *shows*; OCR says what it *says*, which is
  what someone searching for a screenshot's wording actually typed.
- When `IMAGE_TAGGING_ENABLED=true`, the vision LLM is called to produce
  tags / captions. Images exceeding `IMAGE_TAGGING_MAX_IMAGE_DIM` are
  down-scaled first.
- Embeddings and metadata land in a sibling collection named per
  `IMAGE_QDRANT_COLLECTION` (template `{collection}_images`).
- Failures are soft unless `IMAGE_FAIL_ON_EMBED_ERROR` /
  `IMAGE_FAIL_ON_TAG_ERROR` are set.

### Tables — `tables.py`

`tables.py` reads `.csv`, `.tsv`, `.xls`, `.xlsx`, and `.parquet` files
via Pandas. Each row becomes a document whose metadata carries the
configured id/text/metadata columns.

### JSON — `json.py`

`json.py` handles generic `.json` / `.jsonl` / `.ndjson` files. It also
detects the Nextext transcript schema (a JSONL stream whose segments have
`text` plus either `start_ts`/`end_ts` or `start_seconds`/`end_seconds`)
and emits one segment document per line with timing, speaker, and source
metadata preserved.

#### Ingestion granularity

Each Nextext JSONL line becomes exactly one retrievable Qdrant node. The
node's text field contains the segment's prose only (not raw JSON), and
timing/speaker metadata are exposed through the `reference_metadata` dict
so they surface automatically in citation UI.

#### Transcript reference metadata fields

The following fields from a Nextext segment are extracted into
`reference_metadata` and rendered in citations:

| Field | Description |
|---|---|
| `start_ts` | Segment start time in `hh:mm:ss` format (if available) |
| `end_ts` | Segment end time in `hh:mm:ss` format (if available) |
| `speaker` | Speaker name or identifier (if present in segment) |
| `language` | Transcript-text language code (the target for `translate`, the source for `transcribe`) |
| `detected_language` | Auto-detected source-audio language code (equals `language` for `transcribe`; the original source for `translate`) |
| `source_file` | Name of the media file the segment was transcribed from (social and standalone alike; `media_file_hash` carries that file's own hash, while `file_hash` stays the parsed transcript's) |
| `type` | Always `"transcript_segment"` |
| `network` | Always `"nextext"` |

## Ingestion orchestration

The top-level orchestrator is
`docint/core/ingest/ingestion_pipeline.py` (`DocumentIngestionPipeline`).
It is built by `RAG._build_ingestion_pipeline()` and takes:

- the active Qdrant collection,
- the data directory,
- an NER extractor (`docint/utils/ner_client.py`, a thin HTTP client to
  the remote GLiNER service hosted by `vllm-service`) when
  `NER_ENABLED=true`,
- a hate-speech detector when `ENABLE_HATE_SPEECH_DETECTION=true`,
- a progress callback (used by `/ingest/upload` to stream events).

The pipeline iterates files in `INGESTION_BATCH_SIZE` batches. For each
batch:

1. The file extension selects a reader (see above).
2. The reader produces one or more LlamaIndex `Document` objects with
   source metadata attached.
3. The chunker (see [Hierarchical chunking](#hierarchical-chunking))
   produces fine child nodes and optional coarse parent nodes.
4. NER runs in parallel on each fine chunk (when enabled) and annotates
   the chunk metadata with entities and relations.
5. Hate-speech detection runs per chunk (when enabled) and sets a
   `hate_speech_detected` flag in metadata.
6. Chunks are embedded with the dense model (`EMBED_MODEL`) and, for
   hybrid collections, the sparse model (`SPARSE_MODEL`).
7. Embeddings and nodes are upserted to Qdrant and to the SQLite-backed
   KV docstore (`docint/core/storage/sqlite_kvstore.py`) in batches of
   `DOCSTORE_BATCH_SIZE` with retry/backoff governed by
   `DOCSTORE_MAX_RETRIES`, `DOCSTORE_RETRY_BACKOFF_SECONDS`, and
   `DOCSTORE_RETRY_BACKOFF_MAX_SECONDS`.
8. A file-hash ledger is updated so identical files are not
   re-ingested on the next run.

## Hierarchical chunking

`docint/core/storage/hierarchical.py` implements `HierarchicalNodeParser`,
a two-level chunker:

- **Coarse parent chunks** — size `COARSE_CHUNK_SIZE` (default 8192
  tokens).
- **Fine child chunks** — size `FINE_CHUNK_SIZE` (default 8192 tokens)
  with `FINE_CHUNK_OVERLAP` (default 0) overlap. A sentence splitter with
  `SENTENCE_SPLITTER_CHUNK_SIZE` / `_OVERLAP` is used to break text at
  natural boundaries.

Parents and children are linked via `node_id` metadata. When
`PARENT_CONTEXT_RETRIEVAL_ENABLED=true`, retrieval can expand a fine hit
to include its parent context — see
[retrieval-and-agents.md](retrieval-and-agents.md#parent-context-expansion).

Set `HIERARCHICAL_CHUNKING_ENABLED=false` to fall back to flat
sentence-level chunking only.

### Pre-embed re-splitting

Even after hierarchical chunking, individual fine chunks can still
exceed the embedding model's context window — most commonly when an
operator raises `FINE_CHUNK_SIZE` above the embedding service's true
limit, or when a source document has a single very long paragraph.
`docint/utils/embed_chunking.py` runs a pre-embed re-chunking pass
(`resplit_nodes_for_embedding`) right before the embedding API is
called:

Token counting uses the embedding model's authoritative tokenizer when
available. The tokenizer snapshot is fetched from the HF cache by
`docint/utils/embedding_tokenizer.py::build_embedding_token_counter()`
during RAG initialization. When the snapshot is missing or the
tokenizer repo is empty (e.g. OpenAI provider), a WARNING is logged and
the code falls back to the `EMBED_CHAR_TOKEN_RATIO` heuristic.

- Within-budget chunks pass through unchanged.
- Oversize chunks are split into sub-nodes via llama_index's
  `SentenceSplitter` at `chunk_size = EMBED_CTX_TOKENS *
  EMBED_CTX_SAFETY_MARGIN`. Each sub-node gets a fresh UUID,
  `embedding_split=True`, `split_part_index`, `split_total_parts`,
  and `hier.parent_id=<original node id>` so that the existing
  parent-context postprocessor can reconstruct the full parent
  content from the docstore at query time.
- The original oversize parent is kept in the docstore (not in the
  vector store) for citation reconstruction; the vector store holds
  one embedding per sub-node.
- Irreducible single-token streams (e.g. a 60k-character word with
  no whitespace) raise `EmbeddingInputTooLongError` loudly so the
  operator can diagnose the source rather than store a lossy
  prefix-only vector.

Tune the pass via `EMBED_TOKENIZER_REPO`, `EMBED_CTX_TOKENS`,
`EMBED_CHAR_TOKEN_RATIO`, and `EMBED_CTX_SAFETY_MARGIN` — see
[configuration.md](configuration.md#embedding--embeddingconfig).

### Stale embeddings

Qdrant collections ingested before the pre-embed re-splitter landed
may carry prefix-only vectors for their oversize chunks. A dedicated
`docint reingest-stale` CLI to identify and re-ingest affected files
is TODO and will follow in a separate PR. For now, re-ingesting the
source files manually via the UI or the existing `ingest` CLI is the
supported workaround.

## NER and hate-speech

Entity extraction runs during ingestion through
`docint/utils/ner_client.py`, a thin HTTP client to the remote GLiNER
service hosted by `vllm-service`. The model and device live on the
service side; the docint side just configures the endpoint
(`NER_API_BASE`, default `http://vllm-router:4000`; bearer auth via
`NER_API_KEY` when the full vllm-service stack is in use). Each chunk
shorter than `NER_MAX_CHARS` is POSTed with up to `NER_MAX_WORKERS`
concurrent worker threads. Detected entities, relations, and aggregate
statistics end up in node metadata and, post-ingestion, in the NER
cache that powers the `/collections/ner*` endpoints.

Two operator-side deployment shapes for the upstream NER service:

- Full `vllm-service` stack (CUDA): docint reaches the router at
  `http://vllm-router:4000/gliner`; set
  `NER_API_KEY=$OPENAI_API_KEY` to satisfy the router's Bearer auth.
- `vllm-service` NER-only stack (Mac, CPU-only Linux running Ollama for
  chat/embed): docint reaches the GLiNER container directly at
  `http://gliner-ner:8000/gliner`; no Bearer auth needed.

Hate-speech detection is an optional parallel stage governed by
`HateSpeechConfig`. Flagged chunks carry a `hate_speech_detected` flag
that the `/collections/hate-speech` endpoint surfaces in the UI.

## Source staging

Before ingestion begins, `docint/core/storage/sources.py` copies raw
source files into `QDRANT_SRC_DIR / <collection>/` so that the UI
Inspector and `/sources/preview` endpoint can render previews after the
fact. This directory is separate from Qdrant's own storage and must be
mounted (or writable) for the whole ingest run.

## Observability

### Reading a run in the log

`docker logs -f docint-backend-1` narrates a whole run. Every line a job
produces carries the full `job_id`, so one run can be isolated even when
`DOCINT_INGEST_CONCURRENCY` lets two interleave:

```console
$ docker logs docint-backend-1 | grep 3d9f1c72e84b41a6b0d5c7f29ae61b30
```

Four shapes, in order (invented filenames):

```
Ingest job started | job_id=3d9f1c… collection='field-notes' files=3 bytes=18.0 MB by_type=pdf:1,docx:1,csv:1 hybrid=true ner=true hate_speech=false resolve=true
Ingest input 1/3 | job_id=3d9f1c… file='annual-report-2024.pdf' type=pdf bytes=8.0 MB
Job 3d9f1c… (ingest) progress: Core pipeline processing PDF (1/3): annual-report-2024.pdf
Job 3d9f1c… (ingest) progress: Extracting entities: 840/2000 chunks processed
Ingest job completed | job_id=3d9f1c… collection='field-notes' duration=14:22 duration_ms=862431 files_processed=3 files_skipped=1 files_failed=0 docs=3 nodes=1284 entities_minted=214 entities_attached=57 empty=false
```

`hybrid`, `ner` and `hate_speech` are per-request **overrides**, so each has
three values rather than two: `true`, `false`, and `default` — the last meaning
the request specified nothing and the configured default applies. `resolve` is
a plain flag and is only ever `true` or `false`.

A batch of more than 50 files lists the first 50 and then prints
`Ingest inputs truncated | … listed=50 omitted=452`; the header's
`files=` and `bytes=` always cover the whole batch.

The summary reports `duration` and `duration_ms` both — the second is the
exact integer the SPA's ingest card renders, so the two provably agree
rather than nearly agreeing.

### A run has exactly one duration

The server measures it once — from the moment the user started, upload leg
included, through the queue wait and every stage of the job — and both the
backend log's run summary (`Ingest job completed | … duration=00:19
duration_ms=19004 …`) and the ingest card's timer show that same value. The
card gets it from `duration_ms` on the terminal SSE frame, and a reattached
client from the job snapshot's `duration_ms` / `run_started_at`
(`started_at` still means "a worker slot was acquired"). The log prints
both forms for the same reason — a readable one and the exact integer the
card renders, so the two can be compared rather than trusted. Because the
upload happens before the job exists, the SPA reports how long it took as
`upload_elapsed_ms` on `POST /ingest/finalize` — an elapsed duration, never
a timestamp, so no client clock is trusted, and it is clamped server-side.
Deriving a second duration on the client is what previously let one run
report two numbers a second apart.

### Progress and the throttle

The pipeline reports progress per chunk, which is written for a client
that renders the latest message and discards the rest. `core/jobs.py`
tees those messages to the log through a throttle
(`docint/utils/logfmt.py`): a stage announces itself, then heartbeats
once per `LOG_PROGRESS_INTERVAL_S` (default 30) until its counter reaches
its total, and its last observed value is always logged even if the stage
stops short. Warnings pushed by the runner are never throttled.

Set `LOG_PROGRESS_INTERVAL_S=0` to log every progress message. On a large
ingest that is thousands of lines.

Note the two callers differ only in what they pass, not in what they see:
the CLI passes `logger.info` as `RAG.ingest_docs()`'s `progress_callback`
and the job runner passes an SSE publisher — but since the job layer tees,
both now produce a readable log. (Before that tee, the API path was nearly
silent while the SPA saw everything, which is the defect this section used
to describe from the wrong side.)

### Extra telemetry

`INGEST_BENCHMARK_ENABLED=true` adds a per-run throughput line
(`nodes_per_s`, `enrich_batches`, `persist_batches`, batch sizes) for
tuning. It is telemetry, not operator information — the run summary above
is emitted regardless.

Retention is the compose logging driver's job (`docker/compose.yaml`,
`local` driver, 50 MB × 5, compressed). There is no file sink.

## Adding a new reader

1. Create a reader under `docint/core/readers/` that returns LlamaIndex
   `Document` objects.
2. Register it in `DocumentIngestionPipeline._reader_for_extension()`
   (or the equivalent dispatcher) so the new extension is routed to
   your reader.
3. Add the extension to the `default_supported_filetypes` list in
   `load_ingestion_env()` so it passes triage.
4. Write a unit test under `tests/` patterned after
   `tests/test_documents_reader.py` or `tests/test_table_reader.py`.
5. Update [configuration.md](configuration.md) if you introduce new env
   vars, and [api-reference.md](api-reference.md) if you change the
   ingestion response shape.
