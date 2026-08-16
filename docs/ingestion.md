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
`docint/utils/env_cfg.py:736`. Summary by category:

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
groups with that posting at citation time (see "Social Multimodal Media" in
`README.md`), while any other loose audio/video elsewhere in the batch still
goes through the standalone path above. Both require `NEXTEXT_API_BASE`;
when it is unset, audio/video files are skipped with a one-line warning and
the rest of the batch still ingests normally. A pre-made Nextext `.jsonl`
transcript still ingests directly as a structured file if you prefer to
transcribe out of band.

Every other extension is dispatched to the reader that knows how to parse it
(see the next section).

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
- CLIP (`IMAGE_EMBED_MODEL`) produces the dense vector.
- When `IMAGE_OCR_ENABLED` is on, the OCR engine reads the text printed
  inside the image and stores it as `ocr_text` — ahead of the caption in
  the node text and in the search index, since it is what a reader typed.
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
| `source_file` | Name of the original transcript file |
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
7. Embeddings and nodes are upserted to Qdrant and to the Qdrant-backed
   KV docstore (`docint/core/storage/docstore.py`) in batches of
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

- `INGEST_BENCHMARK_ENABLED=true` enables per-batch throughput logs from
  `DocumentIngestionPipeline`.
- `LOG_PATH` controls the rotating log sink (loguru, 5 MB rotation, 3
  file retention — see `docint/utils/logger_cfg.py`).
- Progress callbacks are the mechanism behind the SSE events from
  `POST /ingest/upload`. Library callers can pass their own callback to
  `RAG.ingest_docs()` — the CLI uses `logger.info`.

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
