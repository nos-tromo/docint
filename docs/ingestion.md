# Ingestion pipeline

Ingestion is the write path of Docint: source files on disk become
embedded, chunked, metadata-rich nodes inside Qdrant. This doc walks
through every stage, from file triage to Qdrant persistence.

## Entry points

| Caller | Target | Notes |
|---|---|---|
| `uv run ingest` | `docint/cli/ingest.py` | Prompts for a collection name, reads files from `DATA_PATH`. |
| `POST /ingest` | `docint/core/api.py:1126` | Triggers ingestion over the configured `DATA_PATH`. |
| `POST /ingest/upload` | `docint/core/api.py:1259` | Streaming single-file upload with progress events. |
| UI Ingest page | `docint/ui/ingest.py` | Browser-driven wrapper over `/ingest/upload`. |

All four paths end up calling `RAG.ingest_docs()` in
`docint/core/rag.py`, which owns the whole pipeline.

## Supported file types

The default list lives in `load_ingestion_env()` in
`docint/utils/env_cfg.py:379`. Summary by category:

- **Documents** — `.pdf`, `.docx`, `.md`, `.txt`
- **Tables** — `.csv`, `.tsv`, `.xls`, `.xlsx`, `.parquet`
- **Structured** — `.json`, `.jsonl`, `.ndjson` (generic payloads and
  Nextext transcripts)
- **Images** — `.png`, `.jpg`, `.jpeg`, `.gif`

Only the file types listed above are ingested; all others are silently
skipped. For audio and video, transcribe them in
[Nextext](https://github.com/nos-tromo/nextext) first and upload the
resulting `.jsonl` transcript.

Each extension is dispatched to the reader that knows how to parse it
(see the next section).

## Readers

All readers live under `docint/core/readers/`.

### PDFs — `documents/`

The PDF pipeline is page-level and has its own sub-modules:

| File | Responsibility |
|---|---|
| `documents/triage.py` | Classifies pages as text / scanned / mixed based on `PIPELINE_TEXT_COVERAGE_THRESHOLD`. |
| `documents/layout.py` | Runs Docling's layout analyser on text pages. |
| `documents/ocr.py` | Falls back to OCR for scanned pages. When `PIPELINE_ENABLE_VISION_OCR=true`, the vision LLM is used as a second-stage OCR; otherwise it relies on RapidOCR. |
| `documents/extraction.py` | Extracts text blocks, tables, and images into the intermediate pipeline model. |
| `documents/chunking.py` | Splits the extracted text into coarse parent chunks and fine child chunks. |
| `documents/artifacts.py` | Persists intermediate artifacts under `PIPELINE_ARTIFACTS_DIR` so reruns are incremental. |
| `documents/orchestrator.py` | Glues the stages above into a single per-document run. |
| `documents/reader.py` | The LlamaIndex-compatible reader class (`CorePDFPipelineReader`) used by the ingestion pipeline. |
| `documents/config.py` | Thin re-export of `load_pipeline_config()` from `env_cfg`. |
| `documents/models.py` | Pydantic models shared by the pipeline stages. |

Tuning lives in [`PipelineConfig`](configuration.md#pipeline--pipelineconfig).
Key knobs: `PIPELINE_TEXT_COVERAGE_THRESHOLD`,
`PIPELINE_ENABLE_VISION_OCR`, `PIPELINE_MAX_WORKERS`,
`PIPELINE_FORCE_REPROCESS`, `PIPELINE_VISION_OCR_*`.

### Images — `images.py`

`images.py` and `docint/core/ingest/images_service.py` own the image
ingestion path:

- Images are hashed and, if `IMAGE_CACHE_BY_HASH=true`, embeddings are
  looked up before recomputation.
- CLIP (`IMAGE_EMBED_MODEL`) produces the dense vector.
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
| `language` | Detected or specified language code |
| `source_file` | Name of the original transcript file |
| `type` | Always `"transcript_segment"` |
| `network` | Always `"nextext"` |

## Ingestion orchestration

The top-level orchestrator is
`docint/core/ingest/ingestion_pipeline.py` (`DocumentIngestionPipeline`).
It is built by `RAG._build_ingestion_pipeline()` and takes:

- the active Qdrant collection,
- the data directory,
- an NER extractor (`docint/utils/ner_extractor.py`) when
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

Tune the pass via `EMBED_CTX_TOKENS`, `EMBED_CHAR_TOKEN_RATIO`, and
`EMBED_CTX_SAFETY_MARGIN` — see
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
`docint/utils/ner_extractor.py`, which wraps GLiNER
(`NER_MODEL=gliner-community/gliner_large-v2.5`). Each chunk shorter
than `NER_MAX_CHARS` is processed with up to `NER_MAX_WORKERS` threads.
Detected entities, relations, and aggregate statistics end up in node
metadata and, post-ingestion, in the NER cache that powers the
`/collections/ner*` endpoints.

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
