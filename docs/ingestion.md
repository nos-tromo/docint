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
- **Structured** — `.jsonl`
- **Images** — `.png`, `.jpg`, `.jpeg`, `.gif`
- **Audio / video** — `.mp3`, `.m4a`, `.ogg`, `.wav`, `.mp4`, `.m4v`,
  `.avi`, `.flv`, `.mkv`, `.mov`, `.mpeg`, `.mpg`, `.webm`, `.wmv`

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

### Audio — `audio.py`

`audio.py` wraps OpenAI-Whisper for transcription or translation. The
`WhisperConfig` dataclass (`env_cfg.py:1215`) drives it via
`WHISPER_MAX_WORKERS`, `WHISPER_TASK` (`transcribe` / `translate`) and
`WHISPER_SRC_LANGUAGE`. Audio files are transcribed, split into
sentences, and handed to the chunker as plain-text documents.

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

`json.py` flattens `.jsonl` files into documents. Each line becomes one
document, with selected fields lifted into metadata.

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
