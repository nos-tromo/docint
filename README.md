# Document Intelligence

This is a proof of concept document intelligence platform offering the following features:

1. **Ingestion Pipeline**

   - Process and extract data from various file types, including documents, tables, images, audio, video, and JSON files.
   - Automatically organizes and prepares data for querying.
   - **Scalable Information Extraction**: Uses parallel processing and JSON-mode enforced LLM inference for high-speed entity and relation extraction.

2. **Query Pipeline**

   - Perform batch or single queries on ingested data.
   - Customize queries for batch processing.
   - Outputs results to a designated directory.

3. **CLI, API, and GUI Support**

   - Use the CLI for ingestion and querying tasks.
   - Use the FastAPI service for programmatic access (`/query`, `/collections/*`, `/agent/chat`, `/agent/chat/stream`).
   - Use the Streamlit UI for interactive workflows.

4. **Integration with LLMs and Models**

   - Utilize large language models (LLMs) and other AI models for advanced processing tasks.
   - **Flexible Backend**: Supports OpenAI-compatible APIs (OpenAI, Ollama, llama.cpp, vLLM, etc.) for text and vision tasks.
   - **Provider-Agnostic Architecture**: The system is designed to be backend-agnostic. It uses Ollama for local inference by default, and can switch to any OpenAI-compatible API (`llama.cpp`, vLLM, DeepSeek, etc.) by changing environment variables.
   - **Performance & Decoupling**: Inference runs in a dedicated container (`ollama-server` by default; `llamacpp-server` for llama.cpp profiles), ensuring the application logic remains responsive even during heavy model computations.

5. **Extensibility**

   - Easily extend functionality with additional readers, such as OCR for images, Whisper for audio/video, and table/json enhancements.

6. **Development and Testing**

   - Run the API and Streamlit UI independently or via Docker.
   - Comprehensive unit testing and pre-commit (ruff, mypy) ensure reliability.

## Supported Inference Backends

The platform is provider-agnostic and works with any OpenAI-compatible inference server. The following backends are tested and supported out of the box:

| Backend | Description | Link |
|---------|-------------|------|
| **Ollama** | Easy-to-use local model runner with built-in model management | [ollama/ollama](https://github.com/ollama/ollama) |
| **llama.cpp** | High-performance local inference for GGUF models (bundled in Docker profiles) | [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) |
| **OpenAI API** | Cloud-hosted models (GPT-4o, etc.) via the official API | [openai/openai-python](https://github.com/openai/openai-python) |
| **vLLM** *(planned)* | High-throughput serving engine with PagedAttention | [vllm-project/vllm](https://github.com/vllm-project/vllm) |

Any other server exposing an OpenAI-compatible `/v1/chat/completions` endpoint (e.g., [LocalAI](https://github.com/mudler/LocalAI), [LM Studio](https://github.com/lmstudio-ai)) can also be used by setting the `OPENAI_API_BASE` environment variable.

## Installation

The application can be used both via Docker for containerized environments and directly on the local machine for development and testing purposes.

### Prerequisites

**System Requirements:**

Ensure you have the necessary runtime installed:

- **Docker** (recommended)

- **Python 3.11+** and **uv** (for local development)

Model files are handled automatically by the ingestion pipeline and do not need to be manually pre-loaded.

### Docker Setup (Recommended)

1. **Ensure Docker is installed**

   Refer to the [Docker installation guide](https://docs.docker.com/get-docker/) if needed.

2. **Configure Environment**

   Create the Docker environment file from the example:

   ```bash
   cp .env.docker.example .env.docker
   ```

3. **Choose a Profile**

   > `llama.cpp` images are built for **amd64** architecture systems. Consider using Ollama for local inference when running the
   > app via an arm64 machine.

   Profiles follow the pattern `<hardware>-<backend>`:

   | Profile | Hardware | Inference Backend |
   |---------|----------|-------------------|
   | `cpu-ollama` | CPU | Ollama |
   | `cpu-llamacpp` | CPU | Llama.cpp |
   | `cpu-openai` | CPU | External OpenAI API |
   | `cuda-ollama` | NVIDIA GPU | Ollama |
   | `cuda-llamacpp` | NVIDIA GPU | Llama.cpp |
   | `cuda-openai` | NVIDIA GPU | External OpenAI API |

   CUDA profiles require a CUDA-compatible GPU and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

   Each profile automatically sets `MODEL_PROVIDER` and `OPENAI_API_BASE` to the correct values. For the `*-openai` profiles, set `OPENAI_API_BASE` and `OPENAI_API_KEY` in `.env.docker`.

4. **Start the Services**

   ```bash
   docker compose --profile cpu-ollama up
   ```

   **What's Included:**
   - **Backend**: FastAPI application for RAG and orchestration.
   - **Inference Server**: Ollama, `llama.cpp`, or external OpenAI API (depending on profile).
   - **Qdrant**: Vector database for hybrid (semantic, bm42) search.
   - **Frontend**: Streamlit UI.

   On the first run, required ML models are automatically downloaded into the `model-cache` shared volume. This volume is shared between the backend (which handles downloads) and the inference server (which loads them).

---

### Local Setup (BYO Inference)

1. **Start Infrastructure Services**

   For local Python development (without Docker), you must provide your own inference endpoints.

   - **Qdrant**: Must be running (default: `http://localhost:6333`).
   - **Inference Server**: An OpenAI-compatible server (Ollama, LocalAI, vLLM, or `llama-server`) must be accessible.
     - By default, the app expects Ollama at `http://localhost:11434/v1`. For `llama-server`, use `http://localhost:8080/v1`. Configure `OPENAI_API_BASE` in `.env` accordingly.

2. **Create a Local Environment File**

   Copy the example config and adjust values for your local stack:

   ```bash
   cp .env.example .env
   ```

   At minimum, confirm `MODEL_PROVIDER`, `OPENAI_API_BASE`, `EMBED_MODEL_PROVIDER`, and model IDs (`EMBED_MODEL`, `LLM`, `VLM`) match your stack.

3. **Install Dependencies**

   Navigate to the project root and synchronize dependencies:

   ```bash
   uv sync
   ```

4. **Download Embedding/Rerank Models**

   Pre-download required local models (Embeddings, Sparse, Rerank, Text, Vision, Whisper) to your local cache:

   ```bash
   uv run load-models
   ```

5. **Run the Application**

   The application consists of a FastAPI backend and a Streamlit UI. You can run both locally.

   **Start the Backend:**

   ```bash
   uv run uvicorn docint.core.api:app --reload
   ```

   **Start the Streamlit UI:**

   In a new terminal window:

   ```bash
   uv run docint
   ```

## Configuration

The application is configured via environment variables. Key variables include:

**General:**

- `DOCINT_OFFLINE`: Set to `true` to force offline mode (fails if models aren't cached).
- `NER_ENABLED`: Enable scalable entity/relation extraction during ingestion (default: `true`). Uses parallel execution for maximum throughput.
- `NER_MAX_WORKERS`: Number of parallel workers for entity extraction (default: `4`).
- `ENABLE_HATE_SPEECH_DETECTION`: Enable hate-speech detection during ingestion. When disabled/unset, ingestion behavior remains unchanged (default: `false`).
- `GRAPHRAG_ENABLED`: Enable graph-assisted query expansion during chat/query retrieval (default: `false`).
- `GRAPHRAG_NEIGHBOR_HOPS`: Neighborhood depth for graph expansion (default: `1`).
- `GRAPHRAG_TOP_K_NODES`: Graph node cap used for derived in-memory graph construction (default: `100`).
- `GRAPHRAG_MIN_EDGE_WEIGHT`: Minimum edge weight included in the graph (default: `1`).
- `GRAPHRAG_MAX_NEIGHBORS`: Maximum neighbor entities appended to a retrieval query (default: `6`).
- `RESPONSE_VALIDATION_ENABLED`: Enable a second-pass LLM check that verifies answer grounding and source/query fit, and flags mismatches (default: `false`).
- `SUMMARY_COVERAGE_TARGET`: Minimum document coverage ratio considered sufficient for collection summaries (default: `0.70`).
- `SUMMARY_MAX_DOCS`: Maximum number of documents sampled when building collection summaries (default: `30`).
- `SUMMARY_PER_DOC_TOP_K`: Maximum evidence chunks retrieved per document during summary preparation (default: `4`).
- `SUMMARY_FINAL_SOURCE_CAP`: Maximum number of merged sources returned with a collection summary (default: `24`).
- `PRELOAD_MODELS`: Set to `true` to download all ML models at container startup (default: unset/disabled). Used by `docker-compose.yml` to populate the `model-cache` volume on first run.

**Hosts / Service Endpoints:**

- `BACKEND_HOST`: Backend base URL used by API/UI clients (default: `http://localhost:8000`).
- `BACKEND_PUBLIC_HOST`: Public backend URL used for preview/download links (defaults to `BACKEND_HOST`).
- `QDRANT_HOST`: Qdrant endpoint (default: `http://localhost:6333`).
- `CORS_ALLOWED_ORIGINS`: Comma-separated allow-list for API CORS.

**Model Selection (Environment Variables):**

- `OPENAI_API_KEY`: API key for the LLM provider (default: `sk-no-key-required`).
- `OPENAI_API_BASE`: Base API URL. In Docker, this is set automatically per profile (e.g. `http://ollama-server:11434/v1` for `*-ollama`). For local development, point this to your provider (e.g., `http://localhost:11434/v1`).
- `MODEL_PROVIDER`: Inference provider type (`llama.cpp`, `ollama`, `openai`).
- `EMBED_MODEL_PROVIDER`: Embedding backend selector (`huggingface`, `ollama`, `openai`, or `llama.cpp`). When omitted, Docint defaults to the native embedding backend for `MODEL_PROVIDER`; if no provider-native backend is inferred, it falls back to `huggingface`.
- `LLM`: Repo ID (e.g., `bartowski/Meta-Llama-3-8B-Instruct-GGUF`) for automatic download.
- `EMBED_MODEL`: Embedding model ID for the selected embedding backend. Use an Ollama tag (for example `bge-m3`) with `EMBED_MODEL_PROVIDER=ollama`, an OpenAI embedding model name (for example `text-embedding-3-small`) with `EMBED_MODEL_PROVIDER=openai`, a GGUF spec (for example `ggml-org/bge-m3-Q8_0-GGUF;bge-m3-q8_0.gguf`) with `EMBED_MODEL_PROVIDER=llama.cpp`, or a Hugging Face repo ID (for example `BAAI/bge-m3`) with `EMBED_MODEL_PROVIDER=huggingface`.
- `VLM`: Vision-language model ID (GGUF `repo;filename` for `llama.cpp` or model tag for Ollama/OpenAI).

**Note on Provider Agnosticism:**
The backend logic is standard OpenAI-compatible. The `docker-compose.yml` profiles bundle Ollama and `llama.cpp` servers, but you can also point to any external API by selecting a `*-openai` profile and setting `OPENAI_API_BASE` and `OPENAI_API_KEY` in `.env.docker`.

**Local Models (Hugging Face):**

- `SPARSE_MODEL`: Sparse embedding model (default: `Qdrant/all_miniLM_L6_v2_with_attentions`).
- `RERANK_MODEL`: Cross-encoder reranker (default: `BAAI/bge-reranker-v2-m3`).
- `NER_MODEL`: Local GLiNER model for entity extraction (default: `gliner-community/gliner_large-v2.5`).

See `docint/utils/env_cfg.py` for the full list of configuration options and defaults.

Collection summary responses (`POST /summarize`, `POST /summarize/stream`) include an optional `summary_diagnostics` object:

- `total_documents`: Number of documents sampled for summarization.
- `covered_documents`: Number of sampled documents with extracted evidence.
- `coverage_ratio`: `covered_documents / total_documents` (or `0.0` when no documents are available).
- `uncovered_documents`: Filenames with no extracted evidence.
- `coverage_target`: Configured threshold from `SUMMARY_COVERAGE_TARGET`.

## Image Ingestion and Retrieval

Image ingestion is unified across both paths:

- standalone image files (`.png`, `.jpg`, `.jpeg`, `.gif`) read by `ImageReader`
- images extracted from PDFs by the core pipeline (`artifacts/<doc_id>/images/*.json`)

Both paths call the same shared service: `docint/core/ingest/images_service.py`.

For each image, DocInt stores:

- deterministic `image_id` (SHA-256 of image bytes)
- image embedding vector (named vector, default `image-dense`)
- LLM-generated `llm_description` and `llm_tags`
- normalized metadata (`source_type`, `source_doc_id`, `page_number`, `bbox`, `mime_type`, `width`, `height`, etc.)

By default, image embeddings are written to a per-collection Qdrant collection (`{collection}_images`, e.g. `test-1_images`) and validated/created automatically.

### Image Config Knobs

- `IMAGE_INGESTION_ENABLED` (default `true`)
- `IMAGE_EMBEDDING_ENABLED` (default `true`)
- `IMAGE_TAGGING_ENABLED` (default `true`)
- `IMAGE_QDRANT_COLLECTION` (default `{collection}_images`)
- `IMAGE_QDRANT_VECTOR_NAME` (default `image-dense`)
- `IMAGE_EMBED_MODEL` (default `openai/clip-vit-base-patch32`)
- `IMAGE_CACHE_BY_HASH` (default `true`)
- `IMAGE_FAIL_ON_EMBED_ERROR` (default `false`)
- `IMAGE_FAIL_ON_TAG_ERROR` (default `false`)
- `IMAGE_TAGGING_MAX_IMAGE_DIM` (default `1024`)

If `MODEL_PROVIDER=llama.cpp` and your `llama-server` VLM is started without a
valid `--mmproj`, image requests can fail with `image input is not supported`.
DocInt now treats this as non-retryable, logs it once, and disables image
tagging for the rest of the run. To skip tagging entirely, set
`IMAGE_TAGGING_ENABLED=false`.

### Document Pipeline Config Knobs

- `PIPELINE_TEXT_COVERAGE_THRESHOLD` (default `0.01`)
- `PIPELINE_ARTIFACTS_DIR` (default `artifacts` under the project root)
- `PIPELINE_MAX_RETRIES` (default `2`)
- `PIPELINE_FORCE_REPROCESS` (default `false`)
- `PIPELINE_MAX_WORKERS` (default `4`)
- `PIPELINE_ENABLE_VISION_OCR` (default `true`)
- `PIPELINE_VISION_OCR_TIMEOUT` (default `60.0`)
- `PIPELINE_VISION_OCR_MAX_RETRIES` (default `1`)

## Information Extraction Endpoints

DocInt exposes collection-level IE APIs in addition to raw source payloads:

- `GET /collections/ner`: Raw source-level entities/relations for the selected collection.
- `GET /collections/ner/stats`: Aggregated IE metrics and top entities/relations.
- `GET /collections/ner/search`: Entity search with optional type filter.
- `PIPELINE_VISION_OCR_MAX_IMAGE_DIM` (default `1024`)
- `PIPELINE_VISION_OCR_MAX_TOKENS` (default `4096`)

### Image-to-Image Query

`ImageIngestionService.query_similar_images(...)` can be used to query nearest image neighbors by embedding:

```python
from pathlib import Path

from docint.core.ingest.images_service import ImageIngestionService

service = ImageIngestionService()
matches = service.query_similar_images(Path("query.png"), top_k=5)
```

## Usage

### Ingesting Data

1. **Data Preparation**

   Place files to be ingested in the `~/docint/data` directory. Supported file types include:

   - **Documents**: `.pdf`, `.docx`, `.txt`
   - **Tables**: `.csv`, `.xls`, `.xlsx`
   - **Images**: `.png`, `.jpg`, `.jpeg`, `.gif`
   - **Audio**: `.mp3`, `.wav`
   - **Video**: `.mp4`, `.avi`
   - **JSON**: `.json`
   - **Other**: Additional formats supported via custom readers.

2. **Run the Ingestion Command**

   Start the ingestion process with:

   ```bash
   uv run ingest
   ```

3. **Verify the Ingestion**

   Check the logs or output directory to confirm the data has been organized and prepared for querying.

---

### Querying Data

1. **Prepare Your Queries**

   Create a `queries.txt` file in the `~/docint` directory. Each line represents a single query. If no file is provided, the default summarize prompt is used.

2. **Run the Query Command**

   Execute the following command:

   ```bash
   uv run query
   ```

3. **Review the Results**

   Query results will be saved in the `~/docint/results` directory.
   When GraphRAG is enabled, each result JSON also includes a `graph_debug`
   object with query-expansion details (anchors, neighbors, and applied/no-op reason).

---

### [Development] Launching UI and Backend separately

#### Streamlit UI

1. **Start the Development Server**

   ```bash
   uv run docint
   ```

2. **Access the Frontend**

   Open your browser and navigate to `http://localhost:8501/`.

#### Backend (FastAPI)

1. **Start the Backend Server**

   ```bash
   uv run uvicorn docint.core.api:app --reload
   ```

2. **Verify the Backend**

   The backend will be available at `http://localhost:8000/docs`.

3. **Use Retrieval Filters in Chat**

   Open the `Retrieval filters` panel on the Chat page to restrict retrieval by
   metadata before querying the vector store. Supported filters include MIME
   patterns (for example `image/*` or `application/pdf`), date boundaries on
   timestamp fields, and additional custom metadata rules.

---

### Launching via Docker

1. **Access the Application**

   Navigate to `http://localhost:8501/` in your browser to access the GUI for ingestion and querying tasks.

2. **Stop the Services**

   ```bash
   docker compose down
   ```

---

### Additional Configuration

For additional configuration:

- Local usage: copy `.env.example` to `.env`
- Docker usage: copy `.env.docker.example` to `.env.docker`

Quick start:

```bash
cp .env.example .env
cp .env.docker.example .env.docker
```

Minimal local example (`.env`):

```env
DOCINT_OFFLINE=false
MODEL_PROVIDER=ollama
OPENAI_API_BASE=http://localhost:11434/v1
OPENAI_API_KEY=sk-no-key-required
EMBED_MODEL=bge-m3
LLM=gpt-oss:20b
VLM=qwen3-vl:8b
NER_ENABLED=true
NER_MAX_WORKERS=4
ENABLE_HATE_SPEECH_DETECTION=false
```

## Unit Tests

Run unit tests and pre-commit checks to ensure functionality and lint/type quality:

```bash
uv run pytest
uv run pre-commit run --all-files
```

## Document Processing Pipeline

The enhanced document processing pipeline provides page-level triage, layout analysis, OCR fallback, table/image extraction, and layout-aware chunking. It runs fully offline and handles digital, scanned, and mixed PDFs.
This pipeline is the default PDF ingestion path used by the API/UI/CLI ingestion workflows.

### Pipeline Stages

1. **Page Triage** — Classifies each page as digital (text layer present) or scanned (needs OCR) using a configurable text-coverage heuristic.
2. **Layout Analysis** — Detects layout blocks (text, titles, tables, figures, headers/footers) with reading order and bounding boxes.
3. **OCR / Text Extraction** — Extracts text from the PDF text layer; applies OCR fallback on pages flagged as needing it.
4. **Table & Image Extraction** — Extracts table regions and figure/image regions from layout blocks with best-effort structure detection.
5. **Layout-Aware Chunking** — Chunks text respecting reading order, section headings, and sentence boundaries. Produces stable chunk IDs.
6. **Artifact Persistence** — Writes structured artifacts per document for debugging and reprocessing.

### Artifact Directory Structure

```text
artifacts/{doc_id}/
├── manifest.json                  # Document-level metadata and page triage results
├── pages/{page_index}/
│   ├── layout.json                # Layout blocks for the page
│   └── text.json                  # Text extraction results (PDF text + OCR spans)
├── tables/{table_id}.json         # Table metadata and content (+ .csv if grid available)
├── images/{image_id}.json         # Image metadata
└── chunks.jsonl                   # All chunks with stable IDs and metadata
```

### Document Pipeline Configuration

Pipeline behaviour is controlled via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `PIPELINE_TEXT_COVERAGE_THRESHOLD` | `0.01` | Chars-per-area ratio below which a page needs OCR |
| `PIPELINE_ARTIFACTS_DIR` | `artifacts` | Root directory for artifact output |
| `PIPELINE_MAX_RETRIES` | `2` | Max retry attempts per processing stage |
| `PIPELINE_FORCE_REPROCESS` | `false` | Ignore existing artifacts and reprocess |
| `PIPELINE_MAX_WORKERS` | `4` | Document-level parallelism |
| `PIPELINE_ENABLE_VISION_OCR` | `true` | Enable vision-LLM OCR fallback for scanned pages |
| `PIPELINE_VISION_OCR_TIMEOUT` | `60.0` | Per-request timeout (seconds) for vision OCR calls |
| `PIPELINE_VISION_OCR_MAX_RETRIES` | `1` | Retry count for a single vision OCR call |
| `PIPELINE_VISION_OCR_MAX_IMAGE_DIM` | `1024` | Max image dimension before downscaling for OCR |
| `PIPELINE_VISION_OCR_MAX_TOKENS` | `4096` | Max output tokens for vision OCR responses |

### Programmatic Usage

```python
from docint.core.readers.documents import DocumentPipelineOrchestrator, PipelineConfig

config = PipelineConfig(
    text_coverage_threshold=0.01,
    pipeline_version="1.0.0",
    artifacts_dir="artifacts",
    max_retries=2,
    force_reprocess=False,
    max_workers=4,
      enable_vision_ocr=True,
      vision_ocr_timeout=60.0,
      vision_ocr_max_retries=1,
      vision_ocr_max_image_dimension=1024,
      vision_ocr_max_tokens=4096,
)
orchestrator = DocumentPipelineOrchestrator(config=config)
manifest = orchestrator.process("path/to/document.pdf")

print(f"Pages: {manifest.pages_total}, OCR: {manifest.pages_ocr}, Failed: {manifest.pages_failed}")
print(f"Tables: {manifest.tables_found}, Images: {manifest.images_found}")
```

### Idempotency

The pipeline uses SHA-256 of file bytes as `doc_id`. Re-running on the same file with the same `pipeline_version` skips processing unless `--force` / `PIPELINE_FORCE_REPROCESS=true` is set.

### Error Isolation

Failures on individual pages do not crash the entire document. Failed pages are recorded in `manifest.json` with error details, and processing continues for remaining pages.

### Offline Compliance

Core PDF processing runs locally. Vision OCR fallback uses the configured OpenAI-compatible inference endpoint (local or remote), and can be disabled with `PIPELINE_ENABLE_VISION_OCR=false`. Set `DOCINT_OFFLINE=true` to enforce offline mode for Hugging Face model loading. When `EMBED_MODEL_PROVIDER=huggingface`, `uv run load-models` must be run before startup so the embedding snapshot is present in the local cache.
