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
   - **Flexible Backend**: Supports OpenAI-compatible APIs (OpenAI, Ollama, vLLM, and similar services) for text and vision tasks.
   - **Provider-Agnostic Architecture**: The system is backend-agnostic. It uses Ollama locally by default and can switch to OpenAI, vLLM, LM Studio, or another OpenAI-compatible API by changing the active profile in `config.toml`.
   - **Performance & Decoupling**: Inference runs in a dedicated container (`ollama-server` by default, or the bundled vLLM router for CUDA profiles), ensuring the application logic remains responsive even during heavy model computations.

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
| **OpenAI API** | Cloud-hosted models (GPT-4o, etc.) via the official API | [openai/openai-python](https://github.com/openai/openai-python) |
| **vLLM** | High-throughput OpenAI-compatible serving for CUDA deployments | [vllm-project/vllm](https://github.com/vllm-project/vllm) |

Any other server exposing an OpenAI-compatible `/v1/chat/completions` endpoint (e.g., [LocalAI](https://github.com/mudler/LocalAI), [LM Studio](https://github.com/lmstudio-ai)) can also be used by editing the active profile in `config.toml`.

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

2. **Configure Bootstrap Environment**

   Create the Docker environment file from the example:

   ```bash
   cp .env.docker.example .env.docker
   ```

   `config.toml` is the canonical runtime config. `.env.docker` is now only for
   secrets, `DOCINT_PROFILE` overrides, and Docker-specific proxy or registry
   settings.

   If the machine must reach the internet through an HTTP/HTTPS proxy, add
   `HTTP_PROXY`, `HTTPS_PROXY`, and `NO_PROXY` to `.env.docker`. Include the
   internal Docker hostnames in `NO_PROXY` so backend-to-Qdrant and backend-to-
   inference traffic stays inside the Docker network. A good starting point is:

   ```dotenv
   HTTP_PROXY=http://proxy.example.com:3128
   HTTPS_PROXY=http://proxy.example.com:3128
   NO_PROXY=localhost,127.0.0.1,qdrant,ollama-server,vllm-router,vllm-chat,vllm-embed,vllm-rerank,backend-net
   ```

   If any tooling in your environment only honors lowercase proxy variables,
   duplicate the same values into `http_proxy`, `https_proxy`, and `no_proxy`.

   If your network also blocks direct access to Docker Hub, override the base
   images in `.env.docker` so builds use your registry mirror instead of the
   hard-coded public image names:

   ```dotenv
   PYTHON_SLIM_BOOKWORM_IMAGE=registry.example.com/dockerhub/library/python:3.11-slim-bookworm
   PYTHON_SLIM_IMAGE=registry.example.com/dockerhub/library/python:3.11-slim
   NVIDIA_CUDA_RUNTIME_IMAGE=registry.example.com/dockerhub/nvidia/cuda:13.0.2-cudnn-runtime-ubuntu22.04
   VLLM_OPENAI_IMAGE=registry.example.com/dockerhub/vllm/vllm-openai:latest
   ```

   These variables are consumed during `FROM ...` resolution, which is the step
   that fails when Docker cannot reach Docker Hub to fetch image metadata.

3. **Choose a Profile**

   Profiles follow the pattern `<hardware>-<backend>`:

   | Profile | Hardware | Inference Backend |
   |---------|----------|-------------------|
   | `cpu-ollama` | CPU | Ollama |
   | `cpu-openai` | CPU | External OpenAI API |
   | `cuda-ollama` | NVIDIA GPU | Ollama |
   | `cuda-vllm` | NVIDIA GPU | vLLM with bundled router |
   | `cuda-openai` | NVIDIA GPU | External OpenAI API |

   CUDA profiles require a CUDA-compatible GPU and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

   Runtime behavior comes from `config.toml`. The compose profile chooses the
   container topology, and each backend service pins its own matching
   `DOCINT_PROFILE`. If you use a non-default frontend stack, set
   `DOCINT_PROFILE` in `.env.docker` to the matching `docker-*` profile so the
   frontend resolves the same backend settings.

4. **Start the Services**

   ```bash
   docker compose --env-file .env.docker --profile cpu-ollama up --build
   ```

   Use `--env-file .env.docker` whenever you rely on proxy variables. Docker
   daemon proxy settings alone cover image pulls, but package managers running
   during `docker compose build` and applications inside the containers still
   need the proxy values forwarded explicitly.

   **What's Included:**
   - **Backend**: FastAPI application for RAG and orchestration.
   - **Inference Server**: Ollama, CUDA vLLM via the bundled router, or external OpenAI API (depending on profile).
   - **Qdrant**: Vector database for hybrid (semantic, bm42) search.
   - **Frontend**: Streamlit UI.

   On the first run, required ML models are automatically downloaded into the `model-cache` shared volume. This volume is shared between the backend (which handles downloads) and the inference server (which loads them).

---

### Local Setup (BYO Inference)

1. **Start Infrastructure Services**

   For local Python development (without Docker), you must provide your own inference endpoints.

    - **Qdrant**: Must be running (default: `http://localhost:6333`).
    - **Inference Server**: An OpenAI-compatible server (Ollama, LocalAI, vLLM, or another compatible service) must be accessible.
       - By default, the repo uses the `local-ollama` profile in `config.toml`.

2. **Create a Local Bootstrap Environment File**

   Copy the example config for secrets and optional profile overrides:

   ```bash
   cp .env.example .env
   ```

   Then edit `config.toml` and either change `active_profile` or update the
   selected profile table. Use `DOCINT_PROFILE` in `.env` only when you want a
   temporary override without editing `config.toml`.

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

Runtime configuration now lives in the repo-root [`config.toml`](config.toml).
The file defines named profiles such as `local-ollama`, `local-openai`,
`docker-cpu-ollama`, and `docker-cuda-vllm`. Each profile contains:

- `shared`: inference, models, retrieval, pipeline, and other common runtime knobs
- `backend`, `frontend`, `worker`: role-specific hosts and paths

The canonical loader is [`docint/utils/env_cfg.py`](docint/utils/env_cfg.py).
All app entrypoints bootstrap config through that module before importing
model-dependent libraries.

Only a small env surface remains:

- `OPENAI_API_KEY`: secret for OpenAI-compatible providers that require it
- `HF_TOKEN`: optional Hugging Face auth token
- `DOCINT_PROFILE`: optional override for `active_profile` in `config.toml`
- Docker-only proxy and registry variables in `.env.docker`

Typical workflow:

- Local development: edit `config.toml` and leave `.env` for secrets only
- Docker: select a compose profile and keep `.env.docker` for secrets, proxy settings, and optional `DOCINT_PROFILE` overrides
- One-off experiments: export `DOCINT_PROFILE` instead of editing `active_profile`

Common TOML sections you will edit most often:

- `shared.inference`: provider, API base, thinking settings
- `shared.models`: text, embed, rerank, vision, NER, whisper model IDs
- `shared.retrieval`, `shared.summary`, `shared.pipeline`, `shared.ingestion`
- `<role>.hosts` and `<role>.paths`

Batch-size tuning still works through TOML:

- `shared.ingestion.ingestion_batch_size`
- `shared.ingestion.docstore_batch_size`
- `shared.ingestion.docstore_max_retries`
- `shared.ingestion.docstore_retry_backoff_seconds`
- `shared.ingestion.docstore_retry_backoff_max_seconds`

Collection summary responses (`POST /summarize`, `POST /summarize/stream`) include an optional `summary_diagnostics` object:

- `total_documents`: Number of sampled coverage units. This is documents for standard corpora and posts/chunks for row-heavy social summaries.
- `covered_documents`: Number of sampled coverage units included in the final grounded summary context.
- `coverage_ratio`: `covered_documents / total_documents` (or `0.0` when no documents are available).
- `uncovered_documents`: Filenames with no extracted evidence.
- `coverage_target`: Configured threshold from `SUMMARY_COVERAGE_TARGET`.
- `coverage_unit`: Coverage semantics for the current summary response (`documents`, `posts`, or `chunks`).
- `candidate_count`: Raw candidate sources considered before deduplication.
- `deduped_count`: Candidate sources remaining after deduplication.
- `sampled_count`: Sources retained after diversity filtering and final source capping.

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

These map to `shared.image_ingestion` and `shared.models.image_embed_model` in `config.toml`:

- `enabled` (default `true`)
- `embedding_enabled` (default `true`)
- `tagging_enabled` (default `true`)
- `collection_name` (default `{collection}_images`)
- `vector_name` (default `image-dense`)
- `image_embed_model` (default `openai/clip-vit-base-patch32`)
- `cache_by_hash` (default `true`)
- `fail_on_embedding_error` (default `false`)
- `fail_on_tagging_error` (default `false`)
- `tagging_max_image_dimension` (default `1024`)

### Document Pipeline Config Knobs

These map to `shared.pipeline` and the active role's `paths.artifacts` in `config.toml`:

- `text_coverage_threshold` (default `0.01`)
- `artifacts` path (default under the active role's artifacts directory)
- `max_retries` (default `2`)
- `force_reprocess` (default `false`)
- `max_workers` (default `4`)
- `enable_vision_ocr` (default `true`)
- `vision_ocr_timeout` (default `60.0`)
- `vision_ocr_max_retries` (default `1`)

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

   Create a `queries.txt` file in the `~/docint` directory. Each line represents a single query.

2. **Run the Query Command**

   Execute one of the following commands:

   ```bash
   uv run query
   ```

   The query CLI supports these action flags:

   - `-c`, `--collectionn NAME`: Use `NAME` as the target collection instead of prompting interactively.
   - `-q`, `--query [PATH]`: Run chat queries from `PATH`. If the path is omitted, the CLI falls back to the default `~/docint/queries.txt`. If the file does not exist, query mode is skipped.
   - `-s`, `--summary`: Generate a collection summary using the same backend flow as the frontend analysis page.
   - `-e`, `--entities`: Export a `.txt` file containing the 50 most frequent entities and their mention counts.
   - `-h8`, `--hate-speech`: Export the flagged hate-speech findings as a `.txt` file using the same text format as the frontend download.
   - `-a`, `--all`: Run all of the above actions together.

   Examples:

   ```bash
   uv run query -q
   uv run query -c alpha -q ~/docint/queries.txt
   uv run query --summary
   uv run query --entities --hate-speech
   uv run query -c alpha --all
   ```

3. **Review the Results**

   Query results and export files will be saved as `.txt` files in a per-run subdirectory under `~/docint/results`, using the pattern `~/docint/results/{unixtimestamp_collection_name}`.
   When GraphRAG is enabled, each query result text file also includes a `graph_debug`
   section with query-expansion details (anchors, neighbors, and applied/no-op reason).

4. **Compare Retrieval Modes**

   To compare dense/default, sparse, and hybrid retrieval over the same query set, run:

   ```bash
   uv run query-eval
   ```

   The command reads the same `queries.txt` file used by `uv run query`. You can also use `.json` or `.jsonl` files with optional expectation fields such as `expected_filenames`, `expected_file_hashes`, and `expected_text_ids`. A structured evaluation report is written to `~/docint/results`.

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

   The same panel now also exposes a `Query mode` selector:
   `answer` keeps the normal grounded-answer path, while
   `entity_occurrence` bypasses reranked answer synthesis and returns
   mention-level NER-backed source rows for the best matching entity in the
   active collection.

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
# OPENAI_API_KEY=sk-no-key-required
# DOCINT_PROFILE=local-openai
# HF_TOKEN=hf_your_token_here
```

Minimal local example (`config.toml`):

```toml
active_profile = "local-ollama"

[profiles.local-ollama.shared.inference]
provider = "ollama"
api_base = "http://localhost:11434/v1"

[profiles.local-ollama.shared.models]
embed_model = "bge-m3"
text_model = "gpt-oss:20b"
vision_model = "qwen3.5:9b"
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

Pipeline behaviour is controlled via `shared.pipeline` in `config.toml`:

| TOML key | Default | Description |
|----------|---------|-------------|
| `text_coverage_threshold` | `0.01` | Chars-per-area ratio below which a page needs OCR |
| `max_retries` | `2` | Max retry attempts per processing stage |
| `force_reprocess` | `false` | Ignore existing artifacts and reprocess |
| `max_workers` | `4` | Document-level parallelism |
| `enable_vision_ocr` | `true` | Enable vision-LLM OCR fallback for scanned pages |
| `vision_ocr_timeout` | `60.0` | Per-request timeout (seconds) for vision OCR calls |
| `vision_ocr_max_retries` | `1` | Retry count for a single vision OCR call |
| `vision_ocr_max_image_dimension` | `1024` | Max image dimension before downscaling for OCR |
| `vision_ocr_max_tokens` | `4096` | Max output tokens for vision OCR responses |

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

The pipeline uses SHA-256 of file bytes as `doc_id`. Re-running on the same file with the same `pipeline_version` skips processing unless `--force` is set or `shared.pipeline.force_reprocess = true`.

### Error Isolation

Failures on individual pages do not crash the entire document. Failed pages are recorded in `manifest.json` with error details, and processing continues for remaining pages.

### Offline Compliance

Core PDF processing runs locally. Vision OCR fallback uses the configured OpenAI-compatible inference endpoint (local or remote), and can be disabled with `shared.pipeline.enable_vision_ocr = false`. Offline Hugging Face behaviour is controlled by `shared.runtime.docint_offline` in `config.toml`. When using local model assets, `uv run load-models` must be run before startup so embedding, sparse, rerank, and related model files are present in the local cache.
