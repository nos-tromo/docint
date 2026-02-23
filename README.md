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
   - **Provider-Agnostic Architecture**: The system is designed to be backend-agnostic. While it includes a highly optimized `llama.cpp` server for local inference by default, it can easily switch to any OpenAI-compatible API (Ollama, vLLM, DeepSeek, etc.) by changing environment variables.
   - **Performance & Decoupling**: Inference runs in a dedicated container (`llamacpp-server`), ensuring the application logic remains responsive even during heavy model computations.

5. **Extensibility**

   - Easily extend functionality with additional readers, such as OCR for images, Whisper for audio/video, and table/json enhancements.

6. **Development and Testing**

   - Run the API and Streamlit UI independently or via Docker.
   - Comprehensive unit testing and pre-commit (ruff, mypy) ensure reliability.

## Supported Inference Backends

The platform is provider-agnostic and works with any OpenAI-compatible inference server. The following backends are tested and supported out of the box:

| Backend | Description | Link |
|---------|-------------|------|
| **llama.cpp** | High-performance local inference for GGUF models (bundled in Docker profiles) | [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) |
| **Ollama** | Easy-to-use local model runner with built-in model management | [ollama/ollama](https://github.com/ollama/ollama) |
| **vLLM** *(planned)* | High-throughput serving engine with PagedAttention | [vllm-project/vllm](https://github.com/vllm-project/vllm) |
| **OpenAI API** | Cloud-hosted models (GPT-4o, etc.) via the official API | [openai/openai-python](https://github.com/openai/openai-python) |

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
   | `cpu-llamacpp` | CPU | Llama.cpp (default) |
   | `cpu-ollama` | CPU | Ollama |
   | `cpu-openai` | CPU | External OpenAI API |
   | `cuda-llamacpp` | NVIDIA GPU | Llama.cpp |
   | `cuda-ollama` | NVIDIA GPU | Ollama |
   | `cuda-openai` | NVIDIA GPU | External OpenAI API |

   CUDA profiles require a CUDA-compatible GPU and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

   Each profile automatically sets `MODEL_PROVIDER` and `OPENAI_API_BASE` to the correct values. For the `*-openai` profiles, set `OPENAI_API_BASE` and `OPENAI_API_KEY` in `.env.docker`.

4. **Start the Services**

   ```bash
   docker compose --profile cpu-llamacpp up
   ```

   **What's Included:**
   - **Backend**: FastAPI application for RAG and orchestration.
   - **Inference Server**: `llama.cpp`, Ollama, or external OpenAI API (depending on profile).
   - **Qdrant**: Vector database for hybrid (semantic, bm42) search.
   - **Frontend**: Streamlit UI.

   On the first run, required ML models are automatically downloaded into the `model-cache` shared volume. This volume is shared between the backend (which handles downloads) and the inference server (which loads them).

---

### Local Setup (BYO Inference)

1. **Start Infrastructure Services**

   For local Python development (without Docker), you must provide your own inference endpoints.

   - **Qdrant**: Must be running (default: `http://localhost:6333`).
   - **Inference Server**: An OpenAI-compatible server (Ollama, LocalAI, vLLM, or `llama-server`) must be accessible.
     - By default, the app expects an endpoint at `http://localhost:8080/v1` (or `11434` for Ollama). Configure `OPENAI_API_BASE` in `.env` accordingly.

2. **Install Dependencies**

   Navigate to the project root and synchronize dependencies:

   ```bash
   uv sync
   ```

3. **Download Embedding/Rerank Models**

   Pre-download required local models (Embeddings, Sparse, Rerank, Text, Vision, Whisper) to your local cache:

   ```bash
   uv run load-models
   ```

4. **Run the Application**

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
- `NER_ENABLED`: Enable scalable entity/relation extraction during ingestion (default: `false`). Uses parallel execution for maximum throughput.
- `NER_MAX_WORKERS`: Number of parallel workers for entity extraction (default: `4`).
- `PRELOAD_MODELS`: Set to `true` to download all ML models at container startup (default: unset/disabled). Used by `docker-compose.yml` to populate the `model-cache` volume on first run.

**Model Selection (Environment Variables):**

- `OPENAI_API_KEY`: API key for the LLM provider (default: `sk-no-key-required`).
- `OPENAI_API_BASE`: Base API URL. In Docker, this is set automatically per profile (e.g. `http://llamacpp-server:8080/v1` for `*-llamacpp`). For local development, point this to your provider (e.g., `http://localhost:11434/v1`).
- `LLM`: Repo ID (e.g., `bartowski/Meta-Llama-3-8B-Instruct-GGUF`) for automatic download.
- `EMBED_MODEL`: HuggingFace repo ID for the embedding model (e.g. `ggml-org/bge-m3-Q8_0-GGUF`).

**Note on Provider Agnosticism:**
The backend logic is standard OpenAI-compatible. The `docker-compose.yml` profiles bundle `llama.cpp` and Ollama servers, but you can also point to any external API by selecting a `*-openai` profile and setting `OPENAI_API_BASE` and `OPENAI_API_KEY` in `.env.docker`.

**Local Models (Hugging Face):**

- `SPARSE_MODEL`: Sparse embedding model (default: `Qdrant/all_miniLM_L6_v2_with_attentions`).
- `RERANK_MODEL`: Cross-encoder reranker (default: `BAAI/bge-reranker-v2-m3`).
- `NER_MODEL`: Local GLiNER model for entity extraction (default: `gliner-community/gliner_large-v2.5`).

See `docint/utils/env_cfg.py` for the full list of configuration options and defaults.

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

For additional configuration, populate an `.env` (local usage) or `.env.docker` file in the project's root. Example:

```env
DOCINT_OFFLINE=true
OPENAI_API_KEY=sk-no-key-required
LLM=gpt-4o
EMBED_MODEL=text-embedding-3-small
NER_ENABLED=true
NER_MAX_WORKERS=4
```

## Unit Tests

Run unit tests and pre-commit checks to ensure functionality and lint/type quality:

```bash
uv run pytest
uv run pre-commit run --all-files
```

## Document Processing Pipeline

The enhanced document processing pipeline provides page-level triage, layout analysis, OCR fallback, table/image extraction, and layout-aware chunking. It runs fully offline and handles digital, scanned, and mixed PDFs.

### Pipeline Stages

1. **Page Triage** — Classifies each page as digital (text layer present) or scanned (needs OCR) using a configurable text-coverage heuristic.
2. **Layout Analysis** — Detects layout blocks (text, titles, tables, figures, headers/footers) with reading order and bounding boxes.
3. **OCR / Text Extraction** — Extracts text from the PDF text layer; applies OCR fallback on pages flagged as needing it.
4. **Table & Image Extraction** — Extracts table regions and figure/image regions from layout blocks with best-effort structure detection.
5. **Layout-Aware Chunking** — Chunks text respecting reading order, section headings, and sentence boundaries. Produces stable chunk IDs.
6. **Artifact Persistence** — Writes structured artifacts per document for debugging and reprocessing.

### Artifact Directory Structure

```
artifacts/{doc_id}/
├── manifest.json                  # Document-level metadata and page triage results
├── pages/{page_index}/
│   ├── layout.json                # Layout blocks for the page
│   └── text.json                  # Text extraction results (PDF text + OCR spans)
├── tables/{table_id}.json         # Table metadata and content (+ .csv if grid available)
├── images/{image_id}.json         # Image metadata
└── chunks.jsonl                   # All chunks with stable IDs and metadata
```

### Configuration

Pipeline behaviour is controlled via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `PIPELINE_TEXT_COVERAGE_THRESHOLD` | `0.01` | Chars-per-area ratio below which a page needs OCR |
| `PIPELINE_ARTIFACTS_DIR` | `artifacts` | Root directory for artifact output |
| `PIPELINE_MAX_RETRIES` | `2` | Max retry attempts per processing stage |
| `PIPELINE_FORCE_REPROCESS` | `false` | Ignore existing artifacts and reprocess |
| `PIPELINE_MAX_WORKERS` | `4` | Document-level parallelism |

### Programmatic Usage

```python
from docint.core.pipeline import DocumentPipelineOrchestrator, PipelineConfig

config = PipelineConfig(
    text_coverage_threshold=0.01,
    pipeline_version="1.0.0",
    artifacts_dir="artifacts",
    max_retries=2,
    force_reprocess=False,
    max_workers=4,
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

All processing runs locally. No network calls are made. Set `DOCINT_OFFLINE=true` to enforce offline mode for all Hugging Face model loading.
