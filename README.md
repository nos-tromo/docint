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
   - **Provider-Agnostic Architecture**: The system is designed to be backend-agnostic. It uses Ollama for local inference by default, and can switch to any OpenAI-compatible API (vLLM, DeepSeek, LM Studio, etc.) by changing environment variables.
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

   If the machine must reach the internet through an HTTP/HTTPS proxy, add
   `HTTP_PROXY`, `HTTPS_PROXY`, and `NO_PROXY` to `.env.docker`. Include the
   internal Docker hostnames in `NO_PROXY` so backend-to-Qdrant and backend-to-
   inference traffic stays inside the Docker network. A good starting point is:

   ```dotenv
   HTTP_PROXY=http://proxy.example.com:3128
   HTTPS_PROXY=http://proxy.example.com:3128
   NO_PROXY=localhost,127.0.0.1,qdrant,ollama-server,vllm-router,vllm-chat,vllm-embed,vllm-audio,vllm-rerank,backend-net
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
   VLLM_OPENAI_IMAGE=registry.example.com/dockerhub/vllm/vllm-openai:v0.17.1
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

   Each profile automatically sets `INFERENCE_PROVIDER` and `OPENAI_API_BASE` to the correct values. For the `*-openai` profiles, set `OPENAI_API_BASE` and `OPENAI_API_KEY` in `.env.docker`. The `cuda-vllm` profile routes the backend through an internal nginx service so split chat, embedding, sparse, audio, and rerank upstreams still appear as one OpenAI-compatible endpoint. It also defaults the backend's local auxiliary workloads to `USE_DEVICE=cpu` so GLiNER, CLIP, and similar helpers do not compete with the vLLM workers for VRAM.

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
       - By default, the app expects Ollama at `http://localhost:11434/v1`. Configure `OPENAI_API_BASE` in `.env` accordingly.

2. **Create a Local Environment File**

   Copy the example config and adjust values for your local stack:

   ```bash
   cp .env.example .env
   ```

   At minimum, confirm `INFERENCE_PROVIDER`, `OPENAI_API_BASE`, and model IDs (`EMBED_MODEL`, `TEXT_MODEL`, `VISION_MODEL`) match your stack.

3. **Install Dependencies**

   Navigate to the project root and synchronize dependencies:

   ```bash
   uv sync
   ```

4. **Download Required Model Snapshots**

   Pre-download the model snapshots required by your configured stack. For local backends this includes app-local models such as sparse, rerank, and Whisper; for `INFERENCE_PROVIDER=vllm` it also preloads the provider-served model snapshots used by the bundled vLLM services:

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
- `CHAT_RESPONSE_MODE`: Chat/query response synthesizer mode. Use `auto` to switch social/table-heavy collections to `refine` while keeping other collections on `compact` (default: `auto`).
- `RETRIEVAL_VECTOR_QUERY_MODE`: First-stage retrieval mode. `auto` resolves to `hybrid` when hybrid search is enabled for the collection, otherwise `default`. Supported values: `auto`, `default`, `sparse`, `hybrid`, `mmr` (default: `auto`).
- `RETRIEVAL_HYBRID_ALPHA`: Dense-vs-sparse fusion weight for hybrid retrieval. `1.0` biases dense results, `0.0` biases sparse results (default: `0.5`).
- `RETRIEVAL_SPARSE_TOP_K`: Sparse candidate depth used by sparse and hybrid retrieval (default: `20`).
- `RETRIEVAL_HYBRID_TOP_K`: Final candidate depth retained after dense/sparse fusion in hybrid mode (default: `20`).
- `PARENT_CONTEXT_RETRIEVAL_ENABLED`: When hierarchical chunks are available, retrieve fine chunks first and expand them to their parent context for synthesis (default: `true`).
- `SUMMARY_COVERAGE_TARGET`: Minimum document coverage ratio considered sufficient for collection summaries (default: `0.70`).
- `SUMMARY_MAX_DOCS`: Maximum number of documents sampled when building collection summaries (default: `30`).
- `SUMMARY_PER_DOC_TOP_K`: Maximum evidence chunks retrieved per document during summary preparation (default: `4`).
- `SUMMARY_FINAL_SOURCE_CAP`: Maximum number of merged sources returned with a collection summary (default: `24`).
- `SUMMARY_SOCIAL_CHUNKING_ENABLED`: Enable chunk/post-level summary mode for row-heavy social/table collections (default: `true`).
- `SUMMARY_SOCIAL_CANDIDATE_POOL`: Candidate retrieval depth used by chunk/post-level social summaries (default: `48`).
- `SUMMARY_SOCIAL_DIVERSITY_LIMIT`: Maximum number of retained social summary sources per author/time bucket (default: `2`).
- `INGESTION_BATCH_SIZE`: Size of the in-memory enrichment streaming window (NER/hate-speech). Smaller values reduce crash-loss window during enrichment; larger values can improve throughput (default: `5`).
- `DOCSTORE_BATCH_SIZE`: Size of micro-batches written to docstore/vectorstore after enrichment. Smaller values flush to Qdrant more often; larger values reduce write overhead (default: `100`).
- `INGEST_BENCHMARK_ENABLED`: Emit ingest benchmark logs (elapsed time, docs/nodes throughput, enrichment and persistence batch counters) to help tune batch sizes (default: `false`).
- `DOCSTORE_MAX_RETRIES`: Retry count for transient Qdrant docstore transport errors during ingest/read operations (default: `3`).
- `DOCSTORE_RETRY_BACKOFF_SECONDS`: Initial backoff delay between docstore retries in seconds (default: `0.25`).
- `DOCSTORE_RETRY_BACKOFF_MAX_SECONDS`: Maximum backoff delay between docstore retries in seconds (default: `2.0`).
- `PRELOAD_MODELS`: Set to `true` to download all ML models at container startup (default: unset/disabled). Used by `docker-compose.yml` to populate the `model-cache` volume on first run.

Batch-size tuning guidance:

- `INGESTION_BATCH_SIZE` controls how many parsed nodes are enriched before they are yielded to persistence in streaming mode.
- `DOCSTORE_BATCH_SIZE` controls how many already-enriched nodes are flushed per write call to the persistent stores.
- Increasing only `INGESTION_BATCH_SIZE` raises enrichment throughput but increases in-flight work that can be lost before first persist.
- Increasing only `DOCSTORE_BATCH_SIZE` reduces write-call overhead but increases the amount of post-enrichment work that can be in-flight during a transport interruption.

**Hosts / Service Endpoints:**

- `BACKEND_HOST`: Backend base URL used by API/UI clients (default: `http://localhost:8000`).
- `BACKEND_PUBLIC_HOST`: Public backend URL used for preview/download links (defaults to `BACKEND_HOST`).
- `QDRANT_HOST`: Qdrant endpoint (default: `http://localhost:6333`).
- `CORS_ALLOWED_ORIGINS`: Comma-separated allow-list for API CORS.
- `SESSIONS_PATH`: Directory for the default file-backed chat session store. Local default: `~/docint/sessions`. Docker Compose sets this to `/var/lib/docint/sessions`.
- `SESSION_STORE`: Full SQLAlchemy session-store URL override. Use this if you want something other than the default SQLite file under `SESSIONS_PATH`.
- `USE_DEVICE`: Device used by backend-local auxiliary models such as GLiNER, CLIP, and Whisper. Supported values are `auto`, `cpu`, `mps`, `cuda`, and `cuda:<index>`. The bundled `cuda-vllm` profile defaults this to `cpu`.

**Model Selection (Environment Variables):**

- `OPENAI_API_KEY`: API key for the LLM provider (default: `sk-no-key-required`).
- `OPENAI_API_BASE`: Base API URL. In Docker, this is set automatically per profile (e.g. `http://ollama-server:11434/v1` for `*-ollama`). For local development, point this to your provider (e.g., `http://localhost:11434/v1`).
- `OPENAI_CTX_WINDOW`: Context window advertised to the backend prompt planner. In the bundled `cuda-vllm` profile this is also passed to the vLLM chat server as `--max-model-len`, so it is the preferred single source of truth. If unset, DocInt still falls back to the legacy `CHAT_MAX_MODEL_LEN` knob when present, then to `8192`.
- `OPENAI_ENABLE_THINKING`: Enable OpenAI reasoning/thinking for text models routed through the native OpenAI provider (default: `false`).
- `OPENAI_THINKING_EFFORT`: Reasoning effort used when `OPENAI_ENABLE_THINKING=true`. Supported values: `none`, `minimal`, `low`, `medium`, `high`, `xhigh` (default: `medium`).
- `OPENAI_DIMENSIONS`: Optional embedding dimension override. Only set this for providers and models that support reduced-dimension embeddings; leave it unset for most local OpenAI-compatible backends such as vLLM-served BGE models.
- `INFERENCE_PROVIDER`: Inference provider type (`ollama`, `openai`, `vllm`, or another OpenAI-compatible backend mapped to one of those modes).
- `TEXT_MODEL`: Text/chat model ID served by the configured provider.
- `EMBED_MODEL`: Embedding model ID served by the configured `INFERENCE_PROVIDER`. Use an Ollama tag (for example `bge-m3`) with `INFERENCE_PROVIDER=ollama` or an OpenAI-compatible embedding model name (for example `text-embedding-3-small` or a vLLM-served alias such as `bge-m3-docint`) with `INFERENCE_PROVIDER=openai` or `INFERENCE_PROVIDER=vllm`.
- `VISION_MODEL`: Vision-language model ID served by the configured provider (for example an Ollama tag, an OpenAI model name, or a vLLM-served alias). When the bundled `cuda-vllm` profile routes all chat traffic to a single vLLM chat server, set `VISION_MODEL` equal to `TEXT_MODEL` unless you add a dedicated vision upstream.

**Note on Provider Agnosticism:**
The backend logic is standard OpenAI-compatible. The `docker-compose.yml` profiles bundle Ollama and CUDA vLLM services, but you can also point to any external API by selecting a `*-openai` profile and setting `OPENAI_API_BASE` and `OPENAI_API_KEY` in `.env.docker`.

**Local Models (Hugging Face):**

- `SPARSE_MODEL`: Sparse embedding model. For local hybrid retrieval the default is `Qdrant/all_miniLM_L6_v2_with_attentions`. For `INFERENCE_PROVIDER=vllm`, the default follows `EMBED_MODEL` so the bundled profile can serve dense and sparse retrieval from the same pooling model (`BAAI/bge-m3` by default).
- `RERANK_MODEL`: Cross-encoder reranker (default: `BAAI/bge-reranker-v2-m3`).
- `NER_MODEL`: Local GLiNER model for entity extraction (default: `gliner-community/gliner_large-v2.5`).
- `WHISPER_MODEL`: Audio transcription model. Local backends default to the packaged Whisper `turbo` model; `INFERENCE_PROVIDER=vllm` defaults to the translation-capable `openai/whisper-large-v3`.

See `docint/utils/env_cfg.py` for the full list of configuration options and defaults.

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
- `GET /collections/ner/stats`: Aggregated IE metrics and top entities/relations. Supports `entity_merge_mode=orthographic|exact` and defaults to orthographic condensation.
- `GET /collections/ner/search`: Entity search with optional type filter. Supports `entity_merge_mode=orthographic|exact` and defaults to orthographic condensation.
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
DOCINT_OFFLINE=false
INFERENCE_PROVIDER=ollama
OPENAI_API_BASE=http://localhost:11434/v1
OPENAI_API_KEY=sk-no-key-required
EMBED_MODEL=bge-m3
TEXT_MODEL=gpt-oss:20b
VISION_MODEL=qwen3-vl:8b
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

Core PDF processing runs locally. Vision OCR fallback uses the configured OpenAI-compatible inference endpoint (local or remote), and can be disabled with `PIPELINE_ENABLE_VISION_OCR=false`. Set `DOCINT_OFFLINE=true` to enforce offline mode for Hugging Face model loading. When using local model assets, `uv run load-models` must be run before startup so embedding, sparse, rerank, and related model files are present in the local cache.
