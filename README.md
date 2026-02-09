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

   - Utilize large language models (LLMs) and other AI models for advanced processing tasks like summarization and similarity retrieval.
   - **Offline-First**: Models are cached locally, allowing the system to run without an active internet connection after initial setup.
5. **Extensibility**

   - Easily extend functionality with additional readers, such as OCR for images, Whisper for audio/video, and table/json enhancements.
6. **Development and Testing**

   - Run the API and Streamlit UI independently or via Docker.
   - Comprehensive unit testing and pre-commit (ruff, mypy) ensure reliability.

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

   Decide between the CPU or GPU profile. The GPU profile requires a CUDA-compatible GPU and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) set up.
4. **Start the Services**

   ```bash
   docker compose --profile <cpu|gpu> up
   ```

---

### Local Setup

1. **Start Infrastructure Services**

   Ensure that **Qdrant** is running locally or accessible via network.

   - **Qdrant**: Must be running (default: `http://localhost:6333`).
2. **Install Dependencies**

   Navigate to the project root and synchronize dependencies:

   ```bash
   uv sync
   ```

3. **Download Models**

   Pre-download all required models (Embeddings, Sparse, LLMs, VLMs, NER, Whisper) to your local cache to enable offline functionality:

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

- `DOCINT_OFFLINE`: Set to `true` to force offline mode (fails if models aren't cached).
- `LLAMA_CPP_N_GPU_LAYERS`: Number of layers to offload to GPU. Use `-1` for all layers (full GPU), `0` for CPU only (default: `-1`).
- `LLAMA_CPP_CTX_WINDOW`: Context window size (default: `8192`).
- `LLAMA_CPP_TEMPERATURE`: Sampling temperature (default: `0.0`).
- `ENABLE_IE`: Enable scalable entity/relation extraction during ingestion (default: `false`). Uses parallel execution for maximum throughput.
- `IE_MAX_WORKERS`: Number of parallel workers for entity extraction (default: `4`).

See `docint/utils/env_cfg.py` for the full list of configuration options and defaults.

### GPU Acceleration

**CUDA (NVIDIA GPUs):**

- Set `LLAMA_CPP_N_GPU_LAYERS=-1` to offload all layers to GPU
- Requires CUDA toolkit and compatible NVIDIA GPU
- Docker: Use the `gpu` profile with `docker compose --profile gpu up`

**Metal (Apple Silicon):**
To use Metal acceleration on macOS with Apple Silicon:

```bash
CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python --force-reinstall --no-cache-dir
```

Then set `LLAMA_CPP_N_GPU_LAYERS=-1` to enable GPU acceleration.

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

   Create a `queries.txt` file in the `~/docint` directory. Each line represents a single query. If no file is provided, a default summary query file will be generated.
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
EMBED_MODEL=BAAI/bge-m3
SPARSE_MODEL=Qdrant/all_miniLM_L6_v2_with_attentions
WHISPER_MODEL=turbo
ENABLE_IE=true
IE_MAX_WORKERS=4
LLAMA_CPP_N_GPU_LAYERS=-1
LLAMA_CPP_CTX_WINDOW=8192
LLAMA_CPP_TEMPERATURE=0.1
```

## Unit Tests

Run unit tests and pre-commit checks to ensure functionality and lint/type quality:

```bash
uv run pytest
uv run pre-commit run --all-files
```
