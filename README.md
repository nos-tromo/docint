# Document Intelligence

This is a proof of concept document intelligence platform offering the following features:

1. **Ingestion Pipeline**

   - Process and extract data from various file types, including documents, tables, images, audio, video, and JSON files.
   - Automatically organizes and prepares data for querying.
2. **Query Pipeline**

   - Perform batch or single queries on ingested data.
   - Customize queries for batch processing.
   - Outputs results to a designated directory.
3. **CLI and GUI Support**

   - Use the command-line interface for ingestion and querying tasks.
   - Access a browser-based graphical user interface for interactive workflows.
4. **Integration with LLMs and Models**

   - Utilize large language models (LLMs) and other AI models for advanced processing tasks like summarization and similarity retrieval.
   - **Offline-First**: Models are cached locally, allowing the system to run without an active internet connection after initial setup.
5. **Extensibility**

   - Easily extend functionality with additional readers, such as OCR for images, Whisper for audio/video, and table/json enhancements.
6. **Development and Testing**

   - Run the frontend and backend independently or via Docker.
   - Comprehensive unit testing ensures reliability.

## Installation

The application can be used both via Docker for containerized environments and directly on the local machine for development and testing purposes.

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

   Use `watch` instead of `up` to enable live code synchronization for container development.

---

### Local Setup

1. **Start Infrastructure Services**

   Ensure that **Ollama** and **Qdrant** are running locally or are accessible via network.
   - **Ollama**: Must be running (default: `http://localhost:11434`).
   - **Qdrant**: Must be running (default: `http://localhost:6333`).

2. **Install Dependencies**

   Navigate to the project root and synchronize dependencies:

   ```bash
   uv sync
   ```

3. **Download Models**

   Pre-download the required models to your local cache to enable offline functionality.
   *Note: Ollama must be running for this step to pull the LLM/VLM models.*

   ```bash
   uv run load-models
   ```

4. **Run the Application**

   The application consists of a backend API and a Streamlit frontend. You need to run both.

   **Start the Backend:**

   ```bash
   uv run uvicorn docint.core.api:app --reload
   ```

   **Start the Frontend:**

   In a new terminal window:

   ```bash
   uv run docint
   ```

## Configuration

The application is configured via environment variables. Key variables include:

- `DOCINT_OFFLINE`: Set to `true` to force offline mode (fails if models aren't cached).
- `LLM`: Name of the Ollama model to use (default: `granite4:7b-a1b-h`).
- `EMBED_MODEL`: HuggingFace embedding model ID (default: `BAAI/bge-m3`).
- `SPARSE_MODEL`: Sparse embedding model ID (default: `Qdrant/all_miniLM_L6_v2_with_attentions`).

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

1. **Run the Ingestion Command**

   Start the ingestion process with:

   ```bash
   uv run ingest
   ```

1. **Verify the Ingestion**

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

### [Development] Launching Frontend and Backend separately

#### Frontend (Streamlit)

1. **Start the Development Server**

   ```bash
   uv run streamlit run docint/app.py
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

### Additional Docker Configuration

For additional configurations, populate an `.env.docker` file in the project's root. Example:

```bash
DOCINT_OFFLINE=true
EMBED_MODEL=BAAI/bge-m3
SPARSE_MODEL=Qdrant/all_miniLM_L6_v2_with_attentions
LLM=granite4:7b-a1b-h
VLM=qwen3-vl:8b
WHISPER_MODEL=turbo
RETRIEVE_SIMILARITY_TOP_K=50
```

## Unit Tests

Run unit tests to ensure functionality:

```bash
uv run pytest
```
