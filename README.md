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

2. **Choose a Profile**

   Decide between the CPU or GPU profile. The GPU profile requires a CUDA-compatible GPU and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) set up.

3. **Start the Services**

   ```bash
   docker compose --profile <cpu|gpu> up
   ```

   Use `watch` instead of `up` to enable live code synchronization for container development.

---

### Local Setup

1. **Frontend**

   To set up the frontend, navigate to the `frontend` directory and install the required dependencies:

   ```bash
   cd frontend
   npm install
   ```

2. **Backend**

   For the backend, navigate to the `backend` directory and synchronize dependencies:

   ```bash
   cd backend
   uv sync
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

1. **Run the Ingestion Command**

   Start the ingestion process from `backend` with:

   ```bash
   uv run ingest
   ```

2. **Verify the Ingestion**

   Check the logs or output directory to confirm the data has been organized and prepared for querying.

---

### Querying Data

1. **Prepare Your Queries**

   Create a `queries.txt` file in the `~/docint` directory. Each line represents a single query. If no file is provided, a default summary query file will be generated.
2. **Run the Query Command**

   Execute the following command from `backend`:

   ```bash
   uv run query
   ```

3. **Review the Results**

   Query results will be saved in the `~/docint/results` directory.

---

### [Development] Launching Frontend and Backend separately

#### Frontend

1. **Start the Development Server**

   ```bash
   npm run dev
   ```

2. **Access the Frontend**
   Open your browser and navigate to `http://localhost:5173/`.

#### Backend

1. **Start the Backend Server**

   ```bash
   uv run uvicorn docint.app:app --reload
   ```

2. **Verify the Backend**

   The backend will be available at `http://127.0.0.1:8000/`.

---

### Launching via Docker

1. **Access the Application**

   Navigate to `http://localhost:8080/` in your browser to access the GUI for ingestion and querying tasks.
2. **Stop the Services**

   ```bash
   docker compose down
   ```

---

### Additional Docker Configuration

For additional configurations, populate an `.env.docker` file in the project's root. Example:

```bash
LLM=gpt-oss:120b-cloud
VLM=qwen3-vl:235b-cloud
WHISPER_MODEL=turbo
RETRIEVE_SIMILARITY_TOP_K=50
```

## Unit Tests

Run unit tests from `backend` to ensure functionality:

```bash
uv run pytest
```

## Roadmap

- Expand GUI for ingestion ✅
- Add WhisperReader for audio/video ✅
- Add OCRReader for images ✅
- Improve table and JSON reader functionalities
- Implement explicit summarization features
- Develop GraphRAG
