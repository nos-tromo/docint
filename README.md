# Document Intelligence

## Installation

### Manual setup

#### Frontend

```bash
cd frontend
npm install
```

#### Backend

```bash
cd backend
uv sync
```

### Docker (recommended)

Select whether to use the CPU or GPU (requires a CUDA compatible GPU and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) set up):

```bash
# CPU
docker compose --profile cpu up
# GPU
docker compose --profile gpu up
```

Use `watch` instead of `up` for development and live sync for code changes.

## Usage

### CLI

#### Document ingestion

Place data to be ingested in `~/docint/data` before running the command below.

```bash
uv run ingest
```

#### Querying

Place a `queries.txt` file inside `~/docint` to facilitate batch processing of requests (one query per line; no file provided will create a default file with a summary query). Outputs will be stored to `~/docint/results`.

```bash
uv run query
```

### Run frontend and backend in separate shells

Frontend:

```bash
cd frontend
npm run dev
```

Backend:

```bash
uv run uvicorn docint.app:app --reload
```

Launch the browser app: `http://localhost:5173/`.

### Run via Docker

Launch the browser app: `http://localhost:8080/`. For further configurations, populate an `.env` file in the project's root, e.g.:

```bash
LLM=gpt-oss:120b-cloud
RETRIEVE_SIMILARITY_TOP_K=50
VLM=qwen3-vl:235b-cloud
WHISPER_MODEL=turbo
```

### Unit tests

```bash
uv run pytest
```

## Roadmap

- Expand GUI for ingestion ✅
- Add WhisperReader for audio/video ✅
- Add OCRReader for images ✅
- Explicit summarization feature
- Implement GraphRAG
