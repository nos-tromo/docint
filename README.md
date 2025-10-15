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

### Ingest documents

```bash
uv run ingest
```

### Run via CLI

Place a `queries.txt` file inside `backend/` to facilitate batch processing of requests (one query per line; no file provided will create a default file with a summary query).

```bash
uv run cli
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

Launch the browser app: `http://localhost:8080/`.

## Roadmap

- Explicit summarization feature
- Expand GUI for ingestion
- Add WhisperReader for audio/video
- Add OCRReader for images
- Implement GraphRAG