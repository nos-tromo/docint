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

### Docker

Select whether to use the CPU or GPU (requires a CUDA compatible GPU and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) set up):

```bash
# CPU
docker compose --profile cpu up
# GPU
docker compose --profile gpu up
```

## Usage

### Ingest documents

```bash
uv run ingest
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
