# Document Intelligence

Document Intelligence is a document RAG stack for ingestion, retrieval,
and chat. It ships with:

- a FastAPI backend
- a Streamlit UI
- Qdrant for storage and retrieval
- pluggable inference via Ollama, OpenAI-compatible APIs, or bundled vLLM

## What You Need

- Docker for the containerized setup
- Python 3.11 and `uv` for local development
- an inference backend
  - Docker profiles can provide Ollama or bundled vLLM
  - local development needs an OpenAI-compatible endpoint you manage yourself

## Quick Start With Docker

1. Create the Docker env file:

   ```bash
   cp .env.docker.example .env.docker
   ```

2. Create the shared cache volumes once:

   ```bash
   ./scripts/create_docker_volumes.sh
   ```

3. Pick one profile:

   | Profile | Use when |
   | --- | --- |
   | `cpu-ollama` | CPU-only machine, local Ollama in Docker |
   | `cpu-openai` | CPU-only machine, external OpenAI-compatible API |
   | `cuda-ollama` | NVIDIA GPU, local Ollama in Docker |
   | `cuda-vllm` | NVIDIA GPU, bundled vLLM services |
   | `cuda-openai` | NVIDIA GPU, external OpenAI-compatible API |

4. Start the stack:

   ```bash
   docker compose --env-file .env.docker --profile cpu-ollama up --build
   ```

5. Open the app:

   - UI: <http://localhost:8501>
   - API: <http://localhost:8000>
   - Qdrant: <http://localhost:6333>

### Docker Notes

- `cpu-openai` and `cuda-openai` need `OPENAI_API_KEY` in `.env.docker`.
- `cuda-vllm` needs an NVIDIA GPU and the NVIDIA Container Toolkit.
- First startup may take a while because model assets are downloaded into the
  shared cache volumes.
- If you use an outbound proxy, put the proxy variables in `.env.docker` and
  keep using `--env-file .env.docker` with `docker compose`.

### Shared Docker Volumes

The compose file uses external cache volumes so model artifacts survive
container recreation:

- `docling-cache`
- `huggingface-cache`
- `ollama-cache`
- `whisper-cache`

The helper script creates them with `docker volume create`.

## Local Development

Use this when you want to run the Python services directly instead of through
Docker.

1. Copy the local env file:

   ```bash
   cp .env.example .env
   ```

2. Ensure the required services exist:

   - Qdrant at `http://localhost:6333`
   - an OpenAI-compatible inference endpoint

3. Install dependencies:

   ```bash
   uv sync
   ```

4. Optional: pre-download local model assets:

   ```bash
   uv run load-models
   ```

5. Start the backend:

   ```bash
   uv run uvicorn docint.core.api:app --reload
   ```

6. Start the UI in another terminal:

   ```bash
   uv run docint
   ```

## Common Commands

Ingest data:

```bash
uv run ingest --help
```

Query data:

```bash
uv run query --help
```

Run tests and checks:

```bash
uv run pytest
uv run pre-commit run --all-files
```

Stop the Docker stack:

```bash
docker compose --env-file .env.docker down
```

## Repository Shape

- `docint/core`: backend, ingestion, retrieval, storage, session state
- `docint/agents`: orchestration and tool-using agent flow
- `docint/ui`: Streamlit pages and components
- `tests`: unit tests
