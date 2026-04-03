# Document Intelligence

Document Intelligence is a document RAG stack for ingestion, retrieval,
and chat. It ships with:

- a FastAPI backend
- a Streamlit UI
- Qdrant for storage and retrieval
- pluggable inference via Ollama, OpenAI-compatible APIs, or an external routed vLLM service

## What You Need

- Docker for the containerized setup
- Python 3.11 and `uv` for local development
- an inference backend
  - Docker profiles can provide Ollama
  - vLLM is deployed separately and consumed via one routed base URL
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
   | `cpu-vllm` | CPU-only machine, external routed vLLM service |
   | `cuda-ollama` | NVIDIA GPU, local Ollama in Docker |
   | `cuda-openai` | NVIDIA GPU, external OpenAI-compatible API |
   | `cuda-vllm` | NVIDIA GPU, external routed vLLM service |

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
- `cpu-vllm` and `cuda-vllm` require `OPENAI_API_BASE` in `.env.docker`.
- Deploy the standalone vLLM app first, then start Docint.
- Docint expects the vLLM router to expose one OpenAI-compatible base URL that
  ends in `/v1`, plus the vLLM sparse routes at `/pooling` and `/tokenize`.
- `cuda-vllm` needs an NVIDIA GPU and the NVIDIA Container Toolkit.
- First startup may take a while because model assets are downloaded into the
  shared cache volumes.
- If you use an outbound proxy, put the proxy variables in `.env.docker` and
  keep using `--env-file .env.docker` with `docker compose`.
- Session persistence uses one SQLite file path. Set `SESSIONS_DB_PATH` for
   the normal case or `SESSION_STORE` if you want to supply a full SQLAlchemy
   database URL.

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
   - an OpenAI-compatible inference endpoint, such as an external vLLM service

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

## Standalone vLLM App

Docint provides a basic Ollama service for local inference. A standalone app scaffold for the separate deployment lives in
[vllm-service](https://github.com/nos-tromo/vllm-service/).

Run that stack separately and configure Docint with:

- `INFERENCE_PROVIDER=vllm`
- `OPENAI_API_BASE=https://<router-host>/v1`
- `OPENAI_API_KEY=<token>`
- `TEXT_MODEL`, `VISION_MODEL`, `EMBED_MODEL`, `SPARSE_MODEL`,
  `RERANK_MODEL`, and `WHISPER_MODEL` matching the served model IDs

## Repository Shape

- `docint/core`: backend, ingestion, retrieval, storage, session state
- `docint/agents`: orchestration and tool-using agent flow
- `docint/ui`: Streamlit pages and components
- `tests`: unit tests
