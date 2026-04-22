# Document Intelligence

Document Intelligence is a document RAG stack for ingestion, retrieval,
and chat. It ships with:

- a FastAPI backend
- a Streamlit UI
- Qdrant for storage and retrieval
- pluggable inference via any OpenAI-compatible API or an external routed vLLM service

## What You Need

- Docker for the containerized setup
- Python 3.11 and `uv` for local development
- an inference backend: any OpenAI-compatible endpoint configured via `.env`
  - vLLM is deployed separately and consumed via one routed base URL
  - local development needs an OpenAI-compatible endpoint you manage yourself

## Quick Start With Docker

1. Create the shared env file:

   ```bash
   cp .env.example .env
   ```

2. Create the shared cache volumes once:

   ```bash
   ./scripts/create_docker_volumes.sh
   ```

3. Pick one profile:

   | Profile | Use when |
   | --- | --- |
   | `cpu` | CPU-only machine |
   | `cuda` | NVIDIA GPU machine |

4. Start the stack:

   ```bash
   docker compose --profile cpu up --build
   ```

5. Open the app:

   - UI: <http://localhost:8501>
   - API: <http://localhost:8000>
   - Qdrant: <http://localhost:6333>

### Docker Notes

- Set `INFERENCE_PROVIDER` and `OPENAI_API_BASE` in `.env` — profiles select
  hardware only and do not set a provider.
- The `openai` provider requires `OPENAI_API_KEY` in `.env`.
- The `vllm` provider requires `OPENAI_API_BASE` in `.env`.
  Deploy the standalone vLLM app first, then start Docint. You can use [vllm-service](https://github.com/nos-tromo/vllm-service) to serve text, vision, embedding, reranking and audio endpoints.
  Docint expects the vLLM router to expose one OpenAI-compatible base URL that
  ends in `/v1`, plus the vLLM sparse routes at `/pooling` and `/tokenize`.
- For co-deployed stacks on one server, create one shared external Docker
  network, attach both compose projects to it, and set
  `OPENAI_API_BASE=http://vllm-router:9000/v1`.
- The `cuda` profile needs an NVIDIA GPU and the NVIDIA Container Toolkit.
- First startup may take a while because model assets are downloaded into the
  shared cache volumes.
- If you use an outbound proxy, put the proxy variables in `.env` so Compose,
  image builds, and containers use the same values.
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

Verify that a collection's Qdrant vector store and SQLite KV docstore
are in sync — reports drift (KV-only orphans, Qdrant-only orphans,
broken hierarchical parents) and optionally repairs KV-only orphans:

```bash
uv run verify --collection my_collection
uv run verify --collection my_collection --repair
```

Run tests and checks:

```bash
uv run pytest
uv run pre-commit run --all-files
```

Stop the Docker stack:

```bash
docker compose down
```

## Standalone vLLM App

The standalone deployment lives in
[vllm-service](https://github.com/nos-tromo/vllm-service/).

For a shared-network deployment on one server:

1. Create the shared external network once:

   ```bash
   docker network create inference-net
   ```

2. Start `vllm-service` on that network.
3. Set `INFERENCE_NET=inference-net` in both projects if you use a different name.
4. Configure Docint with:

   ```bash
   INFERENCE_PROVIDER=vllm
   OPENAI_API_BASE=http://vllm-router:9000/v1
   OPENAI_API_KEY=<token>
   ```

Run that stack separately and configure Docint with:

- `INFERENCE_PROVIDER=vllm`
- `OPENAI_API_BASE=https://<router-host>/v1`
- `OPENAI_API_KEY=<token>`
- `TEXT_MODEL`, `VISION_MODEL`, `EMBED_MODEL`, `SPARSE_MODEL`,
  `RERANK_MODEL`, and `WHISPER_MODEL` matching the served model IDs

## Documentation

The [`docs/`](docs/README.md) directory contains the in-repo reference
manual. It complements this README with topic-by-topic deep dives:

- [Getting started](docs/getting-started.md) — install, first ingest,
  first query
- [Architecture](docs/architecture.md) — runtime components and request
  flow
- [Configuration](docs/configuration.md) — every env var grouped by
  dataclass, with defaults
- [API reference](docs/api-reference.md) — every FastAPI route
- [CLI reference](docs/cli-reference.md) — `docint`, `ingest`, `query`,
  `query-eval`, `load-models`
- [Ingestion pipeline](docs/ingestion.md) — readers, chunking, NER,
  storage
- [Retrieval and agents](docs/retrieval-and-agents.md) — orchestrator,
  hybrid retrieval, graph-RAG, validation
- [UI guide](docs/ui-guide.md) — Streamlit pages and components
- [Deployment](docs/deployment.md) — Docker profiles, volumes,
  co-deployment with vLLM / Ollama
- [Development](docs/development.md) — dev workflow, pre-commit,
  pytest layout, CI, extension points

## Repository Shape

- `docint/core`: backend, ingestion, retrieval, storage, session state
- `docint/agents`: orchestration and tool-using agent flow
- `docint/ui`: Streamlit pages and components
- `docs`: in-repo documentation
- `tests`: unit tests
