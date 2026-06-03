# Document Intelligence

Document Intelligence is a document RAG stack for ingestion, retrieval,
and chat. It ships with:

- a FastAPI backend
- a React SPA served by an nginx sidecar that reverse-proxies API
  requests to the backend on the internal Docker network
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

2. Create the shared external networks and volumes once (idempotent):

   ```bash
   make network    # inference-net + data-net
   make volumes    # external Docker volumes
   ```

3. Build and start the stack:

   ```bash
   make build
   make up-dev
   ```

   `make up-dev` layers `docker/compose.override.yaml` so host ports are
   published for local development; `make up` runs the base
   `docker/compose.yaml` alone (production shape, no host ports).

4. Open the app:

   - App: <http://localhost:8080> (override with `FRONTEND_PORT` in `.env`)

   The backend is reachable only via the nginx sidecar — it is no longer
   published on the host. Use `docker compose --env-file .env -f
   docker/compose.yaml exec backend …` to interact with it directly.

   Qdrant is **not** served by this stack — it is provided by the sibling
   `data-plane` project. Start it once with `cd ../data-plane && make up`
   (or `make up-dev` to also publish Qdrant on `localhost:6333`).

### Docker Notes

- docint ships a single CPU-only image. All ML inference (chat,
  embeddings, rerank, NER, CLIP) is delegated to the external
  [vllm-service](https://github.com/nos-tromo/vllm-service) stack.
- Set `INFERENCE_PROVIDER` and `OPENAI_API_BASE` in `.env`.
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
- Large `/ingest/upload` requests are capped by the frontend nginx proxy.
   Override `DOCINT_CLIENT_MAX_BODY_SIZE` in `.env` if your combined multipart
   upload needs more than the default `1g`.
- Session persistence uses one SQLite file path. Set `SESSIONS_DB_PATH` for
  the normal case or `SESSION_STORE` if you want to supply a full SQLAlchemy
  database URL.
- The React SPA does not add an authenticated user header by itself. For
   single-user Docker or local setups, set `DOCINT_DEFAULT_IDENTITY` in `.env`
   so session-backed chat and `/sessions/list` share one owner. If you run
   behind a trusted proxy that injects a user header, set
   `DOCINT_AUTH_HEADER` to that header name instead. Existing legacy sessions
   with `owner = NULL` are backfilled to `DOCINT_DEFAULT_IDENTITY` when the
   backend initializes the session store.

### Shared Docker Volumes

The compose file uses external volumes so model artifacts and backend
state survive container recreation — and so `docker compose down -v`
cannot destroy staged sources or the session database:

- `docling-cache`
- `huggingface-cache`
- `ollama-cache`
- `sessions-storage`
- `source-preview-cache`

The helper script creates them with `docker volume create`.

## Local Development

Use this when you want to run the Python services directly instead of through
Docker.

1. Copy the local env file:

   ```bash
   cp .env.example .env
   ```

2. Ensure the required services exist:

   - Qdrant at `http://localhost:6333` — provided by the sibling
     `data-plane` project (`cd ../data-plane && make up-dev`)
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

6. Start the frontend in another terminal (optional — for live development):

   ```bash
   cd frontend
   pnpm install
   pnpm dev      # → http://localhost:5173 (proxies /api to :8000)
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

Resolve entities — merge duplicate and semantically-similar named entities
(e.g. `USA`/`United States`) for a collection into durable canonical records.
Re-runnable and idempotent; results surface in the NER views under
`entity_merge_mode=resolved`. Tuned by `RES_EMBED_THRESHOLD` (0.86),
`RES_LLM_TIEBREAK` (true), `RES_CASE_NORMALIZE` (true), `RES_VECTOR_K` (5):

Runs in a one-off `backend` container (production is Docker-only), so it
reaches the `qdrant` / `vllm-router` network aliases — bring up data-plane and
vllm-service first.

```bash
make resolve                    # prompts for the collection name
make resolve COLLECTION=mydocs  # non-interactive
# or over HTTP, on the selected collection:
# curl -X POST http://localhost:8000/collections/entities/resolve
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
make stop
```

## Server-Side Exports For Large Collections

The React UI streams collection-wide CSVs from the backend so the browser
never accumulates the whole result set in memory. Two paths exist for jobs
that would otherwise tax the SPA:

```bash
# Server-streamed CSV from anywhere with HTTP access to the backend.
# Selects the active collection first (the SPA does this automatically;
# the example assumes the API is reachable on port 8000).
curl -X POST http://localhost:8000/collections/select \
  -H 'Content-Type: application/json' \
  -d '{"name": "my_collection"}'
curl -O "http://localhost:8000/collections/my_collection/export/entities.csv"
curl -O "http://localhost:8000/collections/my_collection/export/hate-speech.csv"
curl -O "http://localhost:8000/collections/my_collection/export/documents.csv"

# The entities export honours the same merge modes as the Analysis view —
# pass entity_merge_mode=resolved to stream the durable canonical entities
# (run `make resolve` first; falls back to orthographic if not resolved).
curl -O "http://localhost:8000/collections/my_collection/export/entities.csv?entity_merge_mode=resolved"
```

For batch jobs that take many minutes (or shouldn't hold an HTTP
connection open), the `query` CLI runs inside the backend container and
writes the same CSV files to a mounted volume:

```bash
docker compose --env-file .env -f docker/compose.yaml \
  exec backend-cpu query --collection my_collection --all \
  --output /var/lib/docint/sources/my_collection/exports
docker compose cp \
  backend-cpu:/var/lib/docint/sources/my_collection/exports ./exports
```

Both paths share the schemas defined in `docint/utils/csv_stream.py`, so
the streaming endpoint and the CLI produce byte-identical CSVs for the
same collection.

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
- `TEXT_MODEL`, `VISION_MODEL`, `EMBED_MODEL`, `SPARSE_MODEL`, and
  `RERANK_MODEL` matching the served model IDs

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
  co-deployment with vLLM / Ollama, offline image bundles
- [Development](docs/development.md) — dev workflow, pre-commit,
  pytest layout, CI, extension points

## Repository Shape

- `docint/core`: backend, ingestion, retrieval, storage, session state
- `docint/agents`: orchestration and tool-using agent flow
- `frontend/`: React SPA (Vite + TypeScript)
- `docs`: in-repo documentation
- `tests`: unit tests
