# Deployment

Docint ships its Docker assets under `docker/`: a base `compose.yaml`, a
`compose.override.yaml` dev overlay, and two Dockerfiles — a CPU-only
backend image and a React-SPA frontend image. Docint is **CPU-only**: all
ML inference (chat, embeddings, rerank, NER, CLIP) is delegated over HTTP
to the external `vllm-service` stack, so there is no GPU code, no CUDA
image, and no profile toggle. This document explains the services,
volumes, and networks, and the two supported co-deployment patterns with
external inference services.

## Files

| File | Role |
|---|---|
| `docker/compose.yaml` | Services, volumes, networks — production shape, no host ports published. |
| `docker/compose.override.yaml` | Dev overlay that publishes the frontend port; layered by `make up-dev` (not `make up`). |
| `docker/Dockerfile.backend` | CPU-only backend image — a `uv`-assembled virtualenv on the `uv` Python base. |
| `docker/Dockerfile.frontend` | React SPA built with pnpm and served by nginx. |
| `.dockerignore` | Build-context excludes; stays at the repo root so it applies to every image build. |
| `scripts/create_docker_volumes.sh` | Creates the external cache and state volumes (idempotent). |
| `scripts/bundle_images.sh` | Builds and packages a versioned image tarball for offline distribution. |
| `Makefile` | The blessed entry point — convenience targets wrapping the above (`make network`, `volumes`, `build`, `up`, `up-dev`, `bundle`, `stop`, `down`, `resolve`). |
| `.env.example` | Canonical `.env` template. |

The `Makefile` is the entry point for every Docker workflow: it points
`docker compose` at `docker/compose.yaml` for you. A bare `docker compose`
run from the repo root no longer finds the compose file. When a raw
command is needed, the equivalent form is `docker compose --env-file .env
-f docker/compose.yaml [-f docker/compose.override.yaml] <cmd>`.

## Running the stack

Docint builds a single CPU-only backend image and a single frontend
image — there is no `cpu`/`cuda` profile and no `PROFILE` toggle (all ML
inference is remote; see `CLAUDE.md`). The two `make` targets differ only
in whether host ports are published:

```bash
make up        # production shape — docker/compose.yaml alone, no host ports
make up-dev    # layers compose.override.yaml; serves the React SPA on
               # http://localhost:${DOCINT_HOST_PORT:-8080}
```

`make up` runs the base `docker/compose.yaml`, which publishes no host
ports — the backend is reachable only in-network and the SPA only through
the nginx sidecar. `make up-dev` layers `docker/compose.override.yaml` so
the frontend port is published for local development.

> Qdrant is **not** part of this compose project. It is owned by the
> sibling `data-plane` project (`../data-plane/`), which runs Neo4j and
> Qdrant for the whole nos-tromo stack. See
> [Vector store: the `data-plane` project](#vector-store-the-data-plane-project)
> below.

## Services

### `backend`

- Built from `docker/Dockerfile.backend` (CPU-only; `uv`-assembled venv).
- Exposes port `8000` on the compose networks — not host-published.
- Receives the following environment variables from the compose file (on
  top of whatever is in `.env`):

  | Variable | Value |
  |---|---|
  | `QDRANT_HOST` | `http://qdrant:6333` |
  | `QDRANT_SRC_DIR` | `/var/lib/docint/sources` |
  | `SESSIONS_DB_PATH` | `/var/lib/docint/sessions/sessions.sqlite3` |
  | `NO_PROXY` / `no_proxy` | `backend,qdrant,localhost,127.0.0.1,172.16.0.0/12,10.0.0.0/8` (plus `EXTRA_NO_PROXY`) |

- Mounts the `docling-cache`, `huggingface-cache`, `sessions-storage`, and
  `source-preview-cache` volumes.
- Attaches to the `docint-net` network, to `inference-net` with the alias
  `docint-backend` (to reach the vLLM router), and to `data-net` with the
  alias `docint-backend` (to reach the `data-plane` project's Qdrant at
  `http://qdrant:6333`).
- Set `PRELOAD_MODELS=true` to run `load-models` at container start before
  `uvicorn` (warms the HF / Docling caches); otherwise it starts `uvicorn`
  directly.

### `frontend`

- Built from `docker/Dockerfile.frontend`: a multi-stage build that
  compiles the React SPA with pnpm (`node:20-alpine`) and serves the
  static bundle via nginx (`nginx:1.27-alpine`) on container port `80`.
- nginx reverse-proxies API routes to the backend at `backend:8000` over
  `docint-net`, so the backend is never host-published — in dev the SPA is
  the only host-exposed surface.
- The host port is published only by the dev overlay:
  `compose.override.yaml` maps `${DOCINT_HOST_PORT:-8080}:80`, so
  `make up-dev` serves the UI at `http://localhost:8080`.
- Environment: `DOCINT_CLIENT_MAX_BODY_SIZE` (default `1g`) — the nginx
  upload-size cap.
- Attaches to `docint-net` only, and `depends_on` the `backend` so Compose
  starts the backend container first.

## Vector store: the `data-plane` project

Docint uses Qdrant as its vector store, but it no longer runs its own
Qdrant container. Qdrant (and Neo4j) are owned by the sibling
`data-plane` project at `../data-plane/`, which serves them to the whole
nos-tromo stack over the shared `data-net` network.

- The backend reaches Qdrant at `http://qdrant:6333` over `data-net` —
  the `qdrant` hostname is a network alias published by `data-plane`, so
  `QDRANT_HOST=http://qdrant:6333` works unchanged.
- Bring Qdrant up before starting docint:

  ```bash
  cd ../data-plane && make up        # Neo4j + Qdrant on data-net
  cd ../data-plane && make up-dev    # also publishes Qdrant on localhost:6333
  ```

  Use `make up-dev` when running docint's Python services outside Docker
  so the host can reach Qdrant at `http://localhost:6333`.

## Volumes

Backend caches and backend filesystem state are declared
`external: true` so they survive container rebuilds and are **not**
destroyed by `docker compose down -v`. The helper script
`./scripts/create_docker_volumes.sh` creates them idempotently.

| Volume | Scope | Purpose |
|---|---|---|
| `docling-cache` | external | Docling model cache. |
| `huggingface-cache` | external | HF Hub cache (embedding, reranker, NER, image models). |
| `ollama-cache` | external | Ollama model cache, used when Ollama is co-deployed. |
| `sessions-storage` | external | SQLite session database. |
| `source-preview-cache` | external | Raw source files staged for `/sources/preview`. |

## Networks

- `docint-net` — internal bridge network shared by `backend` and
  `frontend`.
- `inference-net` — **external** network declared with
  `name: ${INFERENCE_NET:-inference-net}`. The backend attaches to it
  with the alias `docint-backend` and reaches the vLLM router over it.
- `data-net` — **external** network declared with
  `name: ${DATA_NET:-data-net}`. The backend attaches to it to reach
  the `data-plane` project's Qdrant at `http://qdrant:6333`.

Both external networks must exist before starting the stack. `make
network` creates both idempotently:

```bash
make network
# equivalent to:
docker network create inference-net
docker network create data-net
```

## Dockerfiles

### `docker/Dockerfile.backend`

Multi-stage build, CPU-only:

1. **Builder stage** — starts from the `uv` base image pinned by the
   `UV_IMAGE` build arg (default `ghcr.io/astral-sh/uv:…-python3.11-…`),
   installs build dependencies (`build-essential`, `python3-dev`,
   `zlib1g-dev`), and assembles the virtualenv with `uv sync --locked`
   from `pyproject.toml` / `uv.lock`.
2. **Runtime stage** — starts from the same `uv` base, installs only the
   runtime libraries (`libmagic1`, `libgl1`), copies the prebuilt
   virtualenv and app source, exposes `8000`, and sets the entrypoint to
   `uvicorn docint.core.api:app --host 0.0.0.0 --port 8000`.
3. **Optional model preload** — the entrypoint runs `load-models` before
   `uvicorn` when `PRELOAD_MODELS=true` is set at runtime, warming the
   HF / Docling caches on first start.

### `docker/Dockerfile.frontend`

Multi-stage build: compiles the React SPA with pnpm on `node:20-alpine`,
then serves the static bundle via `nginx:1.27-alpine` on port `80`. nginx
also reverse-proxies API routes to the backend and honors
`DOCINT_CLIENT_MAX_BODY_SIZE` for the upload-size limit. The image carries
no Python or ML dependencies.

## Environment configuration

`.env` at the repository root feeds both Compose variable interpolation
(via `--env-file`) and the backend container (`env_file: ../.env`). Put
provider-specific settings there:

```bash
INFERENCE_PROVIDER=vllm
OPENAI_API_BASE=http://vllm-router:4000/v1
OPENAI_API_KEY=sk-no-key-required
TEXT_MODEL=Qwen/Qwen3.5-2B
```

See [configuration.md](configuration.md) for every variable. A handful
of network, proxy, and runtime overrides are Compose-specific:

| Variable | Purpose |
|---|---|
| `EXTRA_NO_PROXY` | Comma-separated hostnames appended to the backend's `NO_PROXY` / `no_proxy` (must start with a leading comma). |
| `INFERENCE_NET` | Name of the shared external inference network (default `inference-net`). |
| `DATA_NET` | Name of the shared external data network (default `data-net`). |
| `DOCINT_HOST_PORT` | Host port for the React SPA under `make up-dev` (default `8080`). |
| `DOCINT_CLIENT_MAX_BODY_SIZE` | nginx upload-size cap on the frontend (default `1g`). |
| `PRELOAD_MODELS` | When `true`, the backend runs `load-models` at startup before `uvicorn`. |

## Session persistence

The backend defaults `SESSIONS_DB_PATH` to
`/var/lib/docint/sessions/sessions.sqlite3` and mounts the
`sessions-storage` volume over it. Override with `SESSION_STORE` in
`.env` to point at a different SQLAlchemy URL (for example, a PostgreSQL
instance) — this wins over `SESSIONS_DB_PATH` when both are set.

## Co-deployment with a standalone vLLM stack

A separate [`vllm-service`](https://github.com/nos-tromo/vllm-service/)
repository ships an OpenAI-compatible router that Docint can consume.
To run both on a single host with a shared network:

1. Create the shared network once:

   ```bash
   docker network create inference-net
   ```

2. Start `vllm-service` attached to that network.

3. In the Docint `.env`, set:

   ```bash
   INFERENCE_PROVIDER=vllm
   OPENAI_API_BASE=http://vllm-router:4000/v1
   OPENAI_API_KEY=sk-no-key-required
   TEXT_MODEL=Qwen/Qwen3.5-2B
   VISION_MODEL=Qwen/Qwen3.5-2B
   EMBED_MODEL=BAAI/bge-m3
   SPARSE_MODEL=BAAI/bge-m3
   RERANK_MODEL=BAAI/bge-reranker-v2-m3
   INFERENCE_NET=inference-net
   ```

4. Start Docint normally:

   ```bash
   make build
   make up
   ```

For a remote vLLM router, drop `INFERENCE_NET` and point
`OPENAI_API_BASE` at the external URL.

## Co-deployment with Ollama

Similar to vLLM, but point at the Ollama OpenAI-compatible endpoint:

```bash
INFERENCE_PROVIDER=ollama
OPENAI_API_BASE=http://ollama:11434/v1
TEXT_MODEL=gpt-oss:20b
VISION_MODEL=qwen3.5:9b
EMBED_MODEL=bge-m3
```

Mount the `ollama-cache` volume on the Ollama container so downloaded
models are shared across Docint and Ollama.

### Embedding throughput with Ollama

CPU-resident embedding models can take minutes per batch. The default tuning
(`EMBED_BATCH_SIZE=16`, `EMBED_TIMEOUT_SECONDS=1800`) is conservative to
minimize timeout risk on slower machines. Adjust based on your hardware and
corpus size:

- **Slow CPU or very large batches**: increase `EMBED_TIMEOUT_SECONDS` or
  reduce `EMBED_BATCH_SIZE`.
- **Fast CPU or small batches**: consider increasing `EMBED_BATCH_SIZE` to
  `32–64` for better throughput, keeping `timeout × (1 + max_retries) ≤ 3600`
  to avoid potential multi-hour stalls.

### Ollama context window — match the serving ceiling

Ollama serves `bge-m3` with `num_ctx=2048` by default, independent of the
model's 8192-token capacity. `/api/show` does not advertise the runtime value,
so Docint cannot probe it — operators must align the two sides explicitly.

Recommended default when serving embeddings through Ollama:
`EMBED_CTX_TOKENS=2048`.
The pre-embed re-splitter bounds every outbound payload to
`int(2048 × EMBED_CTX_SAFETY_MARGIN) = 1945` tokens, which ollama accepts
unconditionally. Ingest completes — slower than an 8192 budget would imply
because payloads are smaller and more numerous, but correct.

**Reclaim the full 8192-token window** with a one-line custom Modelfile. Docint
ships it at `deploy/Modelfile.bge-m3`:

```bash
ollama pull bge-m3
ollama create docint-bge-m3 -f deploy/Modelfile.bge-m3
```

Then in `.env`:

```
EMBED_MODEL=docint-bge-m3
EMBED_CTX_TOKENS=8192
```

Verify with `ollama show docint-bge-m3 --modelfile` — the output must include
`PARAMETER num_ctx 8192`. Any mismatch between the served `num_ctx` and the
configured `EMBED_CTX_TOKENS` surfaces as an `EmbeddingInputTooLongError` at
ingest time; the error message names the mismatch and points back to this
section.

## Distributable image bundles

For air-gapped hosts, customer deployments, or any environment without
Docker Hub access, build a versioned tarball on a connected machine and
copy it across with the `docker/` directory and `.env`.

### Producing the bundle

`make bundle` wraps build → save:

```bash
make bundle
```

The underlying script is `./scripts/bundle_images.sh`.

This computes `DOCINT_VERSION` as `YYYY-MM-DD-<short-sha>` (override by
exporting it before invocation), tags the buildable services with that
version, then writes the gzipped tarball:

| File | Contents |
|---|---|
| `docint-built-<version>.tar.gz` | Locally-built `docint-backend` and `docint-frontend` images — the only tarball `make bundle` produces. Stateful/remote images (Qdrant) live in the `data-plane` project and ship with its bundle, not this one. |

The compose file references the version through
`image: docint-<service>:${DOCINT_VERSION:-latest}`. The `Makefile`
derives and exports `DOCINT_VERSION` (using the same `YYYY-MM-DD-<short-sha>`
scheme), so `make build` and `make bundle` always produce versioned
tags. The `:latest` fallback only kicks in for direct `docker compose
-f docker/compose.yaml build` invocations without `DOCINT_VERSION`
exported.

### Loading and running the bundle

Ship the tarball produced above, the matching `docker/` directory,
and a `.env` file to the target host. Then:

```bash
docker load -i docint-built-<version>.tar.gz
export DOCINT_VERSION=<version>
docker compose --env-file .env -f docker/compose.yaml up --no-build
```

The bundle target host runs the production shape — `docker/compose.yaml`
without the dev override — so no host ports are published.

The version is embedded in the tarball filename, so the operator just
reads it off the file. Verify with `docker images | grep docint` before
`up`.

> `--no-build` does **not** suppress pulls from a registry. If the
> tagged image isn't loaded locally, Compose still tries to resolve it
> against Docker Hub and errors with a DNS / "no such host" failure on
> offline machines. Always `docker load` first.

## Health checks

- Backend readiness — `GET /collections/list` returns `200` once Qdrant
  is reachable. The UI's sidebar health indicator polls this endpoint.
- Qdrant readiness — `GET http://qdrant:6333/` returns a JSON payload.
- Frontend readiness — nginx serves the React SPA at `/` on container
  port `80` (host `${DOCINT_HOST_PORT:-8080}` under `make up-dev`).

## Teardown

```bash
make stop      # stop containers, keep them
make down      # stop + remove containers (never touches data-plane state)
# or drop internal volumes too:
docker compose --env-file .env -f docker/compose.yaml down --volumes
```

Note that the `external` volumes (`docling-cache`, `huggingface-cache`,
`ollama-cache`, `sessions-storage`, `source-preview-cache`) are **not**
removed by `--volumes` — so `down -v` no longer destroys staged source
files or the SQLite session database. Delete them explicitly with
`docker volume rm` if you want a clean slate.
