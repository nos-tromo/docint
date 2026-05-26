# Deployment

Docint ships its Docker assets under `docker/`: a base `compose.yaml`, a
`compose.override.yaml` dev overlay, and three Dockerfiles covering CPU
and CUDA backends plus a dedicated frontend image. This document
explains the profiles, services, volumes, and networks, and the two
supported co-deployment patterns with external inference services.

## Files

| File | Role |
|---|---|
| `docker/compose.yaml` | Services, profiles, volumes, networks — production shape, no host ports published. |
| `docker/compose.override.yaml` | Dev overlay that publishes host ports; layered automatically by `make up`. |
| `docker/Dockerfile.backend.cpu` | Multi-stage backend image with CPU PyTorch. |
| `docker/Dockerfile.backend.cuda` | Multi-stage backend image with CUDA PyTorch. |
| `docker/Dockerfile.frontend` | Lightweight Streamlit image. |
| `.dockerignore` | Build-context excludes; stays at the repo root so it applies to every image build. |
| `scripts/create_docker_volumes.sh` | Creates the external cache and state volumes (idempotent). |
| `scripts/bundle_images.sh` | Builds and packages versioned image tarballs for offline distribution. |
| `Makefile` | The blessed entry point — convenience targets wrapping the above (`make network`, `make volumes`, `make build`, `make up`, `make stop`, `make bundle`). |
| `.env.example` | Canonical `.env` template. |

The `Makefile` is the entry point for every Docker workflow: it points
`docker compose` at `docker/compose.yaml` for you. A bare `docker compose`
run from the repo root no longer finds the compose file. When a raw
command is needed, the equivalent form is `docker compose --env-file .env
-f docker/compose.yaml [-f docker/compose.override.yaml] [--profile <p>]
<cmd>`.

## Profiles

`docker/compose.yaml` uses Compose profiles to select the backend and
frontend hardware variant.

| Profile | Services enabled |
|---|---|
| `cpu`  | `backend-cpu`, `frontend-cpu`   |
| `cuda` | `backend-cuda`, `frontend-cuda` |

The profile is read from `PROFILE` in `.env` (default `cpu`), so the
`make` targets follow the host's hardware. Override per-invocation on
the command line:

```bash
make up               # uses PROFILE from .env
make up PROFILE=cuda  # override
```

`make up` builds and runs the active profile, layering
`docker/compose.override.yaml` so host ports are published for local
development. The underlying invocation is `docker compose --env-file .env
-f docker/compose.yaml -f docker/compose.override.yaml --profile
<cpu|cuda> up`. The base `docker/compose.yaml` on its own is the
production shape and publishes no host ports.

> Qdrant is **not** part of this compose project. It is owned by the
> sibling `data-plane` project (`../data-plane/`), which runs Neo4j and
> Qdrant for the whole nos-tromo stack. See
> [Vector store: the `data-plane` project](#vector-store-the-data-plane-project)
> below.

## Services

### `backend-cpu` / `backend-cuda`

- Build from `docker/Dockerfile.backend.cpu` or
  `docker/Dockerfile.backend.cuda`.
- Publish port `8000`.
- Share the same volume set via the `x-backend-cpu` YAML anchor.
- Receive the following environment variables at runtime (on top of
  whatever is in `.env`):

  | Variable | Value |
  |---|---|
  | `LOG_PATH` | `/var/log/docint/backend-{cpu|cuda}.log` |
  | `QDRANT_HOST` | `http://qdrant:6333` |
  | `QDRANT_SRC_DIR` | `/var/lib/docint/sources` |
  | `SESSIONS_DB_PATH` | `/var/lib/docint/sessions/sessions.sqlite3` |
  | `USE_DEVICE` | `cpu` on the CPU profile, `auto` on CUDA |
  | `NVIDIA_VISIBLE_DEVICES` (CUDA) | `all` |

- The CUDA variant also declares `deploy.resources.reservations.devices`
  for NVIDIA GPUs and requires the NVIDIA Container Toolkit on the host.
- Both services attach to the `docint-net` network, to `inference-net`
  with the alias `docint-backend` so co-deployed services can reach
  them, and to `data-net` so they can reach the `data-plane` project's
  Qdrant at `http://qdrant:6333`.

### `frontend-cpu` / `frontend-cuda`

- Built from `docker/Dockerfile.frontend` (Streamlit + `loguru`, `pandas`,
  `requests`).
- Runs `streamlit run docint/app.py --server.address 0.0.0.0` on
  port `8501` (host port configurable via `DOCINT_HOST_PORT`).
- Environment: `BACKEND_HOST=http://backend:8000`,
  `BACKEND_PUBLIC_HOST=http://localhost:8000`,
  `STREAMLIT_BROWSER_GATHER_USAGE_STATS=false`,
  `LOG_PATH=/var/log/docint/frontend.log`.
- Mounts the `docint-logs` volume.
- Attaches to `docint-net` only.
- `depends_on` the matching backend (`backend-cpu` / `backend-cuda`),
  so Compose starts the backend container first.

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
| `docint-logs` | internal | Shared log directory (`/var/log/docint`). |
| `docling-cache` | external | Docling model cache. |
| `huggingface-cache` | external | HF Hub cache (embedding, reranker, NER, image models). |
| `ollama-cache` | external | Ollama model cache, used when Ollama is co-deployed. |
| `sessions-storage` | external | SQLite session database. |
| `source-preview-cache` | external | Raw source files staged for `/sources/preview`. |

## Networks

- `docint-net` — internal bridge network shared by `backend-*` and
  `frontend-*`.
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

### `docker/Dockerfile.backend.cpu` / `docker/Dockerfile.backend.cuda`

Multi-stage builds:

1. **Builder stage** — starts from
   `${PYTHON_SLIM_BOOKWORM_IMAGE:-python:3.11.12-slim-bookworm}` (or
   `${NVIDIA_CUDA_RUNTIME_IMAGE}` for the CUDA variant), installs
   system dependencies (`libmagic1`, `libgl1`), copies
   `pyproject.toml` / `uv.lock`, and runs `uv sync` with the matching
   extra.
2. **Runtime stage** — copies the resolved virtualenv, copies the app
   source, and sets `CMD` to
   `uvicorn docint.core.api:app --host 0.0.0.0 --port 8000`.
3. **Optional model preload** — when `PRELOAD_MODELS=true` is passed
   as a build arg, the builder runs `uv run load-models` so the final
   image embeds the HF / Docling caches.

### `docker/Dockerfile.frontend`

Single-stage image that installs a minimal Streamlit runtime
(`streamlit`, `loguru`, `pandas`, `requests`, `python-dotenv`). It is
much smaller than the backend images and does not carry any ML
dependencies.

## Environment configuration

All services share `.env` at the repository root via
`env_file: - .env`. Put provider-specific settings there:

```bash
INFERENCE_PROVIDER=vllm
OPENAI_API_BASE=http://vllm-router:9000/v1
OPENAI_API_KEY=sk-no-key-required
TEXT_MODEL=Qwen/Qwen3.5-2B
```

See [configuration.md](configuration.md) for every variable. A handful
of image / proxy overrides are Compose-specific:

| Variable | Purpose |
|---|---|
| `HTTP_PROXY` / `HTTPS_PROXY` / `NO_PROXY` (and lowercase variants) | Forwarded into image builds and runtime containers through the `x-proxy-build-args` anchor. |
| `PYTHON_SLIM_BOOKWORM_IMAGE` | Override the Python base image. |
| `NVIDIA_CUDA_RUNTIME_IMAGE` | Override the CUDA base image. |
| `INFERENCE_NET` | Name of the shared external network. |
| `USE_DEVICE` | Forced to `cpu` on the CPU profile; defaults to `auto` on CUDA. |

## Session persistence

Docker profiles default `SESSIONS_DB_PATH` to
`/var/lib/docint/sessions/sessions.sqlite3` and mount the
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
   OPENAI_API_BASE=http://vllm-router:9000/v1
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
   make up PROFILE=cuda
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

`make bundle` wraps build → pull → re-tag → save for the active profile:

```bash
make bundle               # uses PROFILE from .env
make bundle PROFILE=cuda  # override
```

The underlying script is `./scripts/bundle_images.sh <profile>`.

This computes `DOCINT_VERSION` as `YYYY-MM-DD-<short-sha>` (override by
exporting it before invocation), tags the buildable services with that
version, then writes the gzipped tarballs:

| File | Contents |
|---|---|
| `docint-built-<profile>-<version>.tar.gz` | Locally-built `docint-backend-*` and `docint-frontend-*` images. |
| `docint-pulled-<profile>-<version>.tar.gz` | Externally-hosted images, re-tagged so the `name:tag@digest` references in `docker/compose.yaml` resolve after `docker load`. Only written when the compose project has registry-only images — docint's no longer does (Qdrant moved to the `data-plane` project), so on a standard build this file is omitted. |

The compose file references the version through
`image: docint-<service>:${DOCINT_VERSION:-latest}`. The `Makefile`
derives and exports `DOCINT_VERSION` (using the same `YYYY-MM-DD-<short-sha>`
scheme), so `make build` and `make bundle` always produce versioned
tags. The `:latest` fallback only kicks in for direct `docker compose
-f docker/compose.yaml build` invocations without `DOCINT_VERSION`
exported.

### Loading and running the bundle

Ship the tarball(s) produced above, the matching `docker/` directory,
and a `.env` file to the target host. Then:

```bash
docker load -i docint-built-cpu-<version>.tar.gz
# only if a pulled tarball was produced:
docker load -i docint-pulled-cpu-<version>.tar.gz
export DOCINT_VERSION=<version>
docker compose --env-file .env -f docker/compose.yaml --profile cpu up --no-build
```

The bundle target host runs the production shape — `docker/compose.yaml`
without the dev override — so no host ports are published.

The version is embedded in the tarball filenames, so the operator just
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
- Frontend readiness — Streamlit exposes `/healthz` on port 8501.

## Teardown

```bash
make stop                                  # stop the active profile's containers
# remove containers, or also drop internal volumes:
docker compose --env-file .env -f docker/compose.yaml --profile cpu down
docker compose --env-file .env -f docker/compose.yaml --profile cpu down --volumes
```

Note that the `external` volumes (`docling-cache`, `huggingface-cache`,
`ollama-cache`, `sessions-storage`, `source-preview-cache`) are **not**
removed by `--volumes` — so `down -v` no longer destroys staged source
files or the SQLite session database. Delete them explicitly with
`docker volume rm` if you want a clean slate.
