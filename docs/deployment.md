# Deployment

Docint ships a `docker-compose.yml` and three Dockerfiles covering CPU
and CUDA backends plus a dedicated frontend image. This document
explains the profiles, services, volumes, and networks, and the two
supported co-deployment patterns with external inference services.

## Files

| File | Role |
|---|---|
| `docker-compose.yml` | Services, profiles, volumes, networks. |
| `Dockerfile.backend.cpu` | Multi-stage backend image with CPU PyTorch. |
| `Dockerfile.backend.cuda` | Multi-stage backend image with CUDA PyTorch. |
| `Dockerfile.frontend` | Lightweight Streamlit image. |
| `scripts/create_docker_volumes.sh` | Creates the external cache volumes (idempotent). |
| `.env.example` | Canonical `.env` template. |

## Profiles

`docker-compose.yml` uses Compose profiles to select the backend
variant and the matching Qdrant image.

| Profile | Services enabled | Qdrant image |
|---|---|---|
| `cpu`  | `backend-cpu`, `qdrant`, `frontend`   | `qdrant/qdrant:v1.17.0` |
| `cuda` | `backend-cuda`, `qdrant-cuda`, `frontend` | `qdrant/qdrant:v1.17.0-gpu-nvidia` |

Start the stack with:

```bash
docker compose --profile cpu up --build
# or
docker compose --profile cuda up --build
```

## Services

### `backend-cpu` / `backend-cuda`

- Build from `Dockerfile.backend.cpu` or `Dockerfile.backend.cuda`.
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
- Both services attach to the `docint-net` network and to
  `inference-net` with the alias `docint-backend` so co-deployed
  services can reach them.

### `frontend`

- Built from `Dockerfile.frontend` (Streamlit + `loguru`, `pandas`,
  `requests`).
- Runs `streamlit run docint/app.py --server.address 0.0.0.0` on
  port `8501`.
- Environment: `BACKEND_HOST=http://backend:8000`,
  `BACKEND_PUBLIC_HOST=http://localhost:8000`,
  `STREAMLIT_BROWSER_GATHER_USAGE_STATS=false`,
  `LOG_PATH=/var/log/docint/frontend.log`.
- Mounts the `docint-logs` volume.
- Attaches to `docint-net` only.

### `qdrant` / `qdrant-cuda`

- Publishes ports `6333` (REST) and `6334` (gRPC).
- Persists data to the `qdrant-storage` and `qdrant-snapshots` volumes.
- Disables telemetry via `QDRANT__TELEMETRY_DISABLED=true`.
- The CUDA variant enables `QDRANT__GPU__INDEXING=1` and maps NVIDIA
  devices.

## Volumes

All backend caches are declared `external: true` so model artifacts
survive container rebuilds. The helper script
`./scripts/create_docker_volumes.sh` creates them idempotently.

| Volume | Scope | Purpose |
|---|---|---|
| `docint-logs` | internal | Shared log directory (`/var/log/docint`). |
| `docling-cache` | external | Docling model cache. |
| `huggingface-cache` | external | HF Hub cache (embedding, reranker, NER, image models). |
| `ollama-cache` | external | Ollama model cache, used when Ollama is co-deployed. |
| `whisper-cache` | external | Whisper model cache. |
| `qdrant-storage` | internal | Qdrant's own storage directory. |
| `qdrant-snapshots` | internal | Qdrant snapshot directory. |
| `qdrant-sources` | internal | Raw source files staged for `/sources/preview`. |
| `sessions-storage` | internal | SQLite session database. |

## Networks

- `docint-net` — internal bridge network shared by `backend-*`,
  `frontend`, and the Qdrant service.
- `inference-net` — **external** network declared with
  `name: ${INFERENCE_NET:-inference-net}`. The backend attaches to it
  with the alias `docint-backend`. Create the network first if you
  want to share it with an external inference stack:

  ```bash
  docker network create inference-net
  ```

## Dockerfiles

### `Dockerfile.backend.cpu` / `Dockerfile.backend.cuda`

Multi-stage builds:

1. **Builder stage** — starts from
   `${PYTHON_SLIM_BOOKWORM_IMAGE:-python:3.11.12-slim-bookworm}` (or
   `${NVIDIA_CUDA_RUNTIME_IMAGE}` for the CUDA variant), installs
   system dependencies (`ffmpeg`, `libmagic1`, `libgl1`), copies
   `pyproject.toml` / `uv.lock`, and runs `uv sync` with the matching
   extra.
2. **Runtime stage** — copies the resolved virtualenv, copies the app
   source, and sets `CMD` to
   `uvicorn docint.core.api:app --host 0.0.0.0 --port 8000`.
3. **Optional model preload** — when `PRELOAD_MODELS=true` is passed
   as a build arg, the builder runs `uv run load-models` so the final
   image embeds the HF / Docling / Whisper caches.

### `Dockerfile.frontend`

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
   WHISPER_MODEL=openai/whisper-large-v3-turbo
   INFERENCE_NET=inference-net
   ```

4. Start Docint normally:

   ```bash
   docker compose --profile cuda up --build
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

## Health checks

- Backend readiness — `GET /collections/list` returns `200` once Qdrant
  is reachable. The UI's sidebar health indicator polls this endpoint.
- Qdrant readiness — `GET http://qdrant:6333/` returns a JSON payload.
- Frontend readiness — Streamlit exposes `/healthz` on port 8501.

## Teardown

```bash
docker compose --profile cpu down          # stop and remove containers
docker compose --profile cpu down --volumes # also remove internal volumes
```

Note that `external` volumes (`docling-cache`, `huggingface-cache`,
`ollama-cache`, `whisper-cache`) are **not** removed by `--volumes` —
delete them explicitly with `docker volume rm` if you want a clean
slate.
