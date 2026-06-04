# Getting started

This guide takes you from a fresh clone to a running stack that can ingest a
document and answer a question about it. It is intentionally short; every
section links into a deeper reference doc when more detail is needed.

The top-level [`README.md`](../README.md) already covers the installation
basics. This page focuses on the **first run** experience and the
extra context a new user needs to understand what they just started.

## Prerequisites

- **Docker** (Engine + Compose v2) — only if you want the containerised
  quick start.
- **Python 3.11** and the [`uv`](https://github.com/astral-sh/uv) package
  manager — required for local development. docint is CPU-only: there is no
  GPU build and no `cpu`/`cuda` profile, because all ML inference is
  delegated to a remote OpenAI-compatible endpoint.
- **Node 20 + pnpm** — only if you want to run the React frontend dev
  server locally.
- A running **inference backend** reachable from Docint:
  - any OpenAI-compatible endpoint (Ollama, vLLM, the public OpenAI API, …),
  - configured via `INFERENCE_PROVIDER`, `OPENAI_API_BASE`, and
    `OPENAI_API_KEY` (see [configuration.md](configuration.md)).
- **Qdrant** — provided by the sibling `data-plane` project
  (`../data-plane/`), not by this stack. Start it with
  `cd ../data-plane && make up` (Docker) or `make up-dev` (also
  publishes Qdrant on `localhost:6333` for local dev).

## Quick start with Docker

1. **Copy the env file** — this seeds your configuration:

   ```bash
   cp .env.example .env
   ```

   Edit `.env` to point at your inference backend. Two common shapes:

   ```bash
   # Using Ollama
   INFERENCE_PROVIDER=ollama
   OPENAI_API_BASE=http://ollama:11434/v1
   TEXT_MODEL=gpt-oss:20b

   # Using the OpenAI API
   INFERENCE_PROVIDER=openai
   OPENAI_API_BASE=https://api.openai.com/v1
   OPENAI_API_KEY=sk-...
   TEXT_MODEL=gpt-4o
   ```

2. **Create the shared networks and volumes** (idempotent):

   ```bash
   make network    # external inference-net + data-net
   make volumes    # external cache + state volumes
   ```

   `make volumes` creates `docling-cache`, `huggingface-cache`,
   `ollama-cache`, `sessions-storage`, and `source-preview-cache` so model
   artifacts and backend state persist across container rebuilds and are
   not destroyed by `docker compose down -v`.

3. **Build and start the stack:**

   ```bash
   make build
   make up-dev   # publishes the React SPA on the host (dev overlay)
   ```

   docint builds a single CPU-only backend image and a React-SPA frontend
   image — there is no profile to choose. `make up` runs the production
   shape (`docker/compose.yaml` alone, no host ports); `make up-dev` layers
   `docker/compose.override.yaml` to publish the frontend port.

4. **Open the app:**

   - React SPA — <http://localhost:8080> (override the host port with
     `DOCINT_HOST_PORT` in `.env`).

   The backend (`:8000`) is reached only through the frontend's nginx
   sidecar — it is not host-published. Qdrant is not served by this stack;
   it comes from the `data-plane` project (see Prerequisites above).

For the complete deployment reference (services, volumes, networks, vLLM
co-deployment) see [deployment.md](deployment.md).

## Quick start for local development

Use this path when you want to iterate on the Python and/or React code
without going through Docker.

1. **Copy the env file** (same as above):

   ```bash
   cp .env.example .env
   ```

2. **Ensure external services are running locally:**

   - Qdrant at `http://localhost:6333` — start the `data-plane`
     project with `cd ../data-plane && make up-dev`, which publishes
     Qdrant on the host port.
   - An OpenAI-compatible endpoint (Ollama, vLLM, OpenAI, …).

3. **Install Python dependencies:**

   ```bash
   uv sync
   ```

   A single environment — no `--extra` flags. docint ships no GPU code, so
   there is no `cpu`/`cuda` split.

4. **(Optional) pre-download model assets:**

   ```bash
   uv run load-models
   ```

   This populates the Hugging Face, Docling, and fastembed caches so the
   first backend startup does not block on downloads.

5. **Start the backend:**

   ```bash
   uv run uvicorn docint.core.api:app --reload
   # or, without auto-reload: uv run docint
   ```

6. **Start the React frontend in another terminal:**

   ```bash
   cd frontend
   pnpm install
   pnpm dev        # → http://localhost:5173 (proxies API calls to :8000)
   ```

   The Vite dev server proxies `/collections`, `/query`, `/stream_query`,
   `/ingest`, … to the backend on `:8000`, so the SPA and API share an
   origin. See [ui-guide.md](ui-guide.md).

## Your first ingest and query

1. **Create a collection** — the SPA sidebar lets you create and select a
   collection. You can also hit the API directly:

   ```bash
   curl -X POST http://localhost:8000/collections/select \
        -H 'Content-Type: application/json' \
        -d '{"name": "demo"}'
   ```

2. **Ingest a file** — from the SPA **Ingest** page (see
   [ui-guide.md](ui-guide.md)), drop a PDF, or use the CLI:

   ```bash
   uv run ingest
   # prompts for a collection name, then reads files from the directory
   # referenced by the DATA_PATH env var.
   ```

3. **Ask a question** — from the SPA **Chat** page, or via the API:

   ```bash
   curl -X POST http://localhost:8000/query \
        -H 'Content-Type: application/json' \
        -d '{"question": "What is this document about?", "retrieval_mode": "stateless"}'
   ```

   The response is a `QueryOut` payload — see
   [api-reference.md](api-reference.md) for the full schema.

## Where to go next

- **Architecture overview:** [architecture.md](architecture.md)
- **All configuration knobs:** [configuration.md](configuration.md)
- **Full API reference:** [api-reference.md](api-reference.md)
- **Deployment & Docker:** [deployment.md](deployment.md)
- **Development workflow:** [development.md](development.md)
