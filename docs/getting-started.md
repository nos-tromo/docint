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
  manager — required for local development.
- A running **inference backend** reachable from Docint:
  - any OpenAI-compatible endpoint (Ollama, vLLM, the public OpenAI API, …),
  - configured via `INFERENCE_PROVIDER`, `OPENAI_API_BASE`, and
    `OPENAI_API_KEY` (see [configuration.md](configuration.md)).
- **Qdrant** — bundled in the `docker compose` profiles, or installed
  separately for local dev.

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

2. **Create the shared cache volumes** (idempotent):

   ```bash
   ./scripts/create_docker_volumes.sh
   ```

   This creates the external volumes `docling-cache`, `huggingface-cache`,
   and `ollama-cache` so model artifacts persist across container
   rebuilds.

3. **Pick a profile and start the stack:**

   | Profile | When to use it |
   |---|---|
   | `cpu`  | Development laptops, CI, CPU-only servers. |
   | `cuda` | Machines with an NVIDIA GPU + NVIDIA Container Toolkit. |

   ```bash
   docker compose --profile cpu up --build
   ```

4. **Open the app:**

   - Streamlit UI — <http://localhost:8501>
   - FastAPI backend — <http://localhost:8000>
   - Qdrant REST — <http://localhost:6333>

For the complete deployment reference (services, volumes, networks, vLLM
co-deployment) see [deployment.md](deployment.md).

## Quick start for local development

Use this path when you want to iterate on the Python code without going
through Docker.

1. **Copy the env file** (same as above):

   ```bash
   cp .env.example .env
   ```

2. **Ensure external services are running locally:**

   - Qdrant at `http://localhost:6333`.
   - An OpenAI-compatible endpoint (Ollama, vLLM, OpenAI, …).

3. **Install dependencies:**

   ```bash
   uv sync --extra cpu   # or --extra cuda on a GPU host
   ```

   The `cpu` / `cuda` extras are mutually exclusive; they pin the matching
   `torch` / `torchvision` index.

4. **(Optional) pre-download model assets:**

   ```bash
   uv run load-models
   ```

   This populates the Hugging Face, Docling, and fastembed caches so the
   first backend startup does not block on downloads.

5. **Start the backend:**

   ```bash
   uv run uvicorn docint.core.api:app --reload
   ```

6. **Start the Streamlit UI in another terminal:**

   ```bash
   uv run docint
   ```

## Your first ingest and query

1. **Create a collection** — the UI sidebar (`docint/ui/sidebar.py`) lets
   you create a new collection. You can also hit the API directly:

   ```bash
   curl -X POST http://localhost:8000/collections/select \
        -H 'Content-Type: application/json' \
        -d '{"name": "demo"}'
   ```

2. **Ingest a file** — from the UI **Ingest** page (see
   [ui-guide.md](ui-guide.md)), drop a PDF or use the CLI:

   ```bash
   uv run ingest
   # prompts for a collection name, then reads files from the directory
   # referenced by the DATA_PATH env var.
   ```

3. **Ask a question** — from the UI **Chat** page, or via the API:

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
