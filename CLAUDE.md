# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies (torch/torchvision are optional extras)
uv sync --extra cpu   # local dev (CPU torch from pytorch-cpu index)
uv sync --extra cuda  # GPU dev (cu130 torch + fastembed-gpu + onnxruntime-gpu)

# Run tests
uv run pytest
uv run pytest tests/test_rag_unit.py          # single file
uv run pytest tests/test_rag_unit.py::test_fn  # single test

# Lint and format (ruff check, ruff format, mypy)
uv run pre-commit run --all-files

# Start backend (needs Qdrant at localhost:6333 + an inference endpoint)
uv run uvicorn docint.core.api:app --reload

# Start Streamlit UI
uv run docint

# CLI tools
uv run ingest --help
uv run query --help
uv run load-models          # pre-download model assets

# Docker (pick a profile: cpu-ollama, cpu-openai, cpu-vllm, cuda-ollama, cuda-openai, cuda-vllm)
docker compose --profile cpu-ollama up --build
```

## Architecture

Document Intelligence is a RAG stack: FastAPI backend + Streamlit UI + Qdrant vector DB + pluggable inference (Ollama, OpenAI-compatible APIs, or external vLLM).

**Request flow:**
```
Streamlit UI (docint/ui/) → FastAPI (docint/core/api.py) → AgentOrchestrator (docint/agents/)
    → understanding → clarification → retrieval → generation
    → RAG engine (docint/core/rag.py) ↔ Qdrant vector store
```

**Key modules:**
- `docint/core/rag.py` — Core RAG engine: ingestion, retrieval, postprocessing (reranking, parent context, source diversity), collection management
- `docint/core/api.py` — FastAPI app with endpoints for chat, ingestion, collections, citations; streams responses
- `docint/agents/orchestrator.py` — Coordinates understanding, clarification, retrieval, and generation agents
- `docint/core/ingest/ingestion_pipeline.py` — Document processing, chunking, metadata extraction
- `docint/core/readers/documents/` — Page-level PDF pipeline: triage, layout analysis, OCR fallback, extraction, chunking
- `docint/core/readers/json.py` — Generic JSON / JSONL reader. Detects Nextext transcripts (JSONL with `text` plus timing keys `start_ts`/`end_ts` or `start_seconds`/`end_seconds`) and routes them to one-node-per-segment ingestion, mirroring the social-table specialized schema pattern; timing/speaker metadata surface via `reference_metadata`.
- `docint/core/storage/` — Qdrant-backed document store, hierarchical node storage, source tracking
- `docint/core/state/` — Session management (SQLite-backed) and citation handling
- `docint/core/ner.py` — Named entity recognition (GLiNER), entity clustering, graph building
- `docint/utils/embed_chunking.py` — Pre-embed re-chunker: bounds oversize chunks to the embedding budget and links sub-nodes back to their parent via `hier.parent_id`
- `docint/utils/embedding_tokenizer.py` — Loads the embedding model's tokenizer from the HF cache for accurate token counting during pre-embed re-chunking; falls back to char-ratio when unavailable
- `docint/utils/env_cfg.py` — **All** environment-backed configuration dataclasses live here (see below)

## Key Conventions

- **Python ≥3.11, <3.12**. Use `uv` for dependency management (`uv add`/`uv remove` to keep `pyproject.toml` and `uv.lock` in sync).
- **Centralized config**: All `os.getenv` calls and config dataclasses must live in `docint/utils/env_cfg.py`. Other modules import from there. If a subpackage needs a short import path, use a thin re-export module.
- **Test synchronization**: Every functional change must include corresponding test updates. Tests are in `tests/` and use pytest. `conftest.py` provides mock stubs for external dependencies like `magic`.
- **Google-style docstrings** for new/modified functions and classes.
- **Pre-commit is mandatory**: always run `uv run pre-commit run --all-files` before finishing work (ruff check, ruff format, mypy).
- Prefer incremental, focused commits. When changes affect both API and UI, update `README.md`.
- Streamlit UI pages live in `docint/ui/`; keep business logic in the API/agents layer, not in UI code.
- **Hidden collection suffixes**: `docint/core/rag.py` defines `HIDDEN_COLLECTION_SUFFIXES` (currently `("_images", "_dockv")`). `RAG.list_collections()` filters these out, which transitively hides them from `/collections/list` and makes `select_collection()` reject them. Extend the tuple rather than adding filters at the UI layer.
