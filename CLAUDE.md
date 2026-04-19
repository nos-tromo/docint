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

## Testing

- Always run the full test suite (`pytest`) after making changes and report pass/fail counts.
- When tests fail, fix the root cause rather than patching tests to match stale/removed code.
- Verify with `pre-commit run --all-files` (mypy, lint, docstrings) before declaring work complete.

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
- `docint/core/readers/audio.py` — Audio transcription via Whisper
- `docint/core/storage/` — Qdrant-backed document store, hierarchical node storage, source tracking
- `docint/core/state/` — Session management (SQLite-backed) and citation handling
- `docint/core/ner.py` — Named entity recognition (GLiNER), entity clustering, graph building
- `docint/utils/env_cfg.py` — **All** environment-backed configuration dataclasses live here (see below)

## Project Context

- WhisperX has been removed from this project; use openai-whisper + pyannote.
- Target torch install is split by extras (cpu/cuda) via conflicts in pyproject.toml.
- Docker base image is pinned to `python:3.11.12-slim-bookworm` across all Dockerfiles.

## Key Conventions

- **Python ≥3.11, <3.12**. Use `uv` for dependency management (`uv add`/`uv remove` to keep `pyproject.toml` and `uv.lock` in sync).
- **Centralized config**: All `os.getenv` calls and config dataclasses must live in `docint/utils/env_cfg.py`. Other modules import from there. If a subpackage needs a short import path, use a thin re-export module.
- **Test synchronization**: Every functional change must include corresponding test updates. Tests are in `tests/` and use pytest. `conftest.py` provides mock stubs for `magic` and `whisper` modules.
- **Google-style docstrings** for new/modified functions and classes.
- **Pre-commit is mandatory**: always run `uv run pre-commit run --all-files` before finishing work (ruff check, ruff format, mypy).
- Prefer incremental, focused commits. When changes affect both API and UI, update `README.md`.
- Streamlit UI pages live in `docint/ui/`; keep business logic in the API/agents layer, not in UI code.

## Docstrings & Style

- All new/modified Python functions must have Google-style docstrings.
- Python 3.11 is the target; prefer explicit types and distinct variable names across branches to satisfy mypy.

## Commits

- Prefer multiple small topical commits over a single catch-all commit.
- Each commit message should describe a single logical change (refactor, fix, feat, docs, test).
