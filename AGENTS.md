# Agent Guidelines

## Scope

These instructions apply to the entire `docint` repository unless a directory introduces its own `AGENTS.md` file with more specific guidance.

## Project overview

- **Monorepo layout** — all Python code lives under `docint/` (there is no separate `backend/` or `frontend/` directory). The single `pyproject.toml` at the project root describes the package and its dependencies.
- The package exposes a **FastAPI** application (`docint/core/api.py`), **CLI** entry points (`ingest`, `query`, `load-models`), a **Streamlit** UI (`docint/app.py`), and an agentic orchestration layer (`docint/agents/`).
- Docker and `docker-compose.yml` are provided for running the full stack; during development we usually run the services directly (see `README.md`).

### Key directories

| Path | Purpose |
|------|---------|
| `docint/core/` | RAG engine (`rag.py`), FastAPI app (`api.py`), ingestion pipeline, readers, storage, and conversation state |
| `docint/core/readers/documents/` | Page-level PDF processing pipeline: triage, layout analysis, OCR fallback, extraction, chunking, artifact persistence |
| `docint/core/ingest/` | Ingestion orchestration (`ingestion_pipeline.py`) and shared image service (`images_service.py`) |
| `docint/core/storage/` | Document store, hierarchical node storage, source tracking |
| `docint/core/state/` | Conversation session management and citation handling |
| `docint/agents/` | Agentic orchestration: clarification, retrieval, generation, tool use |
| `docint/cli/` | CLI entry points for `ingest` and `query` |
| `docint/ui/` | Streamlit page modules (chat, dashboard, sidebar, theme, etc.) |
| `docint/utils/` | Shared utilities: centralized env config, model management, hashing, NER, MIME detection, prompts |
| `tests/` | Unit tests (run with `uv run pytest` from the project root) |

## General expectations

- Prefer incremental, focused changes and keep commits logically scoped.
- **Test synchronization**: Every modification to functional code must be accompanied by corresponding updates to the test suite. Ensure that new features are covered by tests and that existing tests are updated to reflect behavioral changes.
- Preserve or improve existing typing, docstrings, and inline documentation. When adding new functions or classes, include type hints and docstrings that explain non-obvious behavior.
- Use Google-style docstrings consistently when adding or modifying docstrings.
- When modifying behavior that affects both API and UI, document the change in `README.md` or other relevant docs so the manual workflow remains accurate.

## Python / FastAPI

- The project targets **Python ≥ 3.11, < 3.12** (see `requires-python` in `pyproject.toml`). Use 3.11+ features sparingly and only when they are already present elsewhere (e.g., structural pattern matching, `list`/`dict` type hints).
- Dependency management is handled via [`uv`](https://github.com/astral-sh/uv). If you must add or update dependencies, run the change through `uv add`/`uv remove` so both `pyproject.toml` and `uv.lock` stay in sync.
- Follow the existing import style: organize imports by standard library, third party, and local modules. Avoid wildcard imports; prefer explicit exceptions over broad `except Exception` unless the surrounding code already requires it.
- **Centralized environment configuration**: All environment-variable-backed configuration dataclasses and their `load_*` factory functions must live in `docint/utils/env_cfg.py`. Do **not** introduce new config dataclasses or `os.getenv` calls in other modules. Other modules that need those configs should import them from `docint.utils.env_cfg`. When a package needs a convenient short import path (e.g. `docint.core.readers.documents.config`), use a thin re-export module that imports from `env_cfg`.

## Streamlit UI

- The Streamlit app entry point is `docint/app.py`; individual pages live in `docint/ui/`.
- Keep interactions minimal and prefer calling FastAPI endpoints or core library functions for business logic.
- Maintain existing accessibility cues and avoid embedding heavy logic in the UI; move logic to the API/agents where practical.
- When changing Streamlit UI flows, ensure the corresponding FastAPI/agent endpoints support the interaction and add/adjust tests when feasible.

## Testing and verification checklist

- Always finish a task by running the relevant tests and the full pre-commit suite:

  ```bash
  uv run pytest
  uv run pre-commit run --all-files   # ruff check, ruff format, mypy
  ```

- Docker changes: if you modify Dockerfiles or `docker-compose.yml`, ensure the relevant service still builds locally (`docker compose build <service>`).

When in doubt, err on the side of clarifying comments or doc updates so future contributors can follow the workflow without digging through history.
