# Agent Guidelines

## Scope

These instructions apply to the entire `docint` repository unless a directory introduces its own `AGENTS.md` file with more specific guidance.

## Project overview

- The codebase lives under `docint/` and exposes a FastAPI application, ingestion utilities, a lightweight RAG implementation, and an agentic orchestration layer.
- The UI is a Streamlit app (served via `uv run docint` or `uv run streamlit run docint/app.py`).
- Docker and docker-compose files are provided for running the stack end-to-end; during development we usually run the services directly (see `README.md`).

## General expectations

- Prefer incremental, focused changes and keep commits logically scoped.
- **Test Synchronization**: Every modification to functional code must be accompanied by corresponding updates to the test suite. Ensure that new features are covered by tests and that existing tests are updated to reflect behavioral changes.
- Preserve or improve existing typing, docstrings, and inline documentation. When adding new functions or classes, include type hints and docstrings that explain non-obvious behavior.
- Use Google-style docstrings consistently when adding or modifying docstrings.
- When modifying behavior that affects both backend and frontend, document the change in `README.md` or other relevant docs so the manual workflow remains accurate.

## Backend (Python / FastAPI)

- Use Python 3.11+ features sparingly and only when they are already present elsewhere (e.g., structural pattern matching, `list`/`dict` type hints). Keep compatibility with the pinned `requires-python` range in `backend/pyproject.toml`.
- Dependency management is handled via [`uv`](https://github.com/astral-sh/uv). If you must add or update dependencies, run the change through `uv add`/`uv remove` so both `pyproject.toml` and `uv.lock` stay in sync.
- Run unit tests with `uv run pytest` from the `backend/` directory. Add or update tests alongside backend changes when possible.
- Follow the existing style in `backend/docint/`: organize imports by standard library, third party, and local modules; avoid wildcard imports; prefer explicit exceptions over broad `except Exception` unless the surrounding code already requires it.

## Frontend (Streamlit)

- The Streamlit app entry point is `docint/app.py`; keep interactions minimal and prefer calling FastAPI endpoints for business logic.
- Maintain existing accessibility cues and avoid embedding heavy logic in the UI; move logic to the API/agents where practical.
- When changing Streamlit UI flows, ensure the corresponding FastAPI/agent endpoints support the interaction and add/adjust tests when feasible.

## Testing and verification checklist

- Always finish a task by running the relevant tests and the full pre-commit suite (`uv run pytest` + `uv run pre-commit run --all-files` to cover ruff and mypy).
- Backend changes: `cd backend && uv run pytest`.
- Frontend changes: `cd frontend && npm run lint` (and `npm run build` when touching build-affecting files).
- Docker changes: if you modify Dockerfiles or `docker-compose.yml`, ensure the relevant service still builds locally (`docker compose build <service>`).

When in doubt, err on the side of clarifying comments or doc updates so future contributors can follow the workflow without digging through history.
