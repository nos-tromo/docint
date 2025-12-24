# Agent Guidelines

## Scope

These instructions apply to the entire `docint` repository unless a directory introduces its own `AGENTS.md` file with more specific guidance.

## Project overview

- The backend lives in `backend/` and exposes a FastAPI application along with ingestion utilities and a lightweight RAG implementation.
- The frontend lives in `frontend/` and is a Vite/React TypeScript project that uses Chakra UI components.
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

## Frontend (React / TypeScript)

- New components should be written in TypeScript (`.tsx`) and leverage Chakra UI primitives where appropriate. Maintain accessibility attributes already present in neighboring components.
- Keep state management simple (React hooks) unless there is a clear benefit to introducing additional libraries.
- Run `npm run lint` from the `frontend/` directory before submitting changes. For production-oriented changes also ensure `npm run build` succeeds.
- Asset paths and API endpoints are centralized under `frontend/src`; reuse existing helper utilities for network requests instead of duplicating logic.

## Testing and verification checklist

- Backend changes: `cd backend && uv run pytest`.
- Frontend changes: `cd frontend && npm run lint` (and `npm run build` when touching build-affecting files).
- Docker changes: if you modify Dockerfiles or `docker-compose.yml`, ensure the relevant service still builds locally (`docker compose build <service>`).

When in doubt, err on the side of clarifying comments or doc updates so future contributors can follow the workflow without digging through history.
