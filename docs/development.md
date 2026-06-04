# Development workflow

This document covers the day-to-day developer experience for working on
Docint: repo layout, `uv` commands, the React frontend, pre-commit checks,
test layout, CI, and guidelines for extending common surfaces.

## Repository layout

```
docint/
├── docint/              # Python backend package
│   ├── __init__.py
│   ├── agents/          # Agent orchestration layer
│   ├── cli/             # Console scripts (serve, ingest, query, eval, resolve, verify)
│   ├── core/
│   │   ├── api.py       # FastAPI app
│   │   ├── rag.py       # RAG engine
│   │   ├── ner.py       # Entity aggregation + graph building
│   │   ├── entities/    # Entity resolution (canonical store + pipeline)
│   │   ├── retrieval_filters.py
│   │   ├── ingest/      # Ingestion pipeline + image service
│   │   ├── readers/     # PDF pipeline, images, tables, JSON / Nextext transcripts
│   │   ├── storage/     # Docstore, hierarchical parser, source staging
│   │   └── state/       # SQLAlchemy session model
│   └── utils/
│       ├── env_cfg.py   # Centralised env-var configuration
│       └── …
├── frontend/            # React SPA (Vite + TypeScript), served by nginx in prod
│   ├── src/             # routes/, components/, hooks/, api/, stores/, layout/
│   ├── nginx/           # production nginx config
│   └── package.json
├── tests/               # pytest suite (Python backend)
├── scripts/             # Shell helpers
├── docs/                # This documentation tree
├── docker/              # Compose files + Dockerfiles (backend + frontend)
├── pyproject.toml
├── uv.lock
├── README.md
├── AGENTS.md            # Repo-wide agent instructions
└── CLAUDE.md            # Claude Code-specific instructions
```

The Python backend is one package (`docint/`) with `pyproject.toml` at the
repo root. The React SPA is a separate pnpm package under `frontend/` with
its own `package.json` and lockfile.

## Python + `uv`

Docint pins Python **≥ 3.11, < 3.12**. Dependency management is done
exclusively with [`uv`](https://github.com/astral-sh/uv). There are no
optional extras — `uv sync` installs the single environment. (docint is
CPU-only; all ML inference is remote, so there is no `cpu`/`cuda` split.)

| Command | Purpose |
|---|---|
| `uv sync` | Install/refresh the project environment from `uv.lock`. |
| `uv add <package>` | Add a runtime dependency (updates `pyproject.toml` and `uv.lock`). |
| `uv remove <package>` | Remove a runtime dependency. |
| `uv run pytest` | Run the backend test suite. |
| `uv run pre-commit run --all-files` | Run ruff, ruff-format, and mypy. |
| `uv run uvicorn docint.core.api:app --reload` | Start the backend locally with hot-reload. |
| `uv run docint` | Run the backend (uvicorn) via the console script. |

## Frontend (`frontend/`)

The React SPA is a separate pnpm package. Common commands, run inside
`frontend/`:

| Command | Purpose |
|---|---|
| `pnpm install` | Install JS dependencies from `pnpm-lock.yaml`. |
| `pnpm dev` | Vite dev server on `:5173`, proxying API calls to `:8000`. |
| `pnpm build` | Type-check (`tsc -b`) and build to `dist/`. |
| `pnpm test` | Run the Vitest suite. |
| `pnpm lint` | ESLint over `src/`. |

See [ui-guide.md](ui-guide.md) for the app structure (routes, components,
stores, API layer).

## Pre-commit

`.pre-commit-config.yaml` runs three hooks in order:

1. **`ruff check`** — lint.
2. **`ruff format`** — formatter.
3. **`mypy`** — type checker.

Run the suite locally before finishing any change:

```bash
uv run pre-commit run --all-files
```

Pre-commit is **mandatory**. CI will re-run the same hooks and will
block merges if anything is dirty.

## Tests

Backend tests live in `tests/` and run with `uv run pytest`. The
`conftest.py` provides a mock stub for `magic` so the pipeline can be
exercised without its native dependency.

Rough layout:

| Area | Representative files |
|---|---|
| API | `test_api.py` |
| Agents | `test_agents_orchestrator.py`, `test_agents_generation.py`, `test_agents_retrieval.py`, `test_agents_understanding.py`, `test_agents_history.py` |
| CLI | `test_query_cli.py`, `test_eval_cli.py`, `test_resolve_cli.py` |
| Ingestion & readers | `test_ingest.py`, `test_image_ingestion_service.py`, `test_documents_reader.py`, `test_docx_reader.py`, `test_rtf_reader.py`, `test_table_reader.py`, `test_json_reader_nextext.py` |
| NER & entities | `test_app_ner.py`, `test_ner_client.py`, `test_entity_resolution.py`, `test_entity_store.py`, `test_clip_client.py` |
| RAG engine | `test_rag_unit.py`, `test_rag_pipeline_default.py`, `test_rag_list.py`, `test_hierarchical.py`, `test_pipeline.py` |
| Storage & sessions | `test_storage_scroll.py`, `test_sqlite_kvstore.py`, `test_session_manager.py`, `test_session_manager_collection.py` |
| Configuration | `test_env_cfg.py`, `test_model_cfg.py`, `test_openai_cfg.py` |
| Compose guards | `test_vllm_profile_files.py` (asserts the compose ships no bundled vLLM/CUDA services), `test_frontend_proxy_config.py` |
| Utilities | `test_utils_mimetype.py`, `test_utils_misc.py` |

The React SPA has its own Vitest suite next to the source under
`frontend/src` — run it with `cd frontend && pnpm test` (see
[ui-guide.md](ui-guide.md)).

Run the full backend suite:

```bash
uv run pytest
```

Or target a single file / test:

```bash
uv run pytest tests/test_rag_unit.py
uv run pytest tests/test_rag_unit.py::test_parse_query
```

**Rule:** every functional change ships with the matching test update.
The policy is documented in `AGENTS.md` and enforced in review.

## Centralised env configuration

`docint/utils/env_cfg.py` owns every `os.getenv` call in the codebase.
Guidelines:

- Do not call `os.getenv` outside `env_cfg.py`.
- When a module needs a setting, import the appropriate `load_*_env()`
  factory and read the dataclass field.
- If you need a short import path, create a thin re-export module (for
  example `docint/core/readers/documents/config.py`) that imports from
  `env_cfg`.
- New config groups ship as a frozen dataclass plus a
  `load_*_env()` factory. Keep the defaults and the variable names in
  sync with `.env.example` and [configuration.md](configuration.md).

## Adding a new environment variable

1. Pick the dataclass that owns the concern (or add a new one).
2. Add the field to the dataclass with a concrete type.
3. Update `load_*_env()` to read the variable and apply validation.
4. Update `.env.example` with a commented-out example.
5. Update [configuration.md](configuration.md) so the table stays
   complete.
6. Add a test under `tests/test_model_cfg.py` or
   `tests/test_openai_cfg.py` covering the new knob.

## Adding a new FastAPI endpoint

1. Add the route under `docint/core/api.py`, declaring a Pydantic
   request/response model in the same file.
2. Wire the handler to `RAG`, `SessionManager`, or `AgentOrchestrator`
   rather than reaching into Qdrant / SQLite directly.
3. Add a test in `tests/test_api.py`.
4. Update [api-reference.md](api-reference.md).
5. If the SPA consumes it, add a typed wrapper under `frontend/src/api/`
   and (if the dev server needs it) a proxy entry in `vite.config.ts`.

## Adding a new agent or reader

See:

- [retrieval-and-agents.md](retrieval-and-agents.md) for how the
  orchestrator is composed.
- [ingestion.md](ingestion.md) for how readers plug into the pipeline.

In both cases the rule is the same: put business logic in
`docint/core/` or `docint/agents/`, expose it via `api.py`, then wire the
UI in the React app under `frontend/src/` (see [ui-guide.md](ui-guide.md)).

## CI

A single workflow, `.github/workflows/ci.yml`, runs on pushes and pull
requests to `main`. It delegates to the shared reusable pipeline
`nos-tromo/.github/.github/workflows/python-app-ci.yml@v2.3`, which runs:

- **ruff** (lint + format check) and **mypy** (strict) — the same checks
  as pre-commit, so a clean local `pre-commit run --all-files` should pass
  CI.
- **pytest** on Python 3.11 (`uv sync --frozen --group dev`).
- a **Docker image build** to catch packaging regressions.
- a **frontend build** of the React SPA under `frontend/` (pnpm 9.12.0 on
  Node 20).

Dependabot is configured via `.github/dependabot.yml` for dependency
updates.

## Docker development

- `make build` rebuilds the backend and frontend images. `make up` runs
  them in the production shape (no host ports); `make up-dev` layers
  `docker/compose.override.yaml` to publish the frontend port for local
  development.
- To isolate a single-service rebuild after a Python-only change:
  `docker compose --env-file .env -f docker/compose.yaml build backend`.

## Style guide

- **Imports** — stdlib, third party, local; no wildcard imports.
- **Docstrings** — Google-style for new or modified functions/classes.
- **Typing** — full type hints on all functions; use `3.11`
  built-in generics (`list`, `dict`, `tuple`).
- **Error handling** — prefer specific exceptions; avoid broad
  `except Exception` unless the surrounding code already requires it.
- **Emojis** — do not add emojis to source, docs, or UI without an
  explicit request.
- **Commits** — focused, logically scoped, with a short imperative
  summary (for example: `Add parent-context postprocessor`).

See `AGENTS.md` for the full convention reference.
