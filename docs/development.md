# Development workflow

This document covers the day-to-day developer experience for working on
Docint: repo layout, `uv` commands, pre-commit checks, test layout, CI
pipelines, and guidelines for extending common surfaces.

## Repository layout

```
docint/
├── docint/              # Python package (everything lives here)
│   ├── __init__.py
│   ├── app.py           # Streamlit entry point
│   ├── agents/          # Agent orchestration layer
│   ├── cli/             # Console scripts (ingest, query, eval)
│   ├── core/
│   │   ├── api.py       # FastAPI app
│   │   ├── rag.py       # RAG engine
│   │   ├── ner.py       # Named-entity extraction + graph building
│   │   ├── retrieval_filters.py
│   │   ├── ingest/      # Ingestion pipeline + image service
│   │   ├── readers/     # PDF pipeline, images, tables, JSON / Nextext transcripts
│   │   ├── storage/     # Docstore, hierarchical parser, source staging
│   │   └── state/       # SQLAlchemy session model
│   ├── ui/              # Streamlit pages and components
│   └── utils/
│       ├── env_cfg.py   # Centralised env-var configuration
│       └── …
├── tests/               # pytest suite
├── scripts/             # Shell helpers
├── docs/                # This documentation tree
├── docker-compose.yml   # Compose stack
├── Dockerfile.backend.cpu
├── Dockerfile.backend.cuda
├── Dockerfile.frontend
├── pyproject.toml
├── uv.lock
├── README.md
├── AGENTS.md            # Repo-wide agent instructions
└── CLAUDE.md            # Claude Code-specific instructions
```

There is no `backend/` or `frontend/` split — everything is one
package with `pyproject.toml` at the repo root.

## Python + `uv`

Docint pins Python **≥ 3.11, < 3.12**. Dependency management is done
exclusively with [`uv`](https://github.com/astral-sh/uv).

| Command | Purpose |
|---|---|
| `uv sync --extra cpu` | Install deps with the CPU PyTorch index. |
| `uv sync --extra cuda` | Install deps with the CUDA 13.0 PyTorch index. Conflicts with `--extra cpu`. |
| `uv add <package>` | Add a runtime dependency (updates `pyproject.toml` and `uv.lock`). |
| `uv remove <package>` | Remove a runtime dependency. |
| `uv run pytest` | Run the full test suite. |
| `uv run pre-commit run --all-files` | Run ruff, ruff-format, and mypy. |
| `uv run uvicorn docint.core.api:app --reload` | Start the backend locally. |
| `uv run docint` | Start the Streamlit UI. |

The `cpu` and `cuda` extras are declared as conflicting under
`[tool.uv]` in `pyproject.toml`, so `uv` will refuse to mix them.

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

All tests live in `tests/` and run with `uv run pytest`. The
`conftest.py` provides a mock stub for `magic` so the pipeline can be
exercised without its native dependency.

Rough layout:

| Area | Representative files |
|---|---|
| API | `test_api.py` |
| Agents | `test_agents_generation.py`, `test_agents_orchestrator.py` |
| CLI | `test_query_cli.py`, `test_eval_cli.py` |
| Ingestion & readers | `test_ingest.py`, `test_image_ingestion_service.py`, `test_documents_reader.py`, `test_table_reader.py`, `test_json_reader_nextext.py` |
| NER | `test_app_ner.py`, `test_ner_extractor.py` |
| RAG engine | `test_rag_unit.py`, `test_rag_pipeline_default.py`, `test_rag_list.py`, `test_hierarchical.py`, `test_pipeline.py` |
| Storage | `test_docstore.py` |
| Sessions | `test_session_manager.py`, `test_session_manager_collection.py` |
| Streamlit UI | `test_streamlit.py`, `test_ui_analysis.py`, `test_ui_components.py`, `test_ui_chat.py`, `test_ingest_ui.py` |
| Configuration | `test_model_cfg.py`, `test_openai_cfg.py` |
| Utilities | `test_utils_mimetype.py`, `test_utils_misc.py` |
| vLLM profile | `test_vllm_profile_files.py` |

Run the full suite:

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

## Adding a new agent or reader

See:

- [retrieval-and-agents.md](retrieval-and-agents.md) for how the
  orchestrator is composed.
- [ingestion.md](ingestion.md) for how readers plug into the pipeline.

In both cases the rule is the same: put business logic in
`docint/core/` or `docint/agents/`, expose it via `api.py`, then wire
the UI in `docint/ui/`.

## CI workflows

Three GitHub Actions workflows live under `.github/workflows/`:

| File | Triggers | What it runs |
|---|---|---|
| `backend-ci.yml` | Push/PR to `main`/tests. Excludes UI-only paths. | `uv run pytest` (full suite) and a conditional Docker build of `Dockerfile.backend.cpu`. The CUDA image is not built in CI due to runner constraints. |
| `frontend-ci.yml` | Push/PR that touches UI-only paths. | Lightweight install, pytest of `tests/test_streamlit.py`, Docker build of `Dockerfile.frontend`. |
| `codeql.yml` | Scheduled weekly on Sunday plus push to `main`. | CodeQL SAST scan for Python. |

Dependabot is configured via `.github/dependabot.yml` for Python
dependency updates.

## Docker development

- `docker compose --profile cpu up --build` rebuilds the CPU backend
  on the spot.
- `docker compose --profile cpu build backend-cpu` isolates the
  backend rebuild if you only changed Python code.
- Remember to pass proxy variables through `.env` if you sit behind a
  corporate proxy — the Compose file picks them up via the
  `x-proxy-build-args` anchor.

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
