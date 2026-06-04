# CLI reference

Docint ships seven console scripts, all registered in
`pyproject.toml` `[project.scripts]` and installed automatically when
you run `uv sync`.

| Script | Module | Purpose |
|---|---|---|
| `docint` | `docint.cli.serve:main` | Run the FastAPI backend (uvicorn). |
| `ingest` | `docint.cli.ingest:main` | Batch ingest documents into a collection. |
| `resolve` | `docint.cli.resolve:main` | Merge duplicate / semantically-similar entities into durable canonicals. |
| `query` | `docint.cli.query:main` | Run batch chat queries and collection-level exports. |
| `query-eval` | `docint.cli.eval:main` | Corpus retrieval evaluation across retrieval modes. |
| `verify` | `docint.cli.verify:main` | Check Qdrant ↔ docstore consistency (optionally repair). |
| `load-models` | `docint.utils.model_cfg:main` | Pre-download model assets into the local caches. |

All commands respect the environment settings from `.env` /
`docint/utils/env_cfg.py`. See [configuration.md](configuration.md) for
variables referenced below.

## `docint` — run the backend server

```bash
uv run docint
```

Source: `docint/cli/serve.py`. Loads the bind address via
`load_serve_config()` and runs `uvicorn docint.core.api:app` on the
configured host/port (no auto-reload). For local development with
hot-reload, call uvicorn directly:

```bash
uv run uvicorn docint.core.api:app --reload
```

This command runs the **backend only**. The React frontend is served
separately — the Vite dev server (`cd frontend && pnpm dev`) in
development, or the nginx sidecar in Docker. See [ui-guide.md](ui-guide.md).

## `ingest` — batch ingestion

```bash
uv run ingest
```

Source: `docint/cli/ingest.py`. The command:

1. Calls `set_offline_env()` to honour `DOCINT_OFFLINE`.
2. Resolves the data directory from `DATA_PATH` (default `~/docint/data`).
3. **Prompts** interactively for a Qdrant collection name.
4. Runs `RAG.ingest_docs(...)` with `build_query_engine=False` so that
   large reranker and generation models are not loaded on the ingestion
   host.
5. Calls `rag.unload_models()` when done.

Useful env vars:

- `DATA_PATH` — directory that holds the input corpus.
- `INFERENCE_PROVIDER`, `OPENAI_API_BASE`, `EMBED_MODEL` — point the
  embedder at the right backend.
- `NER_ENABLED`, `ENABLE_HATE_SPEECH_DETECTION` — toggle ingestion-time
  extraction.

## `resolve` — entity resolution

```bash
uv run resolve                      # prompts for the collection
uv run resolve --collection demo    # non-interactive
```

Source: `docint/cli/resolve.py`. Merges duplicate and semantically-similar
named entities (e.g. `USA` / `United States`) for a collection into durable
canonical records, so the NER views can group by canonical entity under
`entity_merge_mode=resolved`. Re-runnable and idempotent. Tuned by the
`RES_*` settings in `docint/utils/env_cfg.py` (`RES_EMBED_THRESHOLD`,
`RES_LLM_TIEBREAK`, `RES_CASE_NORMALIZE`, `RES_VECTOR_K`). In Docker the
`make resolve` target runs this in a one-off `backend` container so it
reaches the `qdrant` / `vllm-router` network aliases.

## `query` — batch chat, summaries, exports

Source: `docint/cli/query.py`. The parser (`build_parser()` at
`query.py:40`) accepts:

| Flag | Description |
|---|---|
| `-c NAME`, `--collection NAME` | Use NAME as the collection instead of prompting. |
| `-q [PATH]`, `--query [PATH]` | Run queries from PATH (JSON, JSONL, or one-query-per-line text). When passed without PATH, the default `QUERIES_PATH` file is used. If no file exists, chat queries are skipped. |
| `-s`, `--summary` | Generate a collection summary via the same flow used by the `/summarize` endpoint. |
| `-e`, `--entities` | Export the top 50 entities and their mention counts as a text file. |
| `-h8`, `--hate-speech` | Export flagged hate-speech findings in the same format the frontend uses. |
| `-a`, `--all` | Run chat, summary, entities, and hate-speech exports in one command. |

### Query file formats

`load_queries()` accepts any of:

- **`.json`** — a top-level list of strings or dicts:

  ```json
  [
    "What is this document about?",
    {"query": "Who authored it?", "expected_filenames": ["paper.pdf"]}
  ]
  ```

- **`.jsonl`** — one object (or string) per line.
- **plain text** — one query per non-empty line.

### Output

Results are written to `RESULTS_PATH` (default `~/docint/results`) via
the internal `_store_output()` / `_store_text_output()` /
`_store_csv_output()` helpers (`query.py:167`, `197`, `211`). Chat results
are serialised as JSON, summary/export results as text or CSV as
appropriate.

### Example

```bash
uv run query \
  --collection demo \
  --query ~/docint/queries.jsonl \
  --all
```

## `query-eval` — retrieval evaluation

Source: `docint/cli/eval.py`. Runs each query spec through
`RAG.run_query()` across configured retrieval modes and compares the
retrieved sources to declared expectations.

Query specs can include:

- `expected_filenames: list[str]` — any returned source with a matching
  `filename` counts as a hit.
- `expected_file_hashes: list[str]` — matching `file_hash`.
- `expected_text_ids: list[str]` — matching `reference_metadata.text_id`.

Results are written to `RESULTS_PATH` via `_store_output()`.

### Example

```bash
uv run query-eval
# prompts for the collection, reads queries from QUERIES_PATH,
# compares retrieval results to expectations, writes JSON per run
```

The default query (if no file exists) falls back to the summarisation
prompt at `docint/utils/prompts/summarize.txt`.

## `verify` — docstore consistency

```bash
uv run verify --collection my_collection
uv run verify --collection my_collection --repair
```

Source: `docint/cli/verify.py`. Checks that a collection's Qdrant vector
store and SQLite KV docstore are in sync, reporting drift — KV-only
orphans, Qdrant-only orphans, and broken hierarchical parents. With
`--repair` it removes the KV-only orphans.

## `load-models` — cache pre-population

```bash
uv run load-models
```

Source: `docint/utils/model_cfg.py:main`. Downloads the Docling models
(RapidOCR, layout, table structure, picture classifier, code/formula),
the CLIP image encoder, GLiNER weights, and any Hugging Face models
referenced by `ModelConfig`. Only the assets that are actually needed by
the current `INFERENCE_PROVIDER` are fetched.

Run this once on a fresh machine (or a new cache volume) to avoid
blocking the first backend startup on network downloads. It is also run at
container startup when `PRELOAD_MODELS=true` is set (see the backend
Dockerfile's entrypoint).

## Exit codes and logging

- All commands initialise loguru through `init_logger()`
  (`docint/utils/logger_cfg.py`). Stderr is at `INFO`, rotating file
  logs at `DEBUG` go to `LOG_PATH`.
- `ingest`, `resolve`, `query`, `query-eval`, `verify`, and `load-models`
  return non-zero exit codes on unhandled exceptions. Use the log file for
  diagnostics.
