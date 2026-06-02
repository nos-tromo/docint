# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies (single env, no extras — docint is CPU-only Python;
# all ML inference is delegated to vllm-service over HTTP).
uv sync

# Run tests
uv run pytest
uv run pytest tests/test_rag_unit.py          # single file
uv run pytest tests/test_rag_unit.py::test_fn  # single test

# Lint and format (ruff check, ruff format, mypy)
uv run pre-commit run --all-files

# Start backend (needs Qdrant + an inference endpoint reachable).
# Qdrant comes from the sibling data-plane project: `cd ../data-plane && make up-dev`
uv run uvicorn docint.core.api:app --reload

# Start frontend (Vite dev server)
cd frontend && pnpm install
cd frontend && pnpm dev
cd frontend && pnpm test
cd frontend && pnpm build

# CLI tools
uv run ingest --help
uv run query --help
uv run resolve              # merge duplicate/similar entities for a collection
uv run load-models          # pre-download model assets

# Docker — single CPU image, no profile toggle.
make network   # create the external inference-net + data-net (one-time)
make volumes   # create the external Docker volumes (one-time)
make up        # build + run docint (production shape, no host ports)
make up-dev    # like 'up', but publishes the React SPA on the host
```

## Architecture

Document Intelligence is a RAG stack: FastAPI backend + React SPA + Qdrant vector DB + pluggable inference (Ollama, OpenAI-compatible APIs, or external vLLM).

**All ML inference is remote.** docint ships no GPU code and no local model runtime: chat/embeddings go through the OpenAI-compatible API, reranking through `{RERANK_API_BASE}/rerank`, NER through `{NER_API_BASE}/gliner`, and CLIP image+text embedding through `{CLIP_API_BASE}/clip/*`. All four default to the LiteLLM router alias of the full vllm-service stack; standalone CPU profiles (`ner-only`, `rerank-only`, `clip-only`) live in `vllm-service/docker/compose.*-only.yaml` and let non-CUDA dev hosts override the relevant `*_API_BASE` independently. The runtime container is a single Debian-slim image (no CUDA, no `[cuda]` extra).

**Request flow:**
```
React SPA (frontend/) → FastAPI (docint/core/api.py) → AgentOrchestrator (docint/agents/)
    → understanding → clarification → retrieval → generation
    → RAG engine (docint/core/rag.py) ↔ Qdrant vector store
```

**Key modules:**
- `docint/core/rag.py` — Core RAG engine: ingestion, retrieval, postprocessing (reranking, parent context, source diversity), collection management
- `docint/core/api.py` — FastAPI app with endpoints for chat, ingestion, collections, citations; streams responses
- `docint/agents/orchestrator.py` — Coordinates understanding, clarification, retrieval, and generation agents
- `docint/core/ingest/ingestion_pipeline.py` — Document processing, chunking, metadata extraction
- `docint/core/readers/documents/` — Page-level PDF pipeline: triage, layout analysis, OCR fallback, extraction, chunking
- `docint/core/readers/docx.py` — Word `.docx` reader. Converts via a DOCX-scoped docling `DocumentConverter` (pure-XML `SimplePipeline`, no models or network — airgap-safe) into compact Docling-JSON, so a docx flows through the already-wired `DoclingNodeParser` like a PDF (Markdown fallback if JSON export fails; skips rather than emitting raw bytes if conversion fails). Registered in `ingestion_pipeline.py`'s `file_extractor`, mirroring `rtf.py` — a binary type with no registered extractor gets silently decoded as UTF-8 (the bug this fixed).
- `docint/core/readers/json.py` — Generic JSON / JSONL reader. Detects Nextext transcripts (JSONL with `text` plus timing keys `start_ts`/`end_ts` or `start_seconds`/`end_seconds`) and routes them to one-node-per-segment ingestion, mirroring the social-table specialized schema pattern; timing/speaker metadata surface via `reference_metadata`.
- `docint/core/storage/` — Qdrant-backed document store, hierarchical node storage, source tracking
- `docint/core/state/` — Session management (SQLite-backed) and citation handling
- `docint/core/ner.py` — Entity aggregation / clustering / graph building over already-extracted NER metadata (pure post-processing; no model inference). Merge modes: `exact` (case-insensitive), `orthographic` (alphanumeric-compacted, the default — already collapses `Africa`/`africa`/`Acme Corp`), and `resolved` (groups by durable canonical entity id from the resolution store, falling back to orthographic for unresolved surfaces).
- `docint/core/entities/` — **Entity resolution** (chorus parity), the only way to merge *semantically* similar entities (`USA`/`United States`, `EU`/`European Union`). `resolution.py` is the pure, dependency-injected pipeline (normalize → exact alias → type-blocked vector match ≥ `RES_EMBED_THRESHOLD` → conservative LLM tie-break → mint), mirroring chorus's `ingestion/resolution.py`. `store.py` (`EntityStore`) persists one point per canonical entity in the hidden `{collection}_entities` Qdrant companion (vector = name embedding; payload = `canonical_name`, `type`, `aliases`). Triggered by `RAG.resolve_entities()` (re-runnable, idempotent) via the `resolve` CLI or `POST /collections/entities/resolve`; reuses the existing remote embedding + chat clients (no new model runtime). Tuned by `RES_EMBED_THRESHOLD` (0.86), `RES_LLM_TIEBREAK` (true), `RES_CASE_NORMALIZE` (true), `RES_VECTOR_K` (5) in `env_cfg.py`. Tie-break prompt: `prompts/{en,de}/entity_tiebreak.txt`.
- `docint/utils/ner_client.py` — Thin HTTP client for the remote GLiNER service hosted by `vllm-service` (full stack: `http://vllm-router:4000/gliner` with Bearer auth; ner-only shape: `http://gliner-ner:8000/gliner` with no auth). Replaces the in-process GLiNER runtime previously shipped here.
- `docint/utils/clip_client.py` — Thin HTTP client for the remote CLIP image+text embedding service hosted by `vllm-service`. Same dual-shape posture as the NER client (full stack via router with Bearer auth; `clip-only` shape at `http://clip-embed:8000` with no auth). `RemoteCLIPBackend` satisfies the `ImageEmbeddingBackend` Protocol so `core/ingest/images_service.py` swaps in place. Probes `/clip/dimension` at construction to size Qdrant `_images` collections without burning an embed call. `IMAGE_EMBED_MODEL` is no longer read by docint — set `CLIP_MODEL` on the vllm-service container instead. Override the endpoint via `CLIP_API_BASE` / `CLIP_API_KEY` / `CLIP_TIMEOUT`.
- **Reranking is always remote.** `core/rag.py::RAG.reranker` builds a `VLLMRerankPostprocessor` that POSTs to `{RERANK_API_BASE}/rerank` in the Jina shape (`{model, query, documents, top_n}` → `{results: [{index, relevance_score}]}`) regardless of `INFERENCE_PROVIDER`. Defaults inherit from `OPENAI_API_BASE` / `OPENAI_API_KEY` / `OPENAI_TIMEOUT`; override per-knob with `RERANK_API_BASE` / `RERANK_API_KEY` / `RERANK_TIMEOUT`. The full vllm-service stack exposes `/v1/rerank` via the LiteLLM router; the `rerank-only` deployment shape (CPU container, pairs with `ner-only` for non-CUDA dev) expects `RERANK_API_BASE=http://rerank-cpu:8000`. Transport failure (endpoint unreachable, malformed payload) degrades to original retrieval order (top_n unranked) — no crash, no local fallback model.
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
- Frontend lives in `frontend/`. Keep business logic in the API/agents layer. Frontend dev: pnpm; tests: Vitest.
- **Hidden collection suffixes**: `docint/core/rag.py` defines `HIDDEN_COLLECTION_SUFFIXES` (currently `("_images", "_dockv", "_entities")`). `RAG.list_collections()` filters these out, which transitively hides them from `/collections/list` and makes `select_collection()` reject them. Extend the tuple rather than adding filters at the UI layer. The companions share the base collection's lifecycle — `delete_collection` and the empty-ingestion cleanup remove `{name}_images`/`{name}_entities` alongside it.
- **Locale-aware prompts**: every LLM prompt template lives under `docint/utils/prompts/{en,de}/<name>.txt`. The active locale is `load_language_env().code` (env var `RESPONSE_LANGUAGE`, default `en`). Adding a new prompt = add the file in both `en/` and `de/`. Adding a new locale = create the subdir and translate all 14 files. Non-prompt user-facing strings live in `docint/utils/ui_strings.py` and follow the same env var. JSON output schemas, intent labels, and hate-speech `category` enum stay English in every locale — they are protocol, not prose.
