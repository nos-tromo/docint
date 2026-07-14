# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Data confidentiality — hard rule

**NEVER expose actual production or testing data in any file committed or
pushed to git.** This covers not only file contents but also metadata that
references real data: filenames, file descriptions, social-media account
names or handles, user identifications, sample records, log excerpts, and
screenshots. It applies everywhere git sees — source code, tests, fixtures,
docs, examples, configs, commit messages, and CI files. Use fully synthetic,
invented placeholders instead.

## Commands

```bash
# Install dependencies (single env, no extras — docint is CPU-only Python;
# all ML inference is delegated to vllm-service over HTTP).
uv sync

# Run tests
uv run pytest
uv run pytest tests/test_rag_unit.py          # single file
uv run pytest tests/test_rag_unit.py::test_fn  # single test

# Lint and format (ruff check, ruff format, pyrefly)
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
uv run load-models          # pre-download model assets

# Docker — single CPU image, no profile toggle.
make network   # create the external inference-net + data-net (one-time)
make volumes   # create the external Docker volumes (one-time)
make up        # run docint detached, no build (production shape, no host ports)
make up-dev    # like 'up', but publishes the React SPA on the host (no build)
make dev       # build, then up-dev
make bundle     # airgap image tarball built from the latest annotated release tag (production)
make bundle-dev # airgap tarball of the current working tree (dev/soak)
# Merge duplicate/similar entities for a collection (one-off backend container,
# so it reaches the qdrant/vllm-router aliases — production is Docker-only).
make resolve                    # prompts for the collection name
make resolve COLLECTION=mydocs  # non-interactive
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
- `docint/core/rag.py` — Core RAG engine: ingestion, retrieval, postprocessing (reranking, parent context, source diversity), collection management. The active collection is **per-request**, not a shared singleton: `qdrant_collection` is a property over a `ContextVar` bound by `RAG.collection_scope(physical)`, and `index`/`query_engine` are per-collection thread-safe LRU caches (so concurrent users on different collections don't interfere — see the multi-tenant convention below).
- `docint/core/api.py` — FastAPI app with endpoints for chat, ingestion, collections, citations; streams responses
- `docint/agents/orchestrator.py` — Coordinates understanding, clarification, retrieval, and generation agents
- `docint/core/ingest/ingestion_pipeline.py` — Document processing, chunking, metadata extraction
- `docint/core/ingest/social_linker.py` — Joins a social export's `postings.csv` to its `media.csv` manifest + files (counter-stripped `Media ID`, basename resolution within one flat directory) and routes each linked file to the right backend: still images through `images_service.py` (CLIP); audio/video by delegating per-file Nextext routing to the shared `media_transcribe.py` engine. Every artifact — image embedding, keyframe, transcript segment — is stamped with the parent posting's `posting_uuid`, which `_attach_posting_group` uses to group a post with all its media at retrieval time.
- `docint/core/ingest/media_transcribe.py` — Shared per-file `MediaTranscriber` engine: hash → transcript-cache lookup (the ingest manifest) → bounded-concurrency Nextext round-trip (`NEXTEXT_MAX_CONCURRENCY`, cache misses only) → keyframes to CLIP, transcript to segment `Document`s. Used identically by `social_linker.py` (posting identity, `posting_uuid` link field) and `standalone_media.py` (file-hash identity, no link field) via the path-specific `MediaClip` dataclass; no media decoding or model inference lives here.
- `docint/core/ingest/standalone_media.py` — `StandaloneMediaIngestor`: the standalone audio/video pre-pass, run right after the social linker. Walks the batch tree for audio/video extensions (`MEDIA_FILETYPES`, default `DEFAULT_MEDIA_FILETYPES` in `env_cfg.py`) the linker did not already claim and routes each through `media_transcribe.py`, anchored to the media file's own content hash — no `postings.csv`/`media.csv` required. Automatic whenever `NEXTEXT_API_BASE` is set; a no-op (one-line warning) otherwise. With no posting to link to, transcripts/keyframes retrieve as independent, normally-ranked sources naming the source clip — no `posting_group` cross-modal clustering.
- `docint/core/readers/documents/` — Page-level PDF pipeline: triage, layout analysis, OCR fallback, extraction, chunking
- `docint/core/readers/docx.py` — Word `.docx` reader. Converts via a DOCX-scoped docling `DocumentConverter` (pure-XML `SimplePipeline`, no models or network — airgap-safe) into compact Docling-JSON, so a docx flows through the already-wired `DoclingNodeParser` like a PDF (Markdown fallback if JSON export fails; skips rather than emitting raw bytes if conversion fails). Registered in `ingestion_pipeline.py`'s `file_extractor`, mirroring `rtf.py` — a binary type with no registered extractor gets silently decoded as UTF-8 (the bug this fixed).
- `docint/core/readers/json.py` — Generic JSON / JSONL reader. Detects Nextext transcripts (JSONL with `text` plus timing keys `start_ts`/`end_ts` or `start_seconds`/`end_seconds`) and routes them to one-node-per-segment ingestion, mirroring the social-table specialized schema pattern; timing/speaker metadata surface via `reference_metadata`.
- `docint/core/storage/` — Qdrant-backed document store, hierarchical node storage, source tracking
- `docint/core/state/` — Session management (SQLite-backed) and citation handling
- `docint/core/state/report*.py` — **Report builder**: owner-scoped, server-persisted *curated* reports that let an investigator hand-pick individual chat answers, entity findings (chunk-level, deduped by `chunk_id`), and hate-speech findings instead of exporting a whole collection. `report.py` + `report_item.py` (ORM in the shared session DB; **type-prefixed dedupe keys** — `entity:`/`hate:`/`chat:` — so the same chunk can be distinct evidence under two types while re-adds are no-ops; snapshots are frozen JSON at add-time, immune to re-ingestion). `report_manager.py` (`ReportManager`, mirrors `SessionManager`'s owner-scoped store plumbing; exposed via `RAG.ensure_report_manager()`). `report_render.py` (pure renderers → Markdown / HTML / **PDF via WeasyPrint** / JSON / CSV-bundle; section headings flow through `ui_strings.py`, reuses `csv_stream.py` row builders; `artifact_type`/JSON keys stay English). API: `POST/GET/PATCH/DELETE /reports*` + `GET /reports/{id}/export.{md,html,pdf,json,zip}`. Frontend: the **Report** tab plus an "Add to report" control on each artifact. Entity and hate-speech findings additionally carry an on-demand translation into the snapshot (`translation: {text, target_lang, model}`, frozen at add-time like the rest of the snapshot); every export renders it as an additive **"Machine translation (→ Deutsch)"** block (MD/HTML/PDF) or `translation` column (CSV) beside the original — endonym via `language_endonym()`, never replacing the original — see `docint/utils/translate_client.py`.
- `docint/core/state/collection_ownership.py` + `collection_owner_manager.py` — **Per-user collection ownership** (multi-tenant). `CollectionOwnership` ORM (table `collection_owners` in the shared session DB) is the source of truth for the `(owner, logical_name) ↔ physical_name` mapping; `CollectionOwnerManager` (mirrors `ReportManager`) does `register`/`resolve`/`list_for`/`delete` + legacy backfill to `DOCINT_DEFAULT_IDENTITY`. Physical name = `u{sha256(owner)[:12]}__{logical}` (legacy rows keep the bare name — no Qdrant rename). Exposed via `RAG.ensure_collection_owner_manager()`; the API gate is `_require_owned_collection(logical, principal) -> physical` (404 on cross-owner, like `_get_owned_report`).
- `docint/core/ner.py` — Entity aggregation / clustering / graph building over already-extracted NER metadata (pure post-processing; no model inference). Merge modes: `exact` (case-insensitive), `orthographic` (alphanumeric-compacted, the default — already collapses `Africa`/`africa`/`Acme Corp`), and `resolved` (groups by durable canonical entity id from the resolution store, falling back to orthographic for unresolved surfaces).
- `docint/core/entities/` — **Entity resolution** (chorus parity), the only way to merge *semantically* similar entities (`USA`/`United States`, `EU`/`European Union`). `resolution.py` is the pure, dependency-injected pipeline (normalize → exact alias → type-blocked vector match ≥ `RES_EMBED_THRESHOLD` → conservative LLM tie-break → mint), mirroring chorus's `ingestion/resolution.py`. `store.py` (`EntityStore`) persists one point per canonical entity in the hidden `{collection}_entities` Qdrant companion (vector = name embedding; payload = `canonical_name`, `type`, `aliases`). Triggered by `RAG.resolve_entities()` (re-runnable, idempotent) via the `resolve` CLI or `POST /collections/entities/resolve`; reuses the existing remote embedding + chat clients (no new model runtime). Tuned by `RES_EMBED_THRESHOLD` (0.86), `RES_LLM_TIEBREAK` (true), `RES_CASE_NORMALIZE` (true), `RES_VECTOR_K` (5) in `env_cfg.py`. Tie-break prompt: `prompts/{en,de}/entity_tiebreak.txt`.
- `docint/utils/ner_client.py` — Thin HTTP client for the remote GLiNER service hosted by `vllm-service` (full stack: `http://vllm-router:4000/gliner` with Bearer auth; gliner-only shape: `http://gliner-only:8000/gliner` with no auth). Replaces the in-process GLiNER runtime previously shipped here.
- `docint/utils/clip_client.py` — Thin HTTP client for the remote CLIP image+text embedding service hosted by `vllm-service`. Same dual-shape posture as the NER client (full stack via router with Bearer auth; `clip-only` shape at `http://clip-only:8000` with no auth). `RemoteCLIPBackend` satisfies the `ImageEmbeddingBackend` Protocol so `core/ingest/images_service.py` swaps in place. Probes `/clip/dimension` at construction to size Qdrant `_images` collections without burning an embed call. `IMAGE_EMBED_MODEL` is no longer read by docint — set `CLIP_MODEL` on the vllm-service container instead. Override the endpoint via `CLIP_API_BASE` / `CLIP_API_KEY` / `CLIP_TIMEOUT`.
- `docint/utils/translate_client.py` — Thin, fail-soft wrapper for **on-demand, display-time** snippet translation: nothing ingested or stored is ever translated, and the original is always preserved. Reuses the shared chat model (`OpenAIPipeline.call_chat`) over the same LiteLLM router endpoint as chat — no dedicated translation runtime, no `TRANSLATE_API_BASE` knob. `translate(text) -> TranslateResult` never raises: a transport/model failure degrades to `ok=False` + an `error` token so the caller just keeps showing the original — no crash, no local fallback model. Successful translations are LRU-cached; failures are not. Destination is always the active locale (`RESPONSE_LANGUAGE`); a distinct target-language override is deferred. Model: `TRANSLATE_MODEL` in `env_cfg.py` (defaults to the chat model `TEXT_MODEL`). Locale prompt: `prompts/{en,de}/translate.txt`. API: `POST /translate` (`{text}` → `{ok, translation, model, target_lang, error}`; principal-authenticated but not collection-scoped — it translates text the caller already holds, so there's nothing to leak and no store re-fetch). Frontend: a hover/focus-revealed **Translate** control (`TranslateControl`, composing `@infra/ui`'s `HoverIconAction`) on chat source citations, entity findings, and hate-speech rows — in-app copy is "Translate"/"Translation"; report exports use the fuller "Machine translation" label (see the report-builder bullet above for the additive carry).
- **Reranking is always remote.** `core/rag.py::RAG.reranker` builds a `VLLMRerankPostprocessor` that POSTs to `{RERANK_API_BASE}/rerank` in the Jina shape (`{model, query, documents, top_n}` → `{results: [{index, relevance_score}]}`) regardless of `INFERENCE_PROVIDER`. Defaults inherit from `OPENAI_API_BASE` / `OPENAI_API_KEY` / `OPENAI_TIMEOUT`; override per-knob with `RERANK_API_BASE` / `RERANK_API_KEY` / `RERANK_TIMEOUT`. The full vllm-service stack exposes `/v1/rerank` via the LiteLLM router; the `rerank-only` deployment shape (CPU container, pairs with `gliner-only` for non-CUDA dev) expects `RERANK_API_BASE=http://rerank-only:8000`. Transport failure (endpoint unreachable, malformed payload) degrades to original retrieval order (top_n unranked) — no crash, no local fallback model.
- `docint/utils/embed_chunking.py` — Pre-embed re-chunker: bounds oversize chunks to the embedding budget and links sub-nodes back to their parent via `hier.parent_id`
- `docint/utils/embedding_tokenizer.py` — Loads the embedding model's tokenizer from the HF cache for accurate token counting during pre-embed re-chunking; falls back to char-ratio when unavailable
- `docint/utils/env_cfg.py` — **All** environment-backed configuration dataclasses live here (see below)

## Key Conventions

- **Python ≥3.11, <3.12**. Use `uv` for dependency management (`uv add`/`uv remove` to keep `pyproject.toml` and `uv.lock` in sync).
- **Centralized config**: All `os.getenv` calls and config dataclasses must live in `docint/utils/env_cfg.py`. Other modules import from there. If a subpackage needs a short import path, use a thin re-export module.
- **Test synchronization**: Every functional change must include corresponding test updates. Tests are in `tests/` and use pytest. `conftest.py` provides mock stubs for external dependencies like `magic`.
- **Google-style docstrings** for new/modified functions and classes.
- **Pre-commit is mandatory**: always run `uv run pre-commit run --all-files` before finishing work (ruff check, ruff format, pyrefly).
- Prefer incremental, focused commits. When changes affect both API and UI, update `README.md`.
- Frontend lives in `frontend/`. Keep business logic in the API/agents layer. Frontend dev: pnpm; tests: Vitest.
- **Multi-tenant / per-user isolation (load-bearing)**: Collections, chat sessions, and reports are owner-scoped by the resolved principal (`docint/core/auth/principal.py` reads `DOCINT_AUTH_HEADER`, default `X-Auth-User`, then falls back to `DOCINT_DEFAULT_IDENTITY`, else 401). Collection names are *logical*; the physical Qdrant name is namespaced per owner (`CollectionOwnerManager`), so two users can own the same name. Every collection-scoped endpoint resolves + gates the caller's logical name via `_require_owned_collection` / `_scoped_collection` (404 cross-owner) and binds the active collection **per-request** through `RAG.collection_scope` (a `ContextVar`). `/collections/select` is a non-mutating validation only — clients must send `collection` on every collection-scoped request (the SPA reads `useUiStore.selectedCollection`). The `SessionManager` chat runtime is threaded per request too (`start_session`/`chat`/`stream_chat` take `(session_id, owner)`; no shared `session_id`/`_owner`/`chat_engine`/`chat_memory`). **Never reintroduce process-global active-collection or session state** — it silently breaks (or cross-contaminates) concurrent users. Multi-tenant invariants are guarded by `tests/test_{collection_owner_manager,api_collections_ownership,rag_stateless_concurrency,session_concurrency,multiuser_isolation}.py`.
- **Hidden collection suffixes**: `docint/core/rag.py` defines `HIDDEN_COLLECTION_SUFFIXES` (currently `("_images", "_dockv", "_entities")`). `RAG.list_collections()` filters these out, which transitively hides them from `/collections/list` and makes `select_collection()` reject them. Extend the tuple rather than adding filters at the UI layer. The companions share the base collection's lifecycle — `delete_collection` and the empty-ingestion cleanup remove `{name}_images`/`{name}_entities` alongside it.
- **Locale-aware prompts**: every LLM prompt template lives under `docint/utils/prompts/{en,de}/<name>.txt`. The active locale is `load_language_env().code` (env var `RESPONSE_LANGUAGE`, default `en`). Adding a new prompt = add the file in both `en/` and `de/`. Adding a new locale = create the subdir and translate all 16 files. Non-prompt user-facing strings live in `docint/utils/ui_strings.py` and follow the same env var. JSON output schemas, intent labels, and hate-speech `category` enum stay English in every locale — they are protocol, not prose.
