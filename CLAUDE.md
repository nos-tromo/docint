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

**Likewise, NEVER expose local filepaths from development machines** —
absolute paths or home directories such as `/Users/<name>/...`,
`/home/<name>/...`, or `C:\Users\...` — anywhere git sees. The only
permitted paths are relative project paths starting from the project's
root (e.g. `docker/compose.yaml`).

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
# Build the full-text search index for a collection (payload-only, airgap-safe,
# idempotent). Needed once per collection ingested before search shipped.
make search-index                    # prompts for the collection name
make search-index COLLECTION=mydocs  # non-interactive
make search-index-all                # every collection (one-time backport)
```

## Architecture

Document Intelligence is a RAG stack: FastAPI backend + React SPA + Qdrant vector DB + pluggable inference (Ollama, OpenAI-compatible APIs, or external vLLM).

**All ML inference is remote.** docint ships no GPU code and no local model runtime: chat goes through the OpenAI-compatible API, dense embedding through `{EMBED_API_BASE}/embeddings`, reranking through `{RERANK_API_BASE}/rerank`, NER through `{NER_API_BASE}/gliner`, CLIP image+text embedding through `{CLIP_API_BASE}/clip/*`, and sparse embedding through `{SPARSE_API_BASE}/pooling` + `/tokenize`. The latter five default to the LiteLLM router alias of the full vllm-service stack; standalone CPU profiles (`ner-only`, `rerank-only`, `clip-only`, `embed-only`) live in `vllm-service/docker/compose.*-only.yaml` and let non-CUDA dev hosts override the relevant `*_API_BASE` independently — `embed-only` backs both `EMBED_API_BASE` and `SPARSE_API_BASE` from one bge-m3 instance. The runtime container is a single Debian-slim image (no CUDA, no `[cuda]` extra).

**Request flow:**
```
React SPA (frontend/) → FastAPI (docint/core/api.py) → AgentOrchestrator (docint/agents/)
    → understanding → clarification → retrieval → generation
    → RAG engine (docint/core/rag.py) ↔ Qdrant vector store
```

**Key modules:**
- `docint/core/rag.py` — Core RAG engine: ingestion, retrieval, postprocessing (reranking, parent context, source diversity), collection management. The active collection is **per-request**, not a shared singleton: `qdrant_collection` is a property over a `ContextVar` bound by `RAG.collection_scope(physical)`, and `index`/`query_engine` are per-collection thread-safe LRU caches (so concurrent users on different collections don't interfere — see the multi-tenant convention below).
- `docint/core/api.py` — FastAPI app with endpoints for chat, ingestion, collections, citations; streams responses. **Summaries are read and built through two different methods**: `GET /summarize` only ever reads (200 with the cached payload, 204 when there is none) and queues nothing, while `POST /summarize` queues a build on a miss or an explicit `refresh=true`. Both build their 200 body with `_cached_summary_payload`, so the two can never describe the same cached summary differently. The split is by method rather than a `queue=false` flag because the SPA fires the read whenever the Summary tab opens — a build is minutes of map-reduce, and a handler with no queue branch cannot start one by accident. Note the payload assembly is *not* free (the `_validation_payload` merge may call the text model), which is why the GET handler is a plain `def` and the POST hops through `to_thread.run_sync`. `GET /metrics` exposes Prometheus request counters/histograms (via `prometheus-fastapi-instrumentator`) for the obs-plane scrape target — aggregate only, no document or user data; unauthenticated like `/version`/`/config`; toggle with `METRICS_ENABLED` (default `true`, see `env_cfg.load_metrics_env`).
- `docint/core/jobs.py` — **Ingest job registry**: `IngestJobManager` holds
  runs in memory keyed by `job_id`, owner-scoped, bounded by
  `DOCINT_INGEST_CONCURRENCY` (default 1). `POST /ingest/finalize` queues a job
  and returns `202 {job_id}`; clients consume progress from the
  owner-multiplexed `GET /ingest/jobs/events`, which replays a **collapsed**
  history on connect (started + every warning + latest progress + terminal), so
  a browser reload re-attaches mid-run. A second job for a collection already
  ingesting is refused with 409 carrying the in-flight `job_id` — overlapping
  runs can double-write, since file hashes are only recorded after a run's
  final node batch. Entity resolution runs as a stage *inside* the job, so it
  no longer depends on a client being attached. Jobs survive a browser reload
  but **not** a backend restart (in-memory by design, mirroring Nextext's
  `nextext/api/jobs.py`); the staged files remain on disk and hash dedup makes
  a re-run cheap. The module holds no docint domain imports — the pipeline call
  is injected as a `runner`. **A run's duration is computed once, here, and
  rendered identically in the backend log and the SPA's ingest card**: the
  clock starts at job *creation* (so a queue wait counts) offset by the
  client-reported `upload_elapsed_ms` (so the upload leg counts), and the same
  value is logged, put on the terminal frame as `duration_ms`, and exposed on
  the snapshot alongside `run_started_at`. Do not reintroduce a client-derived
  elapsed for a finished run: two nearly equal durations floored on either
  side of the wire disagree by a whole second whenever their difference
  straddles a boundary, which is how one run came to report `00:18` in the log
  and `00:19` on the card. `started_at` keeps its own meaning (worker slot
  acquired) for queue-depth analysis.
- `docint/agents/orchestrator.py` — Coordinates understanding, clarification, retrieval, and generation agents
- `docint/core/ingest/ingestion_pipeline.py` — Document processing, chunking, metadata extraction
- `docint/core/ingest/social_linker.py` — Joins a social export's `postings.csv` to its `media.csv` manifest + files (counter-stripped `Media ID`, basename resolution within one flat directory) and routes each linked file to the right backend: still images through `images_service.py` (CLIP); audio/video by delegating per-file Nextext routing to the shared `media_transcribe.py` engine. Every artifact — image embedding, keyframe, transcript segment — is stamped with the parent posting's `posting_uuid`, which `_attach_posting_group` uses to group a post with all its media at retrieval time. Artifacts additionally carry the posting's own reference fields (`posting_network`/`posting_author`/`posting_timestamp`/`posting_url`/`posting_text`, built via `build_posting_reference_index` from the `TableReader` postings profile) merged *additively* into their `reference_metadata` — a transcript segment keeps `network: nextext`/`type: transcript_segment`. Pre-existing collections need a re-ingest to gain these fields (no payload migration).
- `docint/core/ingest/media_transcribe.py` — Shared per-file `MediaTranscriber` engine: hash → transcript-cache lookup (the ingest manifest) → bounded-concurrency Nextext round-trip (`NEXTEXT_MAX_CONCURRENCY`, cache misses only) → keyframes to CLIP, transcript to segment `Document`s. Used identically by `social_linker.py` (posting identity, `posting_uuid` link field) and `standalone_media.py` (file-hash identity, no link field) via the path-specific `MediaClip` dataclass; no media decoding or model inference lives here.
- `docint/core/ingest/standalone_media.py` — `StandaloneMediaIngestor`: the standalone audio/video pre-pass, run right after the social linker. Walks the batch tree for audio/video extensions (`MEDIA_FILETYPES`, default `DEFAULT_MEDIA_FILETYPES` in `env_cfg.py`) the linker did not already claim and routes each through `media_transcribe.py`, anchored to the media file's own content hash — no `postings.csv`/`media.csv` required. Automatic whenever `NEXTEXT_API_BASE` is set; a no-op (one-line warning) otherwise. With no posting to link to, transcripts/keyframes retrieve as independent, normally-ranked sources naming the source clip — no `posting_group` cross-modal clustering.
- `docint/core/readers/documents/` — Page-level PDF pipeline: triage, layout analysis, OCR fallback, extraction, chunking. **The vision OCR lane separates "no answer" from "error answer"** (`ocr.py::VisionOCRUnreachable` vs `VisionOCRRejected`): a timeout/connection failure counts toward the 3-consecutive-page budget that disables the lane for the whole document (each such page would otherwise burn a full `PIPELINE_VISION_OCR_TIMEOUT`), while an HTTP error status costs only its own page. Measured on the dev stack, a successful page takes 68–117 s and a rejected one 0.5–1.0 s, arriving in bursts that recover within seconds — conflating the two let a ~5 s upstream blip discard 19 of 30 pages. Both retry paths (`ocr.py`'s half-resolution retry, `images_service._run_with_retries`) wait `_RETRY_BACKOFF_SECONDS`/`RETRY_BACKOFF_SECONDS` before retrying, so the retry does not land inside the same burst; tests that drive those paths must patch `time.sleep` in the relevant module. Outcomes surface as `pages_ocr_failed`/`pages_ocr_skipped` on the manifest — `pages_ocr` counts pages that *needed* OCR, not pages that got it.
- `docint/core/readers/docx.py` — Word `.docx` reader. Drives docling's declarative `MsWordDocumentBackend` directly via `InputDocument` (pure XML, no models or network — airgap-safe; deliberately **not** `DocumentConverter`, whose module eagerly imports the model-backed PDF pipeline → `docling_ibm_models` → torch, which the CPU-only image must not ship — hence the `docling-slim` dependency instead of `docling`) into compact Docling-JSON, so a docx flows through the already-wired `DoclingNodeParser` like a PDF (Markdown fallback if JSON export fails; skips rather than emitting raw bytes if conversion fails). Registered in `ingestion_pipeline.py`'s `file_extractor`, mirroring `rtf.py` — a binary type with no registered extractor gets silently decoded as UTF-8 (the bug this fixed).
- `docint/core/readers/json.py` — Generic JSON / JSONL reader. Detects Nextext transcripts (JSONL with `text` plus timing keys `start_ts`/`end_ts` or `start_seconds`/`end_seconds`) and routes them to one-node-per-segment ingestion, mirroring the social-table specialized schema pattern; timing/speaker metadata surface via `reference_metadata`.
- `docint/core/storage/` — Qdrant-backed document store, hierarchical node storage, source tracking
- `docint/core/state/` — Session management (SQLite-backed) and citation handling
- `docint/core/state/report*.py` — **Report builder**: owner-scoped, server-persisted *curated* reports that let an investigator hand-pick individual chat answers, entity findings (chunk-level, deduped by `chunk_id`), and hate-speech findings instead of exporting a whole collection. `report.py` + `report_item.py` (ORM in the shared session DB; **type-prefixed dedupe keys** — `entity:`/`hate:`/`chat:` — so the same chunk can be distinct evidence under two types while re-adds are no-ops; snapshots are frozen JSON at add-time, immune to re-ingestion). `report_manager.py` (`ReportManager`, mirrors `SessionManager`'s owner-scoped store plumbing; exposed via `RAG.ensure_report_manager()`). `report_render.py` (pure renderers → Markdown / HTML / **PDF via WeasyPrint** / JSON / CSV-bundle; section headings flow through `ui_strings.py`, reuses `csv_stream.py` row builders; `artifact_type`/JSON keys stay English). API: `POST/GET/PATCH/DELETE /reports*` + `GET /reports/{id}/export.{md,html,pdf,json,zip}`. Frontend: the **Report** tab plus an "Add to report" control on each artifact. Entity and hate-speech findings additionally carry an on-demand translation into the snapshot (`translation: {text, target_lang, model}`, frozen at add-time like the rest of the snapshot); every export renders it as an additive **"Machine translation (→ Deutsch)"** block (MD/HTML/PDF) or `translation` column (CSV) beside the original — endonym via `language_endonym()`, never replacing the original — see `docint/utils/translate_client.py`.
- `docint/core/state/collection_ownership.py` + `collection_owner_manager.py` — **Per-user collection ownership** (multi-tenant). `CollectionOwnership` ORM (table `collection_owners` in the shared session DB) is the source of truth for the `(owner, logical_name) ↔ physical_name` mapping; `CollectionOwnerManager` (mirrors `ReportManager`) does `register`/`resolve`/`list_for`/`delete` + legacy backfill to `DOCINT_DEFAULT_IDENTITY`. Physical name = `u{sha256(owner)[:12]}__{logical}` (legacy rows keep the bare name — no Qdrant rename). Exposed via `RAG.ensure_collection_owner_manager()`; the API gate is `_require_owned_collection(logical, principal) -> physical` (404 on cross-owner, like `_get_owned_report`).
- `docint/core/ner.py` — Entity aggregation / clustering / graph building over already-extracted NER metadata (pure post-processing; no model inference). Merge modes: `exact` (case-insensitive), `orthographic` (alphanumeric-compacted, the default — already collapses `Africa`/`africa`/`Acme Corp`), and `resolved` (groups by durable canonical entity id from the resolution store, falling back to orthographic for unresolved surfaces).
- `docint/core/entities/` — **Entity resolution** (chorus parity), the only way to merge *semantically* similar entities (`USA`/`United States`, `EU`/`European Union`). `resolution.py` is the pure, dependency-injected pipeline (normalize → exact alias → type-blocked vector match ≥ `RES_EMBED_THRESHOLD` → conservative LLM tie-break → mint), mirroring chorus's `ingestion/resolution.py`. `store.py` (`EntityStore`) persists one point per canonical entity in the hidden `{collection}_entities` Qdrant companion (vector = name embedding; payload = `canonical_name`, `type`, `aliases`). Triggered by `RAG.resolve_entities()` (re-runnable, idempotent) via the `resolve` CLI or `POST /collections/entities/resolve`; reuses the existing remote embedding + chat clients (no new model runtime). Tuned by `RES_EMBED_THRESHOLD` (0.86), `RES_LLM_TIEBREAK` (true), `RES_CASE_NORMALIZE` (true), `RES_VECTOR_K` (5), `RES_BATCH_SIZE` (defaults to `INGESTION_BATCH_SIZE`; embed/resolve batch cadence, bounded memory on large collections) in `env_cfg.py`. Tie-break prompt: `prompts/{en,de}/entity_tiebreak.txt`.
- `docint/core/search/` — **Full-text keyword search** over chunk text (`POST /search`).
  `index.py` owns the `search_text` payload field and its prefix + lowercase
  text index, plus `write_search_text()` — the batched `batch_update_points`
  writer shared by the ingest hook and the `search-index` backfill, so the
  migration path is exercised by every ingest. `fulltext.py` parses keywords and
  compiles them into native Qdrant filters: one `MatchText` per keyword in
  `must`, so **all** keywords must match, in any order. The field is written
  **payload-only** — never through node metadata, which is rendered into the
  embedding input and serialized into `_node_content`, so stamping it there
  would embed each chunk's text twice and store a third copy. The lowercase
  index is **mandatory, not an optimisation**: un-indexed `MatchText` only
  case-folds ASCII, so German title-case tokens would not match their lowercase
  form. Matching is prefix-based (`Partei` finds `Parteitag`; `tag` does not),
  and coarse parent chunks are excluded — as *not-coarse* rather than *is-fine*,
  because a collection ingested without hierarchical chunking tags nothing and
  requiring `fine` would return zero hits there. No embeddings and no inference
  in the path. The package imports nothing from `core/rag.py` (the text
  extractor is injected), mirroring `core/jobs.py`. Collections ingested before
  this shipped need `make search-index` once — **only as a backport**: ingestion both writes `search_text` and ensures the payload index, so new collections are searchable with no operator step. Both are required; the field without its index is silently case-sensitive on non-ASCII text, which `/search` reports as `not_indexed` rather than pretending to work. Search covers **two lanes** — the collection's chunks and its `_images` companion, whose points carry an image's caption and tags, so a figure or video keyframe is findable by tag; hits are marked `kind` `text`/`image`, and coverage counts both, since an unindexed companion would otherwise read as complete while every image stayed unfindable. Scoping and expanding resolve across both collections. Coverage is **counted, not
  sampled**: `/search` reports `not_indexed` when no point carries the field
  and `partial` (with a `missing` count) while a backfill is incomplete, so a
  half-migrated collection can never masquerade as a complete result set.
  The SPA surfaces this as a **collapsible panel beside the chat** holding the
  query field and the hits — and *only* those. A hit is a **tile**: clicking
  anywhere on it (or Enter/Space) pins it; there is no per-hit checkbox and no
  per-hit Inspector link. Collapsed, the panel's rail is a bare open/shut
  toggle — no counts; it only tints when a scope is live, since that is the one
  state with consequences while hidden. The metadata filters and the
  retrieval-mode toggle are **not** search controls and do not live here — they
  narrow what any answer retrieves against whether or not the panel is open, so
  they hold the right edge of the Chat header row
  (`components/chat/ChatControls.tsx`). Do not move them back into the panel:
  there they read as controls over the keyword index, and stacked at its foot
  the filter panel opened straight over the retrieval control, making the pair
  behave like one control with two faces. Selected hits write the
  session's **scope**: `PUT /sessions/{id}/scope` stores the chunk ids on the
  conversation row, and while a scope is active `/query` and `/stream_query`
  answer **only** from those chunks — `build_query_engine(scoped_node_ids=)`
  swaps in `_ScopedRetriever` and drops every ranking postprocessor, because
  parent-context expansion and link-following would silently widen a
  hand-picked set while the diversity cap and relevance floor would silently
  narrow it. **The scope travels on the query request** (`scope_chunk_ids` on
  `/query` + `/stream_query`), not only on that PUT: the conversation row is
  minted by the first turn, so a selection made before it exists has nowhere to
  be written — pinning it after the answer left the first turn unscoped while
  the banner already claimed otherwise. Present ⇒ it is this turn's scope and
  is pinned to the session; absent ⇒ the stored scope still applies. Both paths
  report back what they did (`retrieval_mode: "scoped"` + `scoped_chunk_count`,
  in `/query`'s body and the final SSE frame), and the SPA flags any turn that
  asked for a scope and did not get one, so a dropped scope can never again
  pass as hand-picked evidence. A scoped turn also skips GraphRAG expansion
  (`RAG.graph_debug_skipped`): it widens retrieval that is not happening, and
  the terms it appends land in the synthesis prompt. An oversize selection is
  refused (422) before the stream opens, never truncated. This
  **replaced the two entity-occurrence chat query modes**, which searched the
  NER aggregate and silently returned whichever entity was most frequent rather
  than the one asked about.
- `docint/utils/ner_client.py` — Thin HTTP client for the remote GLiNER service hosted by `vllm-service` (full stack: `http://vllm-router:4000/gliner` with Bearer auth; gliner-only shape: `http://gliner-only:8000/gliner` with no auth). Replaces the in-process GLiNER runtime previously shipped here.
- `docint/utils/clip_client.py` — Thin HTTP client for the remote CLIP image+text embedding service hosted by `vllm-service`. Same dual-shape posture as the NER client (full stack via router with Bearer auth; `clip-only` shape at `http://clip-only:8000` with no auth). `RemoteCLIPBackend` satisfies the `ImageEmbeddingBackend` Protocol so `core/ingest/images_service.py` swaps in place. Probes `/clip/dimension` at construction to size Qdrant `_images` collections without burning an embed call. `IMAGE_EMBED_MODEL` is no longer read by docint — set `CLIP_MODEL` on the vllm-service container instead. Override the endpoint via `CLIP_API_BASE` / `CLIP_API_KEY` / `CLIP_TIMEOUT`.
- **Images are ordinary retrieval sources, not a side lane** (`core/rag.py::MultimodalRetriever` + `_retrieve_image_nodes`). The image lane is half of the retriever: CLIP candidates become caption nodes that join the text hits *before* ranking, so one reranker pass scores both modalities on one scale, the generator sees images in `context_str` and can cite them, and `CitationNumberingPostprocessor` numbers them like any other source. Everything downstream (parent context, diversity, link-following, synthesis, the citation panel) is modality-blind by construction. A lane outage degrades the answer to text-only; it never fails the query. **Do not reintroduce a post-generation append** — an appended source cannot be cited by the answer that was written without it. A standalone image file exists in *both* collections (`ImageReader` writes its caption as the document's text, `ImageIngestionService` writes the CLIP point), so `MultimodalRetriever` drops the lane's copy when the main-collection node for the same `image_id` was already retrieved. The collection summary draws a document's figures/keyframes from the `_images` companion by document hash (`_summary_image_nodes_for_document`), capped at a third of the per-document budget.
- **Image relevance is gated after the rerank** (`core/rag.py::_image_query_for_clip` + `ImageRelevanceFloorPostprocessor`). CLIP generates candidates; the shared reranker scores their captions and `IMAGE_RERANK_MIN_SCORE` (default `0.05`) drops the rest. The floor runs directly after the reranker and applies to image nodes only — the top-n cut alone cannot protect a sparse collection, where a merely-nearest image takes a slot for lack of competition. The floor deliberately sits on the **reranker** score, never on raw CLIP cosine: measured on a live collection, an unrelated query and a genuinely matching one both land in a ~0.20–0.30 CLIP band (gibberish scored 0.233 against a real hit's 0.280), so no absolute CLIP threshold separates them — reranker scores do (0.12–0.90 when a relevant image exists vs ≤0.0037 when none does). This also puts image scores on the same scale as text sources instead of making junk images look 100× more relevant. Because the deployed checkpoint (`openai/clip-vit-base-patch32`) has an **English-only text tower**, the query is translated to English before embedding whenever `RESPONSE_LANGUAGE` is not English — an untranslated German query ranks near-randomly. The reranker gets the *original* query (bge-reranker-v2-m3 is cross-lingual). Both stages fail soft: a translation outage embeds the untranslated query, and a rerank outage surfaces ungated images rather than blanking the lane — a *wholly* unscored node set is how `VLLMRerankPostprocessor` announces its own swallowed transport error, so the floor stands down instead of reading it as "everything scored below the floor". Image sources carry citation identity like text sources do, and through the same normalizer: `_source_from_payload` knows the `_images` payload shape (`image_id` → `chunk_id`, `source_doc_id` → `file_hash`, `source_path` → filename, caption+tags → body), so `id` is the retrieved point id and `chunk_id` the image content hash. Never hand-build an image source dict beside it.
- `docint/utils/translate_client.py` — Thin, fail-soft wrapper for **on-demand, display-time** snippet translation: nothing ingested or stored is ever translated, and the original is always preserved. Reuses the shared chat model (`OpenAIPipeline.call_chat`) over the same LiteLLM router endpoint as chat — no dedicated translation runtime, no `TRANSLATE_API_BASE` knob. `translate(text) -> TranslateResult` never raises: a transport/model failure degrades to `ok=False` + an `error` token so the caller just keeps showing the original — no crash, no local fallback model. Successful translations are LRU-cached; failures are not. Destination defaults to the active locale (`RESPONSE_LANGUAGE`); pass `target_lang=` to override it when the destination is dictated by a downstream model rather than the operator — image retrieval uses `target_lang="en"` because the CLIP text tower is English-only (see the image-relevance bullet below). Model: `TRANSLATE_MODEL` in `env_cfg.py` (defaults to the chat model `TEXT_MODEL`). Locale prompt: `prompts/{en,de}/translate.txt`. API: `POST /translate` (`{text}` → `{ok, translation, model, target_lang, error}`; principal-authenticated but not collection-scoped — it translates text the caller already holds, so there's nothing to leak and no store re-fetch). Frontend: a hover/focus-revealed **Translate** control (`TranslateControl`, composing `@infra/ui`'s `HoverIconAction`) on chat source citations, entity findings, and hate-speech rows — in-app copy is "Translate"/"Translation"; report exports use the fuller "Machine translation" label (see the report-builder bullet above for the additive carry).
- **Reranking is always remote.** `core/rag.py::RAG.reranker` builds a `VLLMRerankPostprocessor` that POSTs to `{RERANK_API_BASE}/rerank` in the Jina shape (`{model, query, documents, top_n}` → `{results: [{index, relevance_score}]}`) regardless of `INFERENCE_PROVIDER`. Defaults inherit from `OPENAI_API_BASE` / `OPENAI_API_KEY` / `OPENAI_TIMEOUT`; override per-knob with `RERANK_API_BASE` / `RERANK_API_KEY` / `RERANK_TIMEOUT`. The full vllm-service stack exposes `/v1/rerank` via the LiteLLM router; the `rerank-only` deployment shape (CPU container, pairs with `gliner-only` for non-CUDA dev) expects `RERANK_API_BASE=http://rerank-only:8000`. Transport failure (endpoint unreachable, malformed payload) degrades to original retrieval order (top_n unranked) — no crash, no local fallback model.
- **Sparse embedding is always remote, and its wire format is frozen.** `core/rag.py::RemoteSparseEncoder` POSTs to `{SPARSE_API_BASE}/pooling` (`task: token_classify`) and `{SPARSE_API_BASE}/tokenize` regardless of `INFERENCE_PROVIDER` — there is no local fastembed/onnxruntime encoder any more. `SPARSE_API_BASE` / `SPARSE_API_KEY` / `SPARSE_TIMEOUT` inherit from `OPENAI_API_BASE` / `OPENAI_API_KEY` / `OPENAI_TIMEOUT` like the `RERANK_*` knobs; `SPARSE_MODEL` is `BAAI/bge-m3` on every provider. The `embed-only` deployment shape (CPU container, pairs with `gliner-only` / `rerank-only` / `clip-only`, serves dense embedding + sparse weights + tokenization from one bge-m3 instance) expects `SPARSE_API_BASE=http://embed-only:8000`. Its method bodies are intentionally frozen: production collections' sparse vectors depend on encoding this exact way, so changing the request/response shape would silently desync retrieval against already-ingested data. `ENABLE_HYBRID` defaults to **true** when `INFERENCE_PROVIDER=vllm` or `SPARSE_API_BASE` is explicitly set, **false** otherwise (see `resolve_enable_hybrid()` in `env_cfg.py`). Unlike the reranker, sparse encoding is deliberately **not** fail-soft: `RAG.probe_sparse_endpoint()` checks the endpoint before an ingest job's first batch and fails the job cleanly rather than risk writing dense-only points into a hybrid collection.
- **Dense embedding fails loudly and names its own endpoint.** `BudgetedOpenAIEmbedding` (`utils/openai_cfg.py`) classifies the embed endpoint's own faults — unreachable, 401/403, 404, any other HTTP status — into `EmbeddingEndpointError`, whose message carries the resolved base URL, the model, and the knob to change; a context overflow is checked *first* and keeps its existing `EmbeddingInputTooLongError`, and anything else propagates untouched so a docint-side bug is never relabelled an outage. Every embedding path is wrapped, the **query** path included — leaving that one unguarded is what let a one-line config mistake surface as an unexplained chat-stream crash whose traceback was entirely llama_index frames. The commonest fault is `EMBED_API_BASE` missing its `/v1`: the OpenAI SDK appends `/embeddings`, so `http://embed-only:8000` 404s while `SPARSE_API_BASE` on the *same container* is correct bare — one host, two forms. `RAG.probe_embed_endpoint()` runs one embed call beside `probe_sparse_endpoint()` before an ingest stages any file, so the failure costs a second instead of a whole parse pass (vision OCR alone is minutes per page). In the chat stream the same error is reported as SSE code `embedding_unavailable`, never `generation_failed` — the latter reads as a chat-model fault and sends the operator to the wrong service.
- `docint/utils/embed_chunking.py` — Pre-embed re-chunker: bounds oversize chunks to the embedding budget and links sub-nodes back to their parent via `hier.parent_id`
- `docint/utils/embedding_tokenizer.py` — Loads the embedding model's tokenizer from the HF cache for accurate token counting during pre-embed re-chunking; falls back to char-ratio when unavailable
- `docint/utils/env_cfg.py` — **All** environment-backed configuration dataclasses live here (see below)

## Key Conventions

- **Container hardening (deploy ADR 0001)**: both containers run non-root with
  read-only root filesystems — the backend as uid `10001` (`app`,
  `HOME=/home/app`), the frontend on `nginxinc/nginx-unprivileged` as uid `101`
  listening on **:8080** (the edge gateway's `docint-frontend` upstream must
  match). Every runtime write must land on a volume (`/var/lib/docint/*`,
  `/home/app/.cache/*`) or the `/tmp` tmpfs; compose pins
  `DATA_PATH`/`PIPELINE_ARTIFACTS_DIR`/`RESULTS_PATH` to the external
  `pipeline-storage` volume because their `$HOME`-derived defaults are
  unwritable in-container. Volume ownership is normalized automatically: the
  one-shot `volume-permissions` compose service (root, backend image, runs
  before the backend via `depends_on: service_completed_successfully`) chowns
  wrong-owner entries on the five docint volumes to `10001:10001` — an
  image-build chown can't do this, since runtime volume mounts shadow image
  paths. If the sessions DB is still unwritable anyway, the backend **fails at
  startup** with `SessionStoreMigrationError` (`docint/core/state/base.py`):
  session-store column migrations are eager (API lifespan) and fatal, because
  serving with a stale schema 500s every conversations query.
- **Python ≥3.11, <3.12**. Use `uv` for dependency management (`uv add`/`uv remove` to keep `pyproject.toml` and `uv.lock` in sync).
- **Centralized config**: All `os.getenv` calls and config dataclasses must live in `docint/utils/env_cfg.py`. Other modules import from there. If a subpackage needs a short import path, use a thin re-export module.
- **Test synchronization**: Every functional change must include corresponding test updates. Tests are in `tests/` and use pytest. `conftest.py` provides mock stubs for external dependencies like `magic`.
- **Google-style docstrings** for new/modified functions and classes.
- **Pre-commit is mandatory**: always run `uv run pre-commit run --all-files` before finishing work (ruff check, ruff format, pyrefly).
- Prefer incremental, focused commits. When changes affect both API and UI, update `README.md`.
- Frontend lives in `frontend/`. Keep business logic in the API/agents layer. Frontend dev: pnpm; tests: Vitest. The SPA defaults to a light theme (OS-preference driven, toggled via the shared `AppShell` chrome's sidebar slot, tri-state theme toggle and user menu incl. sign-out — `@infra/ui#v0.9.1`); dark remains available via the toggle or an explicit OS dark preference. There is exactly one header row (`AppShell`'s) — no duplicate sidebar headline. It carries the signed-in identity (`user`, from the authenticated `GET /whoami`, undefined while loading/on error) and the release version (`version`, from `GET /version`); neither is echoed anywhere else.
- **Icon actions come from `@infra/ui`, never hand-rolled here** (`DownloadButton`, `DownloadLink`, `NewButton`, `RemoveButton`, `DeleteButton`, the `IconButton`/`IconLink` base, and `HoverIconAction` — the same shell held at `opacity-0` until its row is hovered, which is how a per-row action stays out of a quiet list; the sidebar's session and collection rows use it so their rows measure exactly like the nav rows above them). They are `ghost` — no background until hovered — so they sit quietly beside the other icon controls in a header row. Every one takes a required `label` that drives both `aria-label` and `title`. The icon replaces the **verb**, never the **format**: the full phrase ("Export CSV", "Download session sources (ZIP)") lives in `label`, and only where several downloads sit side by side (the entity graph's JSON / GraphML / HTML) does each keep its format visible as `children` — three identical icons in a row is a guessing game. `RemoveButton` (`×`) takes something out of a list or view; `DeleteButton` (trash) destroys stored data (a collection, a session, a report) and is paired with a `confirm`. A new icon action belongs in infra-ui's `src/icons/` + `src/primitives/iconActions.tsx`, not in a local component here.
- **Icons are drawn, never typed.** `components/common/icons.tsx` holds the SPA-specific set (`CircleIcon`, `CheckCircleIcon`, `CheckAllIcon`, `SlidersIcon`, `ChatContextIcon`, `SingleMessageIcon`); everything shared across apps — `DownloadIcon`, `PlusIcon`, `XIcon`, `TrashIcon`, `ChevronDownIcon`, plus the status markers `CheckIcon`, `WarningIcon`, `InfoIcon`, `StopwatchIcon` and `ExternalLinkIcon`, the `ReportIcon`/`ReportCheckIcon` state pair behind the icon-only "Add to report" toggle, and `SendIcon`/`SearchIcon`/`RefreshIcon` (as the `SendButton`/`SearchButton`/`RefreshButton` actions) — is imported from `@infra/ui`. A pair like that one belongs upstream even though only this app has reports: the icon is the shared part, so it is added to infra-ui, released by a version bump (tag minted on merge) and consumed here by repinning the codeload SHA — never hand-drawn in `components/common/icons.tsx`, which holds only what no other app could use. Never use a text character (`×`, `▾`, `⤓`, `↑`) as a control's symbol: it renders from whatever font the browser and OS fall back to, so weight and size differ on every machine, and in a control carrying no text of its own that drawing *is* the affordance. **The rule covers status markers and catalog copy, not just controls** — `⏱`, `ⓘ` and `↗` additionally carry emoji presentation on some platforms, so they arrive full-colour beside monochrome chrome, and a symbol embedded in an `i18n/{en,de}.ts` string (`'✓ In report'`, `'Open link ↗'`) is one a translator can silently drop or reorder. Drive the icon from state or structure at the call site instead — `MetadataPills` renders `ExternalLinkIcon` from the presence of `href`, never from the label text. A disclosure caret is one `ChevronDownIcon` rotated by class with `aria-expanded` alongside, never a `▾`/`▸` pair.
- **Multi-tenant / per-user isolation (load-bearing)**: Collections, chat sessions, and reports are owner-scoped by the resolved principal (`docint/core/auth/principal.py` reads `DOCINT_AUTH_HEADER`, default `X-Auth-User`, then falls back to `DOCINT_DEFAULT_IDENTITY`, else 401). Collection names are *logical*; the physical Qdrant name is namespaced per owner (`CollectionOwnerManager`), so two users can own the same name. Every collection-scoped endpoint resolves + gates the caller's logical name via `_require_owned_collection` / `_scoped_collection` (404 cross-owner) and binds the active collection **per-request** through `RAG.collection_scope` (a `ContextVar`). `/collections/select` is a non-mutating validation only — clients must send `collection` on every collection-scoped request (the SPA reads `useUiStore.selectedCollection`). The `SessionManager` chat runtime is threaded per request too (`start_session`/`chat`/`stream_chat` take `(session_id, owner)`; no shared `session_id`/`_owner`/`chat_engine`/`chat_memory`). **Never reintroduce process-global active-collection or session state** — it silently breaks (or cross-contaminates) concurrent users. Members of the admin group (from the gateway's `X-Auth-Groups`) operate in another owner's namespace via the `owner` query param (`Principal.effective_owner`); non-admins are unaffected and cross-owner remains 404. Multi-tenant invariants are guarded by `tests/test_{collection_owner_manager,api_collections_ownership,rag_stateless_concurrency,session_concurrency,multiuser_isolation}.py`.
- **Hidden collection suffixes**: `docint/core/rag.py` defines `HIDDEN_COLLECTION_SUFFIXES` (currently `("_images", "_dockv", "_entities")`). `RAG.list_collections()` filters these out, which transitively hides them from `/collections/list` and makes `select_collection()` reject them. Extend the tuple rather than adding filters at the UI layer. The companions share the base collection's lifecycle — `delete_collection` and the empty-ingestion cleanup remove `{name}_images`/`{name}_entities` alongside it.
- **Locale-aware prompts and UI**: The env var `RESPONSE_LANGUAGE` (values `en` | `de`, default `en`) drives the entire app — backend and frontend.
  - **Backend**: LLM prompt templates live under `docint/utils/prompts/{en,de}/<name>.txt`; non-prompt ui_strings (reports, exports) under `docint/utils/ui_strings.py`. Both read `load_language_env().code`. Unknown `RESPONSE_LANGUAGE` values silently fall back to `en`. Adding a new prompt = add in both `en/` and `de/`. Adding a new locale = create the subdir, translate all prompt files (currently 17) and ui_strings entries.
  - **Frontend**: The SPA locale catalog lives in `frontend/src/i18n/` (TypeScript, typed, ~380 keys across all screens). Keys are dot-namespaced by screen (`common.*`, `chat.*`, `ingest.*`, etc.) and interpolate via `{name}` placeholders and the `format()` helper. The `useT()` hook and `LanguageProvider` read `RESPONSE_LANGUAGE` from `/config`. Frontend and backend share key parity — when adding a UI string to the backend, ensure the frontend has its counterpart in both `en/` and `de/` catalogs before the PR.
  - **Protocol invariants**: JSON output schemas, intent labels, hate-speech `category` enum, API field names, and enum values stay English in every locale — they are protocol, not prose. Collections, models, and product names (`docint`, chorus, Nextext, etc.) are never translated.

### Deployment: edge-plane gateway sub-path

The docint SPA is served in production under the canonical `/docint/`
sub-path behind the `edge-plane` gateway, not at its own vhost root. The
`frontend` service joins the external `edge-net` network (alongside its
existing `docint-net` membership) as alias `docint-frontend`, which is how
the gateway reaches it. Vite is built with `base: '/docint/'`, the API base
derives from `BASE_URL` (`VITE_API_BASE_URL` still overrides it verbatim
when set — e.g. for standalone/dev use outside the gateway), the SPA router
uses a matching `basename`, and the frontend's nginx template strips the
`/docint` prefix internally before falling through to the existing
root-anchored locations (SSE endpoints included), redirecting bare `/` to
`/docint/`. NB: every NEW backend route must be registered in BOTH
`frontend/nginx/default.conf`'s proxied-prefix regex and
`frontend/vite.config.ts`'s `API_PREFIXES`, or the SPA fallback silently
serves index.html for it (bit us with `/whoami`). The gateway is the sole production entry point and is what
injects `X-Auth-User` for the backend's trusted-header principal seam
(`docint/core/auth/principal.py`) — production leaves `DOCINT_DEFAULT_IDENTITY`
unset so requests without that header are rejected as unauthenticated. The
dev override's `DOCINT_DEFAULT_IDENTITY` fallback stays dev-only.
