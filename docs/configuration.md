# Configuration reference

Every environment-backed setting in Docint lives in
`docint/utils/env_cfg.py` as a frozen `@dataclass` with a paired
`load_*_env()` factory. This page enumerates those dataclasses, the
environment variables they read, and the defaults baked into each factory.

## Conventions

- **Reading config** — application code imports the `load_*_env()` helper
  from `docint.utils.env_cfg` and uses the returned dataclass. Calls
  to `os.getenv()` outside of `env_cfg.py` are discouraged; see
  [development.md](development.md) for the policy.
- **Overrides** — any variable can be set in `.env` at the repository
  root (loaded by `load_dotenv()` at import time) or in the environment.
- **Booleans** — the factories accept `true`, `1`, `yes` (case-insensitive)
  as true; anything else is false.
- **Offline mode** — if `DOCINT_OFFLINE=1` (the default), Docint enables
  HF / Transformers offline mode so models are loaded from the local
  cache only. See `set_offline_env()` in `env_cfg.py:12`.

## Inference endpoint — `OpenAIConfig`

Loaded by `load_openai_env()` (`env_cfg.py:631`). Controls the
OpenAI-compatible client used for chat, embeddings, and vision.

| Variable | Default | Description |
|---|---|---|
| `INFERENCE_PROVIDER` | `ollama` | One of `ollama`, `openai`, `vllm`. Invalid values raise `ValueError`. |
| `OPENAI_API_BASE` | `http://localhost:11434/v1` | Base URL of the OpenAI-compatible endpoint. |
| `OPENAI_API_KEY` | `sk-no-key-required` | Bearer token. Required for the `openai` provider. |
| `OPENAI_CTX_WINDOW` | `4096` (8192 min when provider is `vllm`) | Context window for the text model. Falls back to `CHAT_MAX_MODEL_LEN` when `vllm`. |
| `OPENAI_DIMENSIONS` | *unset* | Optional override for embedding dimension. |
| `OPENAI_MAX_RETRIES` | `2` | Retry count for OpenAI HTTP calls. |
| `OPENAI_NUM_OUTPUT` | `256` | Max tokens reserved for the model response by the LlamaIndex prompt helper. |
| `OPENAI_REUSE_CLIENT` | `false` | Reuse the OpenAI client across calls. |
| `OPENAI_SEED` | `42` | Sampling seed. |
| `OPENAI_TEMPERATURE` | `0.0` | Sampling temperature. |
| `OPENAI_TOP_P` | `0.1` | Nucleus sampling. |
| `OPENAI_TIMEOUT` | `300.0` | Request timeout in seconds. |
| `OPENAI_ENABLE_THINKING` | `false` | Opt into reasoning/"thinking" mode. |
| `OPENAI_THINKING_EFFORT` | `medium` | One of `none`, `minimal`, `low`, `medium`, `high`, `xhigh`. |

## Embedding — `EmbeddingConfig`

Loaded by `load_embedding_env()` in `env_cfg.py`. Bounds the text
payload sent to the embedding endpoint so oversize chunks are
re-chunked by `docint/utils/embed_chunking.py` before the API call
instead of being silently truncated.

### Token counting strategy

When `EMBED_TOKENIZER_REPO` is non-empty and the tokenizer snapshot exists in the HF cache (via `uv run load-models`), the pre-embed re-chunker uses the embedding model's authoritative tokenizer to count tokens. This is accurate across languages and domains (e.g. CJK, German compounds, transcripts with timestamps).

When the snapshot is missing or `EMBED_TOKENIZER_REPO` is empty (e.g. OpenAI provider), the re-chunker falls back to the `EMBED_CHAR_TOKEN_RATIO` heuristic and emits a WARNING at RAG init. The char-ratio approach is inherently biased toward English and significantly under-counts token budgets for multilingual or token-dense content.

| Variable | Default | Description |
|---|---|---|
| `EMBED_CTX_TOKENS` | `2048` (ollama) / `8191` (openai + `text-embedding-3-*`) / `CHAT_MAX_MODEL_LEN` (vllm) / `8192` (fallback) | Embedding context window in tokens. Must match the provider's serving ceiling. Ollama's default is `num_ctx=2048`; operators who bake a custom Modelfile with `PARAMETER num_ctx 8192` can set this to `8192` to reclaim the full bge-m3 window — see `docs/deployment.md` for the `deploy/Modelfile.bge-m3` recipe. Must be between `256` and `32768`. |
| `EMBED_CHAR_TOKEN_RATIO` | `3.5` | Characters-per-token estimator for mixed-language content (fallback when tokenizer is unavailable). Under-counts tokens intentionally to stay under budget. |
| `EMBED_CTX_SAFETY_MARGIN` | `0.95` | Fraction of `EMBED_CTX_TOKENS` left for the payload after reserving BOS/EOS and estimator slack. Must lie in `(0, 1]`. |
| `EMBED_TIMEOUT_SECONDS` | `1800` (ollama) / `600` (vllm) / `60` (openai) | HTTP request timeout in seconds for the embedding endpoint. Must be positive. Init-time warning logged if `timeout × (1 + max_retries) > 3600` (potential multi-hour stall). |
| `EMBED_BATCH_SIZE` | `16` (ollama) / `64` (vllm) / `100` (openai) | Maximum number of texts sent per embedding API batch. Must be between `1` and `1024`. |
| `EMBED_MAX_RETRIES` | `1` (ollama / vllm) / `2` (openai) | Maximum retries for transient HTTP failures on embedding requests. Must be between `0` and `10`. |

> **Not the same as `OPENAI_CTX_WINDOW`.** `OPENAI_CTX_WINDOW` controls
> the chat LLM only. It does **not** affect the embedding pipeline.
> Use `EMBED_CTX_TOKENS` for that. The two limits are disjoint because
> embedding models and chat models almost never share a context
> window, even when served from the same provider.

## Models — `ModelConfig`

Loaded by `load_model_env()` (`env_cfg.py:512`). Resolves model
identifiers, with provider-specific fallbacks.

| Variable | Default (by provider) | Description |
|---|---|---|
| `EMBED_MODEL` | `bge-m3` (ollama) / `BAAI/bge-m3` (vllm) / `text-embedding-3-small` (openai) | Dense text embedding model. |
| `EMBED_TOKENIZER_REPO` | `BAAI/bge-m3` (ollama / vllm) / `""` (openai) | Hugging Face repository ID of the tokenizer used for offline token counting at ingestion time. Empty string for providers (e.g. `openai`) where the embedding endpoint handles tokenization. Snapshot must be in the HF cache; run `uv run load-models` to populate it. |
| `SPARSE_MODEL` | `BAAI/bge-m3` | Sparse retrieval model. Same value on every provider — sparse encoding is always a remote call (`RemoteSparseEncoder`), so there is no per-provider default to pick. |
| `TEXT_MODEL` | `gpt-oss:20b` (ollama) / `Qwen/Qwen3.5-2B` (vllm) / `gpt-4o` (openai) | Chat / generation model. |
| `VISION_MODEL` | `qwen3.5:9b` (ollama) / `Qwen/Qwen3.5-2B` (vllm) / `gpt-4o` (openai) | General vision model — captions and tags images, and reads them when no `OCR_MODEL` is set. |
| `RERANK_MODEL` | `BAAI/bge-reranker-v2-m3` | Cross-encoder reranker. |
| `NER_MODEL` | `gliner-community/gliner_large-v2.5` | GLiNER NER model. |
| `IMAGE_EMBED_MODEL` | `openai/clip-vit-base-patch32` | Image embedding model (CLIP). |

## Document OCR — `OcrClientConfig`

Loaded by `load_ocr_client_env()`. One endpoint answers every "read this
image" in docint: scanned PDF pages, table regions, and image files alike
(`docint/core/ocr/`). Unset, the general vision endpoint is used and image
OCR stays off, which is exactly the behaviour docint had before this
existed.

| Variable | Default | Description |
|---|---|---|
| `OCR_MODEL` | `""` (unset) | Model id of the document OCR model. Setting it also turns image OCR on by default. A **layout** model (`dots-studio/dots.mocr`, `rednote-hilab/dots.ocr`) returns bounding boxes, categories and tables as HTML; anything else is asked for plain text. |
| `OCR_API_BASE` | `OPENAI_API_BASE` | Endpoint serving that model — e.g. the vllm-service `ocr` backend via the router. |
| `OCR_API_KEY` | `OPENAI_API_KEY` | API key for that endpoint. |
| `OCR_TIMEOUT` | `OPENAI_TIMEOUT` | Client timeout. A page takes a minute or two, not seconds. |

`OCR_MODEL` is the seam a dedicated document-parsing model plugs into
without a code change. It replaces the `TABLE_VLM_*` knobs, which were a
stand-in for exactly this: table structure is now recovered by the same
engine, against the same endpoint, under the same budget.

## Host endpoints — `HostConfig`

Loaded by `load_host_env()` (`env_cfg.py:220`).

| Variable | Default | Description |
|---|---|---|
| `BACKEND_HOST` | `http://localhost:8000` | Internal backend URL used by the frontend container. |
| `BACKEND_PUBLIC_HOST` | `http://localhost:8000` | External URL used for document preview links. It does **not** inherit `BACKEND_HOST`: `env_cfg.py:544` falls back to the same literal default as `BACKEND_HOST` does, not to whatever `BACKEND_HOST` was set to. Setting only `BACKEND_HOST` in production therefore leaves preview links pointing at localhost. Set both. |
| `QDRANT_HOST` | `http://localhost:6333` | Qdrant REST URL. |
| `CORS_ALLOWED_ORIGINS` | `http://localhost:5173,http://127.0.0.1:5173` | Comma-separated CORS origins (the Vite dev server). |

## Vector quantization — `QdrantQuantizationConfig`

Loaded by `load_quantization_env()` (`env_cfg.py`). Controls Qdrant [TurboQuant](https://qdrant.tech/articles/turboquant-quantization/) quantization for every dense vector docint stores (the main collection, the `_images` CLIP companion, and the `_entities` companion; sparse vectors are unaffected). New collections are created quantized by default, and at backend startup a best-effort, **add-only** reconcile upgrades pre-existing collections via `update_collection` (Qdrant re-indexes in the background). Setting `QDRANT_QUANTIZATION=none` stops quantizing new collections and disables the reconcile, but never strips quantization from existing collections — Qdrant retains the original vectors either way, so this is an overlay, not a migration.

Requires Qdrant server ≥ 1.18 (the data-plane stack ships v1.18.3).

| Variable | Default | Description |
|---|---|---|
| `QDRANT_QUANTIZATION` | `turbo` | `turbo` enables TurboQuant; `none` disables quantization for new collections and the startup reconcile. Unknown values fall back to `turbo` with a warning. |
| `QDRANT_TURBOQUANT_BITS` | `bits4` | TurboQuant bit width: `bits4` (8× compression, ~scalar-quantization recall), `bits2` (16×), `bits1_5` (~21×), or `bits1` (32×). Unknown values fall back to `bits4` with a warning. |
| `QDRANT_QUANTIZATION_ALWAYS_RAM` | *unset* | Force quantized vectors to stay in RAM (`true`/`false`). Unset leaves the decision to Qdrant. |

## Identity and authentication — `PrincipalConfig`

Loaded by `load_principal_env()` (`env_cfg.py:574`). Configures request-principal resolution via trusted headers from the gateway or dev fallbacks.

| Variable | Default | Description |
|---|---|---|
| `DOCINT_AUTH_HEADER` | `X-Auth-User` | Trusted header carrying the authenticated principal's username. Set by the gateway (`edge-plane`); in dev, falls back to `DOCINT_DEFAULT_IDENTITY`. |
| `DOCINT_DEFAULT_IDENTITY` | *unset* | Dev-only fallback identity when the trusted header is absent. Production must leave this unset so requests without `X-Auth-User` are rejected (401). Also backfills the owner for pre-existing collection and session rows (legacy migration). |
| `DOCINT_GROUPS_HEADER` | `X-Auth-Groups` | Trusted header carrying the principal's group memberships (comma-separated). Set by the gateway; in dev, falls back to `DOCINT_DEFAULT_GROUPS`. |
| `DOCINT_ADMIN_GROUP` | `admins` | Group name that grants admin-level access. Members of this group (from `X-Auth-Groups`) can access collections, chat sessions, and reports owned by other principals via the `owner` query parameter; non-admin principals are unaffected and cross-owner access remains 404. |
| `DOCINT_DEFAULT_GROUPS` | *unset* | Dev-only fallback group memberships (comma-separated) when the trusted header is absent. E.g. `DOCINT_DEFAULT_GROUPS=admins,devs` makes the dev principal an admin and member of devs. |

## Retrieval — `RetrievalConfig`

Loaded by `load_retrieval_env()` (`env_cfg.py:967`).

| Variable | Default | Description |
|---|---|---|
| `RETRIEVE_TOP_K` | `20` | Top-K documents for dense retrieval. |
| `RETRIEVAL_SPARSE_TOP_K` | `20` | Top-K for sparse retrieval. |
| `RETRIEVAL_HYBRID_TOP_K` | `20` | Final top-K after dense/sparse fusion. |
| `RETRIEVAL_HYBRID_ALPHA` | `0.5` | Dense-vs-sparse fusion weight `[0.0, 1.0]`. |
| `RETRIEVAL_VECTOR_QUERY_MODE` | `auto` | One of `auto`, `default`, `sparse`, `hybrid`, `mmr`. |
| `CHAT_RESPONSE_MODE` | `auto` | Response-synthesiser mode: `auto`, `compact`, `refine`. |
| `RERANK_USE_FP16` | `false` | Use FP16 for the reranker. |
| `PARENT_CONTEXT_RETRIEVAL_ENABLED` | `true` | Expand fine chunks to their hierarchical parent context when available. |
| `PARENT_CONTEXT_SAFETY_MARGIN` | `0.95` | Fraction of `OPENAI_CTX_WINDOW` the parent-context packer may consume before windowing. Clamped to `(0, 1]`; values outside that range fall back to `0.95` with a warning. |
| `SOCIAL_SOURCE_DIVERSITY_LIMIT` | `2` | Cap on retrieved chunks per author/hour bucket on social/table collections, enforced by `SocialSourceDiversityPostprocessor` on the chat/query path. Clamped to a minimum of `1`. |

### Parent-context windowing

`PARENT_CONTEXT_RETRIEVAL_ENABLED=true` expands every reranked fine
sub-node hit to its coarse parent (kept intact in the docstore by the
pre-embed re-splitter in `docint/utils/embed_chunking.py`). To keep
arbitrarily large parents from overflowing the chat context, the
postprocessor packs hits greedily against a budget derived from
`OPENAI_CTX_WINDOW × PARENT_CONTEXT_SAFETY_MARGIN` minus the rendered
prompt template and `OPENAI_NUM_OUTPUT`. Parents that fit are emitted
verbatim; parents that do not fit are emitted as a **windowed slice**
centred on the matched sub-node, tagged with
`parent_context_windowed=True`, `parent_full_chars=<N>`, and
`window_chars=<M>` in metadata. The parent's `node_id` is preserved so
citations still resolve to the original source. `grep
parent_context_windowed` in the logs surfaces when windowing fires and
at what ratio.

If windowing fires frequently and you want the full parent back,
raise the chat context window at the provider. For Ollama, a
Modelfile override is the cleanest path — mirror the
`deploy/Modelfile.bge-m3` pattern already used for embeddings:

```text
FROM gemma4:31b-cloud
PARAMETER num_ctx 32768
```

`ollama create docint-gemma4 -f deploy/Modelfile.gemma4.example`, set
`OPENAI_MODEL=docint-gemma4` and `OPENAI_CTX_WINDOW=32768`, and the
packer automatically scales up.

#### What metadata reaches the chat LLM

The postprocessor adds every non-whitelisted metadata key to each
emitted node's `excluded_llm_metadata_keys`, so the chat prompt only
carries a tight set of locator fields: `filename`, `origin`, `page` /
`page_number`, `start_ts` / `end_ts` / `speaker` / `sentence_index`,
`table` (containing `row_index`), `reference_metadata`, and
`docint_doc_kind`. Everything else (`entities`, `relations`,
`llm_description`, `file_hash`, per-column row dumps, internal
hierarchical / ingest markers) stays in `node.metadata` on the
docstore-side parent but is hidden from the LLM prompt.

Whitelisted values are additionally clamped to
1024 characters and stripped of `\n` / `\r` / `\t` runs before
emission. This bounds a single bulky field (e.g. a social-table
`reference_mapping` column that happens to carry long prose) and
neutralises ingested-content prompt-injection attempts that use
newline-heavy formatting to forge chat-role markers inside
`{metadata_str}\n\n{content}`. The clamped copy is visible in
`response.source_nodes`; the full original value remains on the
docstore parent for graph-building and entity analysis consumers that
read it directly.

The `origin` sub-dict is additionally filtered to `filename`,
`mimetype`, `filetype`, and `page_number` only — deployment-internal
keys (absolute `file_path`, tenant IDs) that a future reader might
add cannot silently leak into the LLM prompt or upstream
provider-retained logs.

**Trust boundary**: ingested document content (the `node.text` itself,
plus whitelisted metadata values that copy through from source rows)
is treated as untrusted input to the chat LLM. Prompt-injection
payloads embedded in ingested CSVs / transcripts will reach the
model. The clamp + control-char scrub closes a narrow set of forged-
formatting vectors but is not a substitute for reviewing ingest
sources you do not control.

## Dense embedding client — `EmbedClientConfig`

Loaded by `load_embed_client_env()` (`env_cfg.py:1495`). Dense embeddings
go through an OpenAI-compatible endpoint on every provider; these knobs
only need to change on a CPU dev host that wants dense embeddings routed
to a dedicated container instead of the default OpenAI-style base.

| Variable | Default | Description |
|---|---|---|
| `EMBED_API_BASE` | inherits `OPENAI_API_BASE` | Base URL of the dense-embedding endpoint. Consumed by the OpenAI SDK, which appends `/embeddings` to it, so the value **must include `/v1`** (e.g. `http://embed-only:8000/v1`) — unlike `SPARSE_API_BASE`, which takes the bare host even when both point at the same container. Omitting it 404s every embedding call: an ingest fails on its pre-flight probe before staging any file, and a chat turn ends with the `embedding_unavailable` SSE code, both logging an `EmbeddingEndpointError` that names the resolved URL and this knob. For the `embed-only` deployment shape (CPU container, pairs with `gliner-only` / `rerank-only` / `clip-only` for non-CUDA dev; serves dense embedding, sparse weights, and tokenization from one bge-m3 instance), set `EMBED_API_BASE=http://embed-only:8000/v1`. |
| `EMBED_API_KEY` | inherits `OPENAI_API_KEY` | Bearer token for the dense-embedding endpoint. `embed-only` requires no auth (trust `inference-net`); the full router requires the master key. |

`EmbedClientConfig` has no timeout field of its own — the dense-embedding
request timeout is `EMBED_TIMEOUT_SECONDS`, documented under
[Embedding — `EmbeddingConfig`](#embedding--embeddingconfig) above.

## Sparse encoder & hybrid retrieval — `SparseClientConfig`

Loaded by `load_sparse_client_env()` (`env_cfg.py:1440`) and
`resolve_enable_hybrid()` (`env_cfg.py:1025`). Sparse embedding is a
remote HTTP call on every provider — `RemoteSparseEncoder` POSTs to
`{SPARSE_API_BASE}/pooling` (`task: token_classify`) and
`{SPARSE_API_BASE}/tokenize`. Its wire format is frozen: production
collections depend on the exact vectors it produces, so its request/
response shape must not change.

| Variable | Default | Description |
|---|---|---|
| `SPARSE_API_BASE` | inherits `OPENAI_API_BASE` | Base URL of the sparse-encoding endpoint. The full vllm-service router exposes `/pooling` and `/tokenize` as LiteLLM pass-throughs against the same base. For the `embed-only` deployment shape (CPU container, pairs with `gliner-only` / `rerank-only` / `clip-only` for non-CUDA dev), set `SPARSE_API_BASE=http://embed-only:8000`. |
| `SPARSE_API_KEY` | inherits `OPENAI_API_KEY` | Bearer token for the sparse endpoint. `embed-only` requires no auth (trust `inference-net`); the full router requires the master key. |
| `SPARSE_TIMEOUT` | inherits `OPENAI_TIMEOUT` | Per-request HTTP timeout in seconds for the sparse endpoint. |
| `ENABLE_HYBRID` | *unset* (see below) | Explicit override for hybrid (dense + sparse) retrieval; `true`/`1`/`yes`/`on` (case-insensitive) enables it, anything else set disables it. When unset, defaults to **true** if `INFERENCE_PROVIDER=vllm` (the router's `/pooling` and `/tokenize` pass-throughs are always present) **or** `SPARSE_API_BASE` is explicitly set (the `embed-only` shape); **false** otherwise — `SPARSE_API_BASE` alone can't signal availability since it silently inherits `OPENAI_API_BASE` and is therefore never actually empty. |

An ingest job calls `RAG.probe_sparse_endpoint()` before its first batch
when hybrid is enabled, and fails the job cleanly if the sparse endpoint
is unreachable — deliberately **not** fail-soft like the reranker,
because a transport failure partway through an ingest would otherwise
write dense-only points into a hybrid collection and corrupt it.

## Rerank client — `RerankClientConfig`

Loaded by `load_rerank_client_env()`. Reranking is always a remote call:
`VLLMRerankPostprocessor` POSTs to `{RERANK_API_BASE}/rerank` in the Jina shape
on every provider. Transport failure degrades to the original retrieval order
rather than failing the query — there is no local fallback model.

| Variable | Default | Description |
|---|---|---|
| `RERANK_API_BASE` | inherits `OPENAI_API_BASE` | Base URL of the rerank endpoint. The full vllm-service router exposes `/v1/rerank`; for the `rerank-only` deployment shape set `http://rerank-only:8000`. |
| `RERANK_API_KEY` | inherits `OPENAI_API_KEY` | Bearer token. `rerank-only` has no auth; the full router requires the master key. |
| `RERANK_TIMEOUT` | inherits `OPENAI_TIMEOUT` | Per-request HTTP timeout in seconds. |

## CLIP client — `CLIPClientConfig`

Loaded by `load_clip_client_env()`. CLIP image+text embedding is a remote
service hosted by vllm-service; docint POSTs to `{CLIP_API_BASE}/clip/embed_*`
and probes `{CLIP_API_BASE}/clip/dimension` at construction to size the
`_images` collection without spending an embed call. Like the NER client and
unlike the others, these do **not** inherit the OpenAI settings.

The CLIP model identity is set on the vllm-service container (`CLIP_MODEL`
there); docint does not read `IMAGE_EMBED_MODEL` any more.

| Variable | Default | Description |
|---|---|---|
| `CLIP_API_BASE` | `http://vllm-router:4000` | Base URL of the CLIP service. For the `clip-only` deployment shape set `http://clip-only:8000`. |
| `CLIP_API_KEY` | *unset* | Sent as `Authorization: Bearer …` when set. Required by the router, absent on `clip-only`. |
| `CLIP_TIMEOUT` | `30.0` | Per-request HTTP timeout in seconds. |

## Entity resolution — `ResolutionConfig`

Loaded by `load_resolution_env()`. Entity resolution is the only mechanism that
merges *semantically* similar entities (`USA` / `United States`); the
`orthographic` merge mode in `core/ner.py` only collapses spelling variants.
The pipeline is normalize → exact alias → type-blocked vector match → LLM
tie-break → mint, persisted one point per canonical entity in the hidden
`{collection}_entities` companion.

Out-of-range values here **raise** at startup rather than warning.

| Variable | Default | Description |
|---|---|---|
| `RES_AUTO_RESOLVE` | `true` | Run resolution as a stage inside the ingest job. When off, resolve on demand via the `resolve` CLI or `POST /collections/entities/resolve`. |
| `RES_BATCH_SIZE` | inherits `INGESTION_BATCH_SIZE` (`50`) | Embed/resolve batch cadence; bounds memory on large collections. Must be ≥ 1. |
| `RES_CASE_NORMALIZE` | `true` | Case-fold surface forms before matching. |
| `RES_EMBED_THRESHOLD` | `0.86` | Cosine floor, in `[0, 1]`, above which a vector match is accepted as the same entity. |
| `RES_LLM_TIEBREAK` | `true` | Ask the chat model to adjudicate near-threshold pairs. Conservative by design: it merges only on an explicit yes. |
| `RES_VECTOR_K` | `5` | Candidates fetched from the entity index per surface form, in `[1, 100]`. |

## Nextext media processing — `NextextConfig`

Loaded by `load_nextext_env()`. docint ships no media runtime: audio and video
are forwarded to a remote [Nextext](https://github.com/nos-tromo/nextext)
service, which returns a transcript plus keyframes. Leaving `NEXTEXT_API_BASE`
unset disables the path — video/audio is skipped fail-soft and images are
unaffected.

| Variable | Default | Description |
|---|---|---|
| `NEXTEXT_API_BASE` | *empty* (disabled) | Base URL of the Nextext API. **Must include Nextext's `/api/v1` prefix** (e.g. `http://nextext-backend:8000/api/v1`): docint calls `{base}/jobs`, so omitting it 404s every request. |
| `NEXTEXT_API_KEY` | *unset* | Sent as `Authorization: Bearer …` when set. Current Nextext does not validate it — forward-compatible no-op. |
| `NEXTEXT_AUTH_HEADER` | `X-Auth-User` | Trusted identity header docint sends. Must match Nextext's own `NEXTEXT_AUTH_HEADER`, which it uses to resolve the caller and without which it answers 401. |
| `NEXTEXT_IDENTITY` | `docint` | Identity sent under that header. Set it empty to send no header and fall back to Nextext's server-side default identity. |
| `NEXTEXT_MAX_CONCURRENCY` | `4` | Clips submitted to Nextext in parallel per ingest (cache misses only). |
| `NEXTEXT_POLL_INTERVAL` | `2.0` | Seconds between job-status polls. |
| `NEXTEXT_POLL_MAX_SECONDS` | `1800.0` | Wall-clock budget per job before the status becomes `timeout`. |
| `NEXTEXT_TIMEOUT` | `30.0` | Per-request HTTP timeout in seconds. |

Keyframe sampling is configured **here**, on the docint side, and forwarded as
per-job options — Nextext has no server-side keyframe knob, and its schema
defaults apply only to callers that omit these fields.

| Variable | Default | Description |
|---|---|---|
| `KEYFRAMES_MAX` | `20` | Hard ceiling on frames per clip. Nextext rejects more than 200. |
| `KEYFRAMES_PER_MINUTE` | `4` | Target frames sampled per minute of video. |
| `KEYFRAME_DEDUP_COSINE` | `0.95` | Client-side near-duplicate pruning before storage, in `[0, 1]`. A frame whose cosine similarity to an already-kept frame reaches this is dropped. |

## Metrics — `MetricsConfig`

Loaded by `load_metrics_env()`.

| Variable | Default | Description |
|---|---|---|
| `METRICS_ENABLED` | `true` | Expose `GET /metrics` (Prometheus request counters and histograms) for the obs-plane scrape target. Aggregate only — no document or user data. Unauthenticated, like `/version` and `/config`. |

## Serve bind — `ServeConfig`

Loaded by `load_serve_config()`. Used by the `serve` entry point; the Docker
image sets its own bind in the container CMD.

| Variable | Default | Description |
|---|---|---|
| `DOCINT_HOST` | `0.0.0.0` | Interface uvicorn binds. |
| `DOCINT_PORT` | `8000` | Port uvicorn binds. |

## Pipeline — `PipelineConfig`

Loaded by `load_pipeline_config()` (`env_cfg.py:850`). Controls the
page-level PDF pipeline in `docint/core/readers/documents/`.

| Variable | Default | Description |
|---|---|---|
| `PIPELINE_VERSION` | `3.4.0` | Semver marker written into pipeline artifacts. |
| `PIPELINE_TEXT_COVERAGE_THRESHOLD` | `0.01` | Chars-per-area threshold used to classify a page as scanned. |
| `PIPELINE_MAX_RETRIES` | `2` | Retry budget per page stage. |
| `PIPELINE_MAX_WORKERS` | `4` | Parallel workers per document. |
| `PIPELINE_FORCE_REPROCESS` | `false` | Ignore cached artifacts. |
| `PIPELINE_OCR_ENABLED` | `true` | Read pages that have no text layer of their own through the OCR engine. |
| `PIPELINE_OCR_TIMEOUT` | inherits `OPENAI_TIMEOUT` | Per-request timeout for an OCR call. Set it only to give OCR a *tighter* budget than the rest of the app — an OCR model takes a minute or two per page where a chat model takes seconds, so a fixed low value cuts every page off mid-flight and surfaces as `Request timed out`. |
| `PIPELINE_OCR_MAX_RETRIES` | `1` | SDK retries per call. |
| `PIPELINE_OCR_MAX_IMAGE_DIM` | `1024` | Longest side of an image sent to a model that reads text only. |
| `PIPELINE_OCR_MAX_PIXELS` | `2007040` | Pixel budget of a page rendered for a *layout* model. Must not **exceed** the server's own cap (vllm-service `OCR_MM_PROCESSOR_KWARGS`, same default) — a model that resizes the render itself reports boxes against an image the caller never saw. Below the cap is safe: it is just a smaller page. |
| `PIPELINE_OCR_MAX_TOKENS` | `4096` | Max tokens the model may generate per call. Raise it for a layout model: a full page of JSON is far longer than a page of text. |
| `PIPELINE_TABLE_OCR` | `true` | Re-read tables whose structure the geometric pass could not recover. |
| `PIPELINE_TABLE_OCR_MAX_IMAGE_DIM` | `1536` | Longest side of a rendered table region — larger than a page's, since a table's digits must stay legible. |
| `PIPELINE_ARTIFACTS_DIR` | `~/docint/artifacts` (via `PathConfig`) | Root dir for pipeline artifacts. |

Both lanes — scanned pages and weak tables — read through the one engine in
`docint/core/ocr/`, against the endpoint configured by `OCR_*` above, and share
its per-document failure budget. Which model is configured decides how much
comes back: a **layout** model (dots.ocr / dots.mocr) returns headings, text,
tables with cells, figures and page furniture, so a scanned page chunks exactly
like a digital one; any other model is asked for plain text and yields one block
per page, which is what this pipeline did before.

The engine distinguishes two failure modes, because they deserve opposite
responses. An endpoint that **never answers** (timeout, connection refused)
would cost a full `PIPELINE_OCR_TIMEOUT` on every remaining page, so after three
consecutive such calls the engine stops calling for the rest of the document. An
endpoint that **answers with an error status** costs about a second and typically
recovers within a few, so it costs its own call and nothing more — a transient
upstream burst must not discard a document.

Outcomes are reported in the pipeline summary as `pages_ocr_read`,
`pages_ocr_failed` (attempted, produced nothing) and `pages_ocr_skipped` (not
attempted after the engine gave up). Note that `pages_ocr` counts pages that
*needed* OCR, not pages that got it — check the other three before concluding a
document was fully read.

## Ingestion — `IngestionConfig`

Loaded by `load_ingestion_env()` (`env_cfg.py:365`). Controls chunking
sizes, batch sizes, and retry behaviour for the ingestion pipeline.

| Variable | Default | Description |
|---|---|---|
| `COARSE_CHUNK_SIZE` | `8192` | Parent chunk token budget. |
| `FINE_CHUNK_SIZE` | `1024` | Child chunk token budget. |
| `FINE_CHUNK_OVERLAP` | `0` | Overlap between child chunks. |
| `SENTENCE_SPLITTER_CHUNK_SIZE` | `1024` | Sentence splitter chunk size (bytes). |
| `SENTENCE_SPLITTER_CHUNK_OVERLAP` | `64` | Sentence splitter overlap. |
| `HIERARCHICAL_CHUNKING_ENABLED` | `true` | Enable two-level parent/child chunking. |
| `INGESTION_BATCH_SIZE` | `50` | Files per ingestion batch. |
| `DOCSTORE_BATCH_SIZE` | `100` | Nodes per docstore upsert. |
| `DOCLING_ACCELERATOR_NUM_THREADS` | `4` | Docling backend thread count. |
| `INGEST_BENCHMARK_ENABLED` | `false` | Emit ingestion throughput logs. |
| `DOCSTORE_MAX_RETRIES` | `3` | Retry budget for docstore upserts. |
| `DOCSTORE_RETRY_BACKOFF_SECONDS` | `0.25` | Initial retry backoff. |
| `DOCSTORE_RETRY_BACKOFF_MAX_SECONDS` | `2.0` | Max retry backoff. |
| `INGEST_FAIL_FAST` | `false` | Abort the run on the first file that fails instead of skipping it. |
| `INGEST_MANIFEST_ENABLED` | `true` | SQLite ingest manifest. Also caches Nextext transcripts by media-file hash, so re-ingesting an unchanged clip skips the round-trip entirely. |
| `INGEST_PIPELINE_OVERLAP_ENABLED` | `false` | Overlap reading and embedding stages instead of running them in sequence. |
| `INGEST_QUEUE_MAX_SIZE` | `4` | Documents buffered between the reader and the embedder when overlap is on. |
| `STREAMING_READERS_ENABLED` | `true` | Readers yield documents as they parse rather than materialising a whole file first, which bounds peak memory on large CSV/JSONL files. |
| `MEDIA_FILETYPES` | see below | Audio/video extensions the standalone media pre-pass claims, comma-separated; a leading dot is added if omitted and entries are lowercased. These route through Nextext, **not** the generic reader whitelist. Default: `.mp4,.mov,.mkv,.webm,.avi,.m4v,.mpg,.mpeg,.mp3,.m4a,.wav,.flac,.aac,.ogg,.opus,.wma`. |

Ingest *jobs* (the server-owned runs behind `POST /ingest/finalize`) take one
knob of their own, read by `load_ingest_concurrency()`:

| Variable | Default | Description |
|---|---|---|
| `DOCINT_INGEST_CONCURRENCY` | `1` | Ingest jobs allowed to run at once, process-wide. The default serialises runs; raising it is opt-in, and overlapping runs over the *same* collection are refused regardless (409). Unparseable values fall back to `1`. |

The default supported file extensions (hard-coded in
`load_ingestion_env`) include `.pdf`, `.docx`, `.txt`, `.md`, `.csv`,
`.tsv`, `.xlsx`, `.xls`, `.parquet`, `.jsonl`, `.jpg`, `.jpeg`, `.png`,
`.gif`, `.mp3`, `.mp4`, `.m4a`, `.m4v`, `.wav`, `.ogg`, `.avi`, `.flv`,
`.mkv`, `.mov`, `.mpeg`, `.mpg`, `.webm`, `.wmv`.

## Image ingestion — `ImageIngestionConfig`

Loaded by `load_image_ingestion_config()` (`env_cfg.py:264`).

| Variable | Default | Description |
|---|---|---|
| `IMAGE_INGESTION_ENABLED` | `true` | Route image files through the image pipeline. |
| `IMAGE_EMBEDDING_ENABLED` | `true` | Compute CLIP embeddings for images. |
| `IMAGE_TAGGING_ENABLED` | `true` | Call the vision LLM for tags/captions. |
| `IMAGE_QDRANT_COLLECTION` | `{collection}_images` | Image-vector collection template. |
| `IMAGE_QDRANT_VECTOR_NAME` | `image-dense` | Vector field name. |
| `IMAGE_CACHE_BY_HASH` | `true` | Cache embeddings keyed by image hash. |
| `IMAGE_FAIL_ON_EMBED_ERROR` | `false` | Treat embedding failures as fatal. |
| `IMAGE_FAIL_ON_TAG_ERROR` | `false` | Treat tagging failures as fatal. |
| `IMAGE_RETRIEVE_TOP_K` | `5` | CLIP candidates the image lane contributes to each query's ranking. |
| `IMAGE_RERANK_MIN_SCORE` | `0.05` | Minimum reranker relevance score for a retrieved image to surface as a source. |
| `IMAGE_TAGGING_MAX_IMAGE_DIM` | `1024` | Max dimension for images sent to the vision tagging endpoint. |
| `IMAGE_OCR_ENABLED` | on when `OCR_MODEL` is set | Read the text *inside* an image (a screenshot, a photographed letter, a slide) and store it as `ocr_text`. |
| `KEYFRAME_OCR_ENABLED` | `false` | Read video keyframes too. Off by default: a clip contributes many frames and only slides tend to carry text. |
| `IMAGE_OCR_MAX_IMAGE_DIM` | `1536` | Longest side of an image sent to be read. |

Images retrieve as ordinary sources: `IMAGE_RETRIEVE_TOP_K` CLIP candidates join
the text hits *before* ranking, so the shared reranker scores both modalities in
one pass and images compete with text chunks for the answer's source slots. The
model therefore sees images in its context and can cite them by number.
`IMAGE_RERANK_MIN_SCORE` then drops the ones that are merely nearest rather than
relevant — the top-n cut alone cannot protect a sparse collection, where an
irrelevant image would take a slot for lack of competition.

Captioning and reading are different questions about the same picture, and both
are asked. The caption says what an image *shows*; `IMAGE_OCR_ENABLED` adds what
it *says*, stored as `ocr_text` on the image point, put ahead of the caption in
the node text and in the full-text search index, so a screenshot is findable by
the words printed in it rather than only through a paraphrase of them. The
same applies to a figure lifted out of a PDF and to a standalone image file —
one code path reads them all.

Turning it on affects **newly ingested** images only: there is no payload
migration, so an existing collection gains `ocr_text` by being re-ingested (file
hashes make an unchanged document cheap to re-run, but the images themselves are
cached by hash — clear the `_images` companion, or ingest into a fresh
collection, to have them read).

The floor is applied to the **reranker** score, not to CLIP similarity, because raw
CLIP cosine is not comparable across queries — measured on a live collection, an
unrelated query and a genuinely matching one both land in a ~0.20–0.30 band, so no
absolute CLIP threshold separates them. Reranker scores do separate: "a relevant
image exists" measured 0.12–0.90 while "nothing relevant exists" measured 0.0037 and
below. Raise the floor to make the image lane stricter; set it to `0` to surface every
CLIP candidate (the pre-gate behavior). If the rerank endpoint is unreachable the lane
degrades to ungated CLIP matches rather than going silent.

Because the deployed CLIP checkpoint (`openai/clip-vit-base-patch32`) has an
English-only text tower, queries are translated to English before embedding whenever
`RESPONSE_LANGUAGE` is not English. The translation reuses the chat model via
`TRANSLATE_MODEL` and is cached; an outage degrades to embedding the untranslated
query. Note this keys off the configured locale, not the language actually typed — a
German query in an English-locale deployment is not translated.

## NER — `NERConfig`

Loaded by `load_ner_env()` (`env_cfg.py:582`).

| Variable | Default | Description |
|---|---|---|
| `NER_ENABLED` | `true` | Run entity/relation extraction during ingestion. |
| `NER_MAX_CHARS` | `1024` | Max chars per node passed to GLiNER. |
| `NER_MAX_WORKERS` | `4` | Parallel NER workers. |

### NER endpoint — `NERClientConfig`

Loaded by `load_ner_client_env()`. NER is a remote service (Ray Serve GLiNER
hosted by vllm-service), reached at `{NER_API_BASE}/gliner`. Unlike the embed,
sparse, rerank and OCR clients, these do **not** inherit `OPENAI_API_BASE` /
`OPENAI_API_KEY`.

| Variable | Default | Description |
|---|---|---|
| `NER_API_BASE` | `http://vllm-router:4000` | Base URL of the GLiNER service. For the `gliner-only` deployment shape (no router), set `http://gliner-only:8000`. |
| `NER_API_KEY` | *unset* | Sent as `Authorization: Bearer …` when set. The router enforces auth (use the master key); the `gliner-only` shape has none, so leave it unset there. |
| `NER_THRESHOLD` | `0.3` | GLiNER confidence floor below which a span is discarded. |
| `NER_TIMEOUT` | `30.0` | Per-request HTTP timeout in seconds. |

## Hate-speech detection — `HateSpeechConfig`

Loaded by `load_hate_speech_env()` (`env_cfg.py:185`).

| Variable | Default | Description |
|---|---|---|
| `ENABLE_HATE_SPEECH_DETECTION` | `false` | Run hate-speech classification per chunk during ingestion. |
| `HATE_SPEECH_MAX_CHARS` | `2048` | Max chars per chunk sent to the detector. |
| `HATE_SPEECH_MAX_WORKERS` | `1` | Parallel hate-speech workers. |

## Graph-RAG — `GraphRAGConfig`

Loaded by `load_graphrag_env()` (`env_cfg.py:141`).

| Variable | Default | Description |
|---|---|---|
| `GRAPHRAG_ENABLED` | `true` | Enable graph-assisted query expansion. |
| `GRAPHRAG_NEIGHBOR_HOPS` | `2` | Graph hops walked for expansion. |
| `GRAPHRAG_TOP_K_NODES` | `50` | Max graph nodes kept in memory. |
| `GRAPHRAG_MIN_EDGE_WEIGHT` | `3` | Min edge weight for graph filtering. |
| `GRAPHRAG_MAX_NEIGHBORS` | `6` | Max neighbours appended to a query. |

## Summarisation — `SummaryConfig`

Loaded by `load_summary_env()` (`env_cfg.py:2145`).

| Variable | Default | Description |
|---|---|---|
| `SUMMARY_COVERAGE_TARGET` | `0.70` | Target document coverage ratio for summaries (clamped to `[0.0, 1.0]`). |
| `SUMMARY_FINAL_SOURCE_CAP` | `24` | Max merged sources in the final answer. |
| `SUMMARY_ON_INGEST` | `true` | Whether a collection summary rebuild runs automatically at the end of an ingest job. Fail-soft: an exception from the rebuild (e.g. an LLM outage) is caught, logged, and reported to the client as a `warning` SSE event (`"Collection summary generation failed."`) — the ingest job still completes normally and its documents are ingested and retrievable regardless. |
| `SUMMARY_MAP_WINDOW_TOKENS` | `3000` | Target token budget per map-stage window (clamped to a minimum of `100`). |
| `SUMMARY_REDUCE_FANIN` | `10` | Max map-stage summaries merged per reduce-stage call (clamped to a minimum of `2`). |
| `SUMMARY_MAX_LLM_CALLS` | `500` | Hard upper bound on the LLM calls a single tree-summary rebuild may issue, as a runaway-cost guard (clamped to a minimum of `1`). Enforced between units, *inside* one unit's window loop (so a single huge transcript cannot issue thousands of map calls on its own), on the intra-unit fold, and across the reduce-fold tiers. Only the one final synthesis call is exempt, so a capped rebuild still produces an answer. Cache hits cost no calls and are resolved before the cap applies. A rebuild that hits the cap is marked `partial: true` in `summary_diagnostics`, which travels through the cache to the API and the SPA's coverage banner; a cap-truncated unit summary is deliberately *not* written to the per-unit map cache, since it would otherwise be stored against the unit's full content fingerprint and served as complete forever. |

`RAG.build_tree_summary()` (called by `POST /summarize`, an ingest job's
post-ingest summary stage, and `uv run query --summary`) replaced the
sampling-based summarizer with a map-reduce ("tree") pipeline:
`docint/core/summary/units.py` partitions every point in the collection into
map units — one per document (grouped by `file_hash`, falling back to
filename), or one per coarse author/hour bucket for row-level social
content — and `docint/core/summary/tree.py` summarizes each unit
independently (windowed across multiple LLM calls at `SUMMARY_MAP_WINDOW_TOKENS`
when a unit is large, with an intra-unit fold to merge those windows), then
folds the per-unit summaries hierarchically, `SUMMARY_REDUCE_FANIN` at a
time, down to one final synthesis call.

Per-unit map results are cached in the collection's own SQLite KV store
(the same `SQLiteKVStore` file that backs the LlamaIndex docstore of
serialized hierarchical nodes, at
`{QDRANT_SRC_DIR}/{collection}/{collection}_kv.db`, under a separate
namespace), keyed by a fingerprint
over the unit's member point ids and their text content. An incremental
re-ingest therefore only re-summarizes units whose content actually
changed; unchanged units are served from cache at no LLM cost, and a
completed rebuild prunes cache entries for units that no longer exist. The
cache validator additionally folds in a fingerprint of the summarize/map/fold
prompts, the `SUMMARY_COVERAGE_TARGET` / `SUMMARY_FINAL_SOURCE_CAP` /
`SUMMARY_MAP_WINDOW_TOKENS` / `SUMMARY_REDUCE_FANIN` knobs above, and the
chat model id — changing any of them invalidates every cached map entry (and
the final cached summary) once, the next time a summary is built. Ingest
already bumps a separate revision counter that invalidates the final cached
summary on every re-ingest, unrelated to this fingerprint.

Summary-rebuild jobs (`kind="summary"`, queued by `POST /summarize` — see
[api-reference.md](api-reference.md#post-summarize)) run under their own
concurrency semaphore, read by `load_summary_concurrency()`:

| Variable | Default | Description |
|---|---|---|
| `DOCINT_SUMMARY_CONCURRENCY` | `1` | Summary-rebuild jobs allowed to run at once, process-wide — bounded separately from `DOCINT_INGEST_CONCURRENCY` so a rebuild's burst of map/reduce LLM calls can never starve (or be starved by) an ingest worker slot. Unparseable values fall back to `1`; values below `1` clamp to `1`. |

## Sessions — `SessionConfig`

Loaded by `load_session_env()` (`env_cfg.py:1105`).

| Variable | Default | Description |
|---|---|---|
| `SESSION_STORE` | *unset* | Full SQLAlchemy URL. If set, wins over `SESSIONS_DB_PATH`. |
| `SESSIONS_DB_PATH` | `~/docint/sessions.sqlite3` | SQLite path used to build a `sqlite:///` URL if `SESSION_STORE` is not set. |

## Frontend — `FrontendConfig`

Loaded by `load_frontend_env()` (`env_cfg.py:111`).

| Variable | Default | Description |
|---|---|---|
| `FRONTEND_COLLECTION_TIMEOUT` | `120` | Seconds the UI will wait for `/collections/list` before falling back. |
| `NER_GRAPH_TOP_K` | `80` | Default node count for the Analysis entity-graph view; the SPA seeds its control from this. |
| `NER_GRAPH_MAX_TOP_K` | `500` | Ceiling for the graph node count (API clamp + UI control max). Raise for large corpora. |
| `DOCINT_CLIENT_MAX_BODY_SIZE` | `1g` | Maximum upload size, in nginx size syntax. Read **twice**: the backend advertises it via `GET /config` so the SPA can size upload batches, and the frontend nginx image reads the same variable to enforce `client_max_body_size`. The two must stay in sync — a backend-only change lets the SPA send batches nginx then rejects. |

All five values above are served to the SPA by `GET /config`, alongside
`RESPONSE_LANGUAGE`.

## Runtime device — `RuntimeConfig`

Loaded by `load_runtime_env()` (`env_cfg.py:1070`).

| Variable | Default | Description |
|---|---|---|
| `USE_DEVICE` | `auto` | Preferred device for local auxiliary models: `auto`, `cpu`, `mps`, `cuda`, or `cuda:<index>`. When set to `cpu`, `CUDA_VISIBLE_DEVICES=""` is forced at import time to prevent accidental GPU context init. |

## Logging — `LoggingConfig`

Loaded by `load_logging_env()` (`docint/utils/env_cfg.py`), applied by
`init_logger()` (`docint/utils/logger_cfg.py`).

| Variable | Default | Description |
|---|---|---|
| `LOG_LEVEL` | `INFO` | Minimum level for the single stderr sink. There is no file sink; under Docker the compose `local` driver owns retention (50 MB × 5, compressed). |
| `LOG_PROGRESS_INTERVAL_S` | `30` | Seconds between heartbeat lines while one long job stage repeats the same progress message. Stage changes and a stage's final count always log regardless. `0` disables throttling and logs every progress message — the debug setting; on a large ingest it is thousands of lines. |

**Queries, answers and document text are never logged** — only their shapes
(retrieval mode, source and image counts, rerank state, duration). One
`Turn complete | …` line is emitted per chat turn on that basis, and the
ingest run summary counts files and nodes rather than naming content. This is
not a switch; there is no level at which document text reaches the log.

Two behaviours are wired here rather than being configurable, because
neither has a defensible "off":

- **Uvicorn's loggers are re-dispatched into loguru**, so `docker logs`
  carries one format on one stream. Only `uvicorn`, `uvicorn.error` and
  `uvicorn.access` are bridged — never the root logger, which would pull
  in httpx, qdrant-client, llama-index and transformers at a volume
  nobody can predict for an airgapped deployment.
- **The container healthcheck's own access lines are dropped.** It probes
  `GET /version` from loopback every 30 s (every 3 s during
  `start_period`), which on a measured run was roughly half of stdout.
  The filter is narrow and fails open: a `/version` that returns 5xx, or
  one from any non-loopback caller, is kept.

See also `INGEST_BENCHMARK_ENABLED` under
[Ingestion](#ingestion--ingestionconfig), which adds per-run throughput
telemetry (`nodes_per_s`, batch counts) on top of the run summary.

## Paths — `PathConfig`

Loaded by `load_path_env()` (`docint/utils/env_cfg.py`). Every path expands `~`.

| Variable | Default | Description |
|---|---|---|
| `DATA_PATH` | `~/docint/data` | Root directory for ingestion inputs. Compose pins it to `/var/lib/docint/pipeline/data` (the `pipeline-storage` volume) — the container's `$HOME` is read-only. |
| `QUERIES_PATH` | `~/docint/queries.txt` | Default query input file for the CLI. |
| `RESULTS_PATH` | `~/docint/results` | Directory for CLI export artifacts. Compose: `/var/lib/docint/pipeline/results`. |
| `PIPELINE_ARTIFACTS_DIR` | `~/docint/artifacts` | Pipeline artifact root (also read by `PipelineConfig`). Compose: `/var/lib/docint/pipeline/artifacts`. |
| `QDRANT_SRC_DIR` | `~/docint/qdrant_sources` | Where raw source files are staged for preview. |
| `HF_HUB_CACHE` | `~/.cache/huggingface/hub` | HF Hub cache path. |

`PathConfig` also exposes a derived `prompts` path pointing at
`docint/utils/prompts/` — it is not overridable by env var.

## Response language — `LanguageConfig`

Loaded by `load_language_env()` in `env_cfg.py`.

| Variable | Default | Description |
|---|---|---|
| `RESPONSE_LANGUAGE` | `en` | Locale for LLM instructions, the hate-speech framing, and user-facing clarification messages. Supported values: `en`, `de`. Switches the active prompt directory from `docint/utils/prompts/en/` to `docint/utils/prompts/de/` and selects the matching entries from `docint/utils/ui_strings.py`. Unknown values fall back silently to `en`. |

The German pack reframes the hate-speech detector around
"Gruppenbezogene Menschenfeindlichkeit" with explicit categories and a
`NICHT kennzeichnen` exclusion list, so mild individual insults
("Du Depp") are not flagged. The JSON output schema is unchanged across
languages.

## Response validation — `ResponseValidationConfig`

Loaded by `load_response_validation_env()` (`env_cfg.py:935`).

| Variable | Default | Description |
|---|---|---|
| `RESPONSE_VALIDATION_ENABLED` | `true` | Run the `ResultValidationResponseAgent` to cross-check answers against sources. |
| `RESPONSE_VALIDATION_SOURCE_BUDGET_CHARS` | `48000` | Total characters of source text shown to the validator, shared across that answer's sources (shortest-first fair share, unused share redistributed). Sources still trimmed are marked inline and the prompt tells the validator not to read hidden text as an unsupported claim. Too small a budget makes the validator flag grounded answers as hallucinated. |

## Corrective retry — `CorrectiveRetryConfig`

Loaded by `load_corrective_retry_env()`.

| Variable | Default | Description |
|---|---|---|
| `CORRECTIVE_RETRY_ENABLED` | `true` | Re-answer once with a reformulated query when validation flags a mismatch *and* the answer is weak. Depends on response validation being on — without a mismatch verdict there is nothing to retry. A triggered retry costs up to three extra LLM round-trips (reformulate, regenerate, re-validate) inside the same request, so a turn that fires it is noticeably slower; untriggered turns are unaffected. |

## Offline mode

- `DOCINT_OFFLINE` — default `1`. When truthy, Docint sets
  `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`,
  `HF_HUB_DISABLE_TELEMETRY=1`, `HF_HUB_DISABLE_SYMLINKS_WARNING=1`, and
  `KMP_DUPLICATE_LIB_OK=TRUE`. See `set_offline_env()` in `env_cfg.py:12`.
