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
| `SPARSE_MODEL` | `Qdrant/all_miniLM_L6_v2_with_attentions` (ollama) / `BAAI/bge-m3` (vllm) | Sparse retrieval model. |
| `TEXT_MODEL` | `gpt-oss:20b` (ollama) / `Qwen/Qwen3.5-2B` (vllm) / `gpt-4o` (openai) | Chat / generation model. |
| `VISION_MODEL` | `qwen3.5:9b` (ollama) / `Qwen/Qwen3.5-2B` (vllm) / `gpt-4o` (openai) | Vision-OCR model. |
| `RERANK_MODEL` | `BAAI/bge-reranker-v2-m3` | Cross-encoder reranker. |
| `NER_MODEL` | `gliner-community/gliner_large-v2.5` | GLiNER NER model. |
| `IMAGE_EMBED_MODEL` | `openai/clip-vit-base-patch32` | Image embedding model (CLIP). |

## Host endpoints — `HostConfig`

Loaded by `load_host_env()` (`env_cfg.py:220`).

| Variable | Default | Description |
|---|---|---|
| `BACKEND_HOST` | `http://localhost:8000` | Internal backend URL used by the frontend container. |
| `BACKEND_PUBLIC_HOST` | `BACKEND_HOST` | External URL used for document preview links. |
| `QDRANT_HOST` | `http://localhost:6333` | Qdrant REST URL. |
| `CORS_ALLOWED_ORIGINS` | `http://localhost:8501,http://127.0.0.1:8501` | Comma-separated CORS origins. |

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

## Pipeline — `PipelineConfig`

Loaded by `load_pipeline_config()` (`env_cfg.py:850`). Controls the
page-level PDF pipeline in `docint/core/readers/documents/`.

| Variable | Default | Description |
|---|---|---|
| `PIPELINE_VERSION` | `1.0.0` | Semver marker written into pipeline artifacts. |
| `PIPELINE_TEXT_COVERAGE_THRESHOLD` | `0.01` | Chars-per-area threshold used to classify a page as scanned. |
| `PIPELINE_MAX_RETRIES` | `2` | Retry budget per page stage. |
| `PIPELINE_MAX_WORKERS` | `4` | Parallel workers per document. |
| `PIPELINE_FORCE_REPROCESS` | `false` | Ignore cached artifacts. |
| `PIPELINE_ENABLE_VISION_OCR` | `true` | Use the vision LLM as an OCR fallback. |
| `PIPELINE_VISION_OCR_TIMEOUT` | `60.0` | Per-request timeout for the vision OCR call. |
| `PIPELINE_VISION_OCR_MAX_RETRIES` | `1` | Per-call retries. |
| `PIPELINE_VISION_OCR_MAX_IMAGE_DIM` | `1024` | Max pixel dimension for rendered OCR images. |
| `PIPELINE_VISION_OCR_MAX_TOKENS` | `4096` | Max tokens the vision LLM may generate per OCR call. |
| `PIPELINE_ARTIFACTS_DIR` | `~/docint/artifacts` (via `PathConfig`) | Root dir for pipeline artifacts. |

## Ingestion — `IngestionConfig`

Loaded by `load_ingestion_env()` (`env_cfg.py:365`). Controls chunking
sizes, batch sizes, and retry behaviour for the ingestion pipeline.

| Variable | Default | Description |
|---|---|---|
| `COARSE_CHUNK_SIZE` | `8192` | Parent chunk token budget. |
| `FINE_CHUNK_SIZE` | `8192` | Child chunk token budget. |
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
| `IMAGE_RETRIEVE_TOP_K` | `5` | Top-K image matches per text query. |
| `IMAGE_TAGGING_MAX_IMAGE_DIM` | `1024` | Max dimension for images sent to the vision tagging endpoint. |

## NER — `NERConfig`

Loaded by `load_ner_env()` (`env_cfg.py:582`).

| Variable | Default | Description |
|---|---|---|
| `NER_ENABLED` | `true` | Run entity/relation extraction during ingestion. |
| `NER_MAX_CHARS` | `1024` | Max chars per node passed to GLiNER. |
| `NER_MAX_WORKERS` | `4` | Parallel NER workers. |

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

Loaded by `load_summary_env()` (`env_cfg.py:1145`).

| Variable | Default | Description |
|---|---|---|
| `SUMMARY_COVERAGE_TARGET` | `0.70` | Target document coverage ratio for summaries (clamped to `[0.0, 1.0]`). |
| `SUMMARY_MAX_DOCS` | `30` | Max documents sampled. |
| `SUMMARY_PER_DOC_TOP_K` | `4` | Max evidence chunks per document. |
| `SUMMARY_FINAL_SOURCE_CAP` | `24` | Max merged sources in the final answer. |
| `SUMMARY_SOCIAL_CHUNKING_ENABLED` | `true` | Use row-level summarisation for social/table collections. |
| `SUMMARY_SOCIAL_CANDIDATE_POOL` | `48` | Candidate retrieval depth for social summaries. |
| `SUMMARY_SOCIAL_DIVERSITY_LIMIT` | `2` | Max sources retained per diversity bucket. |

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

## Runtime device — `RuntimeConfig`

Loaded by `load_runtime_env()` (`env_cfg.py:1070`).

| Variable | Default | Description |
|---|---|---|
| `USE_DEVICE` | `auto` | Preferred device for local auxiliary models: `auto`, `cpu`, `mps`, `cuda`, or `cuda:<index>`. When set to `cpu`, `CUDA_VISIBLE_DEVICES=""` is forced at import time to prevent accidental GPU context init. |

## Paths — `PathConfig`

Loaded by `load_path_env()` (`env_cfg.py:785`). Every path expands `~`.

| Variable | Default | Description |
|---|---|---|
| `DATA_PATH` | `~/docint/data` | Root directory for ingestion inputs. |
| `LOG_PATH` | `<repo>/.logs/docint.log` | Rotating log file. |
| `QUERIES_PATH` | `~/docint/queries.txt` | Default query input file for the CLI. |
| `RESULTS_PATH` | `~/docint/results` | Directory for CLI export artifacts. |
| `PIPELINE_ARTIFACTS_DIR` | `~/docint/artifacts` | Pipeline artifact root (also read by `PipelineConfig`). |
| `QDRANT_SRC_DIR` | `~/docint/qdrant_sources` | Where raw source files are staged for preview. |
| `HF_HUB_CACHE` | `~/.cache/huggingface/hub` | HF Hub cache path. |

`PathConfig` also exposes a derived `prompts` path pointing at
`docint/utils/prompts/` — it is not overridable by env var.

## Response validation — `ResponseValidationConfig`

Loaded by `load_response_validation_env()` (`env_cfg.py:935`).

| Variable | Default | Description |
|---|---|---|
| `RESPONSE_VALIDATION_ENABLED` | `true` | Run the `ResultValidationResponseAgent` to cross-check answers against sources. |

## Offline mode

- `DOCINT_OFFLINE` — default `1`. When truthy, Docint sets
  `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`,
  `HF_HUB_DISABLE_TELEMETRY=1`, `HF_HUB_DISABLE_SYMLINKS_WARNING=1`, and
  `KMP_DUPLICATE_LIB_OK=TRUE`. It also points `FASTEMBED_CACHE_PATH` at
  `HF_HUB_CACHE` when unset. See `set_offline_env()` in `env_cfg.py:12`.

## Proxy & image overrides

Relevant to Docker deployments; see [deployment.md](deployment.md).

| Variable | Purpose |
|---|---|
| `HTTP_PROXY` / `HTTPS_PROXY` / `NO_PROXY` (and lowercase variants) | Forwarded to Compose interpolation, image builds, and containers. |
| `PYTHON_SLIM_BOOKWORM_IMAGE` | Override base image for the backend builder. |
| `NVIDIA_CUDA_RUNTIME_IMAGE` | Override base image for the CUDA backend runtime. |
| `INFERENCE_NET` | Name of the shared external Docker network used by co-deployed inference services (default `inference-net`). |
