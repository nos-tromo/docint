# Remote sparse encoder — design

**Date:** 2026-08-01
**Repos:** `vllm-service` (new service), `docint` (consumer)
**Status:** approved, pending implementation plan

## Problem

`docint/CLAUDE.md` states the invariant:

> **All ML inference is remote.** docint ships no GPU code and no local model runtime.

That is currently false. The sparse half of hybrid retrieval runs a local ONNX
neural network inside the docint backend container.

### Evidence

The live backend process (`uvicorn`, RSS 1.4 GB) has the ONNX Runtime native
libraries mapped into its address space:

```
/app/.venv/lib/python3.11/site-packages/onnxruntime/capi/libonnxruntime_providers_shared.so
/app/.venv/lib/python3.11/site-packages/onnxruntime/capi/onnxruntime_pybind11_state.cpython-311-aarch64-linux-gnu.so
```

`onnxruntime 1.24.2`, `fastembed 0.8.0`, providers
`['AzureExecutionProvider', 'CPUExecutionProvider']`. Encoding two strings
takes 12 ms in-process with no HTTP call. The model is a real 88 MB
`model.onnx` in the shared `huggingface-cache` volume.

### Root cause

`RAG._vector_store()` (`docint/core/rag.py:2549-2560`) forks on the inference
provider, and only one of the two branches is remote:

```python
if self.enable_hybrid and self.openai_inference_provider.lower() == "vllm":
    sparse_encoder = VLLMSparseEncoder(...)           # remote
    vector_store_kwargs["sparse_doc_fn"]   = sparse_encoder.encode_texts
    vector_store_kwargs["sparse_query_fn"] = sparse_encoder.encode_texts
else:
    vector_store_kwargs["fastembed_sparse_model"] = self.sparse_model   # LOCAL
```

The remote path was built for vLLM alone; `ollama` and `openai` kept the
in-process encoder. `enable_hybrid` is a hardcoded `field(default=True)`
(`rag.py:1764`) with no env override, and `api.py:126` constructs
`RAG(qdrant_collection="")` without overriding it, so hybrid is on for the
entire API runtime.

### Where the sparse model is applied

1. **Collection creation** — `rag.py:5385-5400` declares `sparse_vectors_config`
   and picks the IDF modifier. Name-only; no inference.
2. **Ingestion** — `QdrantVectorStore.add()` → `_sparse_doc_fn`
   (`llama_index/vector_stores/qdrant/base.py:334-335`). Every chunk of every
   document is encoded before upsert. This is the bulk of the local compute.
3. **Query** — `_sparse_query_fn` (`base.py:1063, 1114, 1221, 1274`) for
   `HYBRID` and `SPARSE` modes. `vector_store_query_mode` defaults to `"auto"`
   (`rag.py:1837`), which resolves to `hybrid` (`rag.py:4004`), so every chat
   turn encodes its query locally.
4. **Not applied** — the `_images` companion is dense-only
   (`images_service.py:460` passes `enable_hybrid=False`); `_entities` writes
   raw Qdrant points with no sparse vector.

### Impact

`fastembed` is a direct dependency (`pyproject.toml:14`) and pulls
`onnxruntime` transitively (`uv.lock:833`). Both ship in **every** docint
image, including the staging and production images that run
`INFERENCE_PROVIDER=vllm` and never execute the local branch — roughly 1.2 GB
of dead runtime in every airgap bundle.

## Constraints

- **Ollama is the dev shape only.** Staging and production run
  `INFERENCE_PROVIDER=vllm`.
- **Production collections ingested under vLLM exist** (on other hosts). The
  production sparse protocol and its vector space must stay frozen: no changes
  to `docker/compose.yaml`, `docker/litellm.config.yaml`, the `embed` backend,
  or `VLLMSparseEncoder`'s wire format.
- **Cross-provider collection portability is explicitly a non-goal.** Ollama
  and vLLM collections already carry incompatible sparse vectors and will
  continue to.
- **Airgap-first.** Nothing may fetch models at runtime.

## Current sparse topology

| Provider | Sparse model | Protocol | IDF modifier |
|---|---|---|---|
| `vllm` | `BAAI/bge-m3` learned sparse | `POST /pooling` `task:token_classify` + `POST /tokenize`, LiteLLM pass-throughs → `embed:8000` (`litellm.config.yaml:73-78`) | no |
| `ollama` / `openai` | `Qdrant/bm42-all-minilm-l6-v2-attentions` | in-process fastembed ONNX | yes |

The vLLM `embed` backend runs with
`--hf-overrides '{"architectures":["BgeM3EmbeddingModel"]}'`
(`compose.yaml:184`), so `task: token_classify` returns bge-m3's sparse-head
per-token weights.

## Approach

Extract the dev-path sparse encoder into a CPU service in `vllm-service`,
following the established `*-only` pattern (`gliner-only`, `rerank-only`,
`clip-only`), and delete the local branch from docint.

The new service speaks **the same two routes the production router already
passes through**, so docint keeps exactly one sparse client. The `*-only`
pattern's load-bearing property is that the consumer's client code is
identical in both deployment shapes and only `*_API_BASE` changes; serving
BM42 over a second protocol would have broken that.

### Rejected alternatives

**`sparse-only` serving BM42 via fastembed.** Smaller image (no torch), no
re-ingest, no parity risk. Rejected because docint would carry a second sparse
client for a second protocol permanently, purely to serve the dev shape.

**`ENABLE_HYBRID` alone, dense-only on dev.** Zero new infrastructure, same
size saving. Rejected because hybrid retrieval would become untestable before
staging, against production collections reachable only from other hosts.

## Design

### 1. `vllm-service`: new `sparse` CPU server

**`src/sparse_server.py`** — FastAPI serving three routes:

- `POST /pooling` — accepts `{model, task, input: [str]}`, returns
  `{data: [{data: [float]}]}`. Only `task: "token_classify"` is supported;
  any other value is a 400. Computes XLM-R forward → `sparse_linear.pt`
  (`Linear(1024, 1)`) → ReLU, reproducing `BgeM3EmbeddingModel`'s
  token_classify output.
- `POST /tokenize` — accepts `{model, prompt: str}`, returns `{tokens: [int]}`
  from the bge-m3 tokenizer. Response shape must satisfy
  `VLLMSparseEncoder._extract_token_ids` (`rag.py:1607-1642`), which probes
  `token_ids`, `tokens`, `prompt_token_ids`, then nested `data`.
- `GET /health` — for the compose healthcheck.

Model assets load once at startup from `/root/.cache/huggingface/hub` with
`local_files_only=True`. No Bearer gate, matching `rerank-only` and
`gliner-only`; `inference-net` is the trust boundary.

**`docker/Dockerfile.sparse.cpu`** — mirrors `Dockerfile.rerank.cpu`: CPU
torch from `https://download.pytorch.org/whl/cpu`, then `fastapi`, `uvicorn`,
`pydantic`, `transformers`, `sentencepiece`. **No FlagEmbedding** — it pulls
`ir-datasets` → `zlib-state`, which fails to build on aarch64.

**`docker/compose.sparse-only.yaml`** + **`.override.yaml`** — mirror
`compose.rerank-only.yaml`: project name `vllm-service-sparse-only`, one
container aliased `sparse-only` on the external `inference-net`, external
`huggingface-cache` volume, `no-proxy-env`, local logging driver,
`/health` healthcheck with a 180 s `start_period`, `restart: unless-stopped`.

`compose.yaml` and `litellm.config.yaml` are **not** modified.

### 2. `docint`: narrow the gate, delete the local branch

**`docint/utils/env_cfg.py`**

- Add `SPARSE_API_BASE` / `SPARSE_API_KEY` / `SPARSE_TIMEOUT`, resolved by a
  `load_sparse_env()` that mirrors the rerank resolver (`env_cfg.py:1381-1392`):
  inherit `OPENAI_API_BASE` / `OPENAI_API_KEY` / `OPENAI_TIMEOUT` unless
  overridden, `rstrip("/")` the base.
- `load_model_env()`: `default_sparse_model` becomes `"BAAI/bge-m3"` for every
  provider. Today it is `default_embed_model` under vLLM (so already bge-m3)
  and `Qdrant/all_miniLM_L6_v2_with_attentions` otherwise.
- Add `ENABLE_HYBRID`. Its default cannot key off "a sparse base resolved",
  because `SPARSE_API_BASE` inherits `OPENAI_API_BASE` and so is never empty.
  It defaults **true** when `INFERENCE_PROVIDER == "vllm"` (production, where
  the router pass-throughs exist) **or** `SPARSE_API_BASE` is set explicitly
  (dev, pointing at `sparse-only`); **false** otherwise. A plain-OpenAI
  deployment therefore degrades to dense instead of POSTing `/pooling` at an
  endpoint that has no such route.

**`docint/core/rag.py`**

- `_vector_store()` (2549): condition changes from
  `self.openai_inference_provider.lower() == "vllm"` to "a sparse endpoint is
  configured". The `else` branch at 2559 is deleted.
- `sparse_model` property (2343-2374): the fastembed support-list lookup and
  HF-repo mapping are deleted; the property returns `self.sparse_model_id`
  when hybrid is on, else `None`.
- `from fastembed import SparseTextEmbedding` (69) and its `__all__` entry
  (113) are deleted.
- `enable_hybrid` (1764) becomes env-backed.
- `IDF_EMBEDDING_MODELS` (imported at 120, used at 5391) **stays**.
  `qdrant_client/fastembed_common.py:287-295` degrades it to an empty `set()`
  when fastembed is absent, so `modifier` resolves to `None` — which is what
  production already gets, since bge-m3 is not an IDF model. Behaviour is
  preserved for existing production collections.

**`docint/pyproject.toml`** — drop `fastembed>=0.8.0` (line 14). `onnxruntime`
leaves with it as a transitive.

**`docint/utils/model_cfg.py`** — `sparse_model` leaves the `hf_assets` list;
docint no longer needs the sparse model locally.

### 3. Data flow

Unchanged in shape, only in destination. Ingest and query both call
`VLLMSparseEncoder.encode_texts` → `POST {SPARSE_API_BASE}/pooling` and
`/tokenize` → indices/values → Qdrant `text-sparse-new`.

- **Staging / production:** `SPARSE_API_BASE` inherits `OPENAI_API_BASE`, i.e.
  the LiteLLM router — today's behaviour, byte-for-byte.
- **Dev:** `SPARSE_API_BASE=http://sparse-only:8000`.
- **Neither vLLM nor an explicit `SPARSE_API_BASE`** (e.g. plain OpenAI):
  `ENABLE_HYBRID` defaults false, dense-only.

`VLLMSparseEncoder` is renamed `RemoteSparseEncoder` — it now serves a
non-vLLM backend too, though it still speaks the vLLM pooling protocol. Wire
format is untouched.

### 4. Error handling

Deliberately **not** fail-soft, unlike the reranker. A rerank transport
failure degrades to unranked order and costs only quality. A sparse-encoder
failure mid-ingest would silently write dense-only points into a hybrid
collection and corrupt it. `RemoteSparseEncoder` keeps raising.

The new failure mode is a misconfigured endpoint. The check belongs at
**ingest-job start**, not at app import: `api.py:126` builds the `RAG`
instance at module import, and failing there would take the whole backend
down — including `/health`, `/version` and every dense-only operation — over
a dev misconfiguration. So an ingest job with hybrid enabled probes the
sparse endpoint before its first batch and fails the job with a clear message
if it is unreachable. Query-time failures surface as normal request errors.

This mirrors the `*_API_BASE` DNS-failure trap that already bites the
`*-only` shapes — a misconfigured base is easy to miss because the other
remote clients are fail-soft.

### 5. Testing

- **Parity (load-bearing).** Golden token-id and weight fixtures captured from
  the real vLLM `embed` backend, asserted against the CPU server within
  tolerance. Catches drift in CI rather than in dev retrieval. Fixtures must
  be generated on a host with the CUDA stack, so this test lands
  skipped-without-fixtures.
- **Gate.** Sparse endpoint configured → remote encoder wired as both
  `sparse_doc_fn` and `sparse_query_fn`; unset → hybrid off, dense-only.
- **Env resolution.** `SPARSE_*` inheritance from `OPENAI_*`, explicit
  override, `ENABLE_HYBRID` default derivation.
- **Dependency.** Assert `fastembed` and `onnxruntime` are not importable, so
  the local runtime cannot silently return.
- **Collection creation.** `modifier` is `None` for bge-m3, unchanged from
  production today.
- **Server unit tests** in `vllm-service` for both routes, including the
  `task != token_classify` 400 and the `_extract_token_ids`-compatible
  response shape.

## Delivery

Two PRs, in order:

1. **`vllm-service`** — new server, Dockerfile, compose files, README section.
   Purely additive; no consumer yet, nothing to break.
2. **`docint`** — env knobs, gate narrowing, deletions, dependency drop,
   `CLAUDE.md` + `docs/configuration.md` updates. Pinned behind the
   `vllm-service` release.

Dev collections need one re-ingest (BM42 → bge-m3 sparse). Production
collections are untouched.

## Open questions

- Parity fixtures require a CUDA host. Until they exist the parity test is
  skipped, and the CPU server's fidelity to `BgeM3EmbeddingModel` is argued
  from the architecture rather than measured.
- The shared `huggingface-cache` volume holds ~1.1 GB of `docling-project`
  models (layout-heron, CodeFormulaV2, DocumentFigureClassifier ×2). docint
  uses docling only for DOCX via `SimplePipeline` (`readers/docx.py:12` —
  pure XML, no models), so these are likely another repo's or leftovers.
  Out of scope here; worth confirming before an airgap bundle carries them.

## Related findings (separate work)

Surfaced while tracing this; not part of this design.

- `model_cfg.py:49` calls `snapshot_download` with no `allow_patterns`, so
  `uv run load-models` pulls the whole 2.5 GB `BAAI/bge-m3` repo to use ~22 MB
  of tokenizer files, and the whole 2.2 GB `BAAI/bge-reranker-v2-m3` repo
  whose weights docint never loads at all — `rerank_model` is only a string
  posted as `model=` to the remote `/rerank` (`rag.py:2413`).
- `rag.py:2025-2029` logs `Embedding tokenizer loaded from <cache root>
  (repo='BAAI/bge-m3')`, which reads as a model load. It is vocab-file-only
  and correct, but the wording and the cache-root-not-snapshot path make it
  look like local inference.
- `embed_tokenizer_repo` (`env_cfg.py:1003`) is decoupled from `EMBED_MODEL`
  with no consistency check. A non-bge-m3 embedding model silently gets
  bge-m3 token counts while the log asserts "exact token counts".
