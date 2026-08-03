# Consolidating dense + sparse onto one bge-m3 instance — design

**Date:** 2026-08-02
**Repos:** `vllm-service` (serve dense), `docint` (route dense)
**Status:** approved, pending implementation plan
**Extends:** `2026-08-01-remote-sparse-encoder-design.md`
**Amends:** `nos-tromo/vllm-service#78`, `nos-tromo/docint#386` — both open, both to be updated rather than followed up

## Problem

The `sparse-only` CPU shape added by vllm-service#78 loads `BAAI/bge-m3`
to compute sparse weights. On a dev host, Ollama is already serving the
same model for dense embeddings. bge-m3 ends up resident twice:

| Copy | Runtime | Format | Approx RAM |
|---|---|---|---|
| Dense | Ollama | GGUF (`bge-m3:latest`) | ~1.2 GB |
| Sparse | `sparse-only` | fp32 torch (XLM-R-large, 568M params) | ~2.3 GB |

Beyond the memory, it diverges from production topology. On the full
CUDA stack a single vLLM `embed` backend serves both `/v1/embeddings`
and `/pooling`; only the dev shape splits them across two runtimes.

### Why the split exists

`docint/core/rag.py:2325` builds the dense client with
`api_base=self.openai_api_base` — the shared OpenAI-compatible base,
which on dev is Ollama. There is no per-model override for embeddings,
so dense cannot be pointed anywhere else without also moving chat.

Every other remote service in docint already has one: `NER_API_BASE`,
`RERANK_API_BASE`, `CLIP_API_BASE`, and now `SPARSE_API_BASE`, each
inheriting from `OPENAI_*` when unset. Nextext follows the same pattern
with `WHISPER_API_BASE` / `NER_API_BASE` / `DIARIZATION_API_BASE`.
Embeddings are the gap.

## What this does NOT do

**It does not reduce the number of forward passes.** docint makes two
independent HTTP calls for the same text: llama-index's
`/v1/embeddings` for dense, and `RemoteSparseEncoder`'s `/pooling` +
`/tokenize` for sparse. Co-locating them in one container means one
model *load*, not one *pass*.

Fusing them would require a combined endpoint plus a custom embedding
path in docint, because `OpenAIEmbedding`'s contract and Qdrant's
`sparse_doc_fn` are separate call sites that neither knows about the
other. That is a larger change and deliberately out of scope. bge-m3
does produce both vectors from one encoder pass, so the optimisation is
real — it is just not free, and not this.

The win here is RAM (~3.5 GB → ~2.3 GB on dev), one fewer model in
Ollama, and dev topology that matches production.

## Constraints

- **vLLM production must be unaffected.** `EMBED_API_BASE` unset ⇒
  inherit `OPENAI_API_BASE` ⇒ the router ⇒ today's behaviour exactly.
- **`RemoteSparseEncoder`'s wire format stays frozen.** Production
  collections depend on the vectors it produces.
- Airgap-first; no runtime fetches.
- No Bearer gate on the CPU container — `inference-net` is the trust
  boundary, as with the other `*-only` shapes.

## Approach

Extend the CPU container to serve dense embeddings from the model it
already has loaded, add the missing `EMBED_*` override to docint, and
remove `bge-m3` from Ollama so it serves chat only.

Because the shape will then serve dense, sparse, and tokenize, the name
`sparse-only` becomes wrong on arrival. It is renamed **`embed-only`**
as part of the same change, in both open PRs.

### Rejected alternatives

**Shrink the second copy** (fp16, or the ONNX weights already in the
bge-m3 snapshot). Halves the sparse container's footprint with no docint
change and no re-ingest implications — but keeps two runtimes, two model
loads, and the divergence from production topology. A mitigation, not a
fix.

**Drop sparse on dev** (`ENABLE_HYBRID=false`, no container). Zero
duplication and now a first-class supported configuration thanks to
docint#386 — but dev then cannot exercise hybrid retrieval at all, which
is most of why the CPU shape exists.

## Design

### 1. `vllm-service`: dense route on the existing server

`src/sparse_server.py` gains:

- `POST /v1/embeddings` — OpenAI-compatible. Request `{model, input}`
  where `input` is a string or list of strings; response
  `{object: "list", data: [{object: "embedding", index, embedding}],
  model, usage}`. Dense vector = CLS pooling (`last_hidden_state[:, 0]`)
  followed by L2 normalisation, matching FlagEmbedding's `cls`
  sentence-pooling for bge-m3.

`/pooling`, `/tokenize` and `/health` are unchanged. The encoder forward
is shared with the existing `encode_token_weights` seam so both routes
read the same tensor shape and the same tokenizer settings.

Note the two route roots coexist deliberately: the OpenAI SDK appends
`/embeddings` to its base, so dense lives under `/v1`, while the vLLM
pooling protocol is root-anchored. `_vllm_service_root()` in docint
already strips a `/v1` suffix, so **one URL serves both knobs**.

### 2. `vllm-service`: rename `sparse-only` → `embed-only`

Within PR #78, before merge:

| From | To |
|---|---|
| `docker/compose.sparse-only.yaml` | `docker/compose.embed-only.yaml` |
| `docker/compose.sparse-only.override.yaml` | `docker/compose.embed-only.override.yaml` |
| `docker/Dockerfile.sparse.cpu` | `docker/Dockerfile.embed.cpu` |
| `src/sparse_server.py` | `src/embed_server.py` |
| `eval/tests/test_sparse_server.py` | `eval/tests/test_embed_server.py` |
| network alias `sparse-only` | `embed-only` |
| image `vllm-service-sparse-only` | `vllm-service-embed-only` |
| `COMPOSE_SPARSE_ONLY{,_DEV}` | `COMPOSE_EMBED_ONLY{,_DEV}` |
| 6 make targets `*-sparse-only` | `*-embed-only` |
| `bundle_images.sh` case `sparse-only` | `embed-only` |
| `SPARSE_HOST_PORT` | `EMBED_HOST_PORT` (value 8007 unchanged) |
| container env `SPARSE_MODEL` | `EMBED_MODEL` (default `BAAI/bge-m3`) |
| container env `SPARSE_MAX_LENGTH` | `EMBED_MAX_LENGTH` (default 8192) |

The container's own model knobs move to `EMBED_*` because the model now
backs both vector types, and because that is the variable the full
stack's `embed` service already reads — one name for the same model
across both shapes.

This is independent of **docint's** `SPARSE_MODEL`, which stays: that is
the string docint sends as the `model` field in the pooling payload, and
it is a docint-side concern.

`compose.yaml` and `litellm.config.yaml` remain untouched.

### 3. `docint`: the `EMBED_*` override

`docint/utils/env_cfg.py` gains `EmbedClientConfig` +
`load_embed_client_env(default_api_base, default_api_key,
default_timeout)`, mirroring `load_sparse_client_env` exactly — same
inheritance semantics, same `.rstrip("/")`, same blank-key-means-None
handling.

`docint/core/rag.py:2325` changes `"api_base": self.openai_api_base` to
read from the resolved embed config, with `api_key` alongside it.
Everything else in the embedding path is unchanged.

Within PR #386, the `sparse-only` references in `CLAUDE.md`,
`README.md`, `docs/configuration.md` and the `probe_sparse_endpoint`
`RuntimeError` message become `embed-only`.

### 4. Dev configuration

```bash
OPENAI_API_BASE=http://ollama:11434/v1      # chat only now
EMBED_API_BASE=http://embed-only:8000/v1    # dense
SPARSE_API_BASE=http://embed-only:8000/v1   # sparse (/v1 is stripped)
EMBED_MODEL=BAAI/bge-m3                     # was `bge-m3` (Ollama tag)
```

`ollama rm bge-m3` afterwards.

Production sets none of these: `EMBED_API_BASE` and `SPARSE_API_BASE`
inherit the router, and `EMBED_MODEL` is already the HF id there.

### 5. Error handling

Dense stays exactly as fail-soft (or not) as it is today — this changes
where the request goes, not how failures are treated. The existing
oversize-input guard in `BudgetedOpenAIEmbedding` is unaffected, since
it inspects the request before transport.

The `embed-only` container returns 422 via pydantic on a malformed body,
matching the existing routes.

The request's `model` field is **ignored**, and the response echoes the
model the container was actually started with — exactly what `/pooling`
and `/tokenize` already do. Validating it on the dense route alone would
make the three routes behave inconsistently for no gain: the container
serves exactly one model, and a caller naming a different one is a
deployment error that a 400 from a single route would report only
sometimes.

### 6. Testing

- Dense route: response shape matches the OpenAI contract; batch order
  preserved; a single-string `input` and a list both accepted.
- **Dense/sparse consistency**: `/v1/embeddings` and `/pooling` for the
  same text must agree on tokenization — the same truncation and the
  same `max_length`. A divergence here would be silent.
- **Normalisation**: the returned dense vector is unit-length. A missing
  L2 norm changes cosine scores without erroring.
- Dimension: the probe path (`rag.py:5375`, `get_text_embedding("ping")`)
  returns the expected width so collection creation sizes correctly.
- docint: `EMBED_*` inheritance and explicit override, mirroring the
  `SPARSE_*` tests; and that an unset `EMBED_API_BASE` leaves the vLLM
  path byte-identical to today.
- Rename: no `sparse-only` string survives in either repo except where
  it refers to the historical name.

## Migration

Dense vectors change numerically — Ollama's quantised GGUF to fp32
transformers. Dev collections therefore need a delete-and-reingest.

**They already need one** for the BM42 → bge-m3 sparse change in
docint#386. If this ships in the same release, the dense migration is
free. If it ships after dev has re-ingested, it costs a second one.
That is the reason for amending the open PRs rather than following up.

Production (vLLM) needs no re-ingest: it is already serving fp32 bge-m3
dense through the router, and nothing about its path changes.

## Delivery

Amend both open PRs rather than stacking follow-ups:

1. **`vllm-service#78`** — dense route, rename, tests, docs.
2. **`docint#386`** — `EMBED_*` config, one-line client change, rename
   references, docs.

Merge order is unchanged: #78 first, then #386.

## Open questions

- The dense/sparse forward-pass fusion described under *What this does
  not do* remains available as a later optimisation. Worth an issue once
  this lands, so the idea is not lost.
- `vllm-service#74` (bound `/pooling` batch size) applies equally to the
  new dense route — an unbounded `input` list has the same padding
  blow-up. The fix should cover both.
