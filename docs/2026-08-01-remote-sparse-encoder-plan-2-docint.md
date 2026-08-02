# Remote sparse encoder consumer (`docint`) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Delete docint's in-process sparse encoder so the backend ships no local model runtime, routing sparse embedding to a remote endpoint on every inference provider.

**Architecture:** `_vector_store()` currently forks on `INFERENCE_PROVIDER`, using the remote `VLLMSparseEncoder` only for vLLM and falling through to in-process fastembed ONNX otherwise. The fork is replaced by a single remote path gated on `ENABLE_HYBRID`, the encoder is renamed `RemoteSparseEncoder` (wire format unchanged), and `fastembed` — which drags in `onnxruntime` — leaves the dependency tree.

**Tech Stack:** Python 3.11, uv, pytest, ruff + pyrefly via pre-commit, llama-index, Qdrant.

**Design doc:** `docs/2026-08-01-remote-sparse-encoder-design.md`
**Depends on:** `vllm-service` plan 1 (`docs/2026-08-01-remote-sparse-encoder-plan-1-vllm-service.md`) must be merged and released first — the dev box needs a `sparse-only` container to point at.

## Global Constraints

- Repo: `docint`. Branch off `main` as `fix/remote-sparse-encoder`.
- Python `>=3.11,<3.12`. Use `uv add` / `uv remove` — never hand-edit `uv.lock`.
- **All `os.getenv` calls and config dataclasses live in `docint/utils/env_cfg.py`.** Other modules import from there.
- Google-style docstrings on every new or modified function and class.
- `uv run pre-commit run --all-files` (ruff check, ruff format, pyrefly) must pass before every commit. It is **tracked-files-only** — `git add` new files before running it or they are silently skipped.
- Every functional change ships with test updates. Tests live in `tests/`.
- **Production wire format is frozen.** `RemoteSparseEncoder`'s request and response handling must not change — only its name and how it is selected. Production collections ingested under vLLM depend on it.
- Run `make verify` before pushing.

## File Structure

| File | Responsibility |
|---|---|
| `docint/utils/env_cfg.py` (modify) | `SparseClientConfig` + `load_sparse_client_env` + `resolve_enable_hybrid`; unify `default_sparse_model`. |
| `docint/core/rag.py` (modify) | Rename the encoder, narrow the gate, delete the fastembed branch and the support-list mapping, add the ingest probe. |
| `docint/utils/model_cfg.py` (modify) | Drop `sparse_model` from the local HF asset list. |
| `pyproject.toml` (modify) | Remove `fastembed`. |
| `tests/test_env_cfg_sparse.py` (create) | Env resolution + `ENABLE_HYBRID` default derivation. |
| `tests/test_rag_sparse_gate.py` (create) | Gate wiring, probe behaviour, no-local-runtime guard. |
| `CLAUDE.md`, `docs/configuration.md`, `README.md` (modify) | Document the knobs and the invariant restoration. |

---

### Task 1: `SPARSE_*` client configuration

**Files:**
- Modify: `docint/utils/env_cfg.py` (add after `load_rerank_client_env`, which ends at line 1392)
- Create: `tests/test_env_cfg_sparse.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `SparseClientConfig(api_base: str, api_key: str | None, timeout: float)` and `load_sparse_client_env(default_api_base: str, default_api_key: str | None, default_timeout: float) -> SparseClientConfig`.

**Context an implementer needs:**

This deliberately mirrors `load_rerank_client_env` (`env_cfg.py:1346-1392`) — same inheritance semantics, same `rstrip("/")`, same "explicit empty means no auth" key handling. Read that function before writing this one.

- [ ] **Step 1: Write the failing test**

Create `tests/test_env_cfg_sparse.py`:

```python
"""Tests for the SPARSE_* remote sparse-encoder client configuration.

Mirrors the RERANK_* contract: each knob falls back to the active
OpenAI client setting unless explicitly overridden, so the full
vllm-service router works with no configuration while the sparse-only
CPU shape needs only SPARSE_API_BASE.
"""

import pytest

from docint.utils.env_cfg import load_sparse_client_env


def test_sparse_client_inherits_openai_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """With no SPARSE_* set, every field inherits the OpenAI client settings."""
    for name in ("SPARSE_API_BASE", "SPARSE_API_KEY", "SPARSE_TIMEOUT"):
        monkeypatch.delenv(name, raising=False)

    cfg = load_sparse_client_env(
        default_api_base="http://vllm-router:4000/v1",
        default_api_key="sk-master",
        default_timeout=300.0,
    )

    assert cfg.api_base == "http://vllm-router:4000/v1"
    assert cfg.api_key == "sk-master"
    assert cfg.timeout == 300.0


def test_sparse_client_explicit_override_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    """SPARSE_API_BASE points at the sparse-only container; trailing slash is stripped."""
    monkeypatch.setenv("SPARSE_API_BASE", "http://sparse-only:8000/")
    monkeypatch.setenv("SPARSE_TIMEOUT", "45")
    monkeypatch.delenv("SPARSE_API_KEY", raising=False)

    cfg = load_sparse_client_env(
        default_api_base="http://vllm-router:4000/v1",
        default_api_key="sk-master",
        default_timeout=300.0,
    )

    assert cfg.api_base == "http://sparse-only:8000"
    assert cfg.timeout == 45.0


def test_sparse_client_blank_key_disables_auth(monkeypatch: pytest.MonkeyPatch) -> None:
    """The sparse-only shape has no Bearer gate; a blank default means no header."""
    monkeypatch.delenv("SPARSE_API_KEY", raising=False)
    monkeypatch.setenv("SPARSE_API_BASE", "http://sparse-only:8000")

    cfg = load_sparse_client_env(
        default_api_base="http://vllm-router:4000/v1",
        default_api_key="",
        default_timeout=300.0,
    )

    assert cfg.api_key is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_env_cfg_sparse.py -v`
Expected: FAIL — `ImportError: cannot import name 'load_sparse_client_env'`

- [ ] **Step 3: Write minimal implementation**

Add to `docint/utils/env_cfg.py`, immediately after `load_rerank_client_env`:

```python
@dataclass(frozen=True)
class SparseClientConfig:
    """Dataclass for the remote sparse-encoder HTTP client."""

    api_base: str
    api_key: str | None
    timeout: float


def load_sparse_client_env(
    default_api_base: str,
    default_api_key: str | None,
    default_timeout: float,
) -> "SparseClientConfig":
    """Load the remote sparse-encoder client configuration.

    docint reaches sparse embedding over HTTP on every provider. The
    client POSTs to ``{api_base}/pooling`` (``task=token_classify``) and
    ``{api_base}/tokenize``. Defaults mirror the OpenAI client settings —
    the full vllm-service router exposes both as LiteLLM pass-throughs
    against the same base. For the sparse-only deployment shape (CPU
    container hosted by vllm-service), override with
    ``SPARSE_API_BASE=http://sparse-only:8000``.

    Args:
        default_api_base (str): Fallback base URL when ``SPARSE_API_BASE``
            is unset. Typically the active ``OPENAI_API_BASE``.
        default_api_key (str | None): Fallback Bearer token when
            ``SPARSE_API_KEY`` is unset. ``None`` (or empty) disables auth.
        default_timeout (float): Fallback request timeout in seconds.

    Returns:
        SparseClientConfig: Resolved configuration.

        - ``api_base``: Base URL; the encoder appends ``/pooling`` and
          ``/tokenize`` itself.
        - ``api_key``: Bearer token sent as ``Authorization: Bearer ...``
          when set; omitted entirely when ``None``. The sparse-only shape
          requires no auth (trust ``inference-net``); the full router
          requires the master key.
        - ``timeout``: Per-request HTTP timeout in seconds.
    """
    raw_key = os.getenv("SPARSE_API_KEY")
    if raw_key is not None and raw_key.strip():
        api_key: str | None = raw_key.strip()
    elif default_api_key and default_api_key.strip():
        api_key = default_api_key.strip()
    else:
        api_key = None
    return SparseClientConfig(
        api_base=os.getenv("SPARSE_API_BASE", default_api_base).rstrip("/"),
        api_key=api_key,
        timeout=float(os.getenv("SPARSE_TIMEOUT", default_timeout)),
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_env_cfg_sparse.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add docint/utils/env_cfg.py tests/test_env_cfg_sparse.py
git commit -m "feat(sparse): SPARSE_* remote client configuration"
```

---

### Task 2: `ENABLE_HYBRID` and a single sparse model default

**Files:**
- Modify: `docint/utils/env_cfg.py` (`load_model_env` at lines 967-1027; new `resolve_enable_hybrid`)
- Modify: `tests/test_env_cfg_sparse.py`

**Interfaces:**
- Consumes: `load_sparse_client_env` from Task 1.
- Produces: `resolve_enable_hybrid() -> bool`.

**Context an implementer needs:**

`ENABLE_HYBRID` cannot default to "a sparse base resolved" — `SPARSE_API_BASE` inherits `OPENAI_API_BASE` and is therefore never empty. The rule is: **true** when `INFERENCE_PROVIDER == "vllm"` (the router has the pass-throughs) **or** `SPARSE_API_BASE` is set explicitly (the dev box points at `sparse-only`); **false** otherwise, so a plain-OpenAI deployment degrades to dense instead of POSTing `/pooling` at an endpoint with no such route. An explicit `ENABLE_HYBRID` always wins.

Separately, `load_model_env` currently sets `default_sparse_model = default_embed_model` under vLLM (line 1007, i.e. `BAAI/bge-m3`) and leaves it at `Qdrant/all_miniLM_L6_v2_with_attentions` otherwise. Every provider now uses bge-m3 sparse, so that default becomes unconditional.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_env_cfg_sparse.py`:

```python
from docint.utils.env_cfg import load_model_env, resolve_enable_hybrid


@pytest.fixture(autouse=True)
def _clear_hybrid_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Clear every env var that participates in hybrid resolution."""
    for name in ("ENABLE_HYBRID", "SPARSE_API_BASE", "INFERENCE_PROVIDER", "SPARSE_MODEL"):
        monkeypatch.delenv(name, raising=False)


def test_hybrid_on_for_vllm_without_explicit_base(monkeypatch: pytest.MonkeyPatch) -> None:
    """Production: the router already serves /pooling, so hybrid is on by default."""
    monkeypatch.setenv("INFERENCE_PROVIDER", "vllm")
    assert resolve_enable_hybrid() is True


def test_hybrid_on_when_sparse_base_set_explicitly(monkeypatch: pytest.MonkeyPatch) -> None:
    """Dev: pointing at sparse-only opts in, whatever the provider."""
    monkeypatch.setenv("INFERENCE_PROVIDER", "ollama")
    monkeypatch.setenv("SPARSE_API_BASE", "http://sparse-only:8000")
    assert resolve_enable_hybrid() is True


def test_hybrid_off_for_ollama_without_sparse_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    """No sparse endpoint means dense-only, not a POST at a route that 404s."""
    monkeypatch.setenv("INFERENCE_PROVIDER", "ollama")
    assert resolve_enable_hybrid() is False


def test_hybrid_off_for_openai_without_sparse_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    """Plain OpenAI has no /pooling route; degrade to dense."""
    monkeypatch.setenv("INFERENCE_PROVIDER", "openai")
    assert resolve_enable_hybrid() is False


def test_explicit_enable_hybrid_overrides_derivation(monkeypatch: pytest.MonkeyPatch) -> None:
    """An explicit setting always wins over the derived default."""
    monkeypatch.setenv("INFERENCE_PROVIDER", "vllm")
    monkeypatch.setenv("ENABLE_HYBRID", "false")
    assert resolve_enable_hybrid() is False

    monkeypatch.setenv("INFERENCE_PROVIDER", "ollama")
    monkeypatch.setenv("ENABLE_HYBRID", "true")
    assert resolve_enable_hybrid() is True


@pytest.mark.parametrize("provider", ["ollama", "vllm", "openai"])
def test_sparse_model_defaults_to_bge_m3_on_every_provider(
    monkeypatch: pytest.MonkeyPatch,
    provider: str,
) -> None:
    """One sparse model everywhere — the local BM42 default is gone."""
    monkeypatch.setenv("INFERENCE_PROVIDER", provider)
    assert load_model_env().sparse_model == "BAAI/bge-m3"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_env_cfg_sparse.py -v`
Expected: FAIL — `ImportError: cannot import name 'resolve_enable_hybrid'`

- [ ] **Step 3: Write minimal implementation**

Add to `docint/utils/env_cfg.py`:

```python
def resolve_enable_hybrid() -> bool:
    """Decide whether hybrid (dense + sparse) retrieval is enabled.

    Sparse encoding is always remote, so hybrid is only safe where a
    sparse endpoint actually serves ``/pooling`` and ``/tokenize``. That
    cannot be inferred from ``SPARSE_API_BASE`` being non-empty, because
    it inherits ``OPENAI_API_BASE`` and is therefore never empty. Two
    signals stand in for it: the vLLM provider (whose router exposes both
    routes as pass-throughs) and an explicitly set ``SPARSE_API_BASE``
    (the sparse-only deployment shape). An explicit ``ENABLE_HYBRID``
    overrides both.

    Returns:
        bool: True when hybrid retrieval should be enabled.
    """
    explicit = os.getenv("ENABLE_HYBRID")
    if explicit is not None and explicit.strip():
        return explicit.strip().lower() in {"1", "true", "yes", "on"}

    provider = os.getenv("INFERENCE_PROVIDER", "ollama").strip().lower()
    has_explicit_sparse_base = bool(os.getenv("SPARSE_API_BASE", "").strip())
    return provider == "vllm" or has_explicit_sparse_base
```

In `load_model_env`, replace the provider-conditional sparse default. Delete line 1007 (`default_sparse_model = default_embed_model`) and change the signature default at line 971 to:

```python
    default_sparse_model: str = "BAAI/bge-m3",
```

Update the `load_model_env` docstring's `sparse_model` bullet to say the sparse model is bge-m3 on every provider and is served remotely.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_env_cfg_sparse.py -v`
Expected: PASS (9 tests)

- [ ] **Step 5: Check for fallout in the existing suite**

Run: `uv run pytest tests/test_env_cfg.py tests/test_model_cfg.py -v`
Expected: PASS. If a test asserts the old BM42 default, update it to `BAAI/bge-m3` — that assertion is now wrong by design, not a regression.

- [ ] **Step 6: Commit**

```bash
git add docint/utils/env_cfg.py tests/test_env_cfg_sparse.py tests/test_env_cfg.py tests/test_model_cfg.py
git commit -m "feat(sparse): ENABLE_HYBRID knob and one sparse model per provider"
```

---

### Task 3: Rename the encoder and wire its configuration

**Files:**
- Modify: `docint/core/rag.py` (class at line 1491; helper `_vllm_service_root` at 1477; usage at 2550)
- Modify: `tests/test_rag_unit.py` (any reference to the old name)

**Interfaces:**
- Consumes: `load_sparse_client_env` from Task 1.
- Produces: `RemoteSparseEncoder` with the same fields (`api_base`, `model`, `api_key`, `timeout`) and the same `encode_texts(texts: list[str]) -> BatchSparseEncoding`.

**Context an implementer needs:**

This is a rename plus a wiring change, **not** a behaviour change. `encode_texts`, `_pool_token_scores`, `_tokenize`, `_extract_token_ids`, `_coerce_token_scores` and `_build_sparse_vector` keep their bodies byte-for-byte. Production collections depend on the exact vectors these produce. The class still speaks the vLLM pooling protocol — it is renamed because it now also serves a non-vLLM backend, not because the wire format changed.

- [ ] **Step 1: Write the failing test**

Create `tests/test_rag_sparse_gate.py`:

```python
"""Tests for the remote sparse encoder gate.

Sparse embedding is remote on every provider. These tests pin the
selection logic and the frozen wire format; the encoder's request and
response handling must not drift, because production collections were
ingested with it.
"""

import pytest

from docint.core.rag import RemoteSparseEncoder


def test_encoder_appends_pooling_and_tokenize_to_base() -> None:
    """The encoder owns the route suffixes; the config carries only the base."""
    encoder = RemoteSparseEncoder(api_base="http://sparse-only:8000", model="BAAI/bge-m3")
    captured: list[tuple[str, dict[str, object]]] = []

    def _fake_request(url: str, payload: dict[str, object]) -> object:
        captured.append((url, payload))
        if url.endswith("/pooling"):
            return {"data": [{"data": [0.0, 0.7, 0.0]}]}
        return {"tokens": [0, 42, 2]}

    encoder._request_json = _fake_request  # type: ignore[method-assign]
    indices, values = encoder.encode_texts(["alpha"])

    urls = [url for url, _ in captured]
    assert "http://sparse-only:8000/pooling" in urls
    assert "http://sparse-only:8000/tokenize" in urls
    assert indices == [[42]]
    assert values == [[pytest.approx(0.7)]]


def test_encoder_strips_v1_suffix_for_router_base() -> None:
    """Against the router the base ends in /v1, but the routes sit at the root."""
    encoder = RemoteSparseEncoder(api_base="http://vllm-router:4000/v1", model="BAAI/bge-m3")
    captured: list[str] = []

    def _fake_request(url: str, payload: dict[str, object]) -> object:
        captured.append(url)
        if url.endswith("/pooling"):
            return {"data": [{"data": [0.5]}]}
        return {"tokens": [7]}

    encoder._request_json = _fake_request  # type: ignore[method-assign]
    encoder.encode_texts(["alpha"])

    assert captured[0] == "http://vllm-router:4000/pooling"


def test_encoder_drops_non_positive_scores() -> None:
    """ReLU zeroes most tokens; those must not enter the sparse vector."""
    encoder = RemoteSparseEncoder(api_base="http://sparse-only:8000", model="BAAI/bge-m3")

    def _fake_request(url: str, payload: dict[str, object]) -> object:
        if url.endswith("/pooling"):
            return {"data": [{"data": [0.0, 0.9, 0.0, 0.4]}]}
        return {"tokens": [0, 11, 2, 12]}

    encoder._request_json = _fake_request  # type: ignore[method-assign]
    indices, values = encoder.encode_texts(["alpha beta"])

    assert indices == [[11, 12]]
    assert values == [[pytest.approx(0.9), pytest.approx(0.4)]]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_rag_sparse_gate.py -v`
Expected: FAIL — `ImportError: cannot import name 'RemoteSparseEncoder'`

- [ ] **Step 3: Rename the class**

In `docint/core/rag.py`, rename `VLLMSparseEncoder` to `RemoteSparseEncoder` (class definition at line 1491 and its use at line 2550). Update the class docstring to:

```python
    """Adapter that turns remote pooling/tokenize responses into Qdrant sparse vectors.

    Speaks the vLLM pooling protocol — ``POST {root}/pooling`` with
    ``task="token_classify"`` plus ``POST {root}/tokenize`` — against
    either the full vllm-service router (which exposes both as LiteLLM
    pass-throughs to the ``embed`` backend) or the standalone
    ``sparse-only`` CPU container. The wire format is frozen: production
    collections were ingested with it.
    """
```

Leave every method body unchanged. Search for stale references: `grep -rn "VLLMSparseEncoder" docint tests docs` must return nothing afterwards.

- [ ] **Step 4: Wire the configuration**

In `RAG.__post_init__`, near the other client configs, resolve the sparse client and store it:

```python
        self.sparse_client_config = load_sparse_client_env(
            default_api_base=self.openai_api_base,
            default_api_key=self.openai_api_key,
            default_timeout=self.openai_timeout,
        )
```

Add `load_sparse_client_env` and `SparseClientConfig` to the `env_cfg` import block at the top of `rag.py` (alongside `load_rerank_client_env` at line 61), and declare `sparse_client_config: SparseClientConfig | None = field(default=None, init=False, repr=False)` beside the other private fields.

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests/test_rag_sparse_gate.py tests/test_rag_unit.py -v`
Expected: PASS. Fix any `VLLMSparseEncoder` references the rename missed.

- [ ] **Step 6: Commit**

```bash
git add docint/core/rag.py tests/test_rag_sparse_gate.py tests/test_rag_unit.py
git commit -m "refactor(sparse): rename VLLMSparseEncoder to RemoteSparseEncoder"
```

---

### Task 4: Narrow the gate and delete the local branch

**Files:**
- Modify: `docint/core/rag.py` (`_vector_store` at 2530-2561; `sparse_model` property at 2324-2375; imports at 69 and 113; `enable_hybrid` field at 1764)
- Modify: `tests/test_rag_sparse_gate.py`

**Interfaces:**
- Consumes: `RemoteSparseEncoder` and `sparse_client_config` from Task 3; `resolve_enable_hybrid` from Task 2.
- Produces: `RAG._build_sparse_encoder() -> RemoteSparseEncoder` — the single construction point, consumed again by Task 5's probe. Otherwise this task is the deletion.

**Context an implementer needs:**

This is the change that restores the invariant. After it, no code path constructs a local encoder.

`IDF_EMBEDDING_MODELS` (imported at line 120 from `qdrant_client.qdrant_fastembed`, used at 5391) **stays**. Verified: `qdrant_client/fastembed_common.py:8-17` guards the fastembed import and `:287-295` degrades the set to `set()` when it is absent, so `modifier` resolves to `None` — which is what production already gets, since bge-m3 is not an IDF model. Do not "simplify" this by hardcoding `None`; leaving the lookup keeps the behaviour explicit and correct if the sparse model ever changes.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_rag_sparse_gate.py`:

```python
from unittest.mock import MagicMock

from docint.core import rag as rag_module


def _vector_store_kwargs(monkeypatch: pytest.MonkeyPatch, rag: object) -> dict[str, object]:
    """Capture the kwargs RAG passes to QdrantVectorStore."""
    captured: dict[str, object] = {}

    def _fake_store(**kwargs: object) -> object:
        captured.update(kwargs)
        return MagicMock()

    monkeypatch.setattr(rag_module, "QdrantVectorStore", _fake_store)
    rag._vector_store()
    return captured


def test_hybrid_on_wires_remote_encoder_for_docs_and_queries(
    monkeypatch: pytest.MonkeyPatch,
    rag_instance: object,
) -> None:
    """Both sparse callbacks come from the remote encoder — never fastembed."""
    rag_instance.enable_hybrid = True
    kwargs = _vector_store_kwargs(monkeypatch, rag_instance)

    assert "fastembed_sparse_model" not in kwargs
    assert callable(kwargs["sparse_doc_fn"])
    assert callable(kwargs["sparse_query_fn"])
    assert kwargs["enable_hybrid"] is True


@pytest.mark.parametrize("provider", ["ollama", "openai", "vllm"])
def test_remote_encoder_used_on_every_provider(
    monkeypatch: pytest.MonkeyPatch,
    rag_instance: object,
    provider: str,
) -> None:
    """The provider no longer selects the encoder — only ENABLE_HYBRID does."""
    rag_instance.enable_hybrid = True
    rag_instance.openai_inference_provider = provider
    kwargs = _vector_store_kwargs(monkeypatch, rag_instance)

    assert "fastembed_sparse_model" not in kwargs
    assert callable(kwargs["sparse_doc_fn"])


def test_hybrid_off_wires_no_sparse_callbacks(
    monkeypatch: pytest.MonkeyPatch,
    rag_instance: object,
) -> None:
    """Dense-only deployments send no sparse kwargs at all."""
    rag_instance.enable_hybrid = False
    kwargs = _vector_store_kwargs(monkeypatch, rag_instance)

    assert "fastembed_sparse_model" not in kwargs
    assert "sparse_doc_fn" not in kwargs
    assert kwargs["enable_hybrid"] is False
```

Add this fixture to the module. `tests/test_rag_unit.py` has no shared
fixture — it constructs `RAG(qdrant_collection="test")` inline in each test
(e.g. line 118), which works without a live Qdrant because the client is
built lazily. Wrap that same construction:

```python
@pytest.fixture()
def rag_instance() -> object:
    """A RAG built the way tests/test_rag_unit.py builds one (no live Qdrant)."""
    from docint.core.rag import RAG

    return RAG(qdrant_collection="test")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_rag_sparse_gate.py -v`
Expected: FAIL — `fastembed_sparse_model` is present in the kwargs for the non-vLLM providers.

- [ ] **Step 3: Replace the fork in `_vector_store`**

First add the shared construction point to `RAG` (Task 5's pre-ingest probe
calls it too, so the probe can never resolve a different endpoint than the
one ingestion uses):

```python
    def _build_sparse_encoder(self) -> RemoteSparseEncoder:
        """Construct the remote sparse encoder from the resolved config.

        Single construction point, shared by the vector-store wiring and
        the pre-ingest probe, so the two can never drift apart in how
        they resolve the endpoint.

        Returns:
            RemoteSparseEncoder: Encoder bound to the configured endpoint.
        """
        sparse_config = self.sparse_client_config
        return RemoteSparseEncoder(
            api_base=sparse_config.api_base if sparse_config else self.openai_api_base or "",
            api_key=sparse_config.api_key if sparse_config else self.openai_api_key,
            model=self.sparse_model or "",
            timeout=sparse_config.timeout if sparse_config else self.openai_timeout,
        )
```

Then replace lines 2549-2560 with:

```python
        if self.enable_hybrid:
            sparse_encoder = self._build_sparse_encoder()
            vector_store_kwargs["sparse_doc_fn"] = sparse_encoder.encode_texts
            vector_store_kwargs["sparse_query_fn"] = sparse_encoder.encode_texts
```

The `else` branch that set `fastembed_sparse_model` is deleted outright.

- [ ] **Step 4: Simplify the `sparse_model` property**

Replace the body of the `sparse_model` property (2324-2375) with:

```python
    @property
    def sparse_model(self) -> str | None:
        """Return the configured sparse model id for hybrid retrieval.

        The id is passed through to the remote encoder as the ``model``
        field; docint no longer resolves it against a local support list,
        because it no longer runs a local sparse model.

        Returns:
            str | None: The sparse model id, or None when hybrid is off.

        Raises:
            ValueError: If hybrid is enabled but no sparse model is set.
        """
        if not self.enable_hybrid:
            return None
        if self.sparse_model_id is None:
            raise ValueError("sparse_model_id is None")
        return self.sparse_model_id
```

- [ ] **Step 5: Delete the fastembed import**

Remove `from fastembed import SparseTextEmbedding` (line 69) and its `"SparseTextEmbedding"` entry in `__all__` (line 113).

- [ ] **Step 6: Make `enable_hybrid` env-backed**

Change line 1764 from `enable_hybrid: bool = field(default=True)` to:

```python
    enable_hybrid: bool = field(default_factory=resolve_enable_hybrid)
```

Add `resolve_enable_hybrid` to the `env_cfg` import block. `docint/cli/ingest.py:56` passes `enable_hybrid` explicitly and keeps working unchanged.

- [ ] **Step 7: Run the tests**

Run: `uv run pytest tests/test_rag_sparse_gate.py -v`
Expected: PASS

Run: `uv run pytest -q`
Expected: full suite green. Tests that assert the old fastembed wiring are now wrong by design — update them. Tests that fail for any other reason are regressions; stop and investigate.

- [ ] **Step 8: Commit**

```bash
git add docint/core/rag.py tests/
git commit -m "fix(sparse): route sparse encoding remotely on every provider"
```

---

### Task 5: Fail an ingest job with an unreachable sparse endpoint

**Files:**
- Modify: `docint/core/rag.py`
- Modify: `tests/test_rag_sparse_gate.py`

**Interfaces:**
- Consumes: `RAG._build_sparse_encoder() -> RemoteSparseEncoder` (added in Task 4), `sparse_client_config`.
- Produces: `RAG.probe_sparse_endpoint() -> None`, raising `RuntimeError` on failure.

**Context an implementer needs:**

This is deliberately **not** fail-soft, unlike the reranker. A rerank transport failure degrades to unranked order and costs only quality. A sparse failure mid-ingest would write dense-only points into a hybrid collection and corrupt it.

The check must **not** run at import. `docint/core/api.py:126` builds the `RAG` instance at module import; failing there would take down `/health`, `/version` and every dense-only operation over what is typically a dev misconfiguration. Call it at the start of an ingest run, before the first batch. Query-time failures surface as ordinary request errors and need no extra handling.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_rag_sparse_gate.py`:

These tests patch `RemoteSparseEncoder._request_json` — the genuine HTTP
seam, the same one Task 3's tests use — so the real path runs end to end:
`sparse_client_config` resolution → encoder construction → request →
error wrapping. **Do not add a `_sparse_probe_fn` hook to `RAG` for tests
to override.** A test-only backdoor in production code would let all three
tests pass even if `probe_sparse_endpoint` built the encoder with the wrong
config, or never built one at all.

```python
def test_probe_raises_when_sparse_endpoint_unreachable(
    monkeypatch: pytest.MonkeyPatch,
    rag_instance: object,
) -> None:
    """An unreachable endpoint fails the job before any batch is written."""
    rag_instance.enable_hybrid = True

    def _boom(self: object, url: str, payload: dict[str, object]) -> object:
        raise OSError("connection refused")

    monkeypatch.setattr(rag_module.RemoteSparseEncoder, "_request_json", _boom)

    with pytest.raises(RuntimeError, match="sparse"):
        rag_instance.probe_sparse_endpoint()


def test_probe_is_a_noop_when_hybrid_disabled(
    monkeypatch: pytest.MonkeyPatch,
    rag_instance: object,
) -> None:
    """Dense-only ingests must not require a sparse endpoint."""
    rag_instance.enable_hybrid = False

    def _must_not_run(self: object, url: str, payload: dict[str, object]) -> object:
        raise AssertionError("probe must not touch the network when hybrid is off")

    monkeypatch.setattr(rag_module.RemoteSparseEncoder, "_request_json", _must_not_run)
    rag_instance.probe_sparse_endpoint()


def test_probe_passes_when_endpoint_responds(
    monkeypatch: pytest.MonkeyPatch,
    rag_instance: object,
) -> None:
    """A healthy endpoint lets the ingest proceed."""
    rag_instance.enable_hybrid = True

    def _ok(self: object, url: str, payload: dict[str, object]) -> object:
        if url.endswith("/pooling"):
            return {"data": [{"data": [0.5]}]}
        return {"tokens": [7]}

    monkeypatch.setattr(rag_module.RemoteSparseEncoder, "_request_json", _ok)
    rag_instance.probe_sparse_endpoint()


def test_probe_targets_the_configured_sparse_endpoint(
    monkeypatch: pytest.MonkeyPatch,
    rag_instance: object,
) -> None:
    """The probe must use the resolved SPARSE_* config, not some other base.

    Without this, the three tests above would all pass even if the probe
    built its encoder against the wrong endpoint entirely.
    """
    rag_instance.enable_hybrid = True
    rag_instance.sparse_client_config = SparseClientConfig(
        api_base="http://sparse-only:8000",
        api_key=None,
        timeout=12.0,
    )
    seen: list[str] = []

    def _capture(self: object, url: str, payload: dict[str, object]) -> object:
        seen.append(url)
        assert self.timeout == 12.0
        if url.endswith("/pooling"):
            return {"data": [{"data": [0.5]}]}
        return {"tokens": [7]}

    monkeypatch.setattr(rag_module.RemoteSparseEncoder, "_request_json", _capture)
    rag_instance.probe_sparse_endpoint()

    assert "http://sparse-only:8000/pooling" in seen
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_rag_sparse_gate.py -k probe -v`
Expected: FAIL — `AttributeError: 'RAG' object has no attribute 'probe_sparse_endpoint'`

- [ ] **Step 3: Write minimal implementation**

`_build_sparse_encoder` already exists — Task 4 added it. Add only this
method to `RAG`:

```python
    def probe_sparse_endpoint(self) -> None:
        """Verify the sparse endpoint answers before an ingest run starts.

        Sparse encoding is not fail-soft: a transport failure partway
        through an ingest would write dense-only points into a hybrid
        collection and corrupt it. Probing once up front converts that
        into a clean, actionable job failure.

        No-op when hybrid retrieval is disabled.

        Raises:
            RuntimeError: When hybrid is enabled and the configured sparse
                endpoint cannot be reached.
        """
        if not self.enable_hybrid:
            return

        encoder = self._build_sparse_encoder()
        try:
            encoder.encode_texts(["ping"])
        except Exception as exc:
            base = self.sparse_client_config.api_base if self.sparse_client_config else "<unset>"
            logger.error("Sparse endpoint probe failed against {}: {}", base, exc)
            raise RuntimeError(
                f"Hybrid retrieval is enabled but the sparse endpoint at {base} is unreachable: {exc}. "
                "Point SPARSE_API_BASE at a reachable sparse service (the sparse-only shape listens on "
                "http://sparse-only:8000), or set ENABLE_HYBRID=false to ingest dense-only."
            ) from exc
```

- [ ] **Step 4: Call it at ingest start**

Find where an ingest run begins in `rag.py` (the entry point the `IngestJobManager` runner invokes) and call `self.probe_sparse_endpoint()` before the first batch is processed. Read the surrounding code to place it after collection resolution and before any node parsing, so a failure costs no work.

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests/test_rag_sparse_gate.py -v && uv run pytest -q`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add docint/core/rag.py tests/test_rag_sparse_gate.py
git commit -m "fix(sparse): fail the ingest job when the sparse endpoint is unreachable"
```

---

### Task 6: Drop the local runtime from the dependency tree

**Files:**
- Modify: `pyproject.toml` (line 14), `uv.lock`
- Modify: `docint/utils/model_cfg.py` (line 109)
- Modify: `tests/test_rag_sparse_gate.py`

**Interfaces:**
- Consumes: Task 4's deletion of the last fastembed usage.
- Produces: nothing.

**Context an implementer needs:**

`onnxruntime` is transitive via `fastembed` (`uv.lock:833`), so removing the one direct dependency removes both. This is the step that actually shrinks the image and restores the invariant; the guard test is what stops it silently regressing.

`model_cfg.py:109` lists `sparse_model` in `hf_assets`, so `uv run load-models` downloads it locally. docint no longer runs it.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_rag_sparse_gate.py`:

```python
import importlib.util


def test_no_local_model_runtime_is_installed() -> None:
    """docint ships no local model runtime — guard against reintroduction.

    fastembed pulls onnxruntime, which is a full local inference engine.
    Both were removed when sparse encoding moved behind HTTP; this test
    fails loudly if a dependency bump drags either back in.
    """
    assert importlib.util.find_spec("fastembed") is None, "fastembed is back — sparse encoding must stay remote"
    assert importlib.util.find_spec("onnxruntime") is None, "onnxruntime is back — docint runs no local models"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_rag_sparse_gate.py -k local_model_runtime -v`
Expected: FAIL — both are still installed.

- [ ] **Step 3: Remove the dependency**

Run: `uv remove fastembed`
Then confirm `onnxruntime` left with it: `grep -c onnxruntime uv.lock` should drop to 0.

- [ ] **Step 4: Drop the sparse model from local asset loading**

In `docint/utils/model_cfg.py`, remove `(model_config.sparse_model, "sparse"),` from the `hf_assets` list (line 109) and add a short comment above the list noting that sparse embedding is remote, mirroring the existing NER/CLIP comment at lines 102-105.

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests/test_rag_sparse_gate.py -v`
Expected: PASS

Run: `uv run pytest -q`
Expected: full suite green. A failure importing `fastembed` anywhere means Task 4 missed a usage — find it with `grep -rn "fastembed\|SparseTextEmbedding" docint tests`.

- [ ] **Step 6: Verify the import graph is clean**

Run: `uv run python -c "import docint.core.rag; import importlib.util; assert importlib.util.find_spec('onnxruntime') is None; print('no local runtime')"`
Expected: `no local runtime`

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml uv.lock docint/utils/model_cfg.py tests/test_rag_sparse_gate.py
git commit -m "build(sparse): drop fastembed and onnxruntime from the image"
```

---

### Task 7: Documentation

**Files:**
- Modify: `CLAUDE.md`, `docs/configuration.md`, `README.md`

**Interfaces:**
- Consumes: everything above.
- Produces: nothing.

- [ ] **Step 1: Update CLAUDE.md**

- In the "All ML inference is remote" paragraph, add sparse embedding to the list of remote calls, naming `{SPARSE_API_BASE}/pooling` and `/tokenize` and the `sparse-only` CPU shape, matching how NER/rerank/CLIP are described.
- Add a bullet for `RemoteSparseEncoder` near the reranking bullet, stating the wire format is frozen because production collections depend on it.
- Note that `ENABLE_HYBRID` defaults on for vLLM or an explicit `SPARSE_API_BASE`, and off otherwise.

- [ ] **Step 2: Update docs/configuration.md**

Add `SPARSE_API_BASE`, `SPARSE_API_KEY`, `SPARSE_TIMEOUT`, `SPARSE_MODEL` and `ENABLE_HYBRID` to the config table in the same format as the `RERANK_*` rows, including defaults and the inheritance rule.

- [ ] **Step 3: Update README.md**

Note in the deployment/prerequisites section that a CPU dev host needs `sparse-only` alongside `gliner-only`, `rerank-only` and `clip-only`, and that dev collections need one re-ingest because the sparse model changed from BM42 to bge-m3.

- [ ] **Step 4: Verify everything**

Run: `make verify`
Expected: pre-commit (ruff + pyrefly) and the frontend lint/build all pass.

Run: `uv run pytest -q`
Expected: full suite green.

- [ ] **Step 5: Commit and open the PR**

```bash
git add CLAUDE.md docs/configuration.md README.md
git commit -m "docs(sparse): document remote sparse encoding and ENABLE_HYBRID"
git push -u origin fix/remote-sparse-encoder
gh pr create --fill
```

In the PR description, state that dev collections need one re-ingest (BM42 → bge-m3 sparse), that production collections are unaffected, and link the `vllm-service` PR plus the deferred parity-test issue.

---

## Self-review notes

- Spec coverage: `SPARSE_*` knobs (Task 1), `ENABLE_HYBRID` + unified sparse model (2), rename (3), gate narrowing and fastembed-branch deletion (4), ingest probe (5), dependency drop (6), docs (7). The spec's `IDF_EMBEDDING_MODELS` finding is carried into Task 4's context block as an explicit "do not change this" so a well-meaning implementer does not hardcode `None`.
- The parity test lives in the `vllm-service` repo (plan 1, deferred) — not duplicated here.
- Type consistency: `resolve_enable_hybrid`, `load_sparse_client_env`, `SparseClientConfig`, `RemoteSparseEncoder`, `sparse_client_config` and `probe_sparse_endpoint` are spelled identically across every task that defines or consumes them.
- Task 4 Step 1 and Task 5 Step 1 both depend on a `rag_instance` fixture, defined in full at Task 4 Step 1. Verified against the codebase: `tests/test_rag_unit.py` has no shared fixture and builds `RAG(qdrant_collection="test")` inline, so the fixture just wraps that.
- Verified: `torch.load` in plan 1 uses `weights_only=True`. The default (`False`) unpickles arbitrary objects, which would be arbitrary code execution at server start from a file in the shared cache volume.
