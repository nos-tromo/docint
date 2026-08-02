# `EMBED_API_BASE` override + `embed-only` rename (`docint`) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let docint point dense embeddings at a dedicated endpoint, so a dev host stops loading bge-m3 twice.

**Architecture:** docint currently builds its dense embedding client with `api_base=self.openai_api_base` — the shared chat endpoint — so dense cannot move without moving chat. This adds the `EMBED_*` override every other remote service already has (`NER_*`, `RERANK_*`, `CLIP_*`, `SPARSE_*`), changes one client construction site to use it, and renames `sparse-only` references to `embed-only`.

**Tech Stack:** Python 3.11, uv, pytest, ruff + pyrefly via pre-commit, llama-index.

**Design doc:** `docs/2026-08-02-embed-only-dense-consolidation-design.md`

**This AMENDS an open PR** — `nos-tromo/docint#386`, branch `fix/remote-sparse-encoder`, 16 commits. Do not open a new PR.

**Depends on** `vllm-service` plan 3 (`docs/2026-08-02-embed-only-plan-3-vllm-service.md`): the `embed-only` shape must exist before a dev host can point at it. The code and unit tests here do not need it (they mock the transport), but the docs reference it and the merge order is #78 → #386.

## Global Constraints

- Repo: `docint`. Work in the existing worktree at `.claude/worktrees/remote-sparse-encoder`, already on `fix/remote-sparse-encoder`. Confirm `git status` is clean before starting.
- Python `>=3.11,<3.12`. Use `uv add` / `uv remove`; never hand-edit `uv.lock`.
- **All `os.getenv` calls and config dataclasses live in `docint/utils/env_cfg.py`.**
- Google-style docstrings on new/modified functions and classes.
- `uv run pre-commit run --all-files` must pass before each commit (tracked-files-only — `git add` new files first).
- `make verify` must pass before pushing. Frontend deps are already installed in this worktree.
- **Never** expose real data or absolute development-machine paths in anything git sees.
- **`RemoteSparseEncoder`'s method bodies are FROZEN** — production collections depend on the vectors they produce. This plan does not touch them.
- **vLLM production behaviour must not change.** `EMBED_API_BASE` unset ⇒ inherit `OPENAI_API_BASE` ⇒ exactly today's behaviour.

## File Structure

| File | Change |
|---|---|
| `docint/utils/env_cfg.py` | Task 1 — `EmbedClientConfig` + `load_embed_client_env` |
| `tests/test_env_cfg_embed.py` | Task 1 — new |
| `docint/core/rag.py` | Task 2 — field, `__post_init__` wiring, one client-construction change |
| `tests/test_rag_embed_endpoint.py` | Task 2 — new |
| `.env.example`, `README.md`, `CLAUDE.md`, `docs/configuration.md`, `docint/core/rag.py`, `docint/utils/env_cfg.py`, `tests/*` | Task 3 — `sparse-only` → `embed-only` |

---

### Task 1: `EMBED_*` client configuration

**Files:**
- Modify: `docint/utils/env_cfg.py` (add immediately after `load_sparse_client_env`)
- Create: `tests/test_env_cfg_embed.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `EmbedClientConfig(api_base: str, api_key: str | None, timeout: float)` and `load_embed_client_env(default_api_base: str, default_api_key: str | None, default_timeout: float) -> EmbedClientConfig`.

**Context an implementer needs:**

This mirrors `load_sparse_client_env` in the same file exactly — read it first. Same "explicit env value wins → else inherit the passed default → else `None`" key handling, same `.rstrip("/")`, same `float()` coercion.

**One difference that matters:** unlike the sparse base, this value is consumed by the OpenAI SDK (via llama-index's `OpenAIEmbedding`), which appends `/embeddings` to it. So it is expected to end in `/v1`, and `/v1` must **not** be stripped — only a trailing slash.

- [ ] **Step 1: Write the failing test**

Create `tests/test_env_cfg_embed.py`:

```python
"""Tests for the EMBED_* dense-embedding client configuration.

Mirrors the SPARSE_*/RERANK_* contract: each knob falls back to the
active OpenAI client setting unless explicitly overridden, so the full
vllm-service router works with no configuration while a CPU dev host
points dense embeddings at the embed-only container.
"""

import pytest

from docint.utils.env_cfg import load_embed_client_env


def test_embed_client_inherits_openai_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """With no EMBED_* set, every field inherits the OpenAI client settings."""
    for name in ("EMBED_API_BASE", "EMBED_API_KEY", "EMBED_TIMEOUT"):
        monkeypatch.delenv(name, raising=False)

    cfg = load_embed_client_env(
        default_api_base="http://vllm-router:4000/v1",
        default_api_key="sk-master",
        default_timeout=300.0,
    )

    assert cfg.api_base == "http://vllm-router:4000/v1"
    assert cfg.api_key == "sk-master"
    assert cfg.timeout == 300.0


def test_embed_client_preserves_the_v1_suffix(monkeypatch: pytest.MonkeyPatch) -> None:
    """The OpenAI SDK appends /embeddings to this base, so /v1 must survive."""
    monkeypatch.setenv("EMBED_API_BASE", "http://embed-only:8000/v1/")
    monkeypatch.delenv("EMBED_API_KEY", raising=False)

    cfg = load_embed_client_env(
        default_api_base="http://vllm-router:4000/v1",
        default_api_key=None,
        default_timeout=300.0,
    )

    assert cfg.api_base == "http://embed-only:8000/v1"


def test_embed_client_explicit_timeout_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """EMBED_TIMEOUT overrides the inherited OpenAI timeout."""
    monkeypatch.setenv("EMBED_TIMEOUT", "45")
    monkeypatch.delenv("EMBED_API_BASE", raising=False)

    cfg = load_embed_client_env(
        default_api_base="http://vllm-router:4000/v1",
        default_api_key="sk-master",
        default_timeout=300.0,
    )

    assert cfg.timeout == 45.0


def test_embed_client_blank_key_disables_auth(monkeypatch: pytest.MonkeyPatch) -> None:
    """The embed-only shape has no Bearer gate; a blank default means no header."""
    monkeypatch.delenv("EMBED_API_KEY", raising=False)
    monkeypatch.setenv("EMBED_API_BASE", "http://embed-only:8000/v1")

    cfg = load_embed_client_env(
        default_api_base="http://vllm-router:4000/v1",
        default_api_key="",
        default_timeout=300.0,
    )

    assert cfg.api_key is None


def test_embed_client_explicit_key_wins_over_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """An explicitly set EMBED_API_KEY beats the inherited one."""
    monkeypatch.setenv("EMBED_API_KEY", "sk-embed")

    cfg = load_embed_client_env(
        default_api_base="http://vllm-router:4000/v1",
        default_api_key="sk-master",
        default_timeout=300.0,
    )

    assert cfg.api_key == "sk-embed"
```

That last test closes a gap the `SPARSE_*` suite still has — its explicit-key branch is untested. Do not replicate the omission.

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_env_cfg_embed.py -v`
Expected: FAIL — `ImportError: cannot import name 'load_embed_client_env'`

- [ ] **Step 3: Write the implementation**

Add to `docint/utils/env_cfg.py`, immediately after `load_sparse_client_env`:

```python
@dataclass(frozen=True)
class EmbedClientConfig:
    """Dataclass for the remote dense-embedding HTTP client."""

    api_base: str
    api_key: str | None
    timeout: float


def load_embed_client_env(
    default_api_base: str,
    default_api_key: str | None,
    default_timeout: float,
) -> "EmbedClientConfig":
    """Load the remote dense-embedding client configuration.

    Dense embeddings go through an OpenAI-compatible endpoint. Defaults
    mirror the OpenAI client settings, so the full vllm-service router
    needs no configuration. A CPU dev host points this at the
    ``embed-only`` container, which serves dense from the same bge-m3
    instance it uses for sparse — avoiding a second copy of the model
    alongside Ollama.

    The value is consumed by the OpenAI SDK, which appends
    ``/embeddings`` to it, so it is expected to end in ``/v1``; only a
    trailing slash is stripped.

    Args:
        default_api_base (str): Fallback base URL when ``EMBED_API_BASE``
            is unset. Typically the active ``OPENAI_API_BASE``.
        default_api_key (str | None): Fallback Bearer token when
            ``EMBED_API_KEY`` is unset. ``None`` (or empty) disables auth.
        default_timeout (float): Fallback request timeout in seconds.

    Returns:
        EmbedClientConfig: Resolved configuration.
    """
    raw_key = os.getenv("EMBED_API_KEY")
    if raw_key is not None and raw_key.strip():
        api_key: str | None = raw_key.strip()
    elif default_api_key and default_api_key.strip():
        api_key = default_api_key.strip()
    else:
        api_key = None
    return EmbedClientConfig(
        api_base=os.getenv("EMBED_API_BASE", default_api_base).rstrip("/"),
        api_key=api_key,
        timeout=float(os.getenv("EMBED_TIMEOUT", default_timeout)),
    )
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_env_cfg_embed.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add docint/utils/env_cfg.py tests/test_env_cfg_embed.py
git commit -m "feat(embed): EMBED_* client configuration for dense embeddings"
```

---

### Task 2: Route the dense client through `EMBED_*`

**Files:**
- Modify: `docint/core/rag.py` — the `env_cfg` import block (~line 61), a new field beside `sparse_client_config` (~line 1905), `__post_init__` wiring (~line 1987), and the `embed_model` property's `embedding_kwargs` (~line 2325)
- Create: `tests/test_rag_embed_endpoint.py`

**Interfaces:**
- Consumes: `load_embed_client_env`, `EmbedClientConfig` from Task 1.
- Produces: `RAG.embed_client_config: EmbedClientConfig | None`.

**Context an implementer needs:**

`embedding_kwargs` at `rag.py:2325` currently sets `"api_base": self.openai_api_base` and `"api_key": self.openai_api_key`. Both change to read from the resolved embed config. Nothing else in the embedding path changes — `embed_batch_size`, `max_retries`, `model_name`, `reuse_client`, `timeout`, `dimensions` and `context_window` are untouched.

**Ordering matters.** Resolve `embed_client_config` in `__post_init__` *after* the `openai_*` attributes it inherits from are assigned (they are set around line 1972). Place it next to the existing `sparse_client_config` resolution so the two stay together. Resolving before those assignments would silently inherit empty values — a bug no test would catch unless it asserts the resolved base.

The single most important property: **with `EMBED_API_BASE` unset, the resolved base must equal `openai_api_base`**, so a vLLM production deployment is byte-identical to today.

- [ ] **Step 1: Write the failing test**

Create `tests/test_rag_embed_endpoint.py`:

```python
"""Tests that the dense embedding client targets the configured endpoint.

Production (vLLM) sets no EMBED_* vars and must keep inheriting
OPENAI_API_BASE exactly; a CPU dev host overrides EMBED_API_BASE to
reach the embed-only container without moving chat.
"""

import pytest

from docint.core.rag import RAG


@pytest.fixture()
def _clear_embed_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove every EMBED_* var so each test controls its own state."""
    for name in ("EMBED_API_BASE", "EMBED_API_KEY", "EMBED_TIMEOUT"):
        monkeypatch.delenv(name, raising=False)


def test_embed_config_inherits_openai_base_when_unset(
    monkeypatch: pytest.MonkeyPatch,
    _clear_embed_env: None,
) -> None:
    """Production path: unset EMBED_API_BASE must resolve to OPENAI_API_BASE."""
    monkeypatch.setenv("OPENAI_API_BASE", "http://vllm-router:4000/v1")
    rag = RAG(qdrant_collection="test")
    assert rag.embed_client_config is not None
    assert rag.embed_client_config.api_base == "http://vllm-router:4000/v1"
    assert rag.embed_client_config.api_base == rag.openai_api_base


def test_embed_config_uses_explicit_override(
    monkeypatch: pytest.MonkeyPatch,
    _clear_embed_env: None,
) -> None:
    """Dev path: EMBED_API_BASE moves dense without moving chat."""
    monkeypatch.setenv("OPENAI_API_BASE", "http://ollama:11434/v1")
    monkeypatch.setenv("EMBED_API_BASE", "http://embed-only:8000/v1")
    rag = RAG(qdrant_collection="test")
    assert rag.embed_client_config is not None
    assert rag.embed_client_config.api_base == "http://embed-only:8000/v1"
    assert rag.openai_api_base == "http://ollama:11434/v1"


def test_embed_model_client_targets_the_embed_endpoint(
    monkeypatch: pytest.MonkeyPatch,
    _clear_embed_env: None,
) -> None:
    """The constructed embedding client must use the embed base, not the chat base."""
    monkeypatch.setenv("OPENAI_API_BASE", "http://ollama:11434/v1")
    monkeypatch.setenv("EMBED_API_BASE", "http://embed-only:8000/v1")
    captured: dict[str, object] = {}

    from docint.core import rag as rag_module

    def _capture(**kwargs: object) -> object:
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(rag_module, "BudgetedOpenAIEmbedding", _capture)
    rag = RAG(qdrant_collection="test")
    _ = rag.embed_model

    assert captured["api_base"] == "http://embed-only:8000/v1"
```

The third test is the load-bearing one: the first two only prove the config resolved, not that the client actually uses it.

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_rag_embed_endpoint.py -v`
Expected: FAIL — `AttributeError: 'RAG' object has no attribute 'embed_client_config'`

- [ ] **Step 3: Write the implementation**

1. Add `EmbedClientConfig` and `load_embed_client_env` to the `env_cfg` import block in `rag.py`.
2. Declare the field beside `sparse_client_config`:
   ```python
   embed_client_config: EmbedClientConfig | None = field(default=None, init=False, repr=False)
   ```
3. In `__post_init__`, next to the `sparse_client_config` resolution:
   ```python
   self.embed_client_config = load_embed_client_env(
       default_api_base=self.openai_api_base or "",
       default_api_key=self.openai_api_key,
       default_timeout=self.openai_timeout,
   )
   ```
4. In the `embed_model` property's `embedding_kwargs`, replace the two OpenAI values:
   ```python
   "api_base": self.embed_client_config.api_base if self.embed_client_config else self.openai_api_base,
   "api_key": self.embed_client_config.api_key if self.embed_client_config else self.openai_api_key,
   ```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_rag_embed_endpoint.py -v` → PASS (3 tests)
Run: `uv run pytest -q` → full suite green (1239 before this task; expect 1247 after Tasks 1-2).

A failure elsewhere means something else read `openai_api_base` expecting it to be the embedding endpoint. Report it rather than editing the test.

- [ ] **Step 5: Prove the third test has teeth**

Temporarily revert `"api_base"` to `self.openai_api_base`, run `test_embed_model_client_targets_the_embed_endpoint`, confirm it FAILS, then restore. Confirm with `git diff docint/core/rag.py` that the revert is complete.

- [ ] **Step 6: Commit**

```bash
git add docint/core/rag.py tests/test_rag_embed_endpoint.py
git commit -m "feat(embed): route dense embeddings through EMBED_API_BASE"
```

---

### Task 3: Rename references and document

**Files:** `.env.example`, `README.md`, `CLAUDE.md`, `docs/configuration.md`, `docint/core/rag.py`, `docint/utils/env_cfg.py`, `tests/test_rag_sparse_gate.py`, `tests/test_env_cfg_sparse.py`

**Interfaces:** consumes Tasks 1-2; produces nothing consumed later.

**Context an implementer needs:**

The `vllm-service` shape is renamed `sparse-only` → `embed-only`. Every docint reference follows. Verified inventory at the current branch head — `sparse-only` appears in: `.env.example` (3), `README.md` (2), `CLAUDE.md` (2), `docint/core/rag.py` (3, including the `probe_sparse_endpoint` `RuntimeError` message at ~2557), `docint/utils/env_cfg.py` (4), `tests/test_rag_sparse_gate.py` (6), `tests/test_env_cfg_sparse.py` (6), `docs/configuration.md` (3).

**What does NOT rename:** `SPARSE_API_BASE` / `SPARSE_API_KEY` / `SPARSE_TIMEOUT` / `SPARSE_MODEL`, `RemoteSparseEncoder`, `probe_sparse_endpoint`, `resolve_enable_hybrid`, the test file names, and `ENABLE_HYBRID`. Those name the *capability*, not the deployment shape, and renaming them would churn a reviewed PR for no gain.

- [ ] **Step 1: Rename the references**

Replace `sparse-only` with `embed-only` and `http://sparse-only:8000` with `http://embed-only:8000` throughout. In `SPARSE_API_BASE` examples the `/v1` suffix is optional (docint strips it), but write `http://embed-only:8000` for sparse and `http://embed-only:8000/v1` for embed, so the two knobs read distinctly even though either form works for sparse.

Verify with a **case-insensitive** sweep:
```bash
grep -rni "sparse-only" --exclude-dir=.git --exclude-dir=.venv \
  --exclude-dir=.superpowers --exclude-dir=node_modules --exclude-dir=dist . | grep -v "^./uv.lock"
```
Expected: no output. A case-sensitive sweep has already missed an all-caps variable once on this work; do not repeat it.

- [ ] **Step 2: Document `EMBED_*`**

`docs/configuration.md`: add `EMBED_API_BASE`, `EMBED_API_KEY`, `EMBED_TIMEOUT` rows beside the `SPARSE_*` rows, in the same format, stating the inheritance rule and that the value must include `/v1` because the OpenAI SDK appends `/embeddings`.

`.env.example`: add the `EMBED_*` knobs to the sparse block (now the embed block), and state the dev consequence plainly — pointing both `EMBED_API_BASE` and `SPARSE_API_BASE` at `embed-only` means bge-m3 is loaded once instead of twice, and Ollama then serves chat only.

`CLAUDE.md`: update the "All ML inference is remote" bullet so dense embeddings are listed with their own `{EMBED_API_BASE}` endpoint rather than folded into the generic OpenAI-compatible API, matching how NER/rerank/CLIP/sparse are described.

`README.md`: note that a CPU dev host sets both knobs to the same container, and that this replaces Ollama's bge-m3.

- [ ] **Step 3: Update the migration note**

`README.md` already tells dev operators to **delete the collection and ingest again** for the BM42 → bge-m3 sparse change. Dense vectors also change now (Ollama's quantised GGUF → fp32 transformers), so add one sentence saying the same delete-and-reingest covers both — it is one migration, not two. Production (vLLM) still needs none.

- [ ] **Step 4: Verify**

Run: `uv run pytest -q` → green.
Run: `make verify` → green (frontend deps are installed in this worktree).

- [ ] **Step 5: Commit and update the PR**

```bash
git add -A
git commit -m "docs(embed): document EMBED_API_BASE and the embed-only shape"
git push
```

The branch already tracks `origin/fix/remote-sparse-encoder`, so `git push` updates PR #386 in place. **Do not open a new PR.**

Then update the PR body: it now also adds the `EMBED_*` override and points dev's dense embeddings at `embed-only`, so bge-m3 is loaded once rather than twice. Keep the existing merge-gate checklist and update the `sparse-only` mentions in it to `embed-only`.

Report the PR URL and CI status.

---

## Self-review notes

- Spec coverage: `EMBED_*` config (Task 1), client routing (Task 2), rename + docs + migration note (Task 3). The design's "production unaffected" requirement is pinned by `test_embed_config_inherits_openai_base_when_unset`.
- Deliberately NOT renamed: the `SPARSE_*` env vars and the `RemoteSparseEncoder` / `probe_sparse_endpoint` symbols. Stated explicitly in Task 3's context so an implementer does not over-apply the rename to a reviewed PR.
- Type consistency: `EmbedClientConfig`, `load_embed_client_env`, `embed_client_config` are spelled identically across tasks.
- Task 1 adds the explicit-key test the `SPARSE_*` suite omits; the omission is noted in that plan's ledger as pre-existing debt and is not replicated here.
- The design's non-goal (fusing dense and sparse into one forward pass) is not attempted.
