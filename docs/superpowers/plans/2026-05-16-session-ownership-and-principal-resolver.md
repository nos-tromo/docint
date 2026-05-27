# Session Ownership & Principal Resolver Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give every conversation an owner derived from a trusted request header (with a configurable dev fallback) and make session list/history/delete owner-scoped, backfilling legacy rows idempotently.

**Architecture:** A request-scoped principal is resolved by a tiny pure FastAPI dependency that reads a configured trusted header and falls back to a single configured default identity (the same value also seeds the migration backfill). `Conversation` gains a nullable, indexed `owner` column with an idempotent ALTER+backfill migration mirroring the existing `_ensure_turn_validation_columns` pattern; `SessionManager` threads `owner` so list/history/delete are owner-scoped and cross-owner access is reported as not-found. The query path, collection endpoints, and `qdrant_collection` logic are untouched (that is Phase 2).

**Tech Stack:** FastAPI, SQLAlchemy/SQLite, pytest, uv

---

## Phased roadmap (context)

- **Plan 1 of 3 — this document.** Session ownership + the data migration + the principal-resolver seam (env config dataclass, `PrincipalResolver`, `Conversation.owner` column + idempotent backfill, owner-scoped `SessionManager`, and `Depends(resolve_principal)` wired only into the three session endpoints).
- **Plan 2 of 3 — separate plan (NOT here).** Singleton decoupling: extract `CollectionEngineCache` (shared, read-only) + `SessionRuntimeCache` (per-session chat state), delete the destructive global `select_collection()` teardown, and thread the session-resolved collection + principal through every `api.py` endpoint that currently reads the process-global `rag.qdrant_collection`.
- **Plan 3 of 3 — separate plan (NOT here).** Frontend integration: drop the now-defunct global collection-select/persist in `frontend/src/stores/ui.ts`; collection pick becomes "create session"; `session_id` resume contract unchanged. All three plans may land together in one PR.

## File Structure

| File | Create / Modify | Responsibility |
|------|-----------------|----------------|
| `docint/utils/env_cfg.py` | Modify (add `PrincipalConfig` dataclass + `load_principal_env` loader near `HostConfig`/`load_host_env` at line 453) | Single source of the trusted header name and the one configured default identity (dev fallback **and** migration backfill owner). |
| `docint/core/auth/__init__.py` | Create | New `docint.core.auth` package marker. |
| `docint/core/auth/principal.py` | Create | `resolve_principal(request)` FastAPI dependency — the single seam the future auth track replaces. |
| `docint/core/state/conversation.py` | Modify (`Conversation` model, after `collection_name` at line 23) | Add `owner = Column(String, nullable=True, index=True)`. |
| `docint/core/state/base.py` | Modify (add `_ensure_conversation_owner_column`, call it from `_make_session_maker` at line 76) | Idempotent ALTER TABLE + index + backfill of `conversations.owner` to the configured default identity. |
| `docint/core/state/session_manager.py` | Modify (`_load_or_create_convo`, `list_sessions`, `get_session_history`, `delete_session`) | Thread `owner`: stamp new conversations; filter list; treat cross-owner history/delete as not-found. |
| `docint/core/api.py` | Modify (fastapi import line 9; `list_sessions`/`get_session_history`/`delete_session` endpoints at lines 1125/1148/1169) | Add `principal: str = Depends(resolve_principal)` and pass it through; map "not found" to HTTP 404. |
| `tests/test_principal.py` | Create | Unit tests for `load_principal_env` and `resolve_principal`. |
| `tests/test_conversation_owner_migration.py` | Create | Unit tests for `_ensure_conversation_owner_column` (fresh DB, legacy backfill, idempotency). |
| `tests/test_session_manager_ownership.py` | Create | Unit tests for owner-scoped `SessionManager` behaviour. |
| `tests/test_api.py` | Modify (`DummySessionManager` methods at lines 18/26/37; add 2 new tests) | Keep existing `test_sessions_endpoints` green with new `owner` kwarg; add owner-pass-through + 404 + 401 endpoint tests. |

---

## Task 1 — `PrincipalConfig` + `load_principal_env` in `env_cfg.py`

**Files:**
- Create: `tests/test_principal.py` (env loader tests only in this task)
- Modify: `docint/utils/env_cfg.py` — add `PrincipalConfig` dataclass + `load_principal_env` function immediately after `load_host_env` (which ends at line 488; the next `@dataclass(frozen=True)` starts at line 491 — insert between them)

Pattern reference (already in the file): `HostConfig` dataclass at `docint/utils/env_cfg.py:453-460` and `load_host_env` at `:463-488` use `@dataclass(frozen=True)`, a Google-style docstring listing each field, and `os.getenv("ENV_NAME", default_param)` per field. We mirror that exactly. A blank/whitespace-only `DOCINT_DEFAULT_IDENTITY` is normalised to `None` (no fallback ⇒ resolver fails closed) using `.strip()`, matching the `.strip()` convention already used elsewhere in this file (e.g. line 179 `os.getenv("INFERENCE_PROVIDER", "ollama").strip().lower()`).

- [ ] **Step 1: Write the failing env-loader tests.** Create `tests/test_principal.py` with EXACTLY:

```python
"""Tests for principal configuration and the request principal resolver."""

import pytest

from docint.utils.env_cfg import PrincipalConfig, load_principal_env


def test_load_principal_env_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """With no env vars set, the header name defaults and there is no fallback.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
    """
    monkeypatch.delenv("DOCINT_AUTH_HEADER", raising=False)
    monkeypatch.delenv("DOCINT_DEFAULT_IDENTITY", raising=False)

    cfg = load_principal_env()

    assert isinstance(cfg, PrincipalConfig)
    assert cfg.header_name == "X-Auth-User"
    assert cfg.default_identity is None


def test_load_principal_env_reads_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    """Explicit env values override the header name and set a fallback identity.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
    """
    monkeypatch.setenv("DOCINT_AUTH_HEADER", "X-Remote-User")
    monkeypatch.setenv("DOCINT_DEFAULT_IDENTITY", "operator")

    cfg = load_principal_env()

    assert cfg.header_name == "X-Remote-User"
    assert cfg.default_identity == "operator"


def test_load_principal_env_blank_identity_is_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A blank/whitespace ``DOCINT_DEFAULT_IDENTITY`` normalises to ``None``.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
    """
    monkeypatch.setenv("DOCINT_DEFAULT_IDENTITY", "   ")

    cfg = load_principal_env()

    assert cfg.default_identity is None
```

- [ ] **Step 2: Run the test — expect failure (symbol does not exist yet).**

```bash
uv run pytest tests/test_principal.py -v
```
Expected: FAIL — `ImportError: cannot import name 'PrincipalConfig' from 'docint.utils.env_cfg'` (collection error, 0 passed).

- [ ] **Step 3: Add the dataclass + loader to `env_cfg.py`.** Insert the following block in `docint/utils/env_cfg.py` immediately after the `load_host_env` function's closing line (the `)` on line 488) and before the next `@dataclass(frozen=True)` on line 491 (leave the existing two blank lines that already separate top-level definitions):

```python
@dataclass(frozen=True)
class PrincipalConfig:
    """Dataclass for request-principal resolution configuration."""

    header_name: str
    default_identity: str | None


def load_principal_env(
    default_header_name: str = "X-Auth-User",
    default_identity: str | None = None,
) -> PrincipalConfig:
    """Loads request-principal configuration from environment variables.

    The same single ``default_identity`` value is used both as the
    resolver's dev fallback (when the trusted header is absent) and as
    the migration backfill owner for pre-existing conversation rows, so
    that legacy data and pre-auth requests share one identity.

    Args:
        default_header_name (str): Default trusted header carrying the
            authenticated principal.
        default_identity (str | None): Default fallback identity when no
            ``DOCINT_DEFAULT_IDENTITY`` is configured.

    Returns:
        PrincipalConfig: Dataclass containing principal configuration.
        - header_name (str): The trusted header carrying the principal.
        - default_identity (str | None): Fallback / backfill identity, or
          ``None`` when unset (resolver then fails closed with 401).
    """
    raw_identity = os.getenv("DOCINT_DEFAULT_IDENTITY")
    if raw_identity is not None and raw_identity.strip():
        resolved_identity: str | None = raw_identity.strip()
    else:
        resolved_identity = default_identity

    return PrincipalConfig(
        header_name=os.getenv("DOCINT_AUTH_HEADER", default_header_name),
        default_identity=resolved_identity,
    )
```

- [ ] **Step 4: Run the test — expect pass.**

```bash
uv run pytest tests/test_principal.py -v
```
Expected: PASS — 3 passed (the two `resolve_principal` tests are added in Task 2; this run only has the 3 loader tests).

- [ ] **Step 5: Commit.**

```bash
git add docint/utils/env_cfg.py tests/test_principal.py
git commit -m "$(cat <<'EOF'
feat(config): add PrincipalConfig and load_principal_env loader

Single configured identity is both the dev fallback principal and the
migration backfill owner. Header name defaults to X-Auth-User; a blank
DOCINT_DEFAULT_IDENTITY normalises to None so the resolver fails closed.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2 — `resolve_principal` dependency in `docint/core/auth/principal.py`

**Files:**
- Create: `docint/core/auth/__init__.py`
- Create: `docint/core/auth/principal.py`
- Modify: `tests/test_principal.py` — append resolver tests

Behaviour (spec Section 8 error table): header present ⇒ its value; header absent + configured `default_identity` ⇒ that identity; header absent + no configured identity ⇒ `HTTPException(status_code=401)`. The resolver reads config via `load_principal_env()` at call time (not import time) so tests can `monkeypatch.setenv` per test, mirroring how `api.py:42` calls `load_host_env()` rather than caching a module-global config object.

- [ ] **Step 1: Write the failing resolver tests.** Append to `tests/test_principal.py` (after the existing loader tests; add the new imports at the top of the file alongside the existing imports — the final import block must read exactly as shown):

Replace the existing import block at the top of `tests/test_principal.py`:

```python
import pytest

from docint.utils.env_cfg import PrincipalConfig, load_principal_env
```

with:

```python
import pytest
from fastapi import HTTPException
from starlette.requests import Request

from docint.core.auth.principal import resolve_principal
from docint.utils.env_cfg import PrincipalConfig, load_principal_env


def _make_request(headers: dict[str, str] | None = None) -> Request:
    """Build a minimal Starlette ``Request`` with the given headers.

    Args:
        headers (dict[str, str] | None): Header name/value pairs.

    Returns:
        Request: A request object whose ``.headers`` reflects ``headers``.
    """
    raw_headers = [
        (key.lower().encode("latin-1"), value.encode("latin-1"))
        for key, value in (headers or {}).items()
    ]
    scope = {
        "type": "http",
        "method": "GET",
        "path": "/",
        "headers": raw_headers,
    }
    return Request(scope)
```

Then append these test functions to the end of the file:

```python
def test_resolve_principal_returns_header_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A present trusted header is returned verbatim as the principal.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
    """
    monkeypatch.delenv("DOCINT_AUTH_HEADER", raising=False)
    monkeypatch.delenv("DOCINT_DEFAULT_IDENTITY", raising=False)

    request = _make_request({"X-Auth-User": "alice"})

    assert resolve_principal(request) == "alice"


def test_resolve_principal_falls_back_to_default_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the header is absent the configured default identity is used.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
    """
    monkeypatch.delenv("DOCINT_AUTH_HEADER", raising=False)
    monkeypatch.setenv("DOCINT_DEFAULT_IDENTITY", "operator")

    request = _make_request({})

    assert resolve_principal(request) == "operator"


def test_resolve_principal_fails_closed_without_header_or_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No header and no configured fallback must raise HTTP 401.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
    """
    monkeypatch.delenv("DOCINT_AUTH_HEADER", raising=False)
    monkeypatch.delenv("DOCINT_DEFAULT_IDENTITY", raising=False)

    request = _make_request({})

    with pytest.raises(HTTPException) as excinfo:
        resolve_principal(request)
    assert excinfo.value.status_code == 401


def test_resolve_principal_honours_custom_header_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A configured non-default header name is the one consulted.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
    """
    monkeypatch.setenv("DOCINT_AUTH_HEADER", "X-Remote-User")
    monkeypatch.delenv("DOCINT_DEFAULT_IDENTITY", raising=False)

    request = _make_request({"X-Remote-User": "bob"})

    assert resolve_principal(request) == "bob"
```

- [ ] **Step 2: Run the test — expect failure (module does not exist).**

```bash
uv run pytest tests/test_principal.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'docint.core.auth'` (collection error, 0 passed).

- [ ] **Step 3: Create the package marker.** Create `docint/core/auth/__init__.py` with EXACTLY:

```python
"""Authentication seam: request-principal resolution."""
```

- [ ] **Step 4: Create the resolver.** Create `docint/core/auth/principal.py` with EXACTLY:

```python
"""Request-principal resolution dependency.

This module is the single seam the deferred auth track replaces: swap
the header read for a verified-token read and nothing downstream (data
model, ownership queries, endpoint wiring) changes.
"""

from fastapi import HTTPException, Request

from docint.utils.env_cfg import load_principal_env


def resolve_principal(request: Request) -> str:
    """Resolve the calling principal from the configured trusted header.

    Resolution order (spec Section 8):

    1. If the configured trusted header is present, return its value.
    2. Otherwise, if a default identity is configured, return it (the
       dev/pre-auth fallback, also the migration backfill owner).
    3. Otherwise fail closed with HTTP 401.

    Args:
        request (Request): The incoming FastAPI/Starlette request.

    Returns:
        str: The resolved principal identifier.

    Raises:
        HTTPException: With status 401 when neither the trusted header
            nor a configured default identity is available.
    """
    cfg = load_principal_env()
    header_value = request.headers.get(cfg.header_name)
    if header_value:
        return header_value
    if cfg.default_identity:
        return cfg.default_identity
    raise HTTPException(status_code=401, detail="Missing authenticated principal.")
```

- [ ] **Step 5: Run the test — expect pass.**

```bash
uv run pytest tests/test_principal.py -v
```
Expected: PASS — 7 passed (3 loader + 4 resolver).

- [ ] **Step 6: Commit.**

```bash
git add docint/core/auth/__init__.py docint/core/auth/principal.py tests/test_principal.py
git commit -m "$(cat <<'EOF'
feat(auth): add resolve_principal trusted-header dependency

Tiny pure FastAPI dependency: header present -> value; absent + default
identity configured -> default; absent + none -> HTTP 401. This is the
single seam the future auth track replaces.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3 — `Conversation.owner` column

**Files:**
- Modify: `docint/core/state/conversation.py` — `Conversation` model, add column after `collection_name` (line 23)
- Create: `tests/test_conversation_owner_migration.py` — schema assertion only in this task

Reference: the model currently declares `collection_name = Column(String, nullable=True)` at `docint/core/state/conversation.py:23`. The spec (Section 7) mandates `owner = Column(String, nullable=True, index=True)`. `String` is already imported on line 5 (`from sqlalchemy import Column, DateTime, String, Text`), so no import change is needed.

- [ ] **Step 1: Write the failing schema test.** Create `tests/test_conversation_owner_migration.py` with EXACTLY:

```python
"""Tests for the conversations.owner column and its idempotent migration."""

from sqlalchemy import (
    Column,
    DateTime,
    MetaData,
    String,
    Table,
    Text,
    create_engine,
    inspect,
    text,
)

from docint.core.state.base import (
    Base,
    _ensure_conversation_owner_column,
    _make_session_maker,
)
from docint.core.state.conversation import Conversation


def test_conversation_model_declares_indexed_owner_column() -> None:
    """The ORM model must expose a nullable, indexed ``owner`` column."""
    owner_col = Conversation.__table__.columns["owner"]
    assert owner_col.nullable is True
    assert owner_col.index is True
    assert isinstance(owner_col.type, String)
```

- [ ] **Step 2: Run the test — expect failure.**

```bash
uv run pytest tests/test_conversation_owner_migration.py::test_conversation_model_declares_indexed_owner_column -v
```
Expected: FAIL — `ImportError: cannot import name '_ensure_conversation_owner_column' from 'docint.core.state.base'` (collection error; the helper is added in Task 4 — this task only needs the model change, but the test file imports the helper up-front so the collection error is expected until Task 4).

> Note: this test file deliberately imports `_ensure_conversation_owner_column` (added in Task 4) so the file is written once. Run only the single named test in this task; the migration tests are exercised in Task 4. The model change below makes the *model* assertion logic correct; the import is satisfied in Task 4.

- [ ] **Step 3: Add the column to the model.** In `docint/core/state/conversation.py`, change line 23 from:

```python
    collection_name = Column(String, nullable=True)
```

to:

```python
    collection_name = Column(String, nullable=True)
    owner = Column(String, nullable=True, index=True)
```

- [ ] **Step 4: Re-run (still fails at import until Task 4 — that is expected and acceptable).**

```bash
uv run pytest tests/test_conversation_owner_migration.py::test_conversation_model_declares_indexed_owner_column -v
```
Expected: FAIL — still `ImportError: cannot import name '_ensure_conversation_owner_column'`. This is sequenced: Task 4 adds the helper and turns this green. Do **not** commit Task 3 alone (the suite would have a collection error). Proceed directly to Task 4; commit Task 3's model change together with Task 4.

---

## Task 4 — `_ensure_conversation_owner_column` migration + backfill in `base.py`

**Files:**
- Modify: `docint/core/state/base.py` — add `_ensure_conversation_owner_column(engine)` after `_ensure_turn_validation_columns` (which ends at line 60), and call it from `_make_session_maker` right after the existing `_ensure_turn_validation_columns(engine)` call on line 76
- Modify: `tests/test_conversation_owner_migration.py` — add migration tests (file created in Task 3)

Reference: `_ensure_turn_validation_columns` at `docint/core/state/base.py:31-60` is the exact precedent — `try:` / `inspector = inspect(engine)` / `if "<table>" not in inspector.get_table_names(): return` / `existing = {col["name"] for col in inspector.get_columns("<table>")}` / `with engine.begin() as conn: ... conn.execute(text("ALTER TABLE ... ADD COLUMN ..."))` / `except Exception as exc: logger.warning(...)`. `_make_session_maker` at `:64-77` calls `_ensure_turn_validation_columns(engine)` on line 76 immediately before `return sessionmaker(...)`. `text`, `inspect`, `Engine`, `logger` are already imported at the top of `base.py` (lines 5-6). The backfill identity comes from `load_principal_env().default_identity`; when it is `None` there is nothing to backfill to, so the `UPDATE` is skipped (legacy rows keep `owner IS NULL`, consistent with the resolver failing closed in that configuration).

- [ ] **Step 1: Add the migration tests.** Append to `tests/test_conversation_owner_migration.py` (the imports at the top of the file already include everything needed — `Column`, `DateTime`, `MetaData`, `String`, `Table`, `Text`, `create_engine`, `inspect`, `text`, `Base`, `_ensure_conversation_owner_column`, `_make_session_maker`, `Conversation`):

```python
def test_fresh_db_has_owner_column(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """A DB created via _make_session_maker exposes conversations.owner.

    Args:
        tmp_path: Temporary directory provided by pytest.
    """
    db_path = tmp_path / "fresh.db"
    _make_session_maker(f"sqlite:///{db_path}")

    engine = create_engine(f"sqlite:///{db_path}", future=True)
    columns = {c["name"] for c in inspect(engine).get_columns("conversations")}
    assert "owner" in columns
    engine.dispose()


def test_legacy_conversations_table_gets_column_and_backfill(
    monkeypatch,  # type: ignore[no-untyped-def]
) -> None:
    """A pre-existing conversations table without owner is upgraded + backfilled.

    Args:
        monkeypatch: The pytest monkeypatch fixture.
    """
    monkeypatch.setenv("DOCINT_DEFAULT_IDENTITY", "operator")

    engine = create_engine("sqlite:///:memory:", future=True)
    metadata = MetaData()
    # Mirror only the pre-owner conversations schema. The turns FK is
    # intentionally omitted so the legacy table stands alone for the
    # migration assertion (same approach as the turns migration test).
    Table(
        "conversations",
        metadata,
        Column("id", String, primary_key=True),
        Column("created_at", DateTime, nullable=False),
        Column("collection_name", String, nullable=True),
        Column("rolling_summary", Text, nullable=False),
    )
    metadata.create_all(engine)
    with engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO conversations "
                "(id, created_at, collection_name, rolling_summary) "
                "VALUES ('c1', '2026-01-01 00:00:00', 'alpha', '')"
            )
        )

    pre_columns = {c["name"] for c in inspect(engine).get_columns("conversations")}
    assert "owner" not in pre_columns

    _ensure_conversation_owner_column(engine)
    _ensure_conversation_owner_column(engine)  # second call must be a no-op

    post_columns = {c["name"] for c in inspect(engine).get_columns("conversations")}
    assert "owner" in post_columns
    indexes = inspect(engine).get_indexes("conversations")
    assert any("owner" in ix["column_names"] for ix in indexes)

    with engine.begin() as conn:
        backfilled = conn.execute(
            text("SELECT owner FROM conversations WHERE id = 'c1'")
        ).scalar()
    assert backfilled == "operator"
    engine.dispose()


def test_migration_helper_no_op_when_table_missing() -> None:
    """Helper must short-circuit cleanly if conversations doesn't exist yet."""
    engine = create_engine("sqlite:///:memory:", future=True)
    _ensure_conversation_owner_column(engine)  # no exception
    assert "conversations" not in inspect(engine).get_table_names()
    engine.dispose()


def test_backfill_skipped_when_no_default_identity(
    monkeypatch,  # type: ignore[no-untyped-def]
) -> None:
    """With no configured identity the column is added but rows stay NULL.

    Args:
        monkeypatch: The pytest monkeypatch fixture.
    """
    monkeypatch.delenv("DOCINT_DEFAULT_IDENTITY", raising=False)

    engine = create_engine("sqlite:///:memory:", future=True)
    metadata = MetaData()
    Table(
        "conversations",
        metadata,
        Column("id", String, primary_key=True),
        Column("created_at", DateTime, nullable=False),
        Column("collection_name", String, nullable=True),
        Column("rolling_summary", Text, nullable=False),
    )
    metadata.create_all(engine)
    with engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO conversations "
                "(id, created_at, collection_name, rolling_summary) "
                "VALUES ('c1', '2026-01-01 00:00:00', 'alpha', '')"
            )
        )

    _ensure_conversation_owner_column(engine)

    with engine.begin() as conn:
        owner_val = conn.execute(
            text("SELECT owner FROM conversations WHERE id = 'c1'")
        ).scalar()
    assert owner_val is None
    engine.dispose()
```

- [ ] **Step 2: Run the migration tests — expect failure (helper not defined).**

```bash
uv run pytest tests/test_conversation_owner_migration.py -v
```
Expected: FAIL — `ImportError: cannot import name '_ensure_conversation_owner_column' from 'docint.core.state.base'` (collection error, 0 passed).

- [ ] **Step 3: Add the migration helper to `base.py`.** In `docint/core/state/base.py`, insert the following function immediately after `_ensure_turn_validation_columns` (after its closing line 60) and before the `# --- Session maker ---` comment on line 63 (keep the two blank lines that already separate top-level definitions). Add the import for `load_principal_env` at the top of the file alongside the existing imports — the import block (currently lines 1-7) must become exactly:

```python
"""SQLAlchemy declarative base and session factory for state persistence."""

from pathlib import Path

from loguru import logger
from sqlalchemy import Engine, create_engine, inspect, text
from sqlalchemy.orm import declarative_base, sessionmaker

from docint.utils.env_cfg import load_principal_env
```

Then insert this function (between line 60 and the `# --- Session maker ---` comment):

```python
def _ensure_conversation_owner_column(engine: Engine) -> None:
    """Backfill an ``owner`` column onto a pre-existing ``conversations`` table.

    ``Base.metadata.create_all`` only creates missing tables, never adds
    columns to existing ones. Sessions DBs created before ownership
    shipped already have a ``conversations`` table and would otherwise
    have no owner to scope list/history/delete by. This adds the column,
    its index, and idempotently backfills legacy rows to the configured
    default identity (the same value the principal resolver returns
    pre-auth) so existing sessions are owned, not orphaned.
    """
    try:
        inspector = inspect(engine)
        if "conversations" not in inspector.get_table_names():
            return
        existing = {col["name"] for col in inspector.get_columns("conversations")}
        with engine.begin() as conn:
            if "owner" not in existing:
                conn.execute(
                    text("ALTER TABLE conversations ADD COLUMN owner TEXT")
                )
            index_names = {
                ix["name"] for ix in inspector.get_indexes("conversations")
            }
            if "ix_conversations_owner" not in index_names:
                conn.execute(
                    text(
                        "CREATE INDEX IF NOT EXISTS ix_conversations_owner "
                        "ON conversations (owner)"
                    )
                )
            default_identity = load_principal_env().default_identity
            if default_identity:
                conn.execute(
                    text(
                        "UPDATE conversations SET owner = :default "
                        "WHERE owner IS NULL"
                    ),
                    {"default": default_identity},
                )
    except Exception as exc:
        logger.warning(
            "Skipping conversations owner-column migration: {}: {}",
            type(exc).__name__,
            exc,
        )
```

- [ ] **Step 4: Wire the helper into `_make_session_maker`.** In `docint/core/state/base.py`, change the body of `_make_session_maker` (currently lines 73-77) from:

```python
    _ensure_sqlite_parent_dir(db_url)
    engine = create_engine(db_url, future=True)
    Base.metadata.create_all(engine)
    _ensure_turn_validation_columns(engine)
    return sessionmaker(bind=engine, expire_on_commit=False)
```

to:

```python
    _ensure_sqlite_parent_dir(db_url)
    engine = create_engine(db_url, future=True)
    Base.metadata.create_all(engine)
    _ensure_turn_validation_columns(engine)
    _ensure_conversation_owner_column(engine)
    return sessionmaker(bind=engine, expire_on_commit=False)
```

- [ ] **Step 5: Run the full migration + model test file — expect pass.**

```bash
uv run pytest tests/test_conversation_owner_migration.py -v
```
Expected: PASS — 5 passed (`test_conversation_model_declares_indexed_owner_column`, `test_fresh_db_has_owner_column`, `test_legacy_conversations_table_gets_column_and_backfill`, `test_migration_helper_no_op_when_table_missing`, `test_backfill_skipped_when_no_default_identity`).

- [ ] **Step 6: Run the pre-existing state suites to confirm no regression.**

```bash
uv run pytest tests/test_session_manager.py tests/test_session_manager_validation.py -v
```
Expected: PASS — all existing tests still pass (the new column is nullable and the helper is a no-op on `Base.metadata.create_all`-created in-memory DBs that already have `owner`).

- [ ] **Step 7: Commit (Task 3 model change + Task 4 migration together).**

```bash
git add docint/core/state/conversation.py docint/core/state/base.py tests/test_conversation_owner_migration.py
git commit -m "$(cat <<'EOF'
feat(state): add Conversation.owner with idempotent backfill migration

owner is nullable + indexed. _ensure_conversation_owner_column mirrors
_ensure_turn_validation_columns: ALTER + CREATE INDEX + backfill legacy
rows to the configured default identity, guarded and idempotent. Wired
into _make_session_maker.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5 — Owner-scoped `SessionManager`

**Files:**
- Modify: `docint/core/state/session_manager.py`:
  - `_load_or_create_convo` (lines 478-495) — accept `owner: str`, stamp it on new conversations
  - update its two call sites: `start_session` (line 96) and `_persist_turn` (line 530)
  - `list_sessions` (lines 724-751) — accept `owner: str`, filter by it
  - `get_session_history` (lines 753-821) — accept `owner: str`, return `[]` if the conversation is owned by a different principal
  - `delete_session` (lines 823-838) — accept `owner: str`, return `False` if owned by a different principal
- Create: `tests/test_session_manager_ownership.py`

Design decisions, grounded in the read source:
- `_load_or_create_convo` currently builds `Conversation(id=session_id)` and conditionally sets `collection_name` (lines 490-492). We add `conv.owner = owner` on the create branch. We do **not** mutate `owner` on an already-existing conversation (a pre-existing row keeps its migrated/original owner — re-stamping would let a second principal hijack an existing session id).
- `start_session` (line 96) and `_persist_turn` (line 530) are the only `_load_or_create_convo` callers. `start_session` already has `requested_id`; it needs an `owner` to pass through. `_persist_turn` is called internally by `chat`/`stream_chat` after `start_session` has run, so the conversation already exists by then and `_load_or_create_convo` takes the load branch — but we still must pass an `owner` argument for the signature. We thread `owner` into `start_session` and store it on the manager so `_persist_turn` can reuse it without changing `chat`/`stream_chat` signatures (Phase 2 territory). A new `self._owner` runtime field (reset in `reset_runtime`) holds it.
- `list_sessions`/`get_session_history`/`delete_session` are called only from `api.py` (the three endpoints) and tests — confirmed by grep. Adding a required `owner` parameter is type-sound and matches the spec's "prefer explicit required owner".
- "Not found" semantics: `get_session_history` already returns `[]` for a missing conversation (line 765); a cross-owner conversation is treated identically (return `[]`). `delete_session` already returns `False` for a missing conversation (line 838); a cross-owner conversation returns `False`. The API layer (Task 6) maps these to 404. This guarantees no existence leak.

- [ ] **Step 1: Write the failing ownership tests.** Create `tests/test_session_manager_ownership.py` with EXACTLY:

```python
"""Tests for owner-scoped session listing, history, and deletion."""

from typing import Any, Generator, cast
from unittest.mock import MagicMock

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from docint.core.state.base import Base
from docint.core.state.session_manager import SessionManager


@pytest.fixture
def session_manager() -> Generator[SessionManager, None, None]:
    """SessionManager bound to an in-memory SQLite DB.

    Mirrors the fixture in ``test_session_manager.py`` so tests share the
    same RAG mock surface.

    Returns:
        Generator[SessionManager, None, None]: The SessionManager instance.
    """
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    SessionMaker = sessionmaker(bind=engine)

    rag_mock = MagicMock()
    rag_mock.index = None
    rag_mock.qdrant_collection = "test_collection"
    rag_mock.qdrant_host = "http://qdrant:6333"
    rag_mock.embed_model_id = "embed-model"
    rag_mock.sparse_model_id = "sparse-model"
    rag_mock.text_model_id = "text-model"
    rag_mock.retrieve_similarity_top_k = 20
    rag_mock.rerank_top_n = 5
    rag_mock.conversation_summary_prompt = "Summarize turns:\n"
    rag_mock.rewrite_retrieval_query.return_value = "rewritten question"
    rag_mock._infer_collection_profile.return_value = {"coverage_unit": "documents"}
    mode = MagicMock()
    mode.value = "compact"
    rag_mock._resolve_chat_response_mode.return_value = mode
    cast(Any, rag_mock.get_source_by_node_id).return_value = None

    sm = SessionManager(rag=rag_mock)
    sm._SessionMaker = SessionMaker
    yield sm
    engine.dispose()


def _persist_owned_turn(sm: SessionManager, session_id: str, owner: str) -> None:
    """Create an owned conversation and persist one empty turn into it.

    Args:
        sm (SessionManager): The session manager under test.
        session_id (str): The conversation id.
        owner (str): The owning principal.
    """
    with sm._session_scope() as s:
        sm._load_or_create_convo(s, session_id, owner)

    resp_mock = MagicMock()
    resp_mock.metadata = {}
    resp_mock.source_nodes = []
    sm._owner = owner
    sm._persist_turn(
        session_id,
        "hello",
        resp_mock,
        {"response": "world", "reasoning": None},
    )


def test_load_or_create_convo_stamps_owner(
    session_manager: SessionManager,
) -> None:
    """A newly created conversation records the supplied owner.

    Args:
        session_manager (SessionManager): The session manager fixture.
    """
    with session_manager._session_scope() as s:
        conv = session_manager._load_or_create_convo(s, "sess-a", "alice")
        assert conv.owner == "alice"


def test_list_sessions_is_owner_scoped(
    session_manager: SessionManager,
) -> None:
    """list_sessions only returns conversations owned by the caller.

    Args:
        session_manager (SessionManager): The session manager fixture.
    """
    _persist_owned_turn(session_manager, "sess-a", "alice")
    _persist_owned_turn(session_manager, "sess-b", "bob")

    alice_sessions = session_manager.list_sessions("alice")
    bob_sessions = session_manager.list_sessions("bob")

    assert {s["id"] for s in alice_sessions} == {"sess-a"}
    assert {s["id"] for s in bob_sessions} == {"sess-b"}


def test_get_session_history_cross_owner_is_not_found(
    session_manager: SessionManager,
) -> None:
    """B reading A's session history gets an empty list (treated as 404).

    Args:
        session_manager (SessionManager): The session manager fixture.
    """
    _persist_owned_turn(session_manager, "sess-a", "alice")

    own = session_manager.get_session_history("sess-a", "alice")
    cross = session_manager.get_session_history("sess-a", "bob")

    assert len(own) == 2  # user + assistant
    assert cross == []


def test_delete_session_cross_owner_is_not_found(
    session_manager: SessionManager,
) -> None:
    """B deleting A's session returns False and does not delete it.

    Args:
        session_manager (SessionManager): The session manager fixture.
    """
    _persist_owned_turn(session_manager, "sess-a", "alice")

    assert session_manager.delete_session("sess-a", "bob") is False
    # Still present for the real owner.
    assert len(session_manager.get_session_history("sess-a", "alice")) == 2
    # Real owner can delete it.
    assert session_manager.delete_session("sess-a", "alice") is True
    assert session_manager.get_session_history("sess-a", "alice") == []


def test_get_session_history_missing_session_is_empty(
    session_manager: SessionManager,
) -> None:
    """An unknown session id yields an empty history (no existence leak).

    Args:
        session_manager (SessionManager): The session manager fixture.
    """
    assert session_manager.get_session_history("nope", "alice") == []


def test_delete_session_missing_session_is_false(
    session_manager: SessionManager,
) -> None:
    """Deleting an unknown session id returns False.

    Args:
        session_manager (SessionManager): The session manager fixture.
    """
    assert session_manager.delete_session("nope", "alice") is False
```

- [ ] **Step 2: Run the tests — expect failure (signatures don't accept `owner` / `_owner` missing).**

```bash
uv run pytest tests/test_session_manager_ownership.py -v
```
Expected: FAIL — `TypeError: SessionManager._load_or_create_convo() takes 3 positional arguments but 4 were given` (and related failures), 0 passed.

- [ ] **Step 3: Add the `_owner` runtime field.** In `docint/core/state/session_manager.py`, in the `SessionManager` dataclass field block, change (current lines 48-49):

```python
    session_id: str | None = field(default=None, init=False)
    session_store: str = field(default="", init=False)
```

to:

```python
    session_id: str | None = field(default=None, init=False)
    session_store: str = field(default="", init=False)
    _owner: str | None = field(default=None, init=False)
```

- [ ] **Step 4: Reset `_owner` in `reset_runtime`.** Change `reset_runtime` (current lines 65-69) from:

```python
    def reset_runtime(self) -> None:
        """Reset the runtime state."""
        self.session_id = None
        self.chat_engine = None
        self.chat_memory = None
```

to:

```python
    def reset_runtime(self) -> None:
        """Reset the runtime state."""
        self.session_id = None
        self.chat_engine = None
        self.chat_memory = None
        self._owner = None
```

- [ ] **Step 5: Thread `owner` into `start_session`.** Change the signature and the `_load_or_create_convo` call. Current `start_session` signature/docstring/body header (lines 71-97):

```python
    def start_session(self, requested_id: str | None = None) -> str:
        """Start a new chat session.

        After ``RAG.select_collection`` switches collections it deliberately
        resets ``rag.query_engine`` and ``rag.index`` to ``None``. To avoid
        forcing callers to rebuild before every session, this method mirrors
        the lazy-init pattern already used by ``RAG.run_query`` and builds the
        query engine on demand when it is missing.

        Args:
            requested_id (str | None, optional): The ID of the session to start. Defaults to None.

        Returns:
            str: The ID of the started session.
        """
        if not requested_id:
            requested_id = str(uuid.uuid4())
        self.session_id = requested_id

        # Initialize agent context for this session
        self.agent_contexts.setdefault(
            requested_id, AgentTurnContext(session_id=requested_id)
        )

        with self._session_scope() as s:
            self._load_or_create_convo(s, requested_id)
```

becomes:

```python
    def start_session(
        self, requested_id: str | None = None, owner: str | None = None
    ) -> str:
        """Start a new chat session.

        After ``RAG.select_collection`` switches collections it deliberately
        resets ``rag.query_engine`` and ``rag.index`` to ``None``. To avoid
        forcing callers to rebuild before every session, this method mirrors
        the lazy-init pattern already used by ``RAG.run_query`` and builds the
        query engine on demand when it is missing.

        Args:
            requested_id (str | None, optional): The ID of the session to start. Defaults to None.
            owner (str | None, optional): The principal that owns this
                session. Stamped on a newly created conversation and
                reused by ``_persist_turn``. Defaults to None.

        Returns:
            str: The ID of the started session.
        """
        if not requested_id:
            requested_id = str(uuid.uuid4())
        self.session_id = requested_id
        if owner is not None:
            self._owner = owner

        # Initialize agent context for this session
        self.agent_contexts.setdefault(
            requested_id, AgentTurnContext(session_id=requested_id)
        )

        with self._session_scope() as s:
            self._load_or_create_convo(s, requested_id, self._owner)
```

> `owner` is `str | None` here (not required) because `chat`/`stream_chat` call `start_session(session_id)` internally without an owner (Phase 2 wires the principal into the chat path). In that internal case `self._owner` is `None` and `_load_or_create_convo` leaves `owner` unset on a newly created row — exactly today's behaviour for the un-owned chat path, which Phase 2 replaces. The owner-scoped *endpoints* (list/history/delete) never go through `chat` and always pass an explicit principal.

- [ ] **Step 6: Update `_load_or_create_convo`.** Change it (current lines 478-495) from:

```python
    def _load_or_create_convo(self, session: Session, session_id: str) -> Conversation:
        """Load an existing conversation or create a new one.

        Args:
            session (Session): The database session.
            session_id (str): The ID of the session.

        Returns:
            Conversation: The loaded or created conversation.
        """
        conv = session.get(Conversation, session_id)
        if conv is None:
            conv = Conversation(id=session_id)
            if self.rag.qdrant_collection:
                conv.collection_name = cast(Any, self.rag.qdrant_collection)
            session.add(conv)
            session.commit()
        return conv
```

to:

```python
    def _load_or_create_convo(
        self, session: Session, session_id: str, owner: str | None = None
    ) -> Conversation:
        """Load an existing conversation or create a new one.

        A pre-existing conversation keeps its recorded owner; ``owner`` is
        only stamped when the row is created, so a second principal cannot
        rebind an existing session id.

        Args:
            session (Session): The database session.
            session_id (str): The ID of the session.
            owner (str | None, optional): Principal to stamp on a newly
                created conversation. Defaults to None.

        Returns:
            Conversation: The loaded or created conversation.
        """
        conv = session.get(Conversation, session_id)
        if conv is None:
            conv = Conversation(id=session_id)
            if self.rag.qdrant_collection:
                conv.collection_name = cast(Any, self.rag.qdrant_collection)
            if owner is not None:
                conv.owner = cast(Any, owner)
            session.add(conv)
            session.commit()
        return conv
```

- [ ] **Step 7: Update the `_persist_turn` call site.** In `_persist_turn`, change (current line 530):

```python
            conv = self._load_or_create_convo(s, session_id)
```

to:

```python
            conv = self._load_or_create_convo(s, session_id, self._owner)
```

- [ ] **Step 8: Make `list_sessions` owner-scoped.** Change it (current lines 724-751) from:

```python
    def list_sessions(self) -> list[dict[str, Any]]:
        """List all sessions ordered by creation date (descending).

        Returns:
            list[dict[str, Any]]: A list of session dictionaries.
        """
        with self._session_scope() as s:
            convs = s.query(Conversation).order_by(Conversation.created_at.desc()).all()
```

to:

```python
    def list_sessions(self, owner: str) -> list[dict[str, Any]]:
        """List the caller's sessions ordered by creation date (descending).

        Args:
            owner (str): The principal whose sessions to list.

        Returns:
            list[dict[str, Any]]: A list of session dictionaries owned by
                ``owner``.
        """
        with self._session_scope() as s:
            convs = (
                s.query(Conversation)
                .filter(Conversation.owner == owner)
                .order_by(Conversation.created_at.desc())
                .all()
            )
```

(The remainder of the method — the `for c in convs:` loop building `results` — is unchanged.)

- [ ] **Step 9: Make `get_session_history` owner-scoped.** Change its signature/docstring and the conversation lookup (current lines 753-765) from:

```python
    def get_session_history(self, session_id: str) -> list[dict[str, Any]]:
        """Get the full message history for a session.

        Args:
            session_id (str): The ID of the session.

        Returns:
            list[dict[str, Any]]: A list of message dictionaries.
        """
        with self._session_scope() as s:
            conv = s.query(Conversation).filter_by(id=session_id).first()
            if not conv:
                return []
```

to:

```python
    def get_session_history(
        self, session_id: str, owner: str
    ) -> list[dict[str, Any]]:
        """Get the full message history for a session the caller owns.

        A session owned by a different principal is treated as not found
        (empty list), so the API layer can return 404 without leaking
        whether the id exists.

        Args:
            session_id (str): The ID of the session.
            owner (str): The principal requesting the history.

        Returns:
            list[dict[str, Any]]: A list of message dictionaries, or an
                empty list when the session is missing or not owned by
                ``owner``.
        """
        with self._session_scope() as s:
            conv = s.query(Conversation).filter_by(id=session_id).first()
            if not conv or conv.owner != owner:
                return []
```

(The remainder of the method — the `messages = []` / `for t in conv.turns:` loop — is unchanged.)

- [ ] **Step 10: Make `delete_session` owner-scoped.** Change it (current lines 823-838) from:

```python
    def delete_session(self, session_id: str) -> bool:
        """Delete a session.

        Args:
            session_id (str): The ID of the session to delete.

        Returns:
            bool: True if deleted, False otherwise.
        """
        with self._session_scope() as s:
            conv = s.query(Conversation).filter_by(id=session_id).first()
            if conv:
                s.delete(conv)
                s.commit()
                return True
            return False
```

to:

```python
    def delete_session(self, session_id: str, owner: str) -> bool:
        """Delete a session the caller owns.

        A session owned by a different principal is treated as not found
        (returns ``False``), so the API layer can return 404 without
        leaking whether the id exists.

        Args:
            session_id (str): The ID of the session to delete.
            owner (str): The principal requesting the deletion.

        Returns:
            bool: True if deleted, False when missing or not owned by
                ``owner``.
        """
        with self._session_scope() as s:
            conv = s.query(Conversation).filter_by(id=session_id).first()
            if conv and conv.owner == owner:
                s.delete(conv)
                s.commit()
                return True
            return False
```

- [ ] **Step 11: Run the ownership tests — expect pass.**

```bash
uv run pytest tests/test_session_manager_ownership.py -v
```
Expected: PASS — 6 passed.

- [ ] **Step 12: Run the pre-existing session_manager suites — expect failures only in the 3 callers that pass no owner (these are fixed in Task 6 for `api.py`; the unit suites here call the manager directly).**

```bash
uv run pytest tests/test_session_manager.py tests/test_session_manager_validation.py -v
```
Expected: PASS — these files only call `_persist_turn`, `get_session_history(session_id)`?... Check: `test_session_manager.py` calls `get_session_history(session_id)` (line 91, 155) with ONE arg. These will now FAIL with `TypeError: get_session_history() missing 1 required positional argument: 'owner'`. This is expected and is fixed in Step 13 below (these existing tests must be updated to pass an owner — they are part of "every functional change ships test updates").

- [ ] **Step 13: Update the pre-existing tests that call the changed signatures.** Two files call `get_session_history` / `list_sessions` / `delete_session` with the old arity:

In `tests/test_session_manager.py`, change line 91:

```python
    history = session_manager.get_session_history(session_id)
```
to:
```python
    history = session_manager.get_session_history(session_id, owner=None)
```

and line 155:

```python
    history = session_manager.get_session_history("session-social")
```
to:
```python
    history = session_manager.get_session_history("session-social", owner=None)
```

In `tests/test_session_manager_validation.py`, change line 87:

```python
    history = session_manager.get_session_history(session_id)
```
to:
```python
    history = session_manager.get_session_history(session_id, owner=None)
```

and line 126:

```python
    history = session_manager.get_session_history(session_id)
```
to:
```python
    history = session_manager.get_session_history(session_id, owner=None)
```

> Why `owner=None` works here: these legacy fixtures persist turns via `_persist_turn` with `self._owner` left at its default `None`, so the created conversation has `owner IS NULL`. `get_session_history`'s guard is `conv.owner != owner`; with both sides `None` the guard is `None != None` → `False`, so history is returned exactly as before. This keeps the pre-ownership behavioural tests asserting the same thing without weakening the new isolation logic (a real principal string never equals `None`).

- [ ] **Step 14: Re-run all session_manager suites — expect pass.**

```bash
uv run pytest tests/test_session_manager.py tests/test_session_manager_validation.py tests/test_session_manager_ownership.py -v
```
Expected: PASS — all green.

- [ ] **Step 15: Commit.**

```bash
git add docint/core/state/session_manager.py tests/test_session_manager_ownership.py tests/test_session_manager.py tests/test_session_manager_validation.py
git commit -m "$(cat <<'EOF'
feat(state): make session list/history/delete owner-scoped

Thread owner through _load_or_create_convo/start_session/_persist_turn;
new conversations are stamped with the owner. list_sessions filters by
owner; cross-owner history/delete is reported as not-found (empty/False)
so the API can 404 without leaking existence. Existing fixtures updated
to pass owner.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6 — API wiring: `Depends(resolve_principal)` on the three session endpoints

**Files:**
- Modify: `docint/core/api.py`:
  - fastapi import (line 9) — add `Depends`
  - add `from docint.core.auth.principal import resolve_principal` near the existing `docint.core.*` imports (after line 30)
  - `list_sessions` endpoint (lines 1125-1140)
  - `get_session_history` endpoint (lines 1143-1165)
  - `delete_session` endpoint (lines 1168-1186)
- Modify: `tests/test_api.py`:
  - `DummySessionManager.list_sessions` (line 18), `.get_session_history` (line 26), `.delete_session` (line 37) — accept the new `owner` argument so the existing `test_sessions_endpoints` (lines 1646-1669) stays green
  - add two new tests: principal pass-through + 404 on cross-owner, and 401 when no header/fallback

Grounded facts:
- `api.py:9` is exactly `from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile` — `Depends` is **not** present.
- The three endpoints currently wrap the manager call in `try/except Exception` → `HTTPException(status_code=500)` (lines 1138-1140, 1163-1165, 1184-1186). We add explicit not-found mapping: empty history → 404; `delete_session` returning `False` → 404. `list_sessions` has no not-found case (an owner with no sessions legitimately gets `[]`).
- `test_api.py` patches `api_module.rag` with `DummyRAG` whose `ensure_session_manager()` returns a `DummySessionManager` (lines 142-148). The dummy's `list_sessions`/`get_session_history`/`delete_session` (lines 18/26/37) currently take no `owner`; after Task 6 the endpoint passes `principal`, so the dummy signatures must accept it or `test_sessions_endpoints` breaks. `resolve_principal` is a real dependency reading env at call time, so `DummyRAG` need not stub it — but the test must send the header (or set a default identity) or get 401.

- [ ] **Step 1: Add the failing API tests.** Append to `tests/test_api.py` (the file already imports `pytest`, `cast`, `Any`, `TestClient`, `api_module`; no new imports needed):

```python
def test_sessions_endpoints_pass_principal_and_404_on_cross_owner(
    monkeypatch: pytest.MonkeyPatch, client: TestClient
) -> None:
    """Session endpoints forward the resolved principal and 404 cross-owner.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        client (TestClient): The TestClient instance.
    """
    seen: dict[str, Any] = {}

    class OwnerAwareSessions:
        """Session manager stub recording the owner it was called with."""

        def list_sessions(self, owner: str) -> list[dict[str, Any]]:
            seen["list"] = owner
            return [{"id": "s1", "created_at": "2026-01-01", "title": "t"}]

        def get_session_history(
            self, session_id: str, owner: str
        ) -> list[dict[str, Any]]:
            seen["history"] = (session_id, owner)
            # Simulate a cross-owner / missing session: empty history.
            return [] if owner == "bob" else [{"role": "user", "content": "hi"}]

        def delete_session(self, session_id: str, owner: str) -> bool:
            seen["delete"] = (session_id, owner)
            return owner == "alice"

    monkeypatch.setattr(
        api_module.rag, "ensure_session_manager", lambda: OwnerAwareSessions()
    )

    # List forwards the header principal.
    resp = client.get("/sessions/list", headers={"X-Auth-User": "alice"})
    assert resp.status_code == 200
    assert seen["list"] == "alice"
    assert resp.json()["sessions"][0]["id"] == "s1"

    # History for the owner succeeds.
    resp = client.get(
        "/sessions/s1/history", headers={"X-Auth-User": "alice"}
    )
    assert resp.status_code == 200
    assert seen["history"] == ("s1", "alice")
    assert resp.json()["messages"][0]["content"] == "hi"

    # Cross-owner history is 404 (empty -> not found, no existence leak).
    resp = client.get("/sessions/s1/history", headers={"X-Auth-User": "bob"})
    assert resp.status_code == 404

    # Cross-owner delete is 404.
    resp = client.delete("/sessions/s1", headers={"X-Auth-User": "bob"})
    assert resp.status_code == 404
    assert seen["delete"] == ("s1", "bob")

    # Owner delete succeeds.
    resp = client.delete("/sessions/s1", headers={"X-Auth-User": "alice"})
    assert resp.status_code == 200
    assert resp.json()["ok"] is True


def test_sessions_list_401_without_header_or_default(
    monkeypatch: pytest.MonkeyPatch, client: TestClient
) -> None:
    """With no trusted header and no configured default, endpoints 401.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        client (TestClient): The TestClient instance.
    """
    monkeypatch.delenv("DOCINT_AUTH_HEADER", raising=False)
    monkeypatch.delenv("DOCINT_DEFAULT_IDENTITY", raising=False)

    resp = client.get("/sessions/list")
    assert resp.status_code == 401


def test_sessions_list_uses_default_identity_when_no_header(
    monkeypatch: pytest.MonkeyPatch, client: TestClient
) -> None:
    """A configured default identity is used as the owner when no header.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        client (TestClient): The TestClient instance.
    """
    monkeypatch.delenv("DOCINT_AUTH_HEADER", raising=False)
    monkeypatch.setenv("DOCINT_DEFAULT_IDENTITY", "operator")
    seen: dict[str, Any] = {}

    class OwnerAwareSessions:
        """Session manager stub recording the owner it was called with."""

        def list_sessions(self, owner: str) -> list[dict[str, Any]]:
            seen["list"] = owner
            return []

    monkeypatch.setattr(
        api_module.rag, "ensure_session_manager", lambda: OwnerAwareSessions()
    )

    resp = client.get("/sessions/list")
    assert resp.status_code == 200
    assert seen["list"] == "operator"
```

- [ ] **Step 2: Update the existing `DummySessionManager` to accept `owner`.** In `tests/test_api.py`, change `DummySessionManager.list_sessions` (lines 18-24) from:

```python
    def list_sessions(self) -> list[dict[str, Any]]:
        """List all sessions.

        Returns:
            list[dict[str, Any]]: A list of session dictionaries.
        """
        return [{"id": "123", "created_at": "2023-01-01", "title": "Test Chat"}]
```

to:

```python
    def list_sessions(self, owner: str) -> list[dict[str, Any]]:
        """List the caller's sessions.

        Args:
            owner (str): The owning principal.

        Returns:
            list[dict[str, Any]]: A list of session dictionaries.
        """
        return [{"id": "123", "created_at": "2023-01-01", "title": "Test Chat"}]
```

Change `DummySessionManager.get_session_history` (lines 26-35) from:

```python
    def get_session_history(self, session_id: str) -> list[dict[str, Any]]:
        """Get the message history for a session.

        Args:
            session_id (str): The ID of the session.

        Returns:
            list[dict[str, Any]]: A list of message dictionaries.
        """
        return [{"role": "user", "content": "hi"}]
```

to:

```python
    def get_session_history(
        self, session_id: str, owner: str
    ) -> list[dict[str, Any]]:
        """Get the message history for a session.

        Args:
            session_id (str): The ID of the session.
            owner (str): The owning principal.

        Returns:
            list[dict[str, Any]]: A list of message dictionaries.
        """
        return [{"role": "user", "content": "hi"}]
```

Change `DummySessionManager.delete_session` (lines 37-46) from:

```python
    def delete_session(self, session_id: str) -> bool:
        """Delete a session by ID.

        Args:
            session_id (str): The ID of the session.

        Returns:
            bool: True if the session was successfully deleted, False otherwise.
        """
        return True
```

to:

```python
    def delete_session(self, session_id: str, owner: str) -> bool:
        """Delete a session by ID.

        Args:
            session_id (str): The ID of the session.
            owner (str): The owning principal.

        Returns:
            bool: True if the session was successfully deleted, False otherwise.
        """
        return True
```

- [ ] **Step 3: Make the existing `test_sessions_endpoints` send a principal header.** The existing test at `tests/test_api.py:1646-1669` calls the three endpoints with no header; after Task 6 they require a principal. Update its three requests. Change (lines 1652-1669):

```python
    # List
    resp = client.get("/sessions/list")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["sessions"]) == 1
    assert data["sessions"][0]["id"] == "123"

    # History
    resp = client.get("/sessions/123/history")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["messages"]) == 1
    assert data["messages"][0]["content"] == "hi"

    # Delete
    resp = client.delete("/sessions/123")
    assert resp.status_code == 200
    assert resp.json()["ok"] is True
```

to:

```python
    headers = {"X-Auth-User": "tester"}

    # List
    resp = client.get("/sessions/list", headers=headers)
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["sessions"]) == 1
    assert data["sessions"][0]["id"] == "123"

    # History
    resp = client.get("/sessions/123/history", headers=headers)
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["messages"]) == 1
    assert data["messages"][0]["content"] == "hi"

    # Delete
    resp = client.delete("/sessions/123", headers=headers)
    assert resp.status_code == 200
    assert resp.json()["ok"] is True
```

- [ ] **Step 4: Run the API tests — expect failure (endpoints don't take a principal yet).**

```bash
uv run pytest tests/test_api.py::test_sessions_endpoints_pass_principal_and_404_on_cross_owner tests/test_api.py::test_sessions_list_401_without_header_or_default tests/test_api.py::test_sessions_list_uses_default_identity_when_no_header tests/test_api.py::test_sessions_endpoints -v
```
Expected: FAIL — the new pass-through/404/401 tests fail (`TypeError: list_sessions() takes 1 positional argument but 2 were given` once endpoints pass `principal`, or 200 where 401/404 is expected because the endpoint doesn't depend on `resolve_principal`), 0–1 passed.

- [ ] **Step 5: Add the imports to `api.py`.** Change `docint/core/api.py:9` from:

```python
from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
```

to:

```python
from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    UploadFile,
)
```

And add the resolver import immediately after `docint/core/api.py:30` (`from docint.core.retrieval_filters import build_metadata_filters, build_qdrant_filter`), so the import group reads:

```python
from docint.cli import ingest as ingest_module
from docint.core.auth.principal import resolve_principal
from docint.core.rag import RAG, EmptyIngestionError
from docint.core.retrieval_filters import build_metadata_filters, build_qdrant_filter
```

> Note: `from docint.core.auth.principal import resolve_principal` is placed before `from docint.core.rag import ...` to keep this import block alphabetically ordered by module path (`auth` < `rag` < `retrieval_filters`), matching the existing ordering in this block and keeping ruff's import sorter happy.

- [ ] **Step 6: Wire the principal into the three endpoints.** Replace `list_sessions` (lines 1125-1140) with:

```python
@app.get("/sessions/list", response_model=SessionListOut, tags=["Sessions"])
def list_sessions(
    principal: str = Depends(resolve_principal),
) -> dict[str, list[dict]]:
    """List the calling principal's chat sessions.

    Args:
        principal (str): The resolved request principal.

    Returns:
        dict[str, list[dict]]: A dictionary containing the list of sessions.

    Raises:
        HTTPException: If an error occurs while listing sessions.
    """
    try:
        sessions = rag.ensure_session_manager().list_sessions(principal)
        return {"sessions": sessions}
    except Exception as e:
        logger.error("Error listing sessions: {}", e)
        raise HTTPException(status_code=500, detail=str(e))
```

Replace `get_session_history` (lines 1143-1165) with:

```python
@app.get(
    "/sessions/{session_id}/history",
    response_model=SessionHistoryOut,
    tags=["Sessions"],
)
def get_session_history(
    session_id: str, principal: str = Depends(resolve_principal)
) -> dict[str, list[dict]]:
    """Get history for a session owned by the calling principal.

    A session that does not exist or is owned by another principal is
    reported as 404 (no existence leak).

    Args:
        session_id (str): The ID of the session.
        principal (str): The resolved request principal.

    Returns:
        dict[str, list[dict]]: A dictionary containing the session messages.

    Raises:
        HTTPException: 404 when the session is not found for this
            principal; 500 on unexpected errors.
    """
    try:
        messages = rag.ensure_session_manager().get_session_history(
            session_id, principal
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error fetching history: {}", e)
        raise HTTPException(status_code=500, detail=str(e))
    if not messages:
        raise HTTPException(status_code=404, detail="Session not found.")
    return {"messages": messages}
```

Replace `delete_session` (lines 1168-1186) with:

```python
@app.delete("/sessions/{session_id}", tags=["Sessions"])
def delete_session(
    session_id: str, principal: str = Depends(resolve_principal)
) -> dict[str, bool]:
    """Delete a session owned by the calling principal.

    A session that does not exist or is owned by another principal is
    reported as 404 (no existence leak).

    Args:
        session_id (str): The ID of the session to delete.
        principal (str): The resolved request principal.

    Returns:
        dict[str, bool]: A dictionary indicating whether the deletion
            was successful.

    Raises:
        HTTPException: 404 when the session is not found for this
            principal; 500 on unexpected errors.
    """
    try:
        success = rag.ensure_session_manager().delete_session(
            session_id, principal
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error deleting session: {}", e)
        raise HTTPException(status_code=500, detail=str(e))
    if not success:
        raise HTTPException(status_code=404, detail="Session not found.")
    return {"ok": success}
```

> The `except HTTPException: raise` clause is required because `resolve_principal` raises `HTTPException(401)` and the manager call happens inside the `try`; without re-raising, the broad `except Exception` would rewrap the 401 as a 500. (`HTTPException` is a subclass of `Exception`.) `list_sessions` keeps its original shape since it has no 404 path, but it also needs the 401 to propagate — `resolve_principal` is evaluated by FastAPI as a dependency *before* the function body runs, so a 401 there never enters the `try`. For symmetry and safety the history/delete handlers nonetheless guard explicitly because their not-found mapping reads the manager result after the call.

- [ ] **Step 7: Run the API tests — expect pass.**

```bash
uv run pytest tests/test_api.py::test_sessions_endpoints_pass_principal_and_404_on_cross_owner tests/test_api.py::test_sessions_list_401_without_header_or_default tests/test_api.py::test_sessions_list_uses_default_identity_when_no_header tests/test_api.py::test_sessions_endpoints -v
```
Expected: PASS — 4 passed.

- [ ] **Step 8: Run the full `test_api.py` to confirm no regression elsewhere.**

```bash
uv run pytest tests/test_api.py -q
```
Expected: PASS — all tests in the file pass (only the session endpoints changed; `/query`, `/stream_query`, `/collections`, `/ingest` endpoints are untouched).

- [ ] **Step 9: Commit.**

```bash
git add docint/core/api.py tests/test_api.py
git commit -m "$(cat <<'EOF'
feat(api): scope session endpoints to the resolved principal

list/history/delete now Depend(resolve_principal) and forward it to the
session manager. Cross-owner / missing history or delete returns 404
(no existence leak); a missing principal returns 401. Query, collection
and ingest endpoints are unchanged (Phase 2).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7 — Full verification gate

**Files:** none modified unless formatting/lint requires it.

- [ ] **Step 1: Run pre-commit across the repo.**

```bash
uv run pre-commit run --all-files
```
Expected: PASS — `ruff` (check), `ruff-format`, and `mypy` all report success. If `ruff-format` rewrites any of the touched files, re-stage them (`git add <files>`) and include them in the Step 3 commit. If `mypy` flags a type issue in the new code, fix it in place (the new code is written to be `mypy`-clean: `owner: str | None`, `cast(Any, owner)` mirrors the existing `cast(Any, self.rag.qdrant_collection)` on `conversation.collection_name`, and `default_identity: str | None`).

- [ ] **Step 2: Run the full test suite.**

```bash
uv run pytest
```
Expected: PASS — entire suite green, including the four new/updated test files (`tests/test_principal.py`, `tests/test_conversation_owner_migration.py`, `tests/test_session_manager_ownership.py`) and the updated `tests/test_api.py`, `tests/test_session_manager.py`, `tests/test_session_manager_validation.py`.

- [ ] **Step 3: Commit only if pre-commit reformatted files.** If and only if Step 1 modified files:

```bash
git add -u
git commit -m "$(cat <<'EOF'
chore(state): apply ruff-format to session-ownership changes

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

If pre-commit made no changes, skip this commit (do not create an empty commit).

---

## Self-review checklist (for the plan author)

Coverage of the spec's **Phase-1-relevant** requirements (spec sections cited) → task that implements + verifies it:

| # | Spec Phase-1 requirement (section) | Implemented in | Verified by |
|---|------------------------------------|----------------|-------------|
| 1 | Principal config: one dataclass in `env_cfg.py`, header name (`X-Auth-User`) + single configured default identity used as BOTH dev fallback AND backfill owner; loader mirrors file convention (§5 "Principal config", §7) | Task 1 (`PrincipalConfig`, `load_principal_env`) | Task 1 Step 4 (`tests/test_principal.py` loader tests) |
| 2 | `PrincipalResolver`: FastAPI dependency in new `docint/core/auth/principal.py` (+package `__init__.py`); header→value; absent+default→default; absent+none→`HTTPException(401)`; tiny/pure single seam (§5 "PrincipalResolver", §8 error table, §10) | Task 2 (`resolve_principal`) | Task 2 Step 5 (`tests/test_principal.py` resolver tests) |
| 3 | `Conversation.owner = Column(String, nullable=True, index=True)` (§7) | Task 3 (model change) | Task 4 Step 5 (`test_conversation_model_declares_indexed_owner_column`) |
| 4 | Idempotent `_ensure_conversation_owner_column(engine)` mirroring `_ensure_turn_validation_columns`; ALTER+index+`UPDATE ... WHERE owner IS NULL` with configured default; called from `_make_session_maker` (§7, §9 Migration) | Task 4 (helper + wiring) | Task 4 Step 5 (`test_fresh_db_has_owner_column`, `test_legacy_conversations_table_gets_column_and_backfill`, `test_migration_helper_no_op_when_table_missing`, `test_backfill_skipped_when_no_default_identity`) |
| 5 | Ownership scoping in `session_manager.py`: new conversations stamped with `owner`; `list_sessions(owner)` filtered; cross-owner `get_session_history`/`delete_session` treated as not-found (§5 "Ownership enforcement", §6 "List / delete", §9 Isolation) | Task 5 (`_load_or_create_convo`/`start_session`/`_persist_turn`/`list_sessions`/`get_session_history`/`delete_session`) | Task 5 Steps 11 & 14 (`tests/test_session_manager_ownership.py`) |
| 6 | API wiring: add `Depends`; `principal = Depends(resolve_principal)` on the three session endpoints; pass through; cross-owner→404, missing principal→401; query/collection/ingest untouched (§5 "Endpoint threading" scope boundary, §8 error table) | Task 6 (api.py imports + 3 endpoints) | Task 6 Steps 7 & 8 (`tests/test_api.py` pass-through/404/401 + full file) |
| 7 | Tests ship with every functional change; mirror existing style; api integration via existing `TestClient` harness (it exists at `tests/test_api.py:557`) (§9 Testing strategy, project convention) | Tasks 1–6 (each has a failing-test-first step) + existing-test updates in Tasks 5 & 6 | Task 7 Step 2 (`uv run pytest` full suite) |
| 8 | `uv run pre-commit run --all-files` (ruff, ruff-format, mypy) + full pytest pass before completion (§9, §12.6) | Task 7 | Task 7 Steps 1 & 2 |

**Out-of-scope confirmations (must NOT appear in any task):** no `CollectionEngineCache` / `SessionRuntimeCache`; no deletion or rework of `select_collection()`; no changes to `/query`, `/stream_query`, `/collections`, `/ingest`, or any `rag.qdrant_collection` logic; no frontend (`frontend/src/stores/ui.ts`) changes. These are Plans 2 and 3.

**Sequencing invariant:** the suite is green after every commit except the deliberately-paired Task 3↔Task 4 (Task 3's model change is committed *with* Task 4 in Task 4 Step 7, never alone) — this is called out explicitly in Task 3 Step 4.
