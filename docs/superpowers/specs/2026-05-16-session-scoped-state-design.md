# Session-scoped state, ownership & principal seam — Design

- **Date:** 2026-05-16
- **Status:** Approved (design); pending spec review → implementation plan
- **Branch:** `session-scoped-state` (based off `main` @ `c379e67`, post React-SPA merge)
- **Next track (out of scope here):** collection ownership/visibility; real auth credentials/login UI

## 1. Context & problem

The entire backend runs on a single process-global `RAG` instance (`docint/core/api.py:53`,
`rag = RAG(qdrant_collection="")`). That instance holds, as shared mutable state:

- `rag.qdrant_collection` — the "active collection", mutated by `select_collection()`
  (`docint/core/rag.py:5001`), which is **destructive**: it clears docs/nodes, nulls the
  index and query engine, resets session state, and invalidates NER caches.
- A single `SessionManager` (`rag.sessions`); `rag.session_id`, `rag.chat_engine`,
  `rag.chat_memory` all delegate to that one manager (`rag.py:1854-1902`).

Consequences:

1. **Concurrency correctness bug.** Two concurrent users/requests mutate the same active
   collection and session. User B selecting a collection wipes user A's in-flight index,
   query engine, and conversation. This is data corruption, not just a UX annoyance.
2. **Reload does not resume.** The session id survives in the browser
   (`frontend/src/stores/ui.ts`, zustand `persist` of `currentSessionId`) and history is
   durable in SQLite, but the active collection is process-global and deliberately not
   persisted client-side, so a reloaded session cannot query until the user re-picks a
   collection — and re-picking clobbers everyone else.
3. **No isolation.** `Conversation` (`docint/core/state/conversation.py:11`) has no owner
   column. `list_sessions()` (`docint/core/state/session_manager.py`) returns every
   session globally. The deployment target is a multi-user shared instance where users
   must not see each other's sessions.

**Lever already present:** `Conversation.collection_name`
(`conversation.py:23`, nullable) — a session already records its collection; the backend
just never reads it back at query time. The migration pattern
`_ensure_turn_validation_columns` (`docint/core/state/base.py:31`, called from
`_make_session_maker` at `:64`) is a working precedent for additive column migrations.

## 2. Decisions captured

| # | Decision | Rationale |
|---|----------|-----------|
| D1 | Deployment target is a **multi-user shared instance**; users must not see each other's sessions/collections. | Stated requirement; makes ownership mandatory. |
| D2 | This piece includes **full session ownership + the data migration now**; only login credentials/UI is the deferred auth track. | User chose the largest first step closest to the multi-user end state. |
| D3 | Identity via a **trusted principal header + pluggable resolver** with a configurable dev fallback. The auth track later swaps only the header source. | "Auth at the edge" pattern; no data-model or query-path rework later. |
| D4 | **Collection** ownership/visibility is the **next spec**, not this one. The principal seam is designed so it drops in (`list_collections()` / engine cache already receive the principal). | Keeps each implementation plan focused and reviewable. |
| D5 | Decoupling **Approach A**: collection-keyed engine cache (shared, read-only) + per-session chat state, active collection resolved per request from the session. | Only option both correct under concurrency and memory-efficient for a shared instance. |
| D6 | A conversation is **pinned to one collection for its life**. "Switching collection" = starting a new session, not rebinding. | `Conversation.collection_name` is already a single column; matches today's switch-wipes-context behavior; lets us delete the destructive teardown. |

## 3. Goals / Non-goals

**Goals**

- Remove process-global mutable collection/session state; resolve both per request.
- A reloaded/restored session resumes its collection automatically, with no clobber.
- `Conversation` gains an `owner`; list/history/delete are owner-scoped; existing rows
  are migrated.
- A pluggable principal resolver supplies the owner from a trusted header with a
  configurable dev fallback.
- Concurrent users on different sessions/collections never interfere; concurrent users
  on the same collection share a read-only engine safely.

**Non-goals (deliberately deferred)**

- Collection ownership/visibility filtering — next spec (seam designed here).
- Real auth credentials / login UI — auth track (seam is the principal resolver).
- Physical per-user Qdrant namespacing — rejected as YAGNI.
- Frontend redesign — separate track, sequenced after this; changes here are minimal and
  mechanical (drop the now-defunct global collection-select state; collection pick →
  "create session"). No redesign.

## 4. Architecture

```
request → PrincipalResolver (X-Auth-User → user_id; dev fallback)
        → session_id (from payload/route)
        → Conversation row     (owner == principal? else 404)
        → conversation.collection_name        ── per-session active collection
        → CollectionEngineCache[collection]   ── shared, read-only index/query engine
        → SessionRuntimeCache[session_id]     ── per-session chat engine + memory
```

The split that makes this correct and cheap: **retrieval engines are a pure function of
the collection** (read-only), so they are cached once per collection and shared across
all users reading it. Only the conversational, mutable **chat engine/memory** is keyed
per `session_id`. Nothing user-mutable remains process-global. The destructive
`select_collection()` teardown is deleted, not reworked (D6: collection switch = new
session).

## 5. Components (new and changed)

| Unit | Responsibility | Location |
|------|----------------|----------|
| `PrincipalResolver` | FastAPI dependency: read `X-Auth-User` → `user_id`; configurable dev-fallback identity. Auth track later replaces only the header source. | new module + dependency wired into `docint/core/api.py` |
| Principal config | Header name + dev-fallback identity as an `env_cfg` dataclass (all env config lives in `docint/utils/env_cfg.py` per project convention). | `docint/utils/env_cfg.py` |
| `CollectionEngineCache` | Lazy, lock-guarded `collection → (index, query_engine)`. Shared across users. Bounded LRU eviction; `get()` returns a strong ref held for the request. | new, extracted from `RAG` |
| `SessionRuntimeCache` | Lock-guarded `session_id → (chat_engine, chat_memory)` with idle eviction. Turns still persist to SQLite as today. | new, in/near `SessionManager` |
| `Conversation.owner` | New nullable, indexed column + ALTER migration reusing the `_ensure_*_columns` pattern; idempotent backfill to the configured default principal. | `docint/core/state/conversation.py`, `docint/core/state/base.py` |
| Ownership enforcement | `list_sessions` / `get_session_history` / `delete_session` filter by `owner`; cross-owner access → **404** (no existence leak). New conversations stamp `owner` + `collection_name` at creation (`_load_or_create_convo`). | `docint/core/state/session_manager.py` |
| "Bind session to collection" | `POST /collections` stops calling destructive `select_collection()`; collection choice creates a new owned session pinned to that collection. | `docint/core/api.py`, `docint/core/rag.py` |
| Endpoint threading | Every endpoint that today reads the implicit global `rag.qdrant_collection` (chat at `api.py:508/662`; collection-scoped analysis — summarize/NER/hate-speech around `api.py:901+`) must take the collection **explicitly** (from the session, or an explicit param) and the principal. | `docint/core/api.py` |
| Frontend touchpoints | Drop the now-defunct global collection-select/persist in `frontend/src/stores/ui.ts`; collection pick calls "create session"; `session_id` contract unchanged so the resume flow is unaffected. | `frontend/src/stores/ui.ts` + touched call sites only |

**Risk concentration:** the caches are straightforward. The bulk of effort and regression
risk is mechanically threading `principal` + session-resolved `collection` through every
`api.py` endpoint that currently depends on the global active collection.

## 6. Data flow

- **Create session (collection selection):** picking a collection creates a *new* owned
  session pinned to it — `PrincipalResolver → user_id`; insert
  `Conversation(owner=user_id, collection_name=<chosen>)`; return `session_id`. There is
  no global "select" anymore.
- **Query / stream_query** (`api.py:508` / `662`):
  1. Resolver → `principal`.
  2. `payload.session_id` → load `Conversation`; `owner == principal`? else **404**.
  3. `collection = conversation.collection_name`; `None` → **400** ("session has no
     collection; start a new session").
  4. `engine = CollectionEngineCache.get(collection)` (built lazily under lock, shared).
  5. `chat = SessionRuntimeCache.get(session_id)` (rehydrated from SQLite rolling-summary
     / history exactly as `_get_session_context` does today).
  6. Run chat; persist turn via existing `_persist_turn`. No global mutation.
- **Reload / resume:** browser still has `session_id` (`ui.ts` persist); loads
  owner-checked history; next query resolves the collection from
  `conversation.collection_name` automatically — no re-pick, no clobber. This is the
  concrete fix for the original request.
- **List / delete:** `list_sessions(owner=principal)` returns only the caller's sessions;
  delete/history are owner-checked, cross-owner → **404**.

## 7. Data model & migration

- `Conversation` gains `owner = Column(String, nullable=True, index=True)`.
- Add `_ensure_conversation_owner_column(engine)` next to
  `_ensure_turn_validation_columns` (`base.py:31`), called from `_make_session_maker`
  (`base.py:64`). It `ALTER TABLE conversations ADD COLUMN owner TEXT` when absent, then
  runs an idempotent backfill `UPDATE conversations SET owner = :default WHERE owner IS
  NULL`, where `:default` is the configured default/dev principal from `env_cfg`. Guarded
  by column-existence + `WHERE owner IS NULL`; safe on every startup. Create the `owner`
  index in the same helper.
- `_load_or_create_convo` (`session_manager.py`) takes `owner` and stamps it on new
  conversations; new sessions also persist `collection_name` at creation.
- **Legacy data:** all pre-existing sessions become owned by the configured default
  operator (the same identity the resolver returns pre-auth) — isolation-safe, since a
  future real user is not the default principal. A legacy session with `collection_name
  IS NULL` is genuinely unresumable and is surfaced as a clear error (Section 8), not a
  silent failure.

## 8. Concurrency & error handling

FastAPI runs sync endpoints in a threadpool → real thread concurrency → use
`threading.Lock`.

- **`CollectionEngineCache`:** per-key build lock so two concurrent requests for an
  uncached collection build once; afterwards reads are lock-free on an immutable engine
  reference. `get()` returns a strong ref held for the request duration, so LRU eviction
  cannot pull an engine out from under an in-flight query. **Implementation-validation
  point:** confirm the llama-index query engine is safe for concurrent `.query()`; if
  not, pool per collection instead of a single shared instance.
- **`SessionRuntimeCache`:** per-`session_id` lock so two concurrent requests on the same
  session (double-submit, two tabs) cannot corrupt chat memory. Different users/sessions
  remain fully parallel — this is not a global bottleneck.
- Deleting the destructive `select_collection()` teardown removes the worst hazard
  structurally.

**Error modes → explicit status:**

| Condition | Status |
|-----------|--------|
| Header missing + dev fallback configured | resolve to fallback (dev only) |
| Header missing + no fallback configured | **401** (fail closed) |
| Session not found, or owned by another principal | **404** (no existence leak) |
| Session `collection_name` is NULL | **400** ("start a new session") |
| Session's collection deleted from Qdrant | **409** (frontend prompts new session); reuses existing `list_collections()` existence check (`api.py:136`) |
| Engine build fails (Qdrant down) | **503**; failures are not cached (next request retries) |

## 9. Testing strategy

Per project convention: every functional change ships test updates; pytest;
`conftest.py` stubs external deps; frontend uses Vitest; full
`uv run pre-commit run --all-files` (ruff, ruff format, mypy) + `pnpm test` / `pnpm
build` before completion.

- **Migration:** pre-migration DB fixture (conversations without `owner`) → column added,
  legacy rows backfilled to the default principal, idempotent on re-run.
- **Isolation:** A's sessions invisible to B; B's history/delete on A's session → 404.
  Resolver: header → user; absent + fallback → fallback; absent + no fallback → 401.
- **Core regression test for the original bug:** two threads, different
  sessions/collections, interleaved queries → zero cross-talk. Same uncached collection
  ×2 → engine built once. Same session ×2 → serialized, memory intact.
- **Resolution:** a query resolves the session's bound collection; switching collection =
  new session; the old session still queries its original collection.
- **Frontend (Vitest):** `ui.ts` no longer persists/selects a global collection;
  collection pick → "create session"; `session_id` contract unchanged so the resume flow
  is unaffected. Update only touched call sites.
- **Error paths:** each row in Section 8 → expected status code.

## 10. Scope boundaries & seams

| Deferred | Seam that makes it drop-in |
|----------|----------------------------|
| Collection ownership/visibility (next spec) | `list_collections()` + `CollectionEngineCache` already receive the principal — next spec adds a `WHERE owner` filter, no re-plumbing. |
| Real auth credentials / login UI (auth track) | `PrincipalResolver` — swap the header source for a verified token; data model + query path already final. |
| Physical per-user Qdrant namespacing | Rejected (YAGNI). |
| Frontend redesign | Separate track, sequenced after this; changes here are minimal/mechanical, no redesign. |

## 11. Risks

- **R1 — endpoint threading surface.** The main risk: every endpoint reading the global
  active collection must be converted to explicit collection + principal. Mitigated by an
  exhaustive grep of `rag.qdrant_collection` usage and per-endpoint tests.
- **R2 — llama-index query-engine thread safety.** If concurrent `.query()` on one engine
  is unsafe, fall back to a per-collection engine pool (same cache shape, pooled value).
  Flagged as an implementation-validation point.
- **R3 — `RAG` class extraction.** `rag.py` is very large; extracting request-scoped
  state from process-global state is invasive. Mitigated by keeping the cache modules
  independent and well-bounded, and by leaning on the existing test suite plus the new
  concurrency regression test.

## 12. Success criteria

1. Two concurrent users on different collections/sessions never corrupt each other's
   state (regression test passes).
2. Reload/restore resumes a session and queries its bound collection with no re-pick and
   no clobber.
3. `list_sessions`/history/delete are owner-scoped; cross-owner access returns 404.
4. The owner migration adds the column and backfills legacy rows idempotently.
5. The principal resolver is the single seam for the deferred auth track; the
   collection-ownership next spec needs no re-plumbing of the principal path.
6. `uv run pre-commit run --all-files` and the frontend test/build pass.
