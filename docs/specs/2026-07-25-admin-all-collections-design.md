# Admin visibility of all users' collections — design

Date: 2026-07-25
Status: approved for planning
Repos touched: `edge-plane` (trusted-header contract), `docint` (backend + frontend)

## Goal

Users in the Authelia `admins` group get **owner-equivalent access** to every
user's collections in docint: list, open, query/chat over, ingest into, and
delete. Non-admin behavior is unchanged, byte-for-byte, at both the API and
the UI.

Decisions already made with the user:

1. **Scope: full access** — not read-only, not list-only.
2. **UI: owner-grouped sidebar** — admins see their own collections as today,
   plus other users' collections grouped by owner in the same sidebar. No
   separate admin page.
3. **Source of truth: Authelia groups** — the same `admins` group that
   already gates `/grafana` at the edge. No docint-local admin list.

## Why this shape

docint funnels every collection-scoped operation (~40 endpoints) through
three chokepoints: `resolve_principal`
(`docint/core/auth/principal.py` — the seam its docstring names as "the
single seam the deferred auth track replaces"), `CollectionOwnerManager`
(`docint/core/state/collection_owner_manager.py` — `list_for`/`resolve`),
and the API gate helpers (`_require_owned_collection` /
`_resolve_request_collection` / `_scoped_collection` in
`docint/core/api.py`). Widening those chokepoints gives admins uniform
cross-owner access with no endpoint-by-endpoint work.

Alternatives rejected: a parallel `/admin/*` router (duplicates ~40
endpoints, drifts), and impersonation/"act as" (one user at a time —
conflicts with the combined sidebar — and muddies audit). A docint-local
admin list was rejected by decision 3.

## 1. edge-plane: extend the trusted-header contract

Authelia already returns `Remote-Groups` (comma-separated group names) from
`forward_auth`; the gateway currently drops it.

- `caddy/Caddyfile`
  - `strip_identity` snippet: add `request_header -X-Auth-Groups`.
  - `authed` snippet `copy_headers`: add `Remote-Groups>X-Auth-Groups`.
- `scripts/smoke.sh`: in the authenticated whoami step, assert the upstream
  received `X-Auth-Groups` containing the synthetic test user's group, and
  that a client-forged `X-Auth-Groups: admins` never reaches the upstream
  (same spoof pattern as the existing `X-Auth-User: mallory` check).
- `authelia/users.template.yml`: give the synthetic template user a group so
  the smoke assertion has a value to check (synthetic placeholder only).
- `README.md`: document `X-Auth-Groups` in "The trusted-header contract".

This is a pure widening of the contract: upstreams that ignore
`X-Auth-Groups` are unaffected. It ships first and is inert until docint
consumes it.

## 2. docint: identity layer — `Principal`

- `docint/utils/env_cfg.py` — `PrincipalConfig` gains:
  - `groups_header` from `DOCINT_GROUPS_HEADER` (default `X-Auth-Groups`)
  - `admin_group` from `DOCINT_ADMIN_GROUP` (default `admins`)
  - `default_groups` from `DOCINT_DEFAULT_GROUPS` (dev-only, default unset)
- `docint/core/auth/principal.py` — `resolve_principal` returns a frozen
  `Principal` dataclass:
  - `name: str` (resolved exactly as today, same 401 fail-closed rule)
  - `groups: frozenset[str]` parsed from the comma-separated groups header;
    entries stripped of whitespace; empty entries dropped
  - `is_admin` property: `admin_group in groups`
  - Fail closed: missing header, empty value, or unparseable garbage ⇒
    empty `groups` ⇒ not admin. Never an error.
  - Dev fallback: `DOCINT_DEFAULT_GROUPS` applies only when the groups
    header is absent, mirroring `DOCINT_DEFAULT_IDENTITY` (which stays
    unset in production per the existing deployment rule).
- All `principal: str = Depends(resolve_principal)` sites become
  `principal: Principal`; string uses become `principal.name`. Mechanical;
  pyrefly enforces completeness.

## 3. docint: ownership gates and API

- `CollectionOwnerManager.list_all() -> list[tuple[str, str]]` — every
  `(owner, logical_name)` row, sorted by owner then logical name. `list_for`
  is unchanged. Legacy backfilled rows (bare physical names, owner =
  historical default identity) work automatically: everything keys off the
  `collection_owners` table, never name parsing.
- Effective-owner rule in the gate helpers (`_require_owned_collection`,
  `_resolve_request_collection`, `_scoped_collection`): each accepts an
  optional `owner: str | None`.
  - effective owner = `owner` if `owner` is provided **and**
    `principal.is_admin`, else `principal.name`.
  - A non-admin supplying any `owner` other than themselves gets the
    existing indistinguishable **404** — never a 403; collection existence
    must not leak.
  - An admin supplying an unknown owner or collection: 404 (no extra oracle
    beyond what admin listing already grants).
- Scoped endpoints gain an optional `owner` field (query param for GETs,
  body field where a body exists) that threads into those helpers. No other
  per-endpoint logic.
- `GET /collections/list`:
  - No params → `string[]` exactly as today, for everyone (backward
    compatible; old frontend keeps working).
  - `?all=true` and `principal.is_admin` →
    `{"mine": string[], "others": [{"owner": string, "collections": string[]}]}`
    with `mine` excluded from `others`.
  - `?all=true` and not admin → plain `string[]` (silently ignored, no
    error — consistent with the no-leak posture).
- Admin delete/ingest/session operations flow through the same
  effective-owner rule — full owner-equivalent access. Cascades (session
  delete on collection delete) already key off the physical name and need
  no change.

## 4. docint frontend: owner-grouped sidebar

- Admin detection: the frontend calls `GET /collections/list?all=true` on
  load; an object response (vs. a bare array) means admin and carries the
  grouped data in the same round trip. No new whoami endpoint.
- `frontend/src/api/collections.ts` + `hooks/useCollections.ts`: handle both
  response shapes; expose `{mine, others}` (non-admin: `others` empty).
- `frontend/src/layout/Sidebar.tsx`: "My collections" renders exactly as
  today; below it, one group per foreign owner (header = owner name,
  entries = that owner's collections). Selecting a foreign collection stores
  `{owner, name}` in `useUiStore.selectedCollection`; the API layer attaches
  `owner` to scoped requests when it is set and the owner is not the current
  user. Delete control appears on foreign collections too (full access),
  behind the same confirm affordance as own collections.
- Non-admin UI is unchanged.

## 5. Security posture

- `X-Auth-Groups` is trustworthy for the same single reason `X-Auth-User`
  is: production exposes no host ports except the gateway, and the gateway
  strips the header from clients before injecting its own value — verified
  by the extended smoke spoof test. docint itself never authenticates; it
  trusts the seam fail-closed.
- Admin checks are evaluated per request from the header. No admin state is
  cached in sessions, tokens, or the frontend beyond the current response.
- Cross-owner denial remains 404-indistinguishable everywhere.

## 6. Testing

docint (extend the named guard suites; new tests follow their patterns):

- `tests/test_principal.py`: groups parsing (missing/empty/garbage ⇒ not
  admin), `DOCINT_DEFAULT_GROUPS` dev fallback, frozen dataclass.
- `tests/test_collection_owner_manager.py`: `list_all` ordering, legacy
  rows included.
- `tests/test_api_collections_ownership.py` + `tests/test_multiuser_isolation.py`:
  admin can list/resolve/delete cross-owner; non-admin with an `owner`
  param still 404s; `?all=true` shapes for admin and non-admin; no-param
  response byte-identical to today.
- Frontend vitest: dual response-shape parsing; grouped sidebar rendering;
  `owner` attached to scoped calls only for foreign selections.

edge-plane: the extended `scripts/smoke.sh` runs in the existing CI smoke
job (inject + spoof for `X-Auth-Groups`).

## 7. Rollout

1. edge-plane PR: contract widening + smoke + docs. Inert for all current
   upstreams. Release per the federation workflow (bump `VERSION`).
2. docint PR: backend + frontend + tests. Requires the edge-plane release
   in production for admins to be recognized; before that, admins simply
   see their own collections (fail closed).

Each repo keeps its own PR, CI, and release, per the federation's GitHub
Flow.

## Out of scope

- Admin capabilities in chorus/Nextext (same seam exists; separate feature).
- Any UI for managing users/groups (lives in edge-plane `users.yml`).
- Auditing/logging of admin cross-owner actions beyond existing request logs.
