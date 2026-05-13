# Replace the Streamlit frontend with a React SPA

Status: draft
Date: 2026-05-13
Owner: nos-tromo

## Why

The current Streamlit UI in `docint/ui/` has three structural problems:

1. **Reactivity model.** Streamlit re-runs the entire script on every interaction. With ~50 keys in `st.session_state` spread across five pages, this is fragile and forces all conditional logic to defend against partial state. Token-streaming responses fight the rerun loop, which produces visible flicker and a stream of `st.rerun()` log lines on every chat tick.
2. **Log noise.** Each rerun emits its own request and a websocket reload ping. During an SSE chat stream this hits the server log dozens of times per second, drowning out genuine API events.
3. **Look and feel.** Streamlit ships a light theme by default. The current `theme.py` only nudges padding and footer styling; there is no real dark mode and no path to one without rebuilding the chrome.

We will replace the Streamlit UI with a single-page application served as static assets by the existing FastAPI backend.

## Goals

- Preserve every feature currently exposed by `docint/ui/` (5 pages, ~15 endpoints, citations, NER, charts, file uploads, SSE, ZIP downloads).
- Native dark theme. No light-mode parity required.
- Fine-grained reactivity: only the components whose data changed re-render; the server log shows real API traffic only.
- Deploy as a single FastAPI container in production. No second web server, no separate Streamlit process.
- Keep the FastAPI backend as the source of truth. Backend changes are limited to CORS and a static-file mount.

## Non-goals

- No authentication added (matches current trust model — the API has none).
- No mobile-first responsive design. Sensible behavior down to ~1024 px wide is enough; the current Streamlit UI is not mobile-optimized either.
- No backend API changes beyond the two listed in [Backend changes](#backend-changes).
- No design system beyond shadcn/ui defaults. We are not building a brand.
- No internationalization. The current UI is English-only.

## Stack

| Concern | Choice | Rationale |
| --- | --- | --- |
| Build / dev server | Vite | Fast HMR, simple config, native ESM. |
| Package manager | pnpm | Fast, disk-efficient, deterministic. The Docker build assumes `pnpm-lock.yaml`; pin via Corepack so contributors do not need a global install. |
| Framework | React 18 + TypeScript | Largest ecosystem for the components we need (tables, charts, drag-drop, virtualization). TypeScript catches most API-shape drift at compile time. |
| Routing | React Router v6 | Sufficient for five top-level routes; no nested-loader requirements. |
| Styling | Tailwind CSS + shadcn/ui (Radix primitives) | Tailwind handles dark mode with one class; shadcn/ui ships dark-aware components without a runtime theme provider. |
| Server state | TanStack Query v5 | Replaces ~50 `st.session_state` keys with proper cache + invalidation. |
| Client state | Zustand | Tiny, ergonomic, used only for UI-only state (modals, draft input, filter-builder rows). |
| Tables | TanStack Table v8 | Sortable / filterable tables for the inspector and analysis pages. |
| Charts | Recharts | Sufficient for bar charts and density plots; simpler than Visx for this scope. |
| Streaming | Native `EventSource` and `fetch` + `ReadableStream` | No dependency. EventSource handles `/agent/chat/stream` and `/summarize/stream`. `fetch` + reader handles `/ingest/upload` (multipart request, SSE response). |
| Forms | React Hook Form + Zod | Filter builder and ingest form benefit from schema validation. |
| File upload | Native input + drag-drop event handlers | A library like react-dropzone is not necessary for the small surface. |

## Architecture

```
Browser (SPA, static assets)
    │
    │  HTTP / SSE
    ▼
FastAPI (docint/core/api.py)
    ├── /api/...                  ← existing endpoints, unchanged
    ├── /                         ← StaticFiles(html=True) → frontend/dist/
    └── (CORS allowlist gains :5173 for dev)
```

There is no new service. In production the SPA is served by FastAPI from `frontend/dist/`. In development Vite runs on `:5173` and proxies `/api` to FastAPI on `:8000`.

### State boundaries

- **Server state — TanStack Query.** Cached and invalidated on the relevant mutations.
  - `useCollections()` — `GET /collections/list`. Invalidated by collection select, delete, and ingest completion.
  - `useDocuments(collection)` — `GET /collections/documents`.
  - `useSessions()` — `GET /sessions/list`. Invalidated by session create, delete, and chat completion.
  - `useSessionHistory(sessionId)` — `GET /sessions/{id}/history`.
  - `useNer(collection)`, `useNerStats(collection, params)`, `useNerSearch(...)` — `GET /collections/ner*`.
  - `useHateSpeech(collection)` — `GET /collections/hate-speech`.
  - `useIeStats(collection)` — `GET /collections/{collection}/ie-stats`.

- **Client state — Zustand stores, persisted to `localStorage` where indicated.**
  - `useUiStore`: `selectedCollection` (persisted), `currentSessionId` (persisted), `chatInputDraft`, `previewModal`.
  - `useChatFiltersStore`: query mode, retrieval mode, scope, MIME pattern, date range, hate-speech toggle, custom rule rows. Not persisted; matches current Streamlit behavior of resetting filters per session.

- **Streaming state — component-local `useReducer`.** In-flight chat tokens, ingest progress events, summary tokens. Not stored in any global cache; the final settled object is written into TanStack Query as appropriate (e.g., the chat turn is appended to the session-history cache).

### Streaming

The backend already exposes SSE on `/agent/chat/stream`, `/stream_query`, `/summarize/stream`, and `/ingest/upload`. The frontend consumes them with two helpers:

- `streamSse(url, body)` — wraps `EventSource` (or `fetch` + reader where a request body is required, since `EventSource` is GET-only). Yields `{event, data}` objects, finalizes on `done`.
- `streamUpload(url, formData)` — `fetch` POST with `formData`, reads the response body as a `ReadableStream`, parses SSE frames, yields events. Cancels via `AbortController`.

Both helpers feed component reducers. No global state mutation happens until the stream completes.

### Routing

| Route | Page | Replaces |
| --- | --- | --- |
| `/` | Dashboard | `docint/ui/dashboard.py` |
| `/chat` | Chat | `docint/ui/chat.py` |
| `/chat/:sessionId` | Chat (specific session) | (new — Streamlit conflated session selection with sidebar state) |
| `/ingest` | Ingest | `docint/ui/ingest.py` |
| `/analysis` | Analysis | `docint/ui/analysis.py` |
| `/inspector` | Inspector | `docint/ui/inspector.py` |

The sidebar (collections + sessions + nav) is rendered by the layout shell and is shared across routes.

### Component map

```
src/
  main.tsx                    Vite entry
  App.tsx                     Router + QueryClientProvider + theme
  routes/
    Dashboard.tsx
    Chat.tsx
    Ingest.tsx
    Analysis.tsx
    Inspector.tsx
  layout/
    Shell.tsx                 Sidebar + main content slot
    Sidebar.tsx               Collection select, session list, nav
  components/
    chat/
      MessageList.tsx
      StreamingBubble.tsx
      Citation.tsx            Source preview, score, NER chips
      FilterBuilder.tsx       Custom rule rows
      ValidationBanner.tsx
      GraphDebugPanel.tsx
    ingest/
      Dropzone.tsx
      EventTimeline.tsx       Stage icons, per-file rows
    analysis/
      NerTable.tsx
      HateSpeechTable.tsx
      SummaryPanel.tsx
      CoverageBanner.tsx
    inspector/
      DocumentTable.tsx       TanStack Table
      DocumentDetail.tsx
      SessionZipButton.tsx
    common/
      KpiCard.tsx
      ScoreBadge.tsx
      MetadataChips.tsx
      ConfirmPopover.tsx      Delete confirmations
  api/
    client.ts                 fetch wrapper, base URL from VITE_API_BASE_URL
    collections.ts
    sessions.ts
    chat.ts                   includes streamSse helper
    ingest.ts                 includes streamUpload helper
    analysis.ts
    types.ts                  hand-written or generated from openapi
  stores/
    ui.ts                     Zustand: selected collection, current session
    chatFilters.ts
  hooks/
    useCollections.ts
    useSessions.ts
    useSessionHistory.ts
    useNer.ts
    useDocuments.ts
  lib/
    formatScore.ts
    sourceLabel.ts
    aggregateNer.ts           Mirrors docint/ui/components.py helpers
    referencedSources.ts
    csvExport.ts
    zipExport.ts              JSZip — builds session-source archive client-side
  styles/
    globals.css               Tailwind directives + dark base
```

### TypeScript types

Two ways to source them, in order of preference:

1. **Generate from FastAPI OpenAPI** with `openapi-typescript` (one-shot at build time). Adds a lightweight `pnpm gen:types` script that hits `http://localhost:8000/openapi.json` and writes `src/api/types.gen.ts`. Keeps types in lockstep with Pydantic models.
2. **Hand-written `types.ts`** as a fallback if the generated output is awkward for the streaming endpoints (which OpenAPI represents poorly).

Start with hand-written for the streaming envelopes and generated for the rest. Convert opportunistically.

## Backend changes

Two changes only. Both live in `docint/core/api.py`.

1. **CORS allowlist** gains `http://localhost:5173` and `http://127.0.0.1:5173` (configurable via the existing `CORS_ALLOWED_ORIGINS` env var).
2. **Static mount** at the end of route registration:

   ```python
   from pathlib import Path
   from fastapi.staticfiles import StaticFiles

   FRONTEND_DIST = Path(__file__).resolve().parents[2] / "frontend" / "dist"
   if FRONTEND_DIST.is_dir():
       app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")
   ```

   The `if FRONTEND_DIST.is_dir()` guard keeps the API runnable without the SPA built (matters for tests and CI).

No endpoint contracts change. No new endpoints are added.

## Deployment

### Local development

Two terminals:

```bash
uv run uvicorn docint.core.api:app --reload      # :8000
cd frontend && pnpm dev                          # :5173
```

Vite config:

```ts
server: {
  port: 5173,
  proxy: { '/api': 'http://localhost:8000', '/sources': 'http://localhost:8000' }
}
```

(The frontend hits `/collections/list` etc. via `VITE_API_BASE_URL`, which defaults to `''` so requests go to the same origin and the Vite proxy forwards them.)

### Docker

The Streamlit service in `docker-compose.yml` is removed. A new `frontend-builder` build stage compiles the SPA into the FastAPI image:

```dockerfile
FROM node:20-alpine AS frontend-builder
WORKDIR /app
COPY frontend/package.json frontend/pnpm-lock.yaml ./
RUN corepack enable && pnpm install --frozen-lockfile
COPY frontend/ .
RUN pnpm build

FROM python:3.11-slim AS runtime
# ... existing python setup ...
COPY --from=frontend-builder /app/dist /opt/docint/frontend/dist
```

The runtime image now exposes the UI at the same port as the API (`:8000`). `:8501` is freed.

## Migration

1. Land the SPA scaffold in `frontend/` with the dashboard route built (smallest surface).
2. Build chat next — it is the highest-value page and exercises streaming, citations, and validation.
3. Build ingest, analysis, inspector in any order.
4. Delete `docint/ui/`, drop `streamlit` from `pyproject.toml`, drop the Streamlit service from compose. **Keep** the `docint` console script in `pyproject.toml` but repoint it at `uvicorn docint.core.api:app` so existing muscle memory (`uv run docint`, the CLAUDE.md instructions, ops scripts) keeps working.

The Streamlit UI keeps working throughout the migration; the new SPA runs side-by-side at `:5173` (dev) until parity is reached.

## Functionality preservation matrix

| Streamlit feature | New location | Notes |
| --- | --- | --- |
| Dashboard KPI cards | `routes/Dashboard.tsx` → `KpiCard` | Backend `health` is implicit — derive from a successful `/collections/list`. |
| Top-entities chart | `Dashboard` → Recharts `BarChart` | Reads `/collections/{c}/ie-stats`. |
| Recent sessions list | `Dashboard` + sidebar | Click navigates to `/chat/:sessionId`. |
| Chat: streaming bubbles | `Chat` + `StreamingBubble` | EventSource-backed; tokens append, final event commits to `useSessionHistory` cache. |
| Chat: query mode toggle | `Chat` controls bar | answer / entity_occurrence / entity_occurrence_multi. |
| Chat: retrieval mode | `Chat` controls bar | stateless / session. |
| Chat: filter builder | `FilterBuilder` | MIME, date range, custom rows; React Hook Form + Zod. |
| Chat: hate-speech toggle | `Chat` controls bar | Wired into the same metadata-filter payload. |
| Chat: source citations | `Citation` | Score, file label, inline preview via `/sources/preview`. |
| Chat: validation banner | `ValidationBanner` | Reads validation_* fields from the final event. |
| Chat: GraphRAG debug | `GraphDebugPanel` | Collapsible. |
| Chat: entity disambiguation | `EntityCandidatesPanel` | Renders `entity_match_candidates`/`entity_match_groups`. |
| Ingest: drag-drop upload | `Dropzone` | Multi-file. |
| Ingest: progress timeline | `EventTimeline` | Renders `upload_progress`, `file_saved`, `ingestion_progress`, `ingestion_complete`. |
| Ingest: auto-select collection | Side-effect in `Ingest` | Calls `/collections/select` then invalidates `useCollections`. |
| Analysis: NER tab | `NerTable` | Sortable, filterable, CSV export. |
| Analysis: hate-speech tab | `HateSpeechTable` | CSV export. |
| Analysis: streaming summary | `SummaryPanel` | EventSource on `/summarize/stream`. |
| Analysis: coverage diagnostics | `CoverageBanner` | |
| Inspector: document table | `DocumentTable` | TanStack Table; CSV export. |
| Inspector: per-doc detail | `DocumentDetail` | |
| Inspector: session ZIP archive | `SessionZipButton` | JSZip client-side; pulls each source via `/sources/preview`. |
| Sidebar: collection select / delete | `Sidebar` | Confirm popover for delete. |
| Sidebar: session list / select / delete | `Sidebar` | |
| Sidebar: new chat | `Sidebar` | |
| Theme | Tailwind dark mode (always on) | No theme toggle. |

## Risks

- **Build chain.** Adds a Node build step. The runtime image grows by ~30-50 MB after the multi-stage build (only the `dist/` output ships). Acceptable.
- **Type drift.** Pydantic models can change without breaking the API contract visibly. Mitigation: regenerate `types.gen.ts` in CI and fail the build on diffs.
- **SSE behind proxies.** Some corporate proxies buffer SSE. Backend already sets `X-Accel-Buffering: no` — keep it.
- **`/sources/preview` and ZIP download.** Building the session ZIP on the client requires N requests for N sources. If a session references many sources this is slow. Acceptable for now; the current Streamlit ZIP builder is also synchronous and the typical N is small.
- **Citation rendering.** A handful of helpers in `docint/ui/components.py` (`aggregate_ner`, `entity_density_by_document`, etc.) are pure data shaping. Port them to `lib/` in TS — small but tedious.

## Open questions

None. All decisions are made; the spec is implementable as written.

## Out of scope, explicitly

- Auth, RBAC, multi-tenancy.
- Light theme, theme toggle.
- Mobile layouts (works at desktop and laptop widths).
- Real-time collaboration on a session.
- Replacing Recharts with a heavier viz library.
- Replacing TanStack Query with a normalized cache (Redux Toolkit Query, Apollo).
- Server-side rendering or hydration.
