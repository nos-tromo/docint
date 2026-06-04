# UI guide

The user interface is a **React single-page app** (Vite + TypeScript) that
talks to the FastAPI backend over JSON and SSE. It lives in `frontend/`,
is built with pnpm, and in production is served by an nginx sidecar that
reverse-proxies API routes to the backend (see [deployment.md](deployment.md)).
In development it runs on the Vite dev server at `http://localhost:5173`,
which proxies API calls to the backend on `:8000`.

> The previous Streamlit UI has been removed. Everything below describes
> the React app under `frontend/src/`.

## Stack

| Concern | Choice |
|---|---|
| Build / dev server | Vite (`pnpm dev` → `:5173`, `pnpm build` → `dist/`) |
| Language | TypeScript (strict) + React |
| Routing | `react-router-dom` v7 (`src/routes/Router.tsx`) |
| Server state | TanStack Query (`src/api/queryClient.ts`) |
| Client state | Zustand (`src/stores/`) |
| Styling | Tailwind CSS (dark theme) |
| Tests | Vitest + Testing Library (happy-dom) |

`package.json` scripts: `dev`, `build` (`tsc -b && vite build`), `preview`,
`test` (`vitest run`), `test:watch`, `lint` (eslint), `typecheck`.

## Entry & layout

- `index.html` → `src/main.tsx` mounts `<App>` into `#root`.
- `src/App.tsx` wraps the tree in the TanStack Query provider and the
  React Router `BrowserRouter`, then renders `src/routes/Router.tsx`.
- `src/layout/Shell.tsx` is the persistent layout (sidebar + main area);
  `src/layout/Sidebar.tsx` holds the navigation, the collection selector,
  and the session list.

## Routes

| Path | Component | Screen |
|---|---|---|
| `/` | `src/routes/Dashboard.tsx` | Dashboard |
| `/chat`, `/chat/:sessionId` | `src/routes/Chat.tsx` | Chat |
| `/ingest` | `src/routes/Ingest.tsx` | Ingest |
| `/analysis` | `src/routes/Analysis.tsx` | Analysis |
| `/inspector` | `src/routes/Inspector.tsx` | Inspector |

### Dashboard (`src/routes/Dashboard.tsx`)

KPI cards (backend status, collection / document / session counts), a
top-entities chart (`src/components/dashboard/TopEntitiesChart.tsx`) with
top-k / min-mention filters and an entity merge-mode toggle, and a recent-
sessions list. Hooks: `useCollections`, `useDocumentsCount`, `useNerStats`,
`useSessions`.

### Chat (`src/routes/Chat.tsx`)

The primary surface. Streams the answer token-by-token from
`POST /stream_query` over SSE, loads/saves session history
(`/chat/:sessionId`), and builds metadata filters (MIME, date range,
hate-speech-only, custom field/operator/value rules) via
`src/components/chat/FilterBuilder.tsx`. Each exchange renders as a
`ChatTurn` with `Citation`s, a `ValidationBanner` for answer/source
mismatches, and an optional `GraphDebugPanel`. Cancellation uses an
`AbortController`.

### Ingest (`src/routes/Ingest.tsx`)

Drag-and-drop upload (`src/components/ingest/Dropzone.tsx`) into a selected
or newly-created collection. Streams the multipart upload plus SSE progress
events (`src/components/ingest/IngestionStatus.tsx`) for per-file and
per-stage status.

### Analysis (`src/routes/Analysis.tsx`)

Three tabs: **NER** (`EntityInspector` — searchable entities → paginated
sources, with a merge-mode toggle), **Hate Speech** (`HateSpeechTable`),
and **Summary** (`SummaryPanel` with coverage diagnostics). Pre-warms the
NER aggregate when opened.

### Inspector (`src/routes/Inspector.tsx`)

A paginated document table (`src/components/inspector/DocumentTable.tsx`)
over the active collection, plus a per-session ZIP export
(`SessionZipButton`).

## State

- **`src/stores/ui.ts`** (`useUiStore`, Zustand) — selected collection,
  current session id, preview modal, and entity merge mode
  (`exact` | `orthographic` | `resolved`). Persisted to `localStorage`
  (session id + merge mode only; the collection is intentionally not
  persisted).
- **`src/stores/chatFilters.ts`** (`useChatFiltersStore`) — query mode,
  retrieval mode, and the metadata-filter builder state; `buildPayload()`
  serialises it for requests. In-memory only.
- Server state (collections, documents, NER, sessions) is owned by
  TanStack Query hooks under `src/hooks/` (`useCollections`, `useNer`,
  `useSessions`, `useDocuments`).

## API layer

All HTTP lives under `src/api/`:

- `client.ts` — `apiGet` / `apiPost` / `apiDelete`, an `ApiError` type, and
  a base URL from `VITE_API_BASE_URL` (defaults to relative, so the dev
  proxy / nginx handles routing).
- `sse.ts`, `upload.ts` — SSE stream parsers (`streamSse`, `streamUpload`)
  for token streaming and multipart upload progress.
- `chat.ts`, `collections.ts`, `sessions.ts`, `ingest.ts`, `analysis.ts` —
  typed endpoint wrappers; `types.ts` holds the shared types.
- `queryClient.ts` — the TanStack Query client (30s stale time, no retry on
  4xx, no refetch on window focus).

Dev proxy targets are declared in `frontend/vite.config.ts`
(`/collections`, `/sessions`, `/sources`, `/query`, `/stream_query`,
`/summarize`, `/ingest`, `/agent` → `http://localhost:8000`); the
production equivalents live in `frontend/nginx/default.conf`.

## Tests

Vitest specs sit next to their subjects under `frontend/src` (for example
`src/api/sse.test.ts`, `src/stores/ui.test.ts`, `src/routes/Chat.test.tsx`,
`src/layout/Sidebar.test.tsx`). Run them with:

```bash
cd frontend
pnpm test          # run once
pnpm test:watch    # watch mode
```

## Adding a screen

1. Add a route component under `src/routes/` and register its path in
   `src/routes/Router.tsx`.
2. Add a navigation entry in `src/layout/Sidebar.tsx`.
3. Put shared widgets under `src/components/<area>/` and data access in a
   `src/hooks/use*.ts` hook backed by an `src/api/*.ts` wrapper.
4. Add a Vitest spec next to the new files.
