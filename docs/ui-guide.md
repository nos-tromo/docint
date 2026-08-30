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
| Styling | Tailwind CSS v4 + `@infra/ui` semantic tokens (light default, OS-preference/`AppHeader`-toggled dark) |
| Tests | Vitest + Testing Library (happy-dom) |

`package.json` scripts: `dev`, `build` (`tsc -b && vite build`), `preview`,
`test` (`vitest run`), `test:watch`, `lint` (eslint), `typecheck`.

## Entry & layout

- `index.html` → `src/main.tsx` mounts `<App>` into `#root`.
- `src/App.tsx` wraps the tree in the TanStack Query provider and the
  React Router `BrowserRouter`, then renders `src/routes/Router.tsx`.
- `src/layout/Shell.tsx` is the persistent layout: `@infra/ui`'s `AppHeader`
  (federation portal link + light/dark/system theme toggle) on top, then a
  sidebar + main-area row below; `src/layout/Sidebar.tsx` holds the
  navigation, the collection selector, and the session list.

## Routes

| Path | Component | Screen |
|---|---|---|
| `/` | `src/routes/Dashboard.tsx` | Dashboard |
| `/chat`, `/chat/:sessionId` | `src/routes/Chat.tsx` | Chat |
| `/ingest` | `src/routes/Ingest.tsx` | Ingest |
| `/analysis` | `src/routes/Analysis.tsx` | Analysis |
| `/inspector` | `src/routes/Inspector.tsx` | Inspector |
| `/report` | `src/routes/Report.tsx` | Report |

### Dashboard (`src/routes/Dashboard.tsx`)

KPI cards (backend status, collection / document / session counts), a
top-entities chart (`src/components/dashboard/TopEntitiesChart.tsx`) with
top-k / min-mention filters, and a recent-sessions list. Hooks:
`useCollections`, `useDocumentsCount`, `useNerStats`, `useSessions`.

### Chat (`src/routes/Chat.tsx`)

The primary surface. Streams the answer token-by-token from
`POST /stream_query` over SSE, loads/saves session history
(`/chat/:sessionId`), and builds metadata filters (MIME, date range,
hate-speech-only, custom field/operator/value rules) via
`src/components/chat/FilterBuilder.tsx`. Each exchange renders as a
`ChatTurn` with `Citation`s, a `ValidationBanner` for answer/source
mismatches, and an optional `GraphDebugPanel`. Cancellation uses an
`AbortController`.

The metadata-filter disclosure and the retrieval-mode toggle hold the right
edge of the Chat header row (`src/components/chat/ChatControls.tsx`): they
govern what any answer retrieves against, so they belong to the chat rather
than to the search panel, which owns only the query field and the hits. Each
hit is a tile that scopes the chat when clicked, and the panel's collapsed
rail is a bare toggle that tints while a scope is live. Icon-only controls draw from
`src/components/common/icons.tsx`; never a text glyph, whose shape is
whatever font the viewer's browser happens to pick.

A **Search in** picker under the query field chooses what the keywords
match: Text (the chunk body, the default), Author, Network or UUID. The
result is always chunks, so "everything this author wrote" is an ordinary
search whose tiles scope the chat like any other. Filtering by filename is
not a search — use the chat's metadata filters for that.

**Author** deliberately spans everything that names a person — display name,
vanity handle, numeric id, and the copies of all three that an image or
transcript inherits from the posting it came from. An investigator holding
any one of those identifiers can paste it without first working out which
kind it is. Names match on prefixes (`krieg` finds `Krieger`); ids match
exactly, because they are numbers rather than prose.

**UUID** is the sole identifier of a single posting artifact. Paste one and
the hits are that posting plus every image, keyframe and transcript segment
derived from it; dashed or undashed both work. It matches exactly — a uuid
is never a prefix of anything.

The CSV export beside the counts honours the field, and stays available with
a blank query, where it exports the whole filtered collection.

### Ingest (`src/routes/Ingest.tsx`)

Drag-and-drop upload (`src/components/ingest/Dropzone.tsx`) into a selected
or newly-created collection. Streams the multipart upload plus SSE progress
events (`src/components/ingest/IngestionStatus.tsx`) for per-file and
per-stage status. The entity-extraction and hate-speech enrichment options
are `@infra/ui` `ToggleButton`s (lit = on), seeded once per mount from
`GET /config/ingest-defaults`.

### Analysis (`src/routes/Analysis.tsx`)

Three tabs: **NER**, **Hate Speech** (`HateSpeechTable`), and **Summary**
(`SummaryPanel`). Pre-warms the NER aggregate when opened.

The **Summary** tab reads before it builds. On open — and whenever the
selected collection changes — it probes `GET /summarize`, which only ever
reads: a stored summary renders straight away beside a **Refresh** icon,
and `204` (nothing stored) leaves a labelled **Create** button instead. A
build is minutes of map-reduce, so it starts only when an operator presses
one of those two, never because a tab was opened.

Creating and refreshing go through `POST /summarize`, which answers `202`
with a `job_id`; the panel follows it on the shared ingest-job SSE stream
and shows a progress bar (mapped units / total units) while the tree
summarizer runs, then renders the result with its coverage diagnostics once
`summary_completed` arrives. A refresh leaves the previous summary on
screen throughout — blanking the panel for the length of a rebuild would
take away what the operator already had.

The **NER** tab opens with a **Table / Graph** view toggle (only one is shown
at a time):

- **Table** — an `EntitySelect` category + entity picker. The category filter
  re-filters the entity dropdown and pre-selects that category's top entity.
- **Graph** — an interactive, force-directed entity graph rendered by the
  shared `@infra/ui` `ForceGraph` primitive (the same component chorus's
  reactive graph exploration uses; see chorus ADR 0016), fed by
  `GET /collections/ner/graph`. `EntityGraph` (`src/components/analysis/
  EntityGraph.tsx`) is a thin adapter: `src/lib/entityGraphElements.ts` maps
  the NER graph payload onto `ForceGraph`'s `{nodes, edges}` shape and bridges
  docint's single-entity `${text}::${type}` selection key onto the
  primitive's node-id selection API. Nodes are draggable (with collision),
  the canvas zooms (wheel) and pans (background drag), and clicking a node
  selects that entity. `ForceGraph` itself renders the controls row (**Min
  edges** stepper, **Edge length** slider, **Zoom**, **Fit**, **Reset**) and
  the maximize overlay, and adds native `<title>` tooltips, fit-to-view, and
  marquee/shift-click multi-select (docint's single-entity findings panel
  just follows the most recently selected node). A small node-count control
  (`GraphTopKControl`, parent-owned state) sits above the graph. Selecting one
  or more nodes shows a **Remove node**/**Remove {n} nodes** button
  (Backspace/Delete also works); removal is view-only local state in
  `EntityGraph` — the underlying NER data is untouched, and removed nodes
  reappear on the next fetch or top-K change. **Export
  JSON**/**Export GraphML**/**Export HTML** buttons (shown once the graph is
  non-empty) serialize the current view state — respecting node removals —
  via `@infra/ui`'s `toGraphJson`/`toGraphML`/`toGraphHtml` + `downloadText`;
  this is client-side only, with no backend round-trip.

The type legend is computed from the full node set, while the min-edges filter only affects what is drawn—so legend entries can name types whose nodes are currently filtered out; this is a deliberate divergence from the previous local renderer, accepted because filter state is component-internal.

Either selection surface drives the shared **findings table**
(`EntityFindingsTable` → `EntityFinding` rows): one chunk per row, with all
locator/reference metadata flattened into a single Metadata column and an
inline "Add to report" control. `HateSpeechTable` follows the same one-row-per-
finding table shape.

### Inspector (`src/routes/Inspector.tsx`)

A paginated document table (`src/components/inspector/DocumentTable.tsx`)
over the active collection, plus a per-session ZIP export
(`SessionZipButton`). The summary strip above the table
(`DocumentSummary`) reads collection-wide aggregates from
`GET /collections/documents/summary` — so the document/node totals and the
file-type / entity-type breakdown stay accurate regardless of how many pages
the table has lazily loaded (the counts are not derived from the loaded rows).

### Report (`src/routes/Report.tsx`)

The Report Builder surface. Lists the caller's reports and, for the active one,
shows the picked artifacts grouped by type with per-item notes, reordering, and
removal, plus the five export formats. Artifacts are added from elsewhere in
the app: an **"+ Report"** control sits on every chat answer, entity finding,
and hate-speech finding. Switching the active collection releases the active
report, so a report and its evidence always describe the same collection.

The full workflow — snapshot semantics, frozen image evidence, and what each
export contains — is in [reports.md](reports.md).

## Localization

The single env var `RESPONSE_LANGUAGE` (values `en` | `de`, default `en`)
controls the *entire* app — both backend and SPA chrome — with one knob:

- **Backend**: prompts, `ui_strings` in reports, and export captions (PDF
  headers, CSV column names). Unknown values silently fall back to `en`.
- **SPA**: The React interface — buttons, labels, navigation, form hints, and
  error messages — flows from a typed locale catalog (`frontend/src/i18n/`)
  with `en` and `de` as canonical languages, maintained in parity with each new
  feature. The `useT()` hook and `LanguageProvider` read `RESPONSE_LANGUAGE`
  from `GET /config`.

Keys are dot-namespaced by screen (`common.*`, `chat.*`, `ingest.*`, …) and
interpolate via `{name}` placeholders. JSON output schemas, intent labels,
enum values, and API field names stay English in every locale — they are
protocol, not prose. See
[configuration.md](configuration.md#response-language--languageconfig).

### On-demand translation of source content

Chat source citations, entity findings, and hate-speech findings each show a
hover/focus-revealed **Translate** control. Clicking it fetches an on-demand
machine translation into the operator's active locale (`RESPONSE_LANGUAGE`)
and swaps it in for the original in place — a "Translation" label marks the
swapped view, and a second click ("Show original") brings the original back;
the original is always one click away, never discarded. Long chunks stay
clamped to four lines behind a "Show more" toggle in either view. This is a
display-time overlay only: nothing ingested or stored is ever translated.

Translating a finding before adding it to a report carries that translation
into the report's snapshot — see [reports.md](reports.md).

Translation reuses the same chat model as the rest of docint over the same
router endpoint — there is no dedicated translation runtime and no
`TRANSLATE_API_BASE` to configure. Set `TRANSLATE_MODEL` in `.env` to use a
different model than chat's `TEXT_MODEL`; it defaults to `TEXT_MODEL`.
Airgap-safe: no new container and no new network egress target. A
target-language override (translating into a language other than the active
locale) is not yet supported.

## State

- **`src/stores/ui.ts`** (`useUiStore`, Zustand) — selected collection,
  current session id, and preview modal. There is no merge-mode control or
  store field: the UI always requests merged (resolved) entities, while the
  API's `entity_merge_mode` parameter remains for the backend contract.
  Persisted to `localStorage`: the active collection + owner, current
  session id, and the graph node-count override.
- **`src/stores/chatFilters.ts`** (`useChatFiltersStore`) — query mode,
  retrieval mode, the reasoning toggle (off by default; sent as `reasoning`
  on every chat request), and the metadata-filter builder state;
  `buildPayload()` serialises the filters for requests. Persisted to
  `localStorage`.
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
