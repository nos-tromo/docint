# docint — AppShell Adoption Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Adopt `@infra/ui` v0.9.0 in the docint SPA — `AppShell` with the
sidebar as its sidebar slot, `PageHeader` rhythm on titled routes, sign-out
menu, a rebuilt full-width two-column Ingest page, a responsive tile
Dashboard, and `Banner` replacing ad-hoc error text.

**Architecture:** Plan 4 of the federation rollout (design:
`infra-ui/docs/2026-08-04-app-shell-federation-design.md` in the infra-ui
repo). Frontend-only. `Shell.tsx` keeps its data wiring (whoami, version,
ingest SSE stream, PreviewDialog) and delegates chrome to `AppShell`;
`Sidebar.tsx` sheds its own `<aside>` chrome and becomes pure sidebar
content. Chat and Report keep their two-pane `h-full` layouts (they now get
a real height chain from the fixed-chrome shell).

**Tech Stack:** React 19 + Vite + Tailwind v4 + `@infra/ui` (pinned
codeload tarball URL) + vitest.

## Global Constraints

- All frontend commands run inside `frontend/` with pnpm.
- Functionality preserved exactly: collection picker semantics (select /
  delete / stale-selection reconcile), session list + new-chat + delete,
  ingest job lifecycle (queued / interrupted / stream-lost / rerun /
  dismiss), the running-job pulse dot on the Ingest nav item, PreviewDialog,
  theme toggle, user display, i18n en/de parity. Sign-out is the ONE
  addition.
- docint's i18n prefix is camelCase `appHeader.*` (not `appheader.*`).
- Semantic tokens only; no shadows; panels use `Card` (v0.9.0 = opaque
  `bg-muted` tile — KpiCard restyles automatically on the pin bump, which is
  intended). The bespoke `--status-*-fg` tokens stay.
- Known accepted limitation: `AppShell` v0.9.0 forwards no `menuLabel`; the
  user-menu aria-label prefix stays "Account" in both locales.
- Tests stay behavior-based; move assertions with moved elements, never
  delete behavior checks.
- Confidentiality: synthetic data only; no local machine paths committed.
- Working branch: `feature/app-shell` (controller creates it with this plan
  committed; implementers work on it).

---

### Task 1: Bump the `@infra/ui` pin to v0.9.0

**Files:**
- Modify: `frontend/package.json:17`
- Modify: `frontend/pnpm-lock.yaml` (via install)

**Interfaces:**
- Produces: v0.9.0 in `node_modules` (`AppShell`, `PageHeader`, `UserMenu`,
  tile `Card`; `AppHeader` still exported until Task 3). `Card`'s fill went
  `bg-muted/30` → `bg-muted`: `KpiCard` instances restyle opaque — intended,
  it unifies them with the Dashboard's hand-rolled `bg-muted` sections.

- [ ] **Step 1: Bump the pin** — in `frontend/package.json` change

```json
"@infra/ui": "https://codeload.github.com/nos-tromo/infra-ui/tar.gz/v0.8.1",
```

to

```json
"@infra/ui": "https://codeload.github.com/nos-tromo/infra-ui/tar.gz/v0.9.0",
```

- [ ] **Step 2: Install and run the existing gates**

```bash
cd frontend && pnpm install && pnpm lint && pnpm typecheck && pnpm test && pnpm build
```

Expected: all green (docint imports no removed export). If a snapshot/class
assertion on `Card`'s old `bg-muted/30` fails, update that assertion to the
new `bg-muted` — that restyle is the intended v0.9.0 change.

- [ ] **Step 3: Commit**

```bash
git add frontend/package.json frontend/pnpm-lock.yaml
git commit -m "chore(frontend): bump @infra/ui to v0.9.0"
```

---

### Task 2: i18n keys — sign-out + route captions

**Files:**
- Modify: `frontend/src/i18n/en.ts` (or the en catalog file under
  `frontend/src/i18n/` — match where `appHeader.home` lives)
- Modify: the matching de catalog file

**Interfaces:**
- Produces keys later tasks consume: `appHeader.sign_out`, and a
  `<route>.caption` beside each existing `<route>.title`.

- [ ] **Step 1: Add to the en catalog**, next to the existing `appHeader.*`
  keys and each route's `*.title` key respectively:

```ts
  'appHeader.sign_out': 'Sign out',
  'dashboard.caption': 'Corpus and system overview',
  'ingest.caption': 'Upload and index documents',
  'inspector.caption': 'Browse indexed documents',
  'analysis.caption': 'Entities and relations',
  'report.caption': 'Curated findings and exports',
```

- [ ] **Step 2: Add to the de catalog** (same keys, same positions):

```ts
  'appHeader.sign_out': 'Abmelden',
  'dashboard.caption': 'Korpus- und Systemübersicht',
  'ingest.caption': 'Dokumente hochladen und indexieren',
  'inspector.caption': 'Indexierte Dokumente durchsuchen',
  'analysis.caption': 'Entitäten und Relationen',
  'report.caption': 'Kuratierte Ergebnisse und Exporte',
```

(If a route named above has no `*.title` key / no `<h1>` today, still add
the caption key pair — Task 4 skips routes without a title and the parity
test needs both locales either way. If the i18n test enforces key *usage*,
drop exactly the unused caption pair in both locales and note it.)

- [ ] **Step 3: Run the i18n suite**

Run: `cd frontend && pnpm test src/i18n`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add frontend/src/i18n
git commit -m "feat(frontend): i18n keys for AppShell sign-out and route captions"
```

---

### Task 3: Shell swap — `AppShell` with the sidebar slot

**Files:**
- Modify: `frontend/src/layout/Shell.tsx`
- Modify: `frontend/src/layout/Sidebar.tsx:150` (root element only)
- Modify: `frontend/src/routes/Router.tsx` (catch-all route)
- Modify: `frontend/src/layout/Shell.test.tsx`, `frontend/src/layout/Sidebar.test.tsx`
  (assertions that encoded the old chrome)

**Interfaces:**
- Consumes: `AppShell { title, version?, user?, homeLabel?, themeLabels?,
  signOutLabel?, sidebar?, children }`; i18n keys from Task 2.
- Produces: routes render directly inside the canvas `main` (no extra
  wrapper — each route owns its `p-8`); Chat/Report `h-full` layouts now
  resolve against the fixed-height canvas.

- [ ] **Step 1: Rewrite `Shell.tsx`'s returned JSX** (imports: drop
  `AppHeader`, add `AppShell`; hooks unchanged):

```tsx
  return (
    <>
      <AppShell
        title="docint"
        user={whoami?.display_name ?? whoami?.username}
        version={version?.version ? `v${version.version}` : undefined}
        homeLabel={t('appHeader.home')}
        themeLabels={{
          system: t('appHeader.theme_system'),
          light: t('appHeader.theme_light'),
          dark: t('appHeader.theme_dark')
        }}
        signOutLabel={t('appHeader.sign_out')}
        sidebar={<Sidebar />}
      >
        {children}
      </AppShell>
      <PreviewDialog />
    </>
  )
```

Keep the existing explanatory comment about the trusted-header identity on
the `user` prop, and the one about the single global PreviewDialog.

- [ ] **Step 2: Strip the Sidebar's own chrome.** In `Sidebar.tsx` change
  the root element (line 150) from

```tsx
    <aside className="w-72 border-r border-border p-4 flex flex-col gap-4 bg-muted">
```

to

```tsx
    <div className="flex min-h-0 flex-1 flex-col gap-4">
```

(with the matching closing tag) — `AppShell`'s own `<aside>` now provides
width, padding, gap, and scroll; the sidebar sits transparent on the chrome
tint. Also change the nav items' `hover:bg-accent` (line 167) to
`hover:bg-muted` so hover matches the AppShell chrome buttons. Everything
else in the file stays byte-identical.

- [ ] **Step 3: Move the shell out of the router (chorus nesting) and add
  the missing catch-all route.** In `Router.tsx`, remove the `<Shell>`
  wrapper (and its import) so the component renders only `<Routes>…`;
  in `src/App.tsx`, wrap the router usage instead:
  `<Shell><Router /></Shell>` (import `Shell` from `@/layout/Shell`),
  keeping all existing providers around it. Then add `Navigate` to
  `Router.tsx`'s `react-router-dom` import and, as the last route:

```tsx
        <Route path="*" element={<Navigate to="/" replace />} />
```

  NOTE: `Shell` renders no router hooks itself, but it must stay INSIDE
  `BrowserRouter` (the Sidebar uses `NavLink`/`useNavigate`) — place it
  inside the router provider, wrapping `<Router />`.

- [ ] **Step 4: Update the layout tests.** In `Shell.test.tsx` /
  `Sidebar.test.tsx`: identity assertions that found the user as plain text
  now find `getByRole('button', { name: /<the same name>/ })`; assertions
  on the old `<aside>` classes (`bg-muted`, `border-r`, `w-72`) move to
  behavior (nav links present, collection picker present) or assert the new
  AppShell `aside` via `getByRole('complementary')`. Keep every behavioral
  assertion (nav targets, pulse dot, session actions).

- [ ] **Step 5: Run the layout suites, then everything**

Run: `cd frontend && pnpm test src/layout && pnpm lint && pnpm typecheck && pnpm test && pnpm build`
Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add frontend/src/layout frontend/src/routes/Router.tsx
git commit -m "feat(frontend): adopt AppShell chrome with sidebar slot; add catch-all route"
```

---

### Task 4: PageHeader rhythm on titled routes

**Files:**
- Modify: each file under `frontend/src/routes/` that renders a bare
  `<h1 className="text-2xl font-semibold">{t('<route>.title')}</h1>`
  (Dashboard and Ingest do; check Inspector, Analysis, Report, Chat and
  convert exactly those that have such an `<h1>` — do NOT invent titles for
  routes without one; list the skipped routes in your report)

**Interfaces:**
- Consumes: `PageHeader { title, caption? }`; caption keys from Task 2.
- Produces: uniform title+caption rhythm; no route API changes.

- [ ] **Step 1: Swap each bare `<h1>`** for

```tsx
      <PageHeader title={t('<route>.title')} caption={t('<route>.caption')} />
```

adding `PageHeader` to that file's `@infra/ui` import. (In `Ingest.tsx`
this replaces line 208; in `Dashboard.tsx` line 30.) `PageHeader` carries
its own `mb-6`; if the swapped `<h1>` sat in a `space-y-*` stack this
double-spaces slightly — acceptable, do not restructure containers in this
task (Ingest's full rebuild is Task 5).

- [ ] **Step 2: Run the route suites, full gates**

Run: `cd frontend && pnpm test src/routes 2>/dev/null; cd frontend && pnpm lint && pnpm typecheck && pnpm test && pnpm build`
Expected: green (a test querying `heading, level: 1` still passes —
PageHeader renders an `h1`).

- [ ] **Step 3: Commit**

```bash
git add frontend/src/routes
git commit -m "feat(frontend): PageHeader title+caption rhythm on titled routes"
```

---

### Task 5: Ingest rebuilt — full-width two-column, primitives throughout

**Files:**
- Modify: `frontend/src/routes/Ingest.tsx:206-339` (the returned JSX only —
  every hook, memo, effect, and mutation above it stays byte-identical)
- Test: the existing Ingest/ingest-status suites are the spec; adjust only
  DOM-encoding assertions.

**Interfaces:**
- Consumes: `Card`, `Banner`, `Button`, `Input`, `FileList`, `PageHeader`
  from `@infra/ui` (add `Card, Banner, Input` to the import).
- Produces: no state or API changes.

- [ ] **Step 1: Replace the returned JSX** with the two-column canvas
  layout (upload card beside the live status column — kills the
  left-hugging `max-w-3xl` clamp and its dead right gutter):

```tsx
  return (
    <div className="p-8">
      <PageHeader title={t('ingest.title')} caption={t('ingest.caption')} />
      <div className="grid items-start gap-6 lg:grid-cols-[minmax(22rem,28rem)_1fr]">
        <Card className="space-y-4">
          <label className="flex flex-col gap-1 text-sm">
            <span className="text-xs uppercase text-muted-foreground">{t('common.collection')}</span>
            <Input
              list="existing-collections"
              value={run.collection}
              onChange={(e) => run.setCollection(e.target.value)}
              placeholder="my-collection"
            />
            <datalist id="existing-collections">
              {collections?.mine.map((c) => (
                <option key={c} value={c} />
              ))}
            </datalist>
          </label>

          <Dropzone
            disabled={busy}
            onFiles={(v) => {
              setDropError(null)
              run.addFiles(v)
            }}
            onEmpty={() => setDropError(t('ingest.drop_empty'))}
          />

          <FileList
            files={run.files}
            onRemove={(i) => run.removeFile(i)}
            onClear={() => run.clearFiles()}
            labels={{
              files: (n) => t(n === 1 ? 'upload.files_one' : 'upload.files_other', { count: n }),
              clearAll: t('upload.clear_all'),
              remove: t('common.remove')
            }}
          />

          <fieldset className="space-y-1 text-sm" disabled={busy}>
            <label className="flex items-center gap-2">
              <input type="checkbox" checked={run.ner} onChange={(e) => run.setNer(e.target.checked)} />
              {t('ingest.opt_ner')}
            </label>
            <label className="flex items-center gap-2">
              <input
                type="checkbox"
                checked={run.hate}
                onChange={(e) => run.setHate(e.target.checked)}
              />
              {t('ingest.opt_hate')}
            </label>
          </fieldset>

          <Button
            variant="primary"
            onClick={() => void run.start(limitBytes, t)}
            disabled={busy || !run.collection || run.files.length === 0}
          >
            {run.uploading ? t('ingest.busy') : t('ingest.button')}
          </Button>

          {(dropError || run.error) && <Banner variant="danger">{dropError ?? run.error}</Banner>}
        </Card>

        <div className="min-w-0 space-y-4">
          {status.warnings.length > 0 && (
            <ul className="text-sm text-[var(--status-amber-fg)] space-y-1" role="alert">
              {status.warnings.map((w, i) => (
                <li key={i}>{w}</li>
              ))}
            </ul>
          )}

          {queued && !interrupted && (
            <Card className="text-sm text-muted-foreground" role="status">
              {t('ingest.job_queued')}
            </Card>
          )}

          {interrupted && (
            <Card className="text-sm space-y-2" role="status">
              <p className="text-muted-foreground">{t('ingest.job_interrupted')}</p>
              <div className="flex gap-2">
                <Button
                  variant="primary"
                  disabled={rerunMutation.isPending}
                  onClick={() => rerunMutation.mutate()}
                >
                  {t('ingest.job_rerun')}
                </Button>
                <Button variant="secondary" size="sm" onClick={() => run.dismissActive()}>
                  {t('ingest.dismiss')}
                </Button>
              </div>
            </Card>
          )}

          {streamLost && (
            <div className="flex items-center gap-3 text-sm text-[var(--status-amber-fg)]" role="alert">
              <span>{t('ingest.stream_lost')}</span>
              <Button
                variant="secondary"
                size="sm"
                onClick={() => useIngestJobsStore.getState().retryStream()}
              >
                {t('ingest.reconnect')}
              </Button>
            </div>
          )}

          {status.phase !== 'idle' && (
            <div className="space-y-2">
              <IngestionStatus status={status} />
              {(status.phase === 'complete' || status.phase === 'error') && run.activeJobId && (
                <Button
                  variant="secondary"
                  size="sm"
                  onClick={() => dismissMutation.mutate(run.activeJobId!)}
                >
                  {t('ingest.dismiss')}
                </Button>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  )
```

Semantics preserved verbatim: same conditions, same handlers, same i18n
keys, same `role` attributes. Changes are: the `max-w-3xl` clamp and the
nested `max-w-sm` are gone; raw input → `Input` (datalist kept); ad-hoc
`text-[var(--status-red-fg)]` error div → `Banner variant="danger"` inside
the upload card; raw reconnect `<button>` → `Button variant="secondary"
size="sm"`; queued/interrupted boxes → `Card`; warnings use the
theme-tracking `--status-amber-fg` token instead of raw `text-amber-400`.

- [ ] **Step 2: Run the ingest suites, full gates**

Run: `cd frontend && pnpm test src/routes src/components/ingest src/lib 2>/dev/null; cd frontend && pnpm lint && pnpm typecheck && pnpm test && pnpm build`
Expected: green; fix only DOM-encoding assertions (e.g. a query for the old
error div by class — re-target the Banner by `role`/text).

- [ ] **Step 3: Commit**

```bash
git add frontend/src/routes/Ingest.tsx
git commit -m "feat(frontend): rebuild Ingest as full-width two-column canvas with shared primitives"
```

---

### Task 6: Dashboard — responsive tiles

**Files:**
- Modify: `frontend/src/routes/Dashboard.tsx:32,64-109`

**Interfaces:**
- Consumes: `Card` (add to the `@infra/ui` import).
- Produces: no API changes.

- [ ] **Step 1: Make the KPI grid responsive** — line 32:

```tsx
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
```

- [ ] **Step 2: Convert the two hand-rolled sections to `Card`.** The
  Top-entities section (line 64) becomes:

```tsx
      <Card>
        <header className="flex items-center justify-between mb-4">
          <h2 className="text-lg font-semibold text-primary">{t('dashboard.top_entities')}</h2>
          ...unchanged controls...
        </header>
        ...unchanged body...
      </Card>
```

and the Recent-sessions section (line 98):

```tsx
      <Card title={t('dashboard.recent_sessions')}>
        <div className="mt-3">
          ...unchanged body (the empty-state div or the list)...
        </div>
      </Card>
```

(The first section keeps a manual header because of its inline controls —
give its `h2` the tile-title styling `text-lg font-semibold text-primary`
so both cards read identically; the second uses the `title` prop.)

- [ ] **Step 3: Run the suites, full gates, commit**

Run: `cd frontend && pnpm lint && pnpm typecheck && pnpm test && pnpm build`

```bash
git add frontend/src/routes/Dashboard.tsx
git commit -m "feat(frontend): responsive dashboard tiles on the shared Card"
```

---

### Task 7: Release bump + verify

**Files:**
- Modify: `pyproject.toml:3` (`version = "1.1.2"` → `"1.2.0"`)

- [ ] **Step 1: Bump** `[project].version` to `1.2.0`.

- [ ] **Step 2: The full pre-push gate**

```bash
make verify
```

plus `cd frontend && pnpm test` once more. Expected: all green.

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "chore: v1.2.0"
```

- [ ] **Step 4: STOP — do not push.** The controller opens the PR.
