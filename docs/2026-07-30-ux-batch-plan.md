# docint UX Batch Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Four independent frontend UX corrections in docint: prefill the report operator with the signed-in user, reorder the sidebar sections, remove the entity merge-mode control (frontend only), and make dropping a folder on the Ingest dropzone work like the folder picker.

**Architecture:** Frontend-only. Items are independent: the report view gains one field on create (reusing the existing `useWhoami` hook), the sidebar's `NAV` literal is reordered, the merge-mode control and its UI-store slice are deleted while the backend parameter stays (frontend pins `'resolved'`), and `Dropzone`'s drop handler learns to traverse directory entries while preserving each file's relative path. Design: `docs/2026-07-30-ux-batch-design.md`.

**Tech Stack:** React 19 + TypeScript + Vite, zustand (`persist`), TanStack Query, vitest + @testing-library/react (happy-dom), Tailwind v4, `@infra/ui`.

## Global Constraints

- Branch `feature/ux-batch` (already created, holds the design doc). One commit per task. Do not push until the task says so.
- Frontend only — no file under `docint/` (backend) or `tests/` (pytest) changes in any task.
- Backend contract untouched: `entity_merge_mode` stays a valid API parameter; the frontend simply always sends `'resolved'`.
- Every user-visible string goes through the i18n catalog with **both** `frontend/src/i18n/en.ts` and `de.ts` updated (this batch is not expected to add any new string — if you find you need one, add both).
- No real data or local dev-machine filepaths in any commit.
- Per-task validation: `cd frontend && pnpm lint && pnpm test && pnpm build`. Before the final task's commit, also run root `make verify` (`git add` new files first — pre-commit is tracked-only).
- Chores ride along: if `README.md`/`CLAUDE.md` describe the merge-mode control or the dropzone's inputs, correct them in the task that changes the behavior.

---

### Task 1: Report operator prefilled from the signed-in user

**Files:**
- Modify: `frontend/src/api/reports.ts:9` (createReport body type)
- Modify: `frontend/src/routes/Report.tsx` (import + `onCreate`)
- Test: `frontend/src/routes/Report.test.tsx`

**Interfaces:**
- Consumes: `useWhoami()` from `@/hooks/useWhoami` — returns a TanStack Query result whose `data` is `{ username: string; display_name: string | null }` (both loading and error resolve to `data === undefined`).
- Produces: `createReport` accepts an optional `operator` field. Nothing later in this plan depends on it.

- [ ] **Step 1: Write the failing tests**

Append to `frontend/src/routes/Report.test.tsx`. The file's existing `mockFetch()` helper stubs global `fetch` and already returns `{}` for unmatched URLs; these tests capture the POST body and add a `/whoami` response. Add inside the existing `describe('Report view', …)`:

```tsx
  it('creates a report with the signed-in display name as operator', async () => {
    const calls: { url: string; body: unknown }[] = []
    vi.stubGlobal(
      'fetch',
      vi.fn(async (u: string, init?: RequestInit) => {
        const url = String(u)
        if (url.includes('/whoami')) {
          return { ok: true, status: 200, json: async () => ({ username: 'jane.doe', display_name: 'Jane Doe' }) }
        }
        if (url.endsWith('/reports') && init?.method === 'POST') {
          calls.push({ url, body: JSON.parse(String(init.body)) })
          return { ok: true, status: 200, json: async () => ({ ...reportDetail, id: 2 }) }
        }
        if (url.includes('/reports/')) return { ok: true, status: 200, json: async () => reportDetail }
        if (url.endsWith('/reports')) {
          return { ok: true, status: 200, json: async () => ({ reports: [{ ...reportDetail, items: undefined }] }) }
        }
        return { ok: true, status: 200, json: async () => ({}) }
      })
    )
    renderReport()
    fireEvent.click(await screen.findByRole('button', { name: /new report/i }))
    await waitFor(() => expect(calls).toHaveLength(1))
    expect((calls[0].body as { operator?: string }).operator).toBe('Jane Doe')
  })

  it('omits operator when no identity is available', async () => {
    const calls: { body: unknown }[] = []
    vi.stubGlobal(
      'fetch',
      vi.fn(async (u: string, init?: RequestInit) => {
        const url = String(u)
        if (url.includes('/whoami')) return { ok: false, status: 401, json: async () => ({}) }
        if (url.endsWith('/reports') && init?.method === 'POST') {
          calls.push({ body: JSON.parse(String(init.body)) })
          return { ok: true, status: 200, json: async () => ({ ...reportDetail, id: 3 }) }
        }
        if (url.includes('/reports/')) return { ok: true, status: 200, json: async () => reportDetail }
        if (url.endsWith('/reports')) {
          return { ok: true, status: 200, json: async () => ({ reports: [{ ...reportDetail, items: undefined }] }) }
        }
        return { ok: true, status: 200, json: async () => ({}) }
      })
    )
    renderReport()
    fireEvent.click(await screen.findByRole('button', { name: /new report/i }))
    await waitFor(() => expect(calls).toHaveLength(1))
    expect('operator' in (calls[0].body as Record<string, unknown>)).toBe(false)
  })
```

If the "new report" button's accessible name differs, read the existing tests in this file for the actual label/regex and use that — do not invent a label.

- [ ] **Step 2: Run — must fail**

`cd frontend && pnpm vitest run src/routes/Report.test.tsx`
Expected: both new tests fail (the POST body has no `operator`).

- [ ] **Step 3: Widen the API type**

`frontend/src/api/reports.ts`, line 9 — add `operator?: string` to the `createReport` body type:

```ts
export const createReport = (body: {
  title: string
  collection_name?: string | null
  session_id?: string | null
  operator?: string
}) => apiPost<Report>('/reports', body)
```

- [ ] **Step 4: Prefill in `onCreate`**

`frontend/src/routes/Report.tsx` — add the import alongside the other hook imports:

```ts
import { useWhoami } from '@/hooks/useWhoami'
```

Inside `export function Report()`, next to the other hooks:

```ts
  const whoami = useWhoami()
```

Then in `onCreate`, extend the `createReport.mutateAsync` argument (keep the existing `title`/`collection_name` lines as they are):

```ts
      const created = await createReport.mutateAsync({
        title: t('report.untitled_title'),
        collection_name: collection ?? undefined,
        // Create-time default only: the operator field stays editable, and an
        // unknown identity (dev without the gateway, or a failed /whoami) must
        // leave it empty rather than guess.
        operator: whoami.data?.display_name ?? whoami.data?.username
      })
```

`operator: undefined` is dropped by `JSON.stringify`, which is what makes the second test pass — do not add a conditional spread.

- [ ] **Step 5: Run — must pass**

`cd frontend && pnpm vitest run src/routes/Report.test.tsx` → all green (existing tests included).

- [ ] **Step 6: Full frontend check + commit**

```bash
cd frontend && pnpm lint && pnpm test && pnpm build && cd ..
git add frontend/src/api/reports.ts frontend/src/routes/Report.tsx frontend/src/routes/Report.test.tsx
git commit -m "feat: prefill report operator with the signed-in user"
```

---

### Task 2: Sidebar section order

**Files:**
- Modify: `frontend/src/layout/Sidebar.tsx:13-20` (the `NAV` literal)
- Test: `frontend/src/layout/Sidebar.test.tsx` (create if absent)

**Interfaces:**
- Produces: nav order `dashboard, ingest, inspector, chat, analysis, report`. Nothing else depends on it.

- [ ] **Step 1: Write the failing test**

If `frontend/src/layout/Sidebar.test.tsx` exists, add the test below to it, reusing that file's existing render helper and mocks. If it does not exist, create it — `Sidebar` uses `react-router-dom` navigation hooks and the collections/sessions query hooks, so mock those hooks rather than fetch (adjust the mocked module paths only if they differ from the imports at the top of `Sidebar.tsx`):

```tsx
import { render, screen } from '@testing-library/react'
import { describe, it, expect, vi } from 'vitest'
import { MemoryRouter } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { Sidebar } from './Sidebar'

vi.mock('@/hooks/useCollections', () => ({
  useCollections: () => ({ data: { collections: [] }, isLoading: false }),
  useDeleteCollection: () => ({ mutate: vi.fn() }),
  useSelectCollection: () => ({ mutate: vi.fn() })
}))
vi.mock('@/hooks/useSessions', () => ({
  useSessions: () => ({ data: { sessions: [] }, isLoading: false, error: null }),
  useDeleteSession: () => ({ mutate: vi.fn() }),
  sessionsKey: ['sessions']
}))

describe('Sidebar navigation', () => {
  it('orders sections dashboard, ingest, inspector, chat, analysis, report', () => {
    const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    render(
      <QueryClientProvider client={qc}>
        <MemoryRouter>
          <Sidebar />
        </MemoryRouter>
      </QueryClientProvider>
    )
    const hrefs = Array.from(document.querySelectorAll('nav a')).map((a) => a.getAttribute('href'))
    expect(hrefs.slice(0, 6)).toEqual(['/', '/ingest', '/inspector', '/chat', '/analysis', '/report'])
  })
})
```

If the mocked hook shapes don't match what `Sidebar` destructures, read the hooks' real return shapes and mirror them — the test must render the real `Sidebar`, never a stub of it.

- [ ] **Step 2: Run — must fail**

`cd frontend && pnpm vitest run src/layout/Sidebar.test.tsx`
Expected: FAIL — actual order is `/`, `/chat`, `/ingest`, `/analysis`, `/inspector`, `/report`.

- [ ] **Step 3: Reorder `NAV`**

`frontend/src/layout/Sidebar.tsx` — replace the `NAV` literal with:

```ts
const NAV = [
  { to: '/', key: 'nav.dashboard' },
  { to: '/ingest', key: 'nav.ingest' },
  { to: '/inspector', key: 'nav.inspector' },
  { to: '/chat', key: 'nav.chat' },
  { to: '/analysis', key: 'nav.analysis' },
  { to: '/report', key: 'nav.report' }
] as const
```

Nothing else changes: same routes, same i18n keys, Dashboard stays the index route.

- [ ] **Step 4: Run — must pass**, then `cd frontend && pnpm lint && pnpm test && pnpm build`.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/layout/Sidebar.tsx frontend/src/layout/Sidebar.test.tsx
git commit -m "feat: reorder sidebar sections (dashboard, ingest, inspector, chat, analysis, report)"
```

---

### Task 3: Remove the entity merge-mode control (frontend only)

**Files:**
- Delete: `frontend/src/components/common/MergeModeToggle.tsx`, `frontend/src/components/common/MergeModeToggle.test.tsx`
- Modify: `frontend/src/routes/Dashboard.tsx:8,69` (import + mount)
- Modify: `frontend/src/routes/Analysis.tsx:12,33,181,222` (import, store read, mount, prop)
- Modify: `frontend/src/stores/ui.ts` (drop the slice, its setter, `partialize`, and the `migrate` entry)
- Modify: `frontend/src/hooks/useNer.ts:27,34,50,58` (two hooks)
- Modify: `frontend/src/components/analysis/EntityFindingsTable.tsx` (prop + `exportParams`)
- Test: `frontend/src/stores/ui.test.ts`, `frontend/src/routes/Analysis.test.tsx`, `frontend/src/routes/Dashboard.test.tsx`

**Interfaces:**
- Produces: no `entityMergeMode` anywhere in the frontend; `entity_merge_mode: 'resolved'` sent by `useNerGraph`, `useNerSources`, and the findings export. The `EntityMergeMode` type in `@/api/types` and the parameter in `api/collections.ts` REMAIN (backend contract).

- [ ] **Step 1: Write the failing tests**

In `frontend/src/stores/ui.test.ts`, add (adjust the import/access idiom to the file's existing style):

```ts
  it('no longer carries an entity merge mode', () => {
    const state = useUiStore.getState() as Record<string, unknown>
    expect('entityMergeMode' in state).toBe(false)
    expect('setEntityMergeMode' in state).toBe(false)
  })
```

In `frontend/src/routes/Analysis.test.tsx` and `frontend/src/routes/Dashboard.test.tsx`, add one test each, reusing that file's existing render helper:

```tsx
  it('does not render a merge-mode control', async () => {
    // <render via this file's existing helper>
    expect(screen.queryByRole('group', { name: /merge/i })).not.toBeInTheDocument()
  })
```

Add a hook-level assertion in whichever test file covers `useNer` (search for `ner-graph` / `getNerGraph` in `frontend/src`); if no such test exists, assert it via `Analysis.test.tsx`'s fetch stub instead: the request URL/body for the entity graph must carry `entity_merge_mode=resolved`. Write the assertion against the real request the component makes — do not assert on a mock of `useNerGraph` itself.

- [ ] **Step 2: Run — must fail**

`cd frontend && pnpm vitest run src/stores/ui.test.ts src/routes/Analysis.test.tsx src/routes/Dashboard.test.tsx`
Expected: the store test fails (the key exists); the control tests fail (the toggle renders).

- [ ] **Step 3: Delete the component**

```bash
git rm frontend/src/components/common/MergeModeToggle.tsx frontend/src/components/common/MergeModeToggle.test.tsx
```

- [ ] **Step 4: Unmount it**

`frontend/src/routes/Dashboard.tsx`: remove the `import { MergeModeToggle } …` line and the `<MergeModeToggle />` element at ~line 69 (and any wrapper element left empty by the removal — check the surrounding JSX renders sensibly without it).

`frontend/src/routes/Analysis.tsx`: remove the import, the `<MergeModeToggle />` at ~line 181, the `const mergeMode = useUiStore((s) => s.entityMergeMode)` at line 33, and pass the literal in its place at line 222:

```tsx
                entityMergeMode="resolved"
```

- [ ] **Step 5: Strip the store slice**

`frontend/src/stores/ui.ts` — remove from `UiState` the `entityMergeMode` field and the `setEntityMergeMode` signature; remove the `entityMergeMode: 'resolved',` initial value and the `setEntityMergeMode: (mode) => set({ entityMergeMode: mode }),` setter; remove `entityMergeMode: s.entityMergeMode` from `partialize`; and in `migrate`, remove both the `entityMergeMode?: EntityMergeMode` line from the `prior` type and the `entityMergeMode: prior.entityMergeMode ?? 'resolved',` line from the returned object. Bump `version: 3` to `version: 4` so persisted stores drop the dead key on load. If `EntityMergeMode` is now an unused import in this file, remove the import too.

- [ ] **Step 6: Pin `'resolved'` at the request sites**

`frontend/src/hooks/useNer.ts` — in `useNerGraph`: delete `const mergeMode = useUiStore((s) => s.entityMergeMode)`, use `'resolved'` in both the query key and the request:

```ts
    queryKey: ['ner-graph', collection, 'resolved', topKNodes],
```
```ts
        entity_merge_mode: 'resolved',
```

In `useNerSources`: same — delete the `mergeMode` read, use `'resolved'` in the query key (`['ner-sources', collection, entityKey, 'resolved']`) and in the request body. Remove the now-unused `useUiStore` import only if nothing else in the file uses it.

`frontend/src/components/analysis/EntityFindingsTable.tsx` — keep the `entityMergeMode` prop (Analysis now passes the literal) so `exportParams` stays unchanged; do not widen or rename the prop.

- [ ] **Step 7: Run — must pass**

`cd frontend && pnpm test` → green, including the tests from Step 1. Then `pnpm lint && pnpm build`. `pnpm build` failing on an unused import/variable is the expected signal that a leftover reference was missed — fix the reference, don't silence the lint.

- [ ] **Step 8: Chores + commit**

Check for stale prose: `grep -rn "merge mode\|merge_mode\|Orthographic\|orthographic" README.md CLAUDE.md docs/*.md` — if any passage describes the *UI* control, correct it to say the UI always shows resolved entities while the API parameter remains. Leave descriptions of the backend parameter alone.

```bash
git add -A frontend/src README.md CLAUDE.md docs
git commit -m "feat: show merged entities only (drop the merge-mode control)"
```

---

### Task 4: Folder drag-and-drop parity

**Files:**
- Modify: `frontend/src/components/ingest/Dropzone.tsx`
- Test: `frontend/src/components/ingest/Dropzone.test.tsx`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `Dropzone`'s `onFiles(files: File[])` contract unchanged — dropped folders now yield the same `File[]` shape as the folder picker, each file carrying a `webkitRelativePath` equal to its path within the dropped tree.

**Why this is load-bearing:** `frontend/src/api/ingest.ts:44` uploads each file as `f.webkitRelativePath || f.name`. Files from `entry.file()` have an empty `webkitRelativePath`, so without stamping it the dropped tree flattens and two same-named files in different subfolders collide.

- [ ] **Step 1: Write the failing tests**

Append to `frontend/src/components/ingest/Dropzone.test.tsx`. The helpers build a fake entries API — happy-dom has no directory support, so the test constructs `DataTransferItem`-like objects directly. Note `readEntries` is deliberately paginated to catch the truncation bug:

```tsx
type Entry = {
  isFile: boolean
  isDirectory: boolean
  fullPath: string
  file?: (cb: (f: File) => void) => void
  createReader?: () => { readEntries: (cb: (e: Entry[]) => void) => void }
}

function fileEntry(path: string): Entry {
  const f = new File([new Uint8Array([1])], path.split('/').pop() as string)
  return {
    isFile: true,
    isDirectory: false,
    fullPath: path,
    file: (cb) => cb(f)
  }
}

/** Directory whose reader returns `pages` in sequence, then an empty page. */
function dirEntry(path: string, pages: Entry[][]): Entry {
  let i = 0
  return {
    isFile: false,
    isDirectory: true,
    fullPath: path,
    createReader: () => ({
      readEntries: (cb) => cb(i < pages.length ? pages[i++] : [])
    })
  }
}

function dropWith(entries: Entry[]) {
  return {
    preventDefault: () => {},
    dataTransfer: {
      items: entries.map((e) => ({ kind: 'file', webkitGetAsEntry: () => e })),
      files: []
    }
  }
}

describe('Dropzone folder drop', () => {
  it('queues every file in a dropped tree, across readEntries pages, with relative paths', async () => {
    const onFiles = vi.fn()
    render(<Dropzone onFiles={onFiles} />)
    const zone = screen.getByText(/drop/i).closest('div') as HTMLElement

    const tree = dirEntry('/export', [
      [fileEntry('/export/a.pdf'), dirEntry('/export/sub', [[fileEntry('/export/sub/b.pdf')], []])],
      [fileEntry('/export/c.pdf')]
    ])
    fireEvent.drop(zone, dropWith([tree]))

    await waitFor(() => expect(onFiles).toHaveBeenCalled())
    const names = (onFiles.mock.calls[0][0] as File[]).map((f) => f.webkitRelativePath)
    expect(names.sort()).toEqual(['export/a.pdf', 'export/c.pdf', 'export/sub/b.pdf'])
  })

  it('still queues plainly dropped files', async () => {
    const onFiles = vi.fn()
    render(<Dropzone onFiles={onFiles} />)
    const zone = screen.getByText(/drop/i).closest('div') as HTMLElement

    fireEvent.drop(zone, dropWith([fileEntry('/x.pdf')]))
    await waitFor(() => expect(onFiles).toHaveBeenCalled())
    expect((onFiles.mock.calls[0][0] as File[]).map((f) => f.name)).toEqual(['x.pdf'])
  })

  it('falls back to dataTransfer.files when the entries API is unavailable', async () => {
    const onFiles = vi.fn()
    render(<Dropzone onFiles={onFiles} />)
    const zone = screen.getByText(/drop/i).closest('div') as HTMLElement

    const f = new File([new Uint8Array([1])], 'legacy.pdf')
    fireEvent.drop(zone, { preventDefault: () => {}, dataTransfer: { items: [], files: [f] } })
    await waitFor(() => expect(onFiles).toHaveBeenCalledWith([f]))
  })
})
```

If `fireEvent.drop`'s init object doesn't reach the handler with these fields under this repo's testing-library version, build the event with `new Event('drop')` plus `Object.defineProperty(ev, 'dataTransfer', …)` and `fireEvent(zone, ev)` — assert real handler behavior either way.

- [ ] **Step 2: Run — must fail**

`cd frontend && pnpm vitest run src/components/ingest/Dropzone.test.tsx`
Expected: the tree test fails (nothing queued — `dataTransfer.files` is empty for directories, which is exactly the bug); the plain-file test likely fails too for the same reason.

- [ ] **Step 3: Implement the traversal**

`frontend/src/components/ingest/Dropzone.tsx` — add above the component:

```tsx
/** Minimal shape of the non-standard entries API we consume. */
type FsEntry = {
  isFile: boolean
  isDirectory: boolean
  fullPath: string
  file?: (onOk: (f: File) => void, onErr?: (e: unknown) => void) => void
  createReader?: () => { readEntries: (onOk: (e: FsEntry[]) => void, onErr?: (e: unknown) => void) => void }
}

/** `entry.file()` yields an empty webkitRelativePath; ingest.ts uploads each
 *  file as `webkitRelativePath || name`, so stamp the tree path or dropped
 *  folders flatten and same-named files across subfolders collide. */
function withRelativePath(file: File, fullPath: string): File {
  Object.defineProperty(file, 'webkitRelativePath', {
    value: fullPath.replace(/^\//, ''),
    configurable: true
  })
  return file
}

/** readEntries returns at most ~100 entries per call; loop until it yields an
 *  empty page or large folders are silently truncated. */
async function readAllEntries(dir: FsEntry): Promise<FsEntry[]> {
  const reader = dir.createReader?.()
  if (!reader) return []
  const all: FsEntry[] = []
  for (;;) {
    const page = await new Promise<FsEntry[]>((resolve) => reader.readEntries(resolve, () => resolve([])))
    if (!page.length) return all
    all.push(...page)
  }
}

async function collectFiles(entry: FsEntry): Promise<File[]> {
  if (entry.isFile) {
    const file = await new Promise<File | null>((resolve) =>
      entry.file ? entry.file(resolve, () => resolve(null)) : resolve(null)
    )
    return file ? [withRelativePath(file, entry.fullPath)] : []
  }
  if (entry.isDirectory) {
    const children = await readAllEntries(entry)
    const nested = await Promise.all(children.map(collectFiles))
    return nested.flat()
  }
  return []
}
```

Replace `handle` with:

```tsx
  const handle = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    setHover(false)
    if (disabled) return
    // DataTransfer is neutered once this handler returns, so pull every entry
    // out synchronously BEFORE awaiting anything.
    const entries = Array.from(e.dataTransfer.items ?? [])
      .map((item) => (item as unknown as { webkitGetAsEntry?: () => FsEntry | null }).webkitGetAsEntry?.() ?? null)
      .filter((entry): entry is FsEntry => entry !== null)
    if (!entries.length) {
      // No entries API (or no entries): keep the plain-file behavior.
      const list = Array.from(e.dataTransfer.files)
      if (list.length) onFiles(list)
      return
    }
    void Promise.all(entries.map(collectFiles)).then((groups) => {
      const list = groups.flat()
      if (list.length) onFiles(list)
    })
  }
```

- [ ] **Step 4: Run — must pass**

`cd frontend && pnpm vitest run src/components/ingest/Dropzone.test.tsx` → all green, including the pre-existing folder-picker test.

- [ ] **Step 5: Chores**

`grep -rn "drop\|folder" README.md CLAUDE.md docs/*.md | grep -i "ingest\|upload"` — if any passage says only the folder *button* can ingest a directory (or omits folder drops), update it: all three inputs (drop files, drop a folder, pick a folder) now behave identically and preserve the directory tree in the uploaded filenames.

- [ ] **Step 6: Full validation + commit**

```bash
cd frontend && pnpm lint && pnpm test && pnpm build && cd ..
make verify
git add frontend/src/components/ingest/Dropzone.tsx frontend/src/components/ingest/Dropzone.test.tsx README.md CLAUDE.md docs
git commit -m "fix: dropped folders are traversed like the folder picker"
```

- [ ] **Step 7: Push + PR**

```bash
git push -u origin feature/ux-batch
```

Open the PR to `main`, title `docint UX batch: operator prefill, nav order, merged-only entities, folder drops`. Body: the four items, a note that the backend `entity_merge_mode` parameter is intentionally retained, and the validation evidence (frontend suite + `make verify`).

---

## Manual verification (after the PR is up)

Not automatable in the suite, worth one pass in a dev browser: create a report and confirm the operator field arrives prefilled and is still editable; drag a nested folder onto the Ingest dropzone and confirm the staged file list shows subfolder-qualified names matching what the folder button produces.
