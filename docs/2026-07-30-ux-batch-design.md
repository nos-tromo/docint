# docint UX batch — design

Date: 2026-07-30
Status: approved (design); implementation pending
Scope: docint frontend only (four items, one PR)

## Context

Last open item of the first-rollout backlog (see
`infra/` memory: first-rollout backlog). Four independent UX corrections the
owner collected from live use. Decisions locked with the owner:

- Merge-mode removal is **frontend-only**; the backend keeps
  `entity_merge_mode` and its non-resolved code paths for API/debugging use.
- The report operator field is **prefilled on create and remains editable**
  (not a placeholder, not locked to the authenticated identity).
- The "upload directory" item is the **Ingest UI's folder upload**, not the
  backend `DATA_PATH`. Folder drag-and-drop gets fixed; no `DATA_PATH`
  documentation is part of this batch.

Deferred, explicitly out of scope: porting Nextext's tf-idf word frequencies
and wordcloud into Analysis, and Nextext's baked-black wordcloud background.

## 1. Report operator prefill

The report record already carries an `operator` case-metadata field
(`ReportCreateIn`/`ReportUpdateIn`, `docint/core/api.py:820-838`).
`Report.tsx`'s `onCreate` currently posts only `title` and
`collection_name`; it will additionally post
`operator: whoami?.display_name ?? whoami?.username`, reusing the existing
`useWhoami` hook (added for the AppHeader — no new endpoint, no second
fetch). When identity is unknown (dev without the gateway, or a failed
fetch) the field is omitted, so the report is created with an empty
operator rather than a wrong one.

The field stays editable and clearable exactly as today. Existing reports
are untouched — this is a create-time default, not a migration.

Tests: create with a resolved display name sends it; create with no
identity omits the key; a manual edit after creation still persists
(guards against the prefill being re-applied on update).

## 2. Section reorder

`Sidebar.tsx`'s `NAV` array (currently dashboard, chat, ingest, analysis,
inspector, report) becomes:

```
dashboard → ingest → inspector → chat → analysis → report
```

Pure reordering of one array literal: routes, i18n keys, and labels are
unchanged, and Dashboard remains the index route (`to: '/'`). A test
asserts the rendered nav order so an accidental future shuffle fails CI.

## 3. Merged entities only (frontend)

Delete `components/common/MergeModeToggle.tsx` and its test, remove both
mount points (Analysis and Dashboard), and drop the `entityMergeMode`
slice from `stores/ui.ts` — **including its persisted-state migration
path**, so a stale persisted value cannot resurrect the setting for
existing users.

`hooks/useNer.ts` (two call sites) and
`components/analysis/EntityFindingsTable.tsx` send the literal
`'resolved'` for `entity_merge_mode`. The `EntityMergeMode` type and the
API parameter in `api/collections.ts` stay — the backend contract is
unchanged and orthographic/exact remain reachable for API and debugging
use, per the owner's decision.

Tests: both NER hooks request `resolved` unconditionally; neither Analysis
nor Dashboard renders a merge-mode control.

## 4. Folder drag-and-drop parity

Today `Dropzone.tsx` offers three input paths, one of which is silently
broken: dropping a **folder** goes through `handle()`, which reads
`e.dataTransfer.files` — for a directory that yields no usable files, so
nothing is queued and no error is shown. Only the "Or choose a folder"
button (an `input` with `webkitdirectory`) can ingest a tree.

`handle()` will traverse dropped directories via the entries API:

- **Capture entries synchronously.** `DataTransfer` is neutered once the
  event handler returns, so collect `item.webkitGetAsEntry()` for every
  `e.dataTransfer.items` entry *before* any `await`, then traverse.
- **Recurse with a pagination loop.** `DirectoryReader.readEntries()`
  returns at most ~100 entries per call and must be called repeatedly
  until it yields an empty array — a single call silently truncates large
  folders.
- **Preserve the relative path.** This is load-bearing:
  `api/ingest.ts:44` uploads each file as
  `f.webkitRelativePath || f.name`, so the folder *button* reproduces the
  directory tree server-side. Files obtained from `entry.file()` have an
  empty `webkitRelativePath`, so the traversal must stamp the entry's
  `fullPath` (minus the leading slash) onto each file — via
  `Object.defineProperty(file, 'webkitRelativePath', { value: path })` —
  otherwise dropped folders flatten and two `report.pdf` files in
  different subfolders collide on upload.
- **Fall back cleanly.** When `webkitGetAsEntry` is unavailable, keep
  today's `e.dataTransfer.files` behavior so plain multi-file drops are
  unaffected.

After the change all three paths (drop files, drop folder, pick folder)
produce the same `File[]` shape with the same names, and `onFiles`'s
contract is unchanged — no consumer edits.

Non-goal (deliberate): file-count or total-size guard rails on folder
selection. The existing folder button has none, so this change does not
make that worse; adding limits is a separate decision.

Tests: dropping a nested directory queues every file including files
below the readEntries page size, with `webkitRelativePath` preserved;
dropping plain files still works; the no-`webkitGetAsEntry` fallback path
still queues dropped files.

## Testing and shape

One branch (`feature/ux-batch`), one PR, four commits — one per item.
Frontend-only: validation is `pnpm lint && pnpm test && pnpm build` in
`frontend/`, plus root `make verify` (which also runs the untouched
backend suite). Chores ride along: any README/CLAUDE.md claim about the
merge-mode control or the ingest inputs is corrected in the same branch.
