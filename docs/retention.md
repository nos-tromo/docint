# Document retention

Docint is an analysis platform, not a permanent store. Every document is
deleted a fixed period after it was added, and everything built from it goes
no later than it does. The rule is on by default.

| Variable | Default | Meaning |
|---|---|---|
| `DOCUMENT_RETENTION_DAYS` | `365` | Days after its ingest completed a document is deleted, `1` to `3650`. `off` deletes nothing (development only; logged as a warning). |
| `RETENTION_NOTICE_DAYS` | `30` | Days before its deletion an item is listed for its owner, `1` to `365`; also the grace period after the rule takes effect. |

Any other value stops the backend at startup: a typo never deletes on a
schedule nobody chose, and never silently switches retention off.
`COLLECTION_RETENTION`, the idle-window rule this replaces, is no longer read;
a startup that still finds it set logs one warning. Every startup logs one line
saying which period is in force:

```text
Document retention | period=365d notice=30d since=2026-11-05T06:45:00+00:00
```

and records the same in the audit chain (`admin` `retention.config`, with
`period_days`, `notice_days` and `since`; see [audit.md](audit.md)).

## The rule

```text
due = max(ingested + period, period took effect + notice)
```

- **The clock starts when the document is stored**, at the end of its ingest,
  and nothing moves it: reading, chatting with or exporting a document does
  not. There is no extension — no button, flag or admin override keeps an item
  past its date.
- **Per document, per collection.** Re-ingesting the same file into the same
  collection keeps its clock; the same file in another collection has its own.
  Once a document is deleted, a new ingest is a new document with a new clock.
- **Grace.** Nothing is deleted sooner than the notice period after the period
  took effect on the host: the first start of this release, or any change of
  `DOCUMENT_RETENTION_DAYS`. So every owner sees a full notice, also for data
  that is already older than the period when retention starts.

The rule lives in one place, `docint/core/retention.py`, so the API listing,
the operator report and the sweep can never disagree about a date.

## What counts as one document

One uploaded file, keyed by its content hash: every chunk of a PDF, a video
with its transcript segments and keyframes, a social-media export with all its
postings and their images. Images that two documents contain are kept for the
other one.

## Where the clock lives

- **In Qdrant.** Every point of a document carries its ingest time in its
  payload (`docint_ingested_at`, seconds since the epoch; hidden from the
  embedding and the prompts), so a restored collection still knows its
  deadlines.
- **In the sessions store.** The `document_retention` table indexes each
  document (collection, owner, file hash, file name, ingested, due) for the
  notices and the sweep. Each sweep first repairs it from Qdrant: a document
  Qdrant holds and the index lacks gets a row, timed by its payload stamp, else
  the ingest manifest's completion time, else the moment the period took
  effect — never the time of the repair. A restore from a backup therefore
  never earns a fresh clock.
- **Documents from before this release** have no payload stamp; they take the
  manifest's completion time where the manifest has one, else the moment the
  period took effect.

## What is built from documents

Anything derived from documents is deleted no later than the earliest document
it draws on. Where docint cannot tell which documents an item draws on, the
collection's oldest document counts.

| Item | Inherits from |
|---|---|
| Chat session | the documents its answers cite; an answer without citations, a scoped session or one with a topic focus counts as collection-wide |
| Report | each item's sources (a chat answer's citations, a finding's chunk); summaries, topics and the collection overview count as collection-wide |
| Stored extract | collection-wide, and its own `EXTRACT_RETENTION_DAYS` if that comes first |
| Topic run | collection-wide |
| Summary and overview caches, full-text and entity data, docstore nodes, source file, pipeline artifacts | part of the document; deleted with it |

A derived item's date is set when it is made and only ever moves earlier as it
gains sources. An item with nothing to inherit from — built in a collection
without indexed documents, or a chat or report from an older release tied to
no collection — goes one period after it was dated.

**When the period changes**, document dates are recomputed under the new one.
A shorter period moves every chat's and report's date forward by the
difference (never inside the new grace); extracts and topic runs go with the
collection's earliest document at the latest. A longer period keeps the
earlier dates — nothing outlives its sources. Switching to `off` clears the
chat and report dates; switching on again stamps them afresh.

## The sweep

The backend sweeps once a day, starting five minutes after startup. Each sweep:

1. repairs the document index from Qdrant (above);
2. stamps a date on chats and reports from before this release;
3. for each collection with documents due, **skips it if any job is still
   working on it** — an ingest, a summary, an extract, a topics job, or
   upload-time preprocessing — and otherwise deletes those documents (below).
   A collection left without documents is deleted with the collection cascade
   ([api-reference.md](api-reference.md#delete-collectionsname));
4. deletes due chat sessions, reports (with their stored PDFs), extracts and
   topic runs.

Every deletion is recorded in the audit trail first, as `delete` by the system,
with references and never content ([audit.md](audit.md#retention-sweep)). A
deletion whose record cannot be written does not happen. A document deletion
that fails afterwards adds an `error` event; whatever fails is retried by the
next sweep, while the others still go. At most 5,000 due
documents are taken per sweep; the rest follow the next day. Each sweep ends
with one line to grep for:

```text
Retention sweep complete | period=365d documents=12 sessions=3 reports=1 extracts=0 topic_runs=1 collections=1 repaired=0 skipped_busy=0 failed=0
```

The sweep runs inside the backend, not as a separate command, so the process
that deletes is the one whose caches hold the data. While retention is off, or
when the startup could not record the period, the sweep does not start.

## What deleting a document removes

In order, each step a no-op on what is already gone:

1. its points in the collection (by `file_hash`, `media_file_hash` or
   `source_doc_id`) and in the `_dockv` companion;
2. its `_images` points — an image another document also contains loses only
   this document's occurrence;
3. its docstore nodes, and its entries in the summary cache (the collection's
   cached summary is invalidated);
4. its source file, only inside the collection's source directory;
5. its ingest manifest row;
6. its PDF pipeline artifacts, unless another collection still uses them (below);
7. the collection's `_entities` companion — canonical entities record no
   per-document provenance, so the store is rebuilt from what is left: while
   `RES_AUTO_RESOLVE` is on, resolution runs again right after (`combine`
   `entities.resolve`, by the system); until it completes, or with it off until
   the next resolve, entity views show the collection as unresolved;
8. its row in the document index.

While a collection is being worked on by the sweep, requests that would work on
it get `409`.

**PDF artifacts are shared.** They are named by the file's content hash, so two
collections holding the same PDF share one folder. A folder is removed only when
no other collection's ingest manifest records the hash, no other collection's
source files produced it, and no preprocessing worker is reading that PDF right
now. Removing too much costs only a re-read of that PDF. With
`INGEST_MANIFEST_ENABLED=false` nothing says what other collections use, so no
artifact is removed.

## Seeing what is deleted, and when

- **Dashboard** — the *Automatic deletion* card lists the owner's documents,
  chats and reports due within the notice period, grouped by collection, each
  with its date; the sidebar says how many items are due and links to it. See
  [ui-guide.md](ui-guide.md#dashboard-srcroutesdashboardtsx).
- `GET /collections/retention` — the same for the caller; admins pass
  `all=true` for everyone's (recorded as `admin`). See
  [api-reference.md](api-reference.md#get-collectionsretention).
- `GET /config` — `document_retention_days` and `retention_notice_days`.
- `make retention-report` — the operator's view of everything with a date;
  `DAYS=N` shows what another period would do. See
  [cli-reference.md](cli-reference.md#retention-report--what-retention-deletes-and-when).

**Collections without an owner** — ingested with the `ingest` CLI, or from
before ownership on a host without `DOCINT_DEFAULT_IDENTITY` — expire like any
other, but no one is notified; the operator report names them.

## Upgrading

The first start of this release records the period and starts the grace
period: nothing is deleted for the notice period, however old a document is.
Retention is on from then on unless `DOCUMENT_RETENTION_DAYS=off` is set. During
the grace, `make retention-report` shows the operator every date and the
Dashboard shows each owner what falls due.

## Not covered

- **Backups.** A restored collection keeps its deadlines (above); the Qdrant
  snapshots themselves are pruned by `data-plane`.
- **PDF pipeline artifacts** of collections deleted before artifacts joined the
  cascade stay under `PIPELINE_ARTIFACTS_DIR`, as do all artifacts while the
  ingest manifest is off.
