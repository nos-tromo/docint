# Collection retention

Docint keeps every collection until someone deletes it. Collection retention is
the opt-in alternative: set `COLLECTION_RETENTION` and a collection nobody has
worked with for that long falls due for deletion.

| Value | Meaning |
|---|---|
| `off` (default) | Nothing is ever due. |
| `6m`, `12m`, `18m`, `24m` | Due after that many calendar months without activity. |

Any other value keeps retention **off** and logs a warning — a typo never
deletes on a schedule nobody chose. Every startup logs one line saying which
window is in force:

```text
Collection retention | window=6m set_at=2026-09-21T06:45:00+00:00
```

## The rule

```text
due = max(last activity + window, window set + 30 days)
```

- **Calendar months.** A collection last used on 31 August is due at the end of
  February under `6m`, not on 3 March.
- **Notice.** Nothing falls due sooner than 30 days after the window was set.
  Switching retention on, switching it off, or changing the window all restart
  that grace period; restarting with the same window does not. The SPA flags a
  collection from 30 days before it is due, so the grace period guarantees every
  user sees a full notice period.
- **No recorded activity never expires.** A collection with an empty clock is
  never due.

The rule lives in one place, `docint/core/retention.py`, so the API listing and
the operator report can never disagree about a date.

## What counts as activity

The clock is `collection_owners.last_activity_at` in the sessions store. It
moves on:

- every request that works on a collection — each such route passes the
  ownership gate (`_require_owned_collection`), which records the activity:
  chat, search, analysis, the inspector, previews, summaries, extracts and
  exports;
- ingesting into a collection — upload, finalize and `POST /ingest`;
- opening, extending or exporting a report built from the collection.

An admin working in another user's namespace (`?owner=`) counts as activity on
that user's collection. Viewing a collection's data in the app counts as well —
the Dashboard, for one, reads the selected collection's document count on load.

Looking does not count. Listing collections or chat sessions, reading deadlines
(`GET /collections/retention`) and the maintenance commands — `search-index`,
`search-index-all`, `resolve`, `extract` and `retention-report` — never move a
clock, so a one-off backport can never keep every collection alive.

The clock is written at most once an hour per collection, and whether or not
retention is on — switching it on later finds real history. A failed write never
fails the request that caused it; the next request a minute later retries.

## The sweep

While retention is on, the backend sweeps once a day, starting five minutes
after startup. For each collection past its due date it:

1. **skips it if any job is still working on it** — an ingest, a summary, an
   extract, or upload-time preprocessing. Deleting under a running job would
   let the job re-create the collection; the next sweep tries again;
2. **re-reads its clock** and spares it if anyone used it since the sweep
   started;
3. **deletes it** with everything connected (below).

A collection that fails to delete is left intact and retried the next day; the
others still go. Each sweep ends with one line to grep for:

```text
Retention sweep complete | window=6m scanned=12 expired=2 deleted=2 skipped_busy=0 skipped_active=0 failed=0
```

and every deletion is logged by name with its last activity. Logs carry
collection names and counts, never owners or content.

The sweep runs inside the backend, not as a separate command, so the process
that deletes a collection is the one whose caches hold it. Without a recorded
window (the startup could not write it) the sweep does not start.

## What is deleted

The same cascade as deleting a collection by hand (`DELETE /collections/{name}`),
in this order:

1. the Qdrant collection and its hidden companions (`_images`, `_entities`);
2. the source files, and with them the collection's docstore, summary cache,
   ingest manifest and cached transcripts;
3. stored extracts;
4. its PDFs' pipeline artifacts (page text, chunks, tables, figures under
   `PIPELINE_ARTIFACTS_DIR`) that nothing else uses — see below;
5. the collection's chat sessions;
6. the reports built from it — manual deletion removes them too, see
   [reports.md](reports.md#when-the-collection-is-deleted);
7. last, the ownership row.

A failure stops the cascade with the collection still listed, and the next
attempt finishes it: every step is a no-op on what is already gone. While a
collection is being deleted, requests that would work on it get `409`.

**PDF artifacts are shared.** They are named by the file's content hash, so two
collections holding the same PDF share one folder. A folder is removed only when
no other collection's ingest manifest records the hash, no other collection's
source files produced it, and no preprocessing worker is reading that PDF right
now. The collection's own folders are found by its manifest *and* by the source
path each folder records, which is how an upload that was preprocessed but never
ingested is found. Removing too much costs only a re-read of that PDF. With
`INGEST_MANIFEST_ENABLED=false` nothing says what other collections use, so no
artifact is removed. Unlike the steps above, this one is best-effort: a failure
is logged and does not stop the delete.

## Upgrading

Collections that existed before this release have no activity history, so the
upgrade starts every clock at the moment the sessions store is migrated — not at
the collection's creation date. A guess that is too old would delete data. The
consequence: with `COLLECTION_RETENTION=6m`, no existing collection falls due
before six months after the upgrade.

## Before switching it on

See what a window would do, without changing anything:

```bash
make retention-report WINDOW=6m
```

The report lists every collection's last activity and due date, applying the
grace period exactly as switching the window on now would. It moves no clock and
records no window. See [cli-reference.md](cli-reference.md#retention-report--what-retention-deletes-and-when).

## Seeing deadlines

- **Dashboard** — an *Automatic deletion* card lists every collection with its
  deletion date and flags those due within 30 days; the sidebar says how many
  other collections are due within 30 days and links to it. See
  [ui-guide.md](ui-guide.md#dashboard-srcroutesdashboardtsx).
- `GET /collections/retention` — each of the caller's collections with its last
  activity, due date and a `warning` flag; admins pass `all=true` for everyone's.
  See [api-reference.md](api-reference.md#get-collectionsretention).
- `GET /config` — `collection_retention` names the window in force.
- `make retention-report` — the operator's view, without `WINDOW` exactly what
  the running backend acts on.

## Not covered

- **Collections without an ownership row** have no clock and are never due.
  They are the ones ingested with the `ingest` CLI, and pre-ownership
  collections on a host that runs without `DOCINT_DEFAULT_IDENTITY`.
  `make retention-report` names them.
- **Reports with no collection** are tied to none and are kept. So are reports
  whose collection was deleted before reports joined the cascade — until a new
  collection of the same name is deleted, since reports are matched by owner and
  collection name.
- **PDF pipeline artifacts** of collections deleted before artifacts joined the
  cascade stay under `PIPELINE_ARTIFACTS_DIR`, as do all artifacts while the
  ingest manifest is off.
- **A deletion record.** What was deleted is in the backend log only.
