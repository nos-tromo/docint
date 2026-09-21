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

> **Status.** This release records activity and computes deadlines. The sweep
> that deletes due collections is not active yet, so the clock has real history
> behind it before anything is removed.

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
that user's collection.

Looking does not count. Listing collections or chat sessions, reading deadlines
(`GET /collections/retention`) and the maintenance commands — `search-index`,
`search-index-all`, `resolve`, `extract` and `retention-report` — never move a
clock, so a one-off backport can never keep every collection alive.

The clock is written at most once an hour per collection, and whether or not
retention is on — switching it on later finds real history. A failed write never
fails the request that caused it; the next request a minute later retries.

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

- `GET /collections/retention` — each of the caller's collections with its last
  activity, due date and a `warning` flag; admins pass `all=true` for everyone's.
  See [api-reference.md](api-reference.md#get-collectionsretention).
- `GET /config` — `collection_retention` names the window in force.
- `make retention-report` — the operator's view, without `WINDOW` exactly what
  the running backend acts on.

## Not covered

Collections without an ownership row have no clock and are never due. They are
the ones ingested with the `ingest` CLI, and pre-ownership collections on a host
that runs without `DOCINT_DEFAULT_IDENTITY`. `make retention-report` names them.
