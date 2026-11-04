# Data-protection audit log

docint records every processing of collection data it performs — collection,
alteration, query, disclosure (including transfer to inference), combination
and erasure, the categories §76 BDSG distinguishes — as one hash-chained JSON
line per operation. This page lists what is recorded, how an event is shaped,
which routes and commands record what, and how to configure it.

The lines are written to stdout and nowhere else. Storing, retaining,
verifying and granting access to them is the observability plane's job: see
obs-plane `docs/audit.md` for the separate audit store, its one-year retention
and its access model. Nothing on this page duplicates that.

## Contents

- [How it works](#how-it-works)
- [What is recorded](#what-is-recorded)
- [How an event is shaped](#how-an-event-is-shaped)
- [Outcomes, refusals and what is not recorded](#outcomes-refusals-and-what-is-not-recorded)
- [Fail closed](#fail-closed)
- [Stated purpose](#stated-purpose)
- [Routes](#routes)
- [Background jobs](#background-jobs)
- [Model calls](#model-calls)
- [Streaming routes](#streaming-routes)
- [Retention sweep](#retention-sweep)
- [Command-line tools](#command-line-tools)
- [Configuration](#configuration)
- [Limits](#limits)
- [For developers](#for-developers)

## How it works

- **The library is the federation's.** `docint/utils/audit_event.py` (the
  event format, the hash chain, `verify_chain`) and
  `docint/utils/audit_runtime.py` (the output stream, the purpose header,
  request-id cleaning, id chunking) are vendored byte-identically from
  `nos-tromo/.github` `configs/audit/`; the hub's
  `scripts/validate_audit_lib.py --consumer-root .` checks both copies. Never
  edit them here — change the canonical file and copy it again.
  `docint/core/audit.py` is docint's glue: one `Auditor` per process, the
  request gate every audited route goes through, the `Operation` record, and
  the command-line session.
- **One chain per process.** The API opens its chain in the FastAPI lifespan
  (`_start_audit` in `docint/core/api.py`), never at import, and closes it
  last on shutdown. Every command-line run opens a chain of its own and
  closes it however the command ends. Each chain starts with the library's
  `lifecycle`/`start` event and one `admin`/`config.audit` event stating the
  operating mode in the keys every app of the federation uses —
  `{"edge_secret_enforced": …, "default_identity_enabled": …, "require_purpose": […]}`,
  `default_identity_enabled` being true only while the dev fallback is in
  effect (set, and no edge secret); a command's chain adds `"command"`. Never
  a secret or a path. A `lifecycle`/`heartbeat` follows every 60 seconds and a
  `lifecycle`/`shutdown` ends a clean chain.
- **Where the lines go.** `AUDIT_OUTPUT` selects the stream. In the backend
  container compose sets `AUDIT_OUTPUT: container`: every process in the
  container — the API and any command started with `docker compose exec` —
  writes its chain to the container's main stdout (`/proc/1/fd/1`) under one
  container-wide lock (`/tmp/audit-stream.lock`, on the backend's `media-tmp`
  volume, which `volume-permissions` empties before every start). The lines
  therefore land in the backend's own log, and two writers never interleave.
  Unset (a `uv run` host), they go to the process's own stdout.
- **stdout is the audit log's.** The operational log is loguru's single
  stderr sink, uvicorn's loggers included (see
  [configuration.md](configuration.md#logging--loggingconfig)); nothing else
  in the backend writes to stdout. The log collector routes every line
  carrying `"stream":"audit"` to the audit store.

## What is recorded

| Action | In docint |
|---|---|
| `collect` | Files staged by an upload (`ingest.upload`) and every ingest run into a collection (`ingest.run`): through the job queued by `POST /ingest/finalize`, the synchronous `POST /ingest`, and the `ingest` command. |
| `modify` | A session's scope or topic changed; reports created, edited, assigned and filled; a collection summary built; a topic run built, discovered or aligned; the full-text search backfill (`search-index`). |
| `query` | Chat and agent turns, keyword search, one chunk's text, the NER views, the hate-speech list, the documents list, session listings and histories, report listings and reports, the topics views, a summary read, a source preview or a thumbnail. |
| `disclose` | Every download (`recipient="download:<format>"`: CSV exports, report exports, extract archives, a single source's extract, a session's source files, a stored report PDF) and every file written for someone to take away (`recipient="file:extract"`, `"file:report_pdf"`, and the command-line tools' result files). |
| `ai_call` | A model call: the answer of a chat or agent turn (`inference:chat`), a snippet translation, a summary read's validation, and the transfers of a batch operation — one event per stage, with counts ([Model calls](#model-calls)). |
| `combine` | Entity resolution: merging entity surfaces into durable canonicals (`entities.resolve`), whether from the API, after an ingest, or from `resolve`; and the social linker joining a social export's postings with their comments and media inside an ingest run (`ingest.social_link`). |
| `delete` | A collection, a session, a report, a report item, a stored extract; the retention sweep's purges (as the system); `verify --repair`. |
| `admin` | The chain's `config.audit` event, and an admin's listing of every owner's collections (`collections.list_all`, `collections.retention_all`). |
| `lifecycle` | Written by the library itself: `start`, `heartbeat`, `shutdown`. |

docint writes no `auth` events: authentication is the edge gateway's
(edge-plane `docs/identity-contract.md`).

## How an event is shaped

Every event carries `service: "docint"`, the package version as
`service_version`, and the chain fields the library adds. The fields docint
fills:

**`actor`** — who caused it:

| Caller | `actor` |
|---|---|
| A request whose identity header the backend believes (`X-Auth-User`; with `DOCINT_EDGE_SECRET` set, only beside the gateway's secret) | `{"user": "<X-Auth-User>", "via": "edge"}` |
| The dev fallback identity (`DOCINT_DEFAULT_IDENTITY`, honoured only while no edge secret is set) | `{"user": "<default identity>", "via": "system"}` |
| An admin working in another owner's namespace (`?owner=`) | `{"user": "<admin>", "via": "edge", "on_behalf_of": "<owner>"}` |
| A non-admin whose `owner` parameter names someone else | refused with 403 and recorded as `denied`, `on_behalf_of` naming the owner asked for |
| A command-line run | `{"user": "<AUDIT_CLI_USER>", "via": "cli"}` (`null` when unset) |
| The retention sweep, a job no request started, the chain's own events | `{"user": null, "via": "system"}` |

The actor mapping is `actor_for` in `docint/core/audit.py`; whether a name
came from a header or the default identity is `Principal.via`
(`docint/core/auth/principal.py`). A request without an identity (401) is
never recorded: it has no actor, and recording it would let anyone on the
networks flood the log.

**`request_id`** — the gateway's `X-Request-ID`, kept only when it is 1–128
characters of letters, digits and `._:-` (`clean_request_id`). Jobs carry the
id of the request that queued them.

**`purpose`** — the stated `case_ref` and `reason` ([Stated purpose](#stated-purpose)).
Jobs carry the purpose of the request that queued them.

**`target`** — what was touched, by reference only. One kind per event,
always with `count` (the number of rows returned or touched, also when some
row carries no identifier); `container` is the logical collection.

| `kind` | Each id is | Recorded by |
|---|---|---|
| `chunk` | A Qdrant point id — a chunk in the collection or an image point in its `_images` companion (what `GET /search/chunk` and a chat scope address); the reader-minted `chunk_id` when a row has no point id | answers' sources, search hits, NER and hate-speech findings, histories, topic passages; the full-text index backfill (count only) |
| `document` | A file hash | documents list, upload, ingest run, source preview, session sources ZIP, topic documents |
| `entity` | `<text>::<type>`, the key the SPA builds (a graph node's cluster key in the graph) | NER stats, entity search, entity graph, entities CSV |
| `image` | An image's content hash (`image_id`) | thumbnail |
| `posting` | A social posting's uuid | the social linker's joins inside an ingest run (`ingest.social_link`) |
| `session` | A session id | session listing, scope and topic changes, session delete |
| `report` | A report id | report listing, create, update, assign, overview refresh, delete, reorder, stored-PDF download |
| `report_item` | The item's dedupe key — `entity:<chunk id>`, `hate:<chunk id>`, `chat:<session>:<turn>`, `summary:<collection>`, `topics:<collection>` — or its item id where only that is known | report reads and exports, item add/note/remove |
| `topic` | A topic id | topics overview, topic runs (count only) |
| `extract` | A stored extract's id | extract download and delete |
| `collection` | A logical collection name | collection delete, retention purge, admin listings (count only) |
| `node` | A node id in the collection's docstore or its Qdrant collection | `verify --repair`: the orphans removed |
| `text` | — (count only) | a translated snippet |
| one unit kind, or `mixed` | An extract unit's key: a file hash for a `document` or `media` unit, the posting uuid for a `posting`, the image id for an `image`; several kinds at once make a `mixed` target whose ids carry their kind (`posting:<uuid>`, `document:<hash>`) | extract builds, a single source's extract |

A target with more than 200 ids is not cut: the event carries the first 200,
and each further 200 follow in a continuation event — the same action, the
operation `<operation>.ids`, `parent_event_id` pointing at the first event,
and a target holding that chunk (`Auditor.emit`). The first event's `count`
counts every id.

**`criteria`** — the caller's parameters: collection, filters, keywords,
the entity searched for, limits, modes, flags, and counts that say how an
operation ended. Never document text, a question, an answer, a transcript, a
note or a file. A chat question or agent message is referenced as
`question_sha256`, a translated snippet as `text_sha256`; a report update
names the fields changed, never their values; a report item's note is never
recorded. Keyword search terms and the entity searched for are recorded as
given: they are the search criteria. The library caps every value.

**`recipient`** — where data went: `download:<format>` for a download,
`file:extract` / `file:report_pdf` / the command-line tools' `file:<what>`
for a file written to be taken away, `inference:<task>` for a model call.

**`ai`** — on `ai_call` events: the model alias asked for; for an answer or a
translation, `output_sha256` of the text produced; for an answer, `sources`,
the ids of the passages the model was given. docint's model calls report no
token counts.

**`parent_event_id`** — an `ai_call` points at the event of the operation it
belongs to (the turn's `query`, the run's `collect`, the resolution's
`combine`, the build's `modify`), the social linker's `combine` at its run's
`collect`, and a continuation at its first event.

## Outcomes, refusals and what is not recorded

- **`ok`** — written when the operation completes: for a read or a download
  after the data is gathered and before the response goes out, for a write
  after it is committed.
- **`error`** — the operation began and then failed: the route answered 5xx
  or raised. The global error handlers (`docint/core/errors.py`) record every
  audited operation of a failing request that has no outcome yet, with its
  criteria and never the exception's text. A job records its failure itself,
  with static codes only: `cancelled` for a cancelled run, `refused` /
  `endpoint_error` with the topics service's code. A broken-off stream adds
  `incomplete: true`.
- **`denied`** — the gate refused the request: 403 for a non-admin's
  `owner` naming someone else, 428 for a missing purpose that
  `AUDIT_REQUIRE_PURPOSE` requires. The event has the operation's action and
  name and no target.
- **Not recorded:** a malformed `X-Audit-Purpose` (400 — there is no valid
  operation yet), an unauthenticated request (401), and every other client
  error (404, 409, 413, 422 …), which returns no data. A cross-owner name is a
  404 exactly like a missing one, so such a probe is not recorded either.

The gate is a route dependency (`audited`), so it runs before the route's
body and a refusal usually comes before the route's own checks. Two kinds of
route open it later: the report routes check that the report belongs to the
request's collection first (a 404 there records nothing), and
`POST /collections/{name}/topics` and the admin listings open it in their
body, once the action or the `all=true` branch is known (`audit.begin`).

A non-admin's `owner` parameter is refused only on audited routes; on the
others (the job status routes, `GET /collections/list` without `all=true` …)
it is ignored as before, and the request resolves against the caller's own
namespace.

## Fail closed

An event that cannot be written fails the operation it records
(`AuditWriteError`, or `AuditTrailClosed` when no chain is open —
`AUDIT_FAILURES` in `docint/core/audit.py`):

- **A request** answers the generic 500 and does not return its data.
- **A stream** ends with the generic error frame instead of the frame that
  carries the result ([Streaming routes](#streaming-routes)).
- **A job** is marked failed. Its optional stages (entity resolution, the
  summary, topics) swallow their own errors as warnings, but re-raise an
  audit failure, so a run never completes unrecorded.
- **A command** exits non-zero.
- **Startup** stops on an unknown action in `AUDIT_REQUIRE_PURPOSE`, an
  `AUDIT_OUTPUT` naming neither mode, or a container stream that cannot be
  opened.

Recording a failure that is already being handled — the error handlers, a
broken stream, a client gone mid-answer — cannot fail anything further: a
write that fails there is logged on stderr and the failure goes on
(`settle_quietly`).

## Stated purpose

A request states why it is made in the `X-Audit-Purpose` header: the
`application/x-www-form-urlencoded` serialization of up to two fields,
`case_ref` and `reason` — what a browser's
`new URLSearchParams({case_ref, reason}).toString()` produces:

```http
X-Audit-Purpose: case_ref=case-0001&reason=Review+of+the+flagged+postings
```

The value must be ASCII (percent-encode the fields), at most 4096
characters, each field at most 256 characters once decoded and free of
control characters, with neither field repeated; a blank field counts as
absent and other keys are ignored. Anything else is answered 400 with a
generic detail and records nothing. A purpose is *stated* when at least one
field is set; it is recorded on every event of the operation, and never in
the operational log — it is personal data in its own right.

`AUDIT_REQUIRE_PURPOSE` lists the actions that need a stated purpose. It is
**empty in the code** (tests, a bare `uvicorn`); the SPA states the purpose
through its purpose bar ([below](#how-the-spa-sends-the-purpose)). When it is
set, a route is refused (428, one `denied` event) when *any* action its
operation may record is listed — a chat turn records `query` and `ai_call`, a
finalize `collect`, `ai_call`, `combine` and `modify`, a summary request
`query`, `modify` and `ai_call`. A job runs with the purpose of the request
that queued it, so the requirement is checked once, at the request.
`GET /config` names the actions in force as `require_purpose`.

A command-line run states its purpose in `AUDIT_CASE_REF` and `AUDIT_REASON`
under the same rules; a run that needs one and has none records a `denied`
event and exits with code 2, and a malformed value exits with code 2 before
any chain opens.

### How the SPA sends the purpose

The SPA's header holds a **purpose bar** (`@infra/ui`'s `PurposeBar`, scope
`docint`) where the user states a case reference, a reason, or both. It is
kept for the browser tab's session, per app (the apps share one origin behind
the gateway), and signing out clears it. While `GET /config` lists any
`require_purpose` actions the empty bar is a warning and a blank purpose
cannot be saved.

- **One way out.** Every request leaves through `apiFetch`
  (`frontend/src/api/client.ts`): `apiGet`/`apiPost`/`apiPut`/`apiPatch`/
  `apiDelete`, both SSE helpers, and the XHR upload, which goes through the
  same `withPurpose`. A stated purpose rides along as `X-Audit-Purpose`.
- **A 428 asks.** The refused request opens the bar's dialog ("This action
  needs a stated purpose") and is sent once more with the purpose saved there
  — to the same URL, owner included, with the same body. Cancelled, the
  refusal is the request's error ("A purpose must be stated first"), and the
  queries it left refused are fetched again as soon as a purpose is stated.
- **No browser-made requests to audited routes.** An `<a href>`, `<img src>`,
  `<iframe src>`, `window.open` or `EventSource` request cannot carry a
  header. Downloads, previews and thumbnails are therefore fetched through
  `apiFetch` and handed to the browser as a file (`frontend/src/api/files.ts`):
  saved under the name the server's `Content-Disposition` gives it, opened in
  a new tab from what was fetched (the report's HTML view, a preview's "Open
  in new tab"), or shown in an `<img>` or `<iframe>` from memory.

Every audited route the SPA calls, and how it carries the purpose.
"Required" means refused without a purpose under
`AUDIT_REQUIRE_PURPOSE=query,disclose`; the other routes carry a purpose when
one is stated, and it is recorded, but they are not refused without one.

| Route | Records | Required | Sent by |
|---|---|---|---|
| `POST /stream_query`, `POST /agent/chat/stream` | `query` + `ai_call` | yes | `streamSse` |
| `POST /search`, `GET /search/chunk` | `query` | yes | `apiPost`, `apiGet` |
| `GET` / `POST /summarize` | `query` (+ `modify`, `ai_call`) | yes | `apiGetOrNull`, `apiPost` |
| `GET /collections/ner/stats`, `/ner/sources`, `/ner/graph`, `GET /collections/hate-speech`, `GET /collections/documents` | `query` | yes | `apiGet` |
| `GET /collections/{name}/topics` and its `/evidence`, `/documents`, `/chunks` | `query` | yes | `apiGet`, `apiGetOrNull` |
| `GET /sessions/list`, `GET /sessions/{id}/history` | `query` | yes | `apiGet` |
| `GET /reports`, `GET /reports/unassigned`, `GET /reports/{id}` | `query` | yes | `apiGet` |
| `GET /collections/{name}/export/documents.csv`, `ner-sources.csv`, `hate-speech.csv` | `disclose` | yes | download button → `downloadFile` (was a link) |
| `GET /search/export.csv` | `disclose` | yes | download button → `downloadFile` (was a link) |
| `GET /sessions/{id}/sources.zip` | `disclose` | yes | download button → `downloadFile` (was a link) |
| `POST /collections/{name}/extracts` | `disclose` (the job) | yes | `apiPost` |
| `GET /collections/{name}/extracts/{id}/download` | `disclose` | yes | download button → `downloadFile` (was a link) |
| `GET /collections/{name}/sources/{id}/extract.zip` | `disclose` | yes | `fetchFile` (was a bare `fetch`) |
| `GET /reports/{id}/export.{md,zip,json}` | `disclose` | yes | export menu → `downloadFile` (was a link) |
| `GET /reports/{id}/export.html` | `disclose` | yes | export menu → `viewFile`, opened in a new tab (was a `target="_blank"` link) |
| `POST /reports/{id}/pdf` | `disclose` (the job) | yes | `apiPost` |
| `GET /reports/{id}/pdf` | `disclose` | yes | export menu and the finished render's own download → `downloadFile` (was a link and an anchor click) |
| `GET /sources/preview` | `query` | yes | preview dialog → `fetchFile`, framed or shown from memory; its "Open in new tab" opens what was fetched (was an `<iframe>`/`<img>` `src`, a bare `fetch` and a link) |
| `GET /sources/thumbnail` | `query` | yes | evidence thumbnails and the stored-image preview → `fetchFile`, cached per image (was an `<img src>`) |
| `POST /ingest/upload` | `collect` + `ai_call` | no | `streamUpload` (XHR, `setRequestHeader`; resent after a `428` with the same body) |
| `POST /ingest/finalize` | `collect` + `ai_call`, `combine`, `modify` | no | `apiPost` |
| `PUT` / `DELETE /sessions/{id}/scope`, `DELETE /sessions/{id}/topic` | `modify` | no | `apiPut`, `apiDelete` |
| `POST /reports`, `PATCH /reports/{id}`, `POST …/assign`, `…/collection-overview/refresh`, `…/items`, `…/items/batch`, `…/items/reorder`, `PATCH …/items/{id}` | `modify` | no | `apiPost`, `apiPatch` |
| `POST /collections/{name}/topics` | `modify` + `ai_call` | no | `apiPost` |
| `DELETE /collections/{name}`, `/sessions/{id}`, `/reports/{id}`, `/reports/{id}/items/{id}`, `/collections/{name}/extracts/{id}` | `delete` | no | `apiDelete` |
| `POST /translate` | `ai_call` | no | `apiPost` |
| `GET /collections/list?all=true`, `GET /collections/retention?all=true` | `admin`, for an admin only | no | `apiGet` |

The admin owner context (`?owner=`) is added by `apiFetch` too, so a download
an admin makes in another owner's namespace is recorded `on_behalf_of` that
owner like any other request — the session sources ZIP, which used to go out
without it, included. The routes the SPA reads that record nothing (`/config`,
`/version`, `/whoami`, the job status routes and stream, the PDF status,
extract listings, document counts and summary, the session's topic) carry the
purpose too, and ignore it.

## Routes

Every route of `docint/core/api.py`, grouped by its OpenAPI tag. "—" means
not recorded, with the reason. Queued work is recorded by the job when it
ends ([Background jobs](#background-jobs)).

**Meta**

| Route | Recorded as | Target and notes |
|---|---|---|
| `GET /config` | — | Deploy-time configuration, served without a principal — including `require_purpose`, the actions the running chain requires a stated purpose for. |
| `GET /config/ingest-defaults` | — | Configuration. |
| `GET /version` | — | Liveness. |
| `GET /health` | — | Readiness. |
| `GET /whoami` | — | The caller's own identity. |
| `GET /metrics` | — | Aggregate request counters, no principal. |

**Collections**

| Route | Recorded as | Target and notes |
|---|---|---|
| `GET /collections/list` | `admin` `collections.list_all` — only for an admin's `all=true` | `collection`, count = every other owner's collections; criteria `owners`. The caller's own names are not recorded. |
| `GET /collections/retention` | `admin` `collections.retention_all` — only for an admin's `all=true` | `collection`, count = rows; criteria `owners`. The caller's own clock is not recorded. |
| `POST /collections/select` | — | An ownership check returning the name it was given. |
| `DELETE /collections/{name}` | `delete` `collection.delete` | `collection` [name]; criteria `sessions_deleted`, `reports_deleted`. |

**Query**

| Route | Recorded as | Target and notes |
|---|---|---|
| `POST /query` | `query` `chat.query` + `ai_call` `chat.answer` | `chunk`: the sources returned. Criteria `question_sha256`, `session_id`, `retrieval_mode`, `retrieval_target`, `metadata_filters`, `scope_chunks`, `topic_id`, `reasoning`. The `ai_call` (`inference:chat`) carries the answer's hash and the same sources. |
| `POST /stream_query` | as `POST /query` | Written before the final frame ([Streaming routes](#streaming-routes)). |
| `POST /search` | `query` `search.fulltext` | `chunk`: the hits on the page. Criteria `keywords` (parsed), `field`, `metadata_filters`, `limit`, `page`. |
| `GET /search/chunk` | `query` `search.chunk` | `chunk` [id]. |
| `GET /search/export.csv` | `disclose` `export.search_csv`, `download:csv` | `chunk`: every exported row. Criteria `keywords`, `field`, `metadata_filters`, `session_id`, `marked`. |
| `POST /summarize` | cache hit: `query` `summary.read` (+ `ai_call` `summary.validate` when the validator ran); miss: queues a summary build | `chunk`: the summary's sources. A queued build records `modify` `summary.build` when it ends. |
| `GET /summarize` | `query` `summary.read` (+ `summary.validate`) | As above; a `204` miss returns nothing and records nothing. |
| `GET /collections/ner` | `query` `ner.sources_all` | `chunk`: every NER-bearing source. Criteria `refresh`. |
| `GET /collections/ner/stats` | `query` `ner.stats` | `entity`: the ranked entities. Criteria `top_k`, `min_mentions`, `entity_type`, `include_relations`, `entity_merge_mode`. |
| `GET /collections/ner/search` | `query` `ner.search` | `entity`: the matches. Criteria `q`, `entity_type`, `limit`, `entity_merge_mode`. |
| `GET /collections/ner/graph` | `query` `ner.graph` | `entity`: the graph's nodes. Criteria `top_k_nodes`, `min_edge_weight`, `entity_merge_mode`. |
| `GET /collections/ner/sources` | `query` `ner.sources` | `chunk`: the page's rows. Criteria `entity_key`, `entity_text`, `entity_type`, `entity_merge_mode`, `limit`, `page`. |
| `POST /collections/ner/warm` | — | Fills a cache and returns nothing. |
| `GET /collections/hate-speech` | `query` `hate_speech.list` | `chunk`: the findings returned. Criteria `category`, `min_confidence`, `limit`, `page`. |
| `POST /collections/entities/resolve` | `combine` `entities.resolve` + transfers | `entity`, count = entities touched; criteria `processed`, `minted`, `attached`, `skipped`. Transfers `entities.resolve.embed` / `.tiebreak` beneath it. |
| `GET /collections/documents` | `query` `documents.list` | `document`: the rows returned. Criteria `limit`, `page`. |
| `GET /collections/documents/count` | — | A count. |
| `GET /collections/documents/summary` | — | Totals and type breakdowns; no names. |
| `GET /collections/{name}/export/documents.csv` | `disclose` `export.documents_csv`, `download:csv` | `document`. |
| `GET /collections/{name}/export/entities.csv` | `disclose` `export.entities_csv`, `download:csv` | `entity`. Criteria `top_k`, `min_mentions`, `entity_type`, `entity_merge_mode`. |
| `GET /collections/{name}/export/ner-sources.csv` | `disclose` `export.ner_sources_csv`, `download:csv` | `chunk`. Criteria `entity_text`, `entity_type`, `entity_merge_mode`. |
| `GET /collections/{name}/export/hate-speech.csv` | `disclose` `export.hate_speech_csv`, `download:csv` | `chunk`. Criteria `category`, `min_confidence`. |
| `POST /collections/{name}/extracts` | queues an extract build | The build records `disclose` `extract.build`, `file:extract`, when the archive is stored. |
| `GET /collections/{name}/extracts` | — | The store's records: ids, counts and file names, no content. |
| `GET /collections/{name}/extracts/{extract_id}/download` | `disclose` `extract.download`, `download:zip` | `extract` [id]; criteria `source`, `counts`. |
| `DELETE /collections/{name}/extracts/{extract_id}` | `delete` `extract.delete` | `extract` [id]. |
| `GET /collections/{name}/sources/{source_id}/extract.{fmt}` | `disclose` `extract.source`, `download:<fmt>` | The units rendered ([unit targets](#how-an-event-is-shaped)); criteria `source`, `reference_number`, `operator`. |
| `POST /collections/{name}/topics` | queues a topics run | The run records `modify` `topics.<action>` when it ends. The gate checks `modify` and `ai_call`. |
| `GET /collections/{name}/topics` | `query` `topics.overview` | `topic`: the run's topics (labels, keywords and summaries come from passages). Criteria `run_id`. A `204` records nothing. |
| `GET /collections/{name}/topics/{topic_id}/evidence` | `query` `topics.evidence` | `chunk`. Criteria `topic_id`, `run_id`, `limit`. |
| `GET /collections/{name}/topics/{topic_id}/documents` | `query` `topics.documents` | `document`. Criteria `topic_id`, `run_id`, `limit`. |
| `GET /collections/{name}/topics/{topic_id}/chunks` | `query` `topics.chunks` | `chunk`. Criteria `topic_id`, `run_id`, `limit`, `page`. |

**Sessions**

| Route | Recorded as | Target and notes |
|---|---|---|
| `GET /sessions/list` | `query` `sessions.list` | `session`. A session's title is drawn from its first question. |
| `GET /sessions/{session_id}/history` | `query` `session.history` | `chunk`: every source the turns cite. Criteria `session_id`, `messages`. |
| `GET /sessions/{session_id}/sources.zip` | `disclose` `export.session_sources`, `download:zip` | `document`: the files written into the archive. Criteria `session_id`. |
| `DELETE /sessions/{session_id}` | `delete` `session.delete` | `session` [id]. |
| `PUT /sessions/{session_id}/scope` | `modify` `session.scope.set` | `chunk`: the pinned ids. Criteria `session_id`. |
| `DELETE /sessions/{session_id}/scope` | `modify` `session.scope.clear` | `session` [id]. |
| `GET /sessions/{session_id}/topic` | — | The session's own setting (run, topic id, label). |
| `DELETE /sessions/{session_id}/topic` | `modify` `session.topic.clear` | `session` [id]. |

**Reports**

| Route | Recorded as | Target and notes |
|---|---|---|
| `POST /reports` | `modify` `report.create` | `report` [id]; criteria `collection_overview` (whether the document overview was captured). |
| `GET /reports` | `query` `reports.list` | `report`: titles and case metadata are the investigator's text. |
| `GET /reports/unassigned` | `query` `reports.list_unassigned` | `report`. |
| `POST /reports/unassigned/{report_id}/assign` | `modify` `report.assign` | `report` [id]; criteria `collection_overview`. |
| `GET /reports/{report_id}` | `query` `report.get` | `report_item`: the report's items. Criteria `report_id`. |
| `PATCH /reports/{report_id}` | `modify` `report.update` | `report` [id]; criteria `fields` — the names of the fields set, never their values. |
| `POST /reports/{report_id}/collection-overview/refresh` | `modify` `report.overview_refresh` | `report` [id]. |
| `DELETE /reports/{report_id}` | `delete` `report.delete` | `report` [id]. |
| `POST /reports/{report_id}/items` | `modify` `report.item_add` | `report_item` [dedupe key]; criteria `report_id`, `artifact_type`. |
| `POST /reports/{report_id}/items/batch` | `modify` `report.items_add` | `report_item`: every requested key; criteria `report_id`, `added`, `updated`, `skipped`. |
| `PATCH /reports/{report_id}/items/{item_id}` | `modify` `report.item_note` | `report_item` [dedupe key]; criteria `report_id`, `item_id`, `cleared`. Never the note. |
| `DELETE /reports/{report_id}/items/{item_id}` | `delete` `report.item_remove` | `report_item` [item id]: the frozen snapshot is erased with it. Criteria `report_id`. |
| `POST /reports/{report_id}/items/reorder` | `modify` `report.items_reorder` | `report` [id]; criteria `items`. |
| `GET /reports/{report_id}/export.{md,html,json,zip,pdf}` | `disclose` `export.report_<fmt>`, `download:<fmt>` | `report_item`: the items rendered. Criteria `report_id`. The HTML is served inline and still counts as a download. |
| `POST /reports/{report_id}/pdf` | queues a PDF render | The render records `disclose` `report.pdf_render`, `file:report_pdf`, when the PDF is stored. |
| `GET /reports/{report_id}/pdf` | `disclose` `report.pdf_download`, `download:pdf` | `report` [id]; criteria `pages`, `rendered_at` (the stored PDF was rendered from the report as it stood then). |
| `GET /reports/{report_id}/pdf/status` | — | Job status and the stored file's record. |

**Agent**

| Route | Recorded as | Target and notes |
|---|---|---|
| `POST /agent/chat` | `query` `agent.query` + `ai_call` `agent.answer` | `chunk`: the sources. Criteria `question_sha256`, `session_id`, `reasoning`. A clarification instead of an answer records the query with `count: 0` and `clarification: true`, plus `ai_call` `agent.understand` when a chat model analysed the message. |
| `POST /agent/chat/stream` | as `POST /agent/chat` | Written before the `done` frame. |

**Ingestion**

| Route | Recorded as | Target and notes |
|---|---|---|
| `POST /ingest/upload` | `collect` `ingest.upload` | `document`: the hashes of the files saved. Criteria `preprocess`, `bytes`. Written before `upload_complete`; a batch cut short is recorded as `error` with the files that did land. |
| `GET /ingest/staged` | — | Status of the caller's own staged batch. |
| `POST /ingest/finalize` | queues the ingest run | The run records `collect` `ingest.run` and its stages. The gate checks `collect`, `ai_call`, `combine` and `modify`. |
| `GET /ingest/jobs/events` | — | Job status stream. |
| `GET /ingest/jobs` | — | Job status. |
| `GET /ingest/jobs/{job_id}` | — | Job status. |
| `POST /ingest/jobs/{job_id}/cancel` | — | Control only; the run itself records its end as `error` with `cancelled: true`. |
| `DELETE /ingest/jobs/{job_id}` | — | Dismisses a finished job from memory; no stored data. |
| `POST /ingest` | `collect` `ingest.run` + `ingest.social_link` + transfers | `document`: the files ingested; criteria `source: "data_dir"`, `hybrid`, the run's counts, `empty`. As the ingest job below. The gate checks `collect`, `ai_call` and `combine`. |

**Sources**

| Route | Recorded as | Target and notes |
|---|---|---|
| `GET /sources/preview` | `query` `source.preview` | `document` [file hash]. |
| `GET /sources/thumbnail` | `query` `source.thumbnail` | `image` [image id]; criteria `stored_thumbnail`. |

**Translate**

| Route | Recorded as | Target and notes |
|---|---|---|
| `POST /translate` | `ai_call` `translate.snippet`, `inference:chat` | `text`, count 1. Criteria `text_sha256`, `chars`, `target_lang`, `same_language`, `cached` (a cached translation sent nothing this time). `ai`: the model, and the translation's hash. `error` when the model failed. |

A new route is not finished until it is in this table, recorded or not.

## Background jobs

A job queued by a request carries that request's actor, purpose and request
id (`IngestJobState.audit`) and records with them when it ends; every event
names the job in `criteria.job_id`. A job no request started records as the
system.

| Job | Events |
|---|---|
| Ingest run (`POST /ingest/finalize`) | `collect` `ingest.run` — `document`: the hashes of the files the run worked on; criteria `source: "upload"`, the run's options (`hybrid`, `ner`, `hate_speech`, `summary`, `topics`), its counts (`files_processed`, `files_skipped`, `files_failed`, `docs`, `nodes`) and `empty`. Beneath it: `combine` `ingest.social_link` when the batch held a social export — `posting`: the uuids of the postings joined; criteria `dossiers`, `comments`, `media_linked` — and one `ai_call` `ingest.<stage>` per stage. Then, each as an operation of its own: `combine` `entities.resolve` when entity resolution follows the ingest, `modify` `summary.build` when the summary stage runs, `modify` `topics.build` or `topics.align` for the topics stage, each with its transfers beneath it. |
| Summary build (`POST /summarize` on a miss) | `modify` `summary.build` — `chunk`: the summary's sources; criteria `llm_calls`; transfers `summary.build.summary`. |
| Topics run (`POST /collections/{name}/topics`) | `modify` `topics.<action>` — `topic`, count = topics; criteria `run_id`, `added`, `assigned`, `deferred`; transfers `topics.<action>.fit` / `.labels`. A refusal is `error` with `refused: <code>`. |
| Extract (`POST /collections/{name}/extracts`) | `disclose` `extract.build`, `file:extract` — the units rendered; criteria `target`, `reference_number`, `operator`, the stored extract's id, `counts`. An empty collection stores nothing and records nothing. |
| Report PDF (`POST /reports/{report_id}/pdf`) | `disclose` `report.pdf_render`, `file:report_pdf` — `report_item`: the items rendered; criteria `report_id`, `pages`. |

A run that fails records `error` with what it had sent so far; a cancelled
run adds `cancelled: true`.

## Model calls

**A chat or agent turn** records one `ai_call` for its answer
(`chat.answer`, `agent.answer`, `inference:chat`), with the answer's hash and
the sources the model was given. It stands for all of the turn's model calls:
the turn's question, its rewrite and the retrieved passages also go to
inference for the query embedding, the rerank, the image lane's text
embedding and translation, response validation and the corrective retry. A
summary read's validation is `summary.validate`; a snippet translation is
`translate.snippet` itself.

**A batch operation** — an ingest run, entity resolution, a summary build, a
topics run — records one `ai_call` per stage it sent data from, not one per
call: `target` `{kind: <item>, count: <items sent>}`, `criteria.requests`,
`recipient`, `ai.model_alias`, and `parent_event_id` pointing at the
operation's event. The operation names it `<operation>.<stage>` — for an
ingest run `ingest.<stage>`.

| Stage | Recipient | Model alias | One item is | Counted in |
|---|---|---|---|---|
| `embed` | `inference:embed` | the embedding model | a text | `BudgetedOpenAIEmbedding` (`docint/utils/openai_cfg.py`) |
| `sparse` | `inference:sparse` | the sparse model | a text | `RemoteSparseEncoder.encode_texts` (`docint/core/rag.py`) |
| `ner` | `inference:ner` | `gliner` | a text | `docint/utils/ner_client.py` |
| `hate_speech` | `inference:chat` | the hate-speech model | a chunk or transcript window | `DocumentIngestionPipeline._complete_hate_speech` |
| `caption` | `inference:vision` | the vision model | an image | `OpenAIPipeline.call_vision` |
| `ocr` | `inference:ocr` | the OCR model | a page, region or image | `DocumentOcrEngine` (`docint/core/ocr/engine.py`) |
| `clip` | `inference:clip` | `clip` | an image | `RemoteCLIPBackend` (`docint/utils/clip_client.py`) |
| `transcribe` | `inference:asr` | `nextext` | a media file (sent to Nextext) | `NextextClient.process_media` |
| `summary` | `inference:chat` | the text model | a map-reduce prompt | `RAG.build_tree_summary` |
| `labels` | `inference:chat` | the text model | a topic | `RAG._topic_label_complete` |
| `fit` | `inference:topics` | `topics` | a text | `TopicsClient.fit` |
| `tiebreak` | `inference:chat` | the text model | an entity tie-break | `RAG.resolve_entities` |
| `translate` | `inference:chat` | `TRANSLATE_MODEL` | a snippet (cache misses only) | `_translate_cached` (`docint/utils/translate_client.py`) |

Counts are calls made — data sent — whether or not the model answered.

The counting is a meter (`docint/utils/audit_meter.py`): the clients report
every call with `count_transfer`, and the operation binds a `TransferMeter`
around its work (`audit.metered`). The meter travels in a context variable,
so it follows the work into anyio worker threads; the thread pools the work
fans out to submit through `in_context` so their tasks count for whoever
submitted them. A call made with no meter bound — a chat turn's retrieval —
is not counted there; the request records its model call itself.

**Upload-time preprocessing.** With `INGEST_PREPROCESS_ON_UPLOAD` on, an
upload starts each file's heavy stage (layout and OCR, caption, CLIP,
transcription) before any run exists. Those calls count into the
collection's *staged* meter, which the next ingest run of that collection
takes over and records beneath its own `collect` event — that run is what
consumes the results. A batch that is never finalized is not recorded beyond
its `ingest.upload` event, whose `preprocess: true` says its files were sent;
deleting the collection drops the staged counts, and a restart loses them.

## Streaming routes

- **`POST /stream_query`, `POST /agent/chat/stream`** — the answer streams
  token by token; the turn's events are written when it is complete, before
  the final frame that carries the sources. A write that fails ends
  `/stream_query` with the generic error frame (`generation_failed`) instead
  of that frame; `/agent/chat/stream`, which has no error frame, breaks off
  without its `done` frame. A stream that breaks off — an error, a client
  gone mid-answer — is recorded as `error` with `incomplete: true`. The
  answer tokens already streamed cannot be recalled; the sources never
  leave.
- **`POST /ingest/upload`** — the files are saved as the stream runs; the
  `collect` event is written once the batch is on disk, before
  `upload_complete`. A write that fails ends the stream with the
  `save_failed` error frame, without a file name; the files stay staged, and
  no run can be finalized unrecorded either.

## Retention sweep

Each collection the sweep deletes ([retention.md](retention.md)) records
`delete` `retention.purge` as the system, after the cascade:
`collection` [logical name]; criteria `owner`, `window_months`,
`last_activity_at`, `sessions_deleted`, `reports_deleted`. A purge that fails
records `error`; the sweep counts it failed and the next one retries.

## Command-line tools

Every command that processes collection data opens its own chain, with the
actor `{"user": <AUDIT_CLI_USER>, "via": "cli"}` and the purpose from
`AUDIT_CASE_REF` / `AUDIT_REASON`:

| Command | Recorded as |
|---|---|
| `resolve` | `combine` `entities.resolve` + `entities.resolve.<stage>` |
| `extract` | `disclose` `extract.build`, `file:extract` — the units written; recorded before the archive is |
| `topics build` / `discover` / `align` | `modify` `topics.<op>` + `topics.<op>.<stage>` |
| `topics status` | `query` `topics.overview` |
| `search-index` | `modify` `search_index.build` — `chunk`, count = points written; criteria the scan's counts (and the image companion's) |
| `search-index-all` | one chain for the run, one `search_index.build` per collection (`container`: the physical name) |
| `ingest` | `collect` `ingest.run` (`source: "data_dir"`) + `ingest.social_link` + `ingest.<stage>`, as the API's run |
| `query` | per query `query` `chat.query`, `ai_call` `chat.answer` and `disclose` `export.query_result` (its result file); `--summary`: `modify` `summary.build` with its transfers, then `disclose` `export.summary`; `--entities`: `disclose` `export.entities`; `--hate-speech`: `disclose` `export.hate_speech` — every file `file:query` |
| `query-eval` | per (query, mode) `query` `eval.query` (by question hash, with the mode) and `ai_call` `eval.answer`; the result file `disclose` `export.eval_results`, `file:query-eval` |
| `verify --repair` | `delete` `verify.repair` — `node`: the orphan nodes removed |
| `verify` (without `--repair`), `retention-report`, `load-models`, `docint` | — they read counts, node ids and the retention clock only, or serve the API |

The make targets run the commands inside the running backend with
`docker compose exec`, so their lines land in the backend's log
(`AUDIT_OUTPUT=container`): a one-off container can exit before the log
collector has found it, and its lines would be lost without a trace. The
backend must be up. They pass the operator's host account and the purpose:

```bash
make resolve COLLECTION=mydocs CASE_REF=case-0001 REASON="Merge duplicate entities"
make extract COLLECTION=mydocs CASE_REF=case-0001
```

`AUDIT_CLI_USER` is `$SUDO_USER`, else `id -un`, on the host; an empty
`CASE_REF`/`REASON` states nothing. `make retention-report` keeps its
one-off container: it reads only the retention clock.

## Configuration

| Variable | Read by | Default | Description |
|---|---|---|---|
| `AUDIT_OUTPUT` | `AuditConfig` | *unset* | `stdout` (or unset) for the process's own stdout; `container` for the container's main stdout under the container-wide lock. Compose sets `container` on the backend. Any other value stops startup. |
| `AUDIT_REQUIRE_PURPOSE` | `AuditConfig` | *empty* | Comma-separated actions that need a stated purpose (`collect`, `modify`, `query`, `disclose`, `combine`, `delete`, `admin`, `ai_call`, `auth`). Empty in this release. An unknown name stops startup. |
| `AUDIT_CLI_USER` | `AuditCliConfig` | *unset* | The operator's host account, for a command's actor. The make targets set it. |
| `AUDIT_CASE_REF` | `AuditCliConfig` | *unset* | A command's case reference. Per run (`make … CASE_REF=…`), never in `.env`. |
| `AUDIT_REASON` | `AuditCliConfig` | *unset* | A command's reason. Per run (`make … REASON=…`), never in `.env`. |

All five are in `docint/utils/env_cfg.py` (`load_audit_env`,
`load_audit_cli_env`); see also [configuration.md](configuration.md#data-protection-audit-log--auditconfig-auditcliconfig).
The identity settings that decide the actor are in
[configuration.md](configuration.md#identity-and-authentication--principalconfig).

## Limits

- **No correlation id on inference calls.** docint has no single HTTP client
  for inference — each lane builds its own (the OpenAI SDK, llama-index,
  `httpx`, `urllib`) — so the request's `X-Request-ID` is not forwarded to the
  router. A model call is tied to its operation by `parent_event_id` instead.
- **A chat turn's model calls are one event.** Its `ai_call` stands for the
  answer and the retrieval-side calls; the embedding, rerank and validation
  calls of a turn are not counted one by one.
- **Entity annotations made inside an ingest run** — NER tagging each chunk
  with the entities it mentions — are the run's `ingest.ner` transfer, not a
  `combine`: they annotate a chunk rather than join data sets. The joins that
  are (`ingest.social_link`, `entities.resolve`) are recorded as `combine`.
- **Already-streamed tokens** of an answer whose events then fail to write
  cannot be taken back.
- **A never-finalized upload's preprocessing** is recorded only as the
  upload's `preprocess: true`.

## For developers

- Open an operation with the route dependency
  `op: Operation = Depends(audited("<action>", "<operation>", also=(...)))`
  (or `audit.begin(...)` once the operation is known), set
  `op.container` / `op.criteria` from the request, and call `op.record(...)`
  once the data is gathered — after the work's own error handling, before
  the response. Model calls go beneath it with `op.child("ai_call", …)` or,
  for batch work, `with audit.metered(op) as meter:` and
  `op.transfers(meter)`.
- Work that swallows its own exceptions re-raises `audit.AUDIT_FAILURES`
  first.
- An ingest run is recorded by `docint/core/audit_ingest.py`, for the API's
  jobs and the `ingest` command alike, so a run reads the same however it
  was started.
- A new thread pool or thread whose work calls a model runs that work
  through `audit_meter.in_context(fn)`, once per submission or thread
  (`threading.Thread` and `ThreadPoolExecutor` do not copy the caller's
  context, so the calls would go uncounted); a new inference client reports
  each call with `count_transfer`.
- Tests: the autouse `audit_stream` fixture (`tests/conftest.py`) gives every
  test a chain on a `StringIO` — the lifespan's and the commands' chains land
  on it too — and runs `verify_chain` over everything written at teardown.
