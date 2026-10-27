# Topics

Topics are docint's scalable collection clustering. A remote service fits a
topic model to a deterministic sample of a collection's chunks. docint then
places every chunk of the collection: a chunk the fit put in a topic keeps it,
and any other joins its nearest topic if it clears that topic's admission
floor, or stays unassigned. The topics, their centres and their
floors form a **run**, stored on disk; a chunk's membership is two payload keys
on its Qdrant point.

The fit and the labels are the only model calls. Sampling, assignment and every
count are docint's own, so whatever the collection's size, a build costs one
fit of at most `TOPICS_FIT_MAX_POINTS` chunks, a label request per topic, and
payload writes.

## Runs

A collection has at most one **active** run, named by `active.json` in its
store directory (see [The store and its lock](#the-store-and-its-lock)). Three
actions work on it, each as a background job:

| Action | What it does | Refused when |
|---|---|---|
| `build` | Samples the collection, fits a new run, assigns every eligible chunk to it, activates it and deletes every other run. | The sample is too small (`too_few_chunks`), or the collection was deleted and created again while the build ran (`collection_changed`). |
| `discover` | Fits the active run's unassigned chunks and appends the topics that are new. Existing topics are untouched. | There is no active run (`no_run`), the dense vectors changed size since the build (`dimension_changed`), too few chunks are unassigned (`too_few_unassigned`), or the collection was deleted and created again while the discover ran (`collection_changed`). |
| `align` | Assigns every eligible chunk the active run has not placed yet: chunks ingested since, or left behind by an interrupted job. | `no_run`, `dimension_changed` or `collection_changed`, as for discover. |

A refusal is not a failure: the job completes with `empty: true` and a warning
carrying the code (see [Jobs and events](#jobs-and-events)).

**Eligible** chunks are the points of the main collection except coarse parent
chunks (`not_coarse_condition()`). A job never reads or writes the `_images`
companion. A point without a dense vector is written as unassigned, so it
cannot stay pending forever.

### Build

1. **Sample.** Every eligible point id is ranked by the SHA-256 of
   `<TOPICS_SEED>:<point id>`, and the ids with the smallest hashes are read,
   at most `TOPICS_FIT_MAX_POINTS` of them and never more than the service
   accepts at the collection's vector size (see
   [The fit service](#the-fit-service)).
2. **Prepare.** A chunk's text is its `search_text`, or its node text where that
   is blank, with control characters removed and whitespace collapsed to single
   spaces. It is then cut to `200,000,000 ÷ n` characters for the `n` chunks
   ranked into the sample, kept between 2,000 and 8,000. A fit of up to 25,000
   chunks therefore sees whole 1,024-token chunks, and the largest fit sends no
   more text in all than 100,000 chunks of 2,000 characters. A chunk whose
   prepared text is shorter than `TOPICS_MIN_TEXT_CHARS` leaves the sample.
   Points with an identical vector are fitted once: the first by point id
   represents the others.
3. **Size.** The minimum topic size is `TOPICS_MIN_TOPIC_SIZE`, or, at `0`,
   `round(√n / 3)` between 2 and 10 for a sample of `n` chunks, and from 813
   chunks on `n // 1000` between 10 and 100. A sample smaller than
   `max(8, 2 × minimum topic size)` is refused: 8 is the fewest points the
   service fits. The refusal says how many chunks the sample held and how many
   a fit needs.
4. **Fit.** The sample is sent to the service. A fit that finds no topic leaves
   the collection's current run as it was, with the warning `none_found`.
5. **Topics.** A topic's centre is the normalised mean of its fit members' unit
   vectors, stored at float32 precision, and its floor is set from the same
   members (see [Assignment](#assignment)). Its keywords are the service's:
   single words, at most 10 per topic, each found in at least two texts, stop
   words left out (see [The fit service](#the-fit-service)). With
   `TOPICS_LABELS` on, the chat model (`TEXT_MODEL`) writes each topic a label
   of at most 80 characters and a one-sentence description, from its keywords
   and the first 800 characters of its five fit members nearest the centre,
   four topics at a time. The label names what the passages share, never the
   title or author of the work they come from. The prompt is
   `docint/utils/prompts/{en,de}/topic_label.txt`, and the reply is constrained
   by a JSON schema while the endpoint accepts one. A topic whose label cannot
   be written is named by its first four keywords.
6. **Assign.** If the collection's ownership row still has the creation time
   it had when the build started, the run is saved, then every eligible chunk
   is written (see [Assignment](#assignment)).
7. **Activate.** If the creation time is still the same, the run becomes
   active and every other run directory is deleted. A last align pass then
   places the chunks ingested while the build ran.

A build that fails or is cancelled before it activates deletes its run, and the
previous run stays active. The chunks it had already written name the deleted
run, so they count as pending for the active one until the next align places
them again.

### Discover

Discover works on the active run's **unassigned pool**: the eligible chunks
whose `docint_topic` is `<run_id>:-1`.

1. Pending chunks are aligned first. Then the pool is aligned again against the
   current topics, which places chunks an interrupted earlier discover left
   behind; in the steady state it writes nothing.
2. The pool is sampled, prepared and fitted like a build's collection, except
   that at `TOPICS_MIN_TOPIC_SIZE=0` its minimum topic size is never below the
   build's, counted up to 10, so a small pool cannot add topics finer than the
   build made. A pool too small to fit is refused with `too_few_unassigned`.
3. A pool sample member stays in its pool topic only where that topic is its
   nearest. Each pool topic gets a provisional centre from all its members,
   and the member's cosine to it, computed as in alignment, must be the
   highest among all centres, existing and provisional; a tie goes to an
   existing topic, or else to the lowest pool-fit id. A pool topic left with
   fewer members than the minimum topic size is dropped, and each remaining
   pool topic's centre, floor and fit size come from the members that stayed.
   The pool also holds chunks of existing topics that fell under their floors,
   and the fit may put them in a pool topic; one nearer an existing topic is
   set aside, so it neither joins the new topic nor lowers its floor.
4. A pool topic is kept only when the cosine of its centre to every existing
   centre is below `TOPICS_MERGE_MIN_SIMILARITY`. The kept topics are labelled
   and appended with the ids `max + 1`, `max + 2`, … in the order of their
   pool-fit ids. When none is kept, the job ends with the warning `none_new`.
5. If the collection's ownership row still has the creation time it had when
   the discover started, the extended run is saved and the pool is placed
   against it: the members that stayed in a kept topic take it, and so do
   their exact-duplicate copies. Every other pool chunk, a member set aside
   included, is aligned against all the run's topics, old and new; one that is
   not admitted stays unassigned and is not rewritten.
6. If the creation time has changed by then, the discover ends refused with
   `collection_changed`; the chunks it wrote stay.

Existing topic ids, centres and floors never change, and the run keeps its id.

### Align

Align writes every **pending** chunk — every eligible chunk whose
`docint_topic_run` is not the active run's id — through the alignment rule
below, in up to five forward passes. The ingest stage runs the same pass at the
end of an ingest job (see [The ingest stage](#the-ingest-stage)).

## Assignment

Every eligible chunk gets exactly one value:

- a sample member that the fit placed in a topic keeps that topic (in a
  discover, only where that topic is its nearest; see [Discover](#discover));
- a copy dropped from the sample as an exact duplicate takes its
  representative's topic;
- every other chunk is **aligned**: the fit's outliers, a discover's members
  set aside, the chunks the sample did not reach or that were too short to
  join it, and every chunk ingested later. Its L2-normalised vector is
  compared by cosine, in float64, with every unit centre; the highest wins, a
  tie going to the lowest topic id, and the chunk is admitted only when that
  cosine reaches the topic's floor. Otherwise it is unassigned (`-1`).

A topic's **floor** is the `TOPICS_ADMIT_PERCENTILE`-th percentile (default 5)
of its fit members' cosines to its centre; a discovered topic counts only the
members that stayed in it. At the default, an aligned chunk joins its nearest
topic when it is at least as close to that centre as all but the farthest 5% of
the topic's own fit members. A higher percentile admits fewer chunks.

## Payload keys and indexes

| Key | Value |
|---|---|
| `docint_topic` | `"<run_id>:<topic_id>"`, or `"<run_id>:-1"` for an unassigned chunk |
| `docint_topic_run` | `"<run_id>"` |

Both are written **payload-only**: `batch_update_points` with one
`SetPayloadOperation` per topic per page of 256 points, `wait=True`, retried on
transient errors. An assignment never touches a point's vectors, its text, its
`_node_content` or its full-text index. Points are selected by an id filter
(`HasIdCondition`) rather than a list of points, so a point deleted during a
pass is skipped instead of failing the batch.

`docint_topic`, `docint_topic_run` and `file_hash` carry KEYWORD indexes, and an
index of the wrong kind is replaced. Every job ensures them before it reads the
collection, and a process ensures them once per collection before its first
topics read: a Qdrant server needs the index to facet on a key (local mode
needs none). Every count is exact:
per-topic sizes and per-document counts are facets with `exact=True`, totals are
exact counts.

A topic is an ordinary metadata filter. The rule
`{"field": "docint_topic", "operator": "eq", "value": "<run_id>:<topic_id>"}`
on `/query`, `/stream_query` or `/search` narrows an answer or a search to that
topic's chunks; it is the rule the SPA's topic focus sends. Image points in the
`_images` companion carry no topic keys, so such a rule selects text chunks
only.

## The store and its lock

```text
{TOPICS_DIR}/
  <collection dir>/
    .lock
    active.json        {"run_id": "<run_id>"}
    <run_id>/
      centres.npy      float32, one row per topic, loaded without pickle
      run.json         the run's settings, provenance and topics
```

- **The collection directory** is the physical collection name when it fully
  matches `[A-Za-z0-9._-]{1,128}`, and `~` followed by the name's SHA-256
  otherwise, as the extract store names its directories. Every path is checked
  to resolve under `TOPICS_DIR`.
- **Run ids** are `YYYYMMDD-HHMMSS-<6 hex>`, the time in UTC, minted once a
  build's topics are fitted and labelled, just before the run is first saved.
  An id is checked against that pattern before it joins a path.
- **Writes** are atomic (written beside the target, then renamed) and readable
  by the process user only. A run is saved centres first and `run.json` last; a
  reader that meets new centres beside the old record reads once more.
- **`run.json`** records the settings the build ran with (`params`), the dense
  vector's name and dimension, the embedding model, the stop-word list's count
  and SHA-256, the ownership row's creation time (`owner_created_at`),
  `created_at` and `updated_at`, an entry for the build's fit and one for each
  discover that added topics (its kind, time, sample size, the characters it
  kept of each text (`text_chars`), the count and SHA-256 of the stop words it
  sent and how many of them its own texts added (`stop_words`), topic and
  outlier counts, and the versions, runtime and timings the service reported; a
  discover's entry adds how many topics it kept; a discover that finds nothing
  new adds none), and the topics:
  id, label, description, keywords, origin (`build` or `discover`), when it was
  added, its fit size, its floor, and its first and last seen.
- **The lock** is an exclusive `fcntl.flock` on `.lock`. Every job takes it
  before it reads anything and holds it to the end, the wait on the fit
  included. It works across processes, so a CLI run and an API job never write
  one collection at once. A job that finds it held waits, polling once a second
  and reporting the `waiting` stage; the ingest stage does not wait.
- **A recreated collection.** A collection deleted and created again under the
  same name has a new ownership row. A run whose `owner_created_at` differs
  from it counts as absent: reads answer as if there were no run, and the next
  job on the collection, the ingest stage included, deletes it.
- **Deletion.** Deleting a collection, by hand or by retention, removes its
  whole directory. A build or discover still running on it is refused by the
  creation-time check (`collection_changed`) between its fit and the writes
  that follow, and again before a build activates or a discover completes; a
  collection replaced while those writes run still receives the rest of them,
  which stay.

## The fit service

docint ships no topic model. The fit runs in vllm-service's `topics` service at
`{TOPICS_API_BASE}/topics/fit`: through the router on the full stack, or
directly on the `topics-only` shape. How the service fits into vllm-service is
described in that repository's `docs/api-reference.md` § Topics and
`docs/deployment-shapes.md` § Topics-only. docint relies on the contract below.

**What docint sends.** One multipart request per fit, every part a file part:

| Part | Filename | Content |
|---|---|---|
| `meta` | `meta.json` | `n`, `dim`, `seed`, `min_topic_size`, `nr_topics`, `top_n_words`, `min_df` and `stop_words`, every key present |
| `texts` | `texts.json` | A UTF-8 JSON array of `n` strings, non-ASCII characters left unescaped |
| `vectors` | `vectors.f32` | `n × dim` little-endian float32, row-major |

It carries `Authorization: Bearer <TOPICS_API_KEY>` when the key is set.
`top_n_words` is fixed at 10 and `min_df` at 2. `nr_topics` is `0`, or
`TOPICS_MAX_TOPICS + 1`, because the service counts its outlier class as a
topic. `stop_words` are the NLTK English and German lists that llama-index-core
ships inside its wheel, merged, so nothing is fetched at runtime, plus a short
supplement of words those lists lack (auxiliaries such as `would` and `wurde`,
`said`, honorifics such as `mr`, citation fragments such as `et al` and `doi`).
Each fit then adds the words its own texts call for, found the way the
service's vectorizer finds words (lowercased, two or more word characters):
every number found in at least two texts, and every word found in more than 80%
of the texts (`max_df`), which cannot tell one topic from another. When the
service's limits leave room for only some, the most widespread go first.

**The limits it obeys.** docint keeps every request within these, and checks
them before anything is sent:

- `seed` in `0 ≤ seed < 2³²`, `min_topic_size` at least 2, `nr_topics` 0 or at
  least 2, `top_n_words` 1 to 30, `min_df` at least 1;
- at most 10,000 stop words, each at most 64 characters (a longer word is left
  out of the list);
- `dim` from 1 to 8,192, and every vector value finite;
- `n` at least `max(8, 2 × min_topic_size)`: a smaller sample is refused
  before it is sent (see [Build](#build));
- `n` at most `min(150000, 150000 × 1024 // dim)`: the service's default
  capacity of 150,000 points of 1,024 values each. docint samples no more than
  that;
- `meta.json` at most 1 MiB, and `texts.json` at most
  `min(n × 16384 + 65536, 536870912)` bytes. When the prepared texts are
  larger, each is cut to a shared byte budget.

**What it expects back.** `labels` (one topic id per input, `-1` for an
outlier), `outliers`, `topics` (each `id`, `size` and `keywords` as
`[word, score]` pairs), `versions`, `runtime` and `timings`. docint checks the
answer: one label per input, unique non-negative topic ids, labels that name a
listed topic, sizes and an outlier count that agree with the labels, and finite
scores. Topic ids arrive in ascending order, not by size, and docint orders
topics by its own counts. An answer with every label `-1` and no topics is
valid: a build then warns `none_found`, a discover `none_new`.

**One fit at a time.** The service runs one fit at a time and answers any other
fit as busy. docint sends a fit that was answered busy again, ten attempts in
all, each after the wait the service asks for (30 s when it names none), and
then fails the job with `topics_busy`. A request waits up to `TOPICS_TIMEOUT`
(default 3600 s) for its answer, which arrives only when the fit is done. The
call runs on a background thread while the job reports the `fitting` stage
every 5 s. A job cancelled during the fit abandons the call; the service
finishes that fit before it takes another.

**Endpoint faults.** When the service cannot produce a fit, the job pushes one
`warning` with a static code and message, then fails with `topics_failed`:

| Code | What it means |
|---|---|
| `topics_unreachable` | The service could not be reached, the upload timed out (HTTP 408), or it did not answer within `TOPICS_TIMEOUT`. |
| `topics_busy` | The service stayed busy with another fit through every retry. |
| `topics_too_large` | The service refused a sample this large. Lower `TOPICS_FIT_MAX_POINTS`. |
| `topics_rejected` | The service refused the request otherwise, for example for a missing or wrong `TOPICS_API_KEY` on the router, or a `TOPICS_API_BASE` that names no fit service. |
| `topics_bad_response` | The service answered with a body that breaks the contract above. |
| `topics_failed` | Any other fault of the service. |

The warning names no endpoint. The backend's log line for the failed job
carries the full error, which names the fit URL (without credentials) and, for
a configuration fault, the setting to change; the CLI prints the same message.
The router's HTTP 400 `no_db_connection` authentication error names
`TOPICS_API_KEY` and `TOPICS_API_BASE` as settings to check, without echoing
the rejected key.

## Determinism

- **The sample** depends only on the collection's eligible points (their ids,
  texts and vectors) and the settings. Points are ranked by a hash of the seed
  and the point id, not by scroll order, and the sample is sent sorted by point
  id.
- **The fit** is deterministic on the service's side: the same input bytes and
  settings give the same labels and keywords on the same image and CPU class.
- **The assignment** computes against centres held at the float32 values the
  store saves, so a chunk aligned during a build lands where it would land
  after the run is read back. Ties go to the lowest topic id.

Two builds of an unchanged collection with the same settings, against the same
service image and CPU class, therefore produce the same topics, keywords and
assignment, under new run ids: the id carries a random suffix. Labels are
written by the chat model and repeat only as far as its endpoint does; nothing
else depends on them. `run.json` records what a result depends on: the
settings, the dense vector's name and size, the embedding model, the SHA-256 of
the bundled stop-word list and of each fit's full list, which follows from the
sample's texts, and the library versions and runtime the service reported.

## Jobs and events

A topics job runs from any of three places:

- the Analysis screen's **Topics** tab (see [In the app](#in-the-app));
- `POST /collections/{name}/topics` with `{"action": "build" | "discover" | "align"}`
  (see [api-reference.md](api-reference.md#topics));
- `uv run topics <op> <collection>` or `make topics OP=<op> COLLECTION=<name>`,
  which also takes `status` (see
  [cli-reference.md](cli-reference.md#topics--topic-runs)).

Over HTTP it is a `kind="topics"` job on the shared registry and the
owner-multiplexed stream at `GET /ingest/jobs/events`, with its own concurrency
bound (`DOCINT_TOPICS_CONCURRENCY`, default 1), so it never takes an ingest,
summary or extract slot. One topics job per collection is in flight at a time,
and a second is refused with 409; an ingest of the same collection may run
beside it. `POST /ingest/jobs/{job_id}/cancel` stops it at its next progress
frame.

| Frame | Payload |
|---|---|
| `topics_started` | `collection`, `action` |
| `topics_progress` | `message`, `stage`, `done`, `total` (`0` when the stage has no total) |
| `warning` | `message`, `code`; `found` and `needed` for `too_few_chunks` and `too_few_unassigned` |
| `topics_completed` | `collection`, `empty`, `duration_ms`; `artifact` (`action`, `run_id`, `topics`, `added`, `assigned`, `deferred`) once the job did something |
| `topics_cancelled` | `collection`, `duration_ms` |
| `error` | `message` (`Topics run failed.`), `code` (`topics_failed`), `duration_ms` |

Every job opens with `starting`, and `waiting` appears only while another job
holds the lock. A build then reports `scanning`, `sampling`, `fitting`,
`labelling`, `assigning` and `aligning`, in that order. A discover reports
`aligning` first, for the pending chunks and the unassigned pool, then
`scanning` through `labelling` as a build does, `assigning` for the pool's
re-align, and `aligning` again; an align reports `aligning` alone. A stage with
nothing to do may send no frame, and `labelling` sends none while
`TOPICS_LABELS` is off. A frame's `message` is static English, such as
`Aligning topics: 1200/4000 chunks processed`; the SPA writes its own copy
from `stage`, `done` and `total`. In the artifact, `topics` counts the run's
topics after the job, `added` the topics it added, and `assigned` the chunks
it placed in a topic (chunks written as unassigned are not counted).

Warning codes:

| Code | The job | Meaning |
|---|---|---|
| `too_few_chunks` | completes, `empty` | The collection has too few chunks with enough text to build topics; `found` says how many the sample held, `needed` how many a fit needs. |
| `too_few_unassigned` | completes, `empty` | Too few unassigned chunks to look for new topics; `found` and `needed` as for `too_few_chunks`. |
| `no_run` | completes, `empty` | There is no active run to discover on or align. |
| `dimension_changed` | completes, `empty` | The dense vectors changed size since the build; rebuild. |
| `collection_changed` | completes, `empty` | The collection was replaced while a build, discover or align ran. |
| `none_found` | completes, `empty` | The fit found no topics; the current run stays. |
| `none_new` | completes, `empty` | No pool topic was new; the unassigned chunks stay unassigned. |
| `topics_unreachable` … `topics_failed` | fails | An endpoint fault (see [The fit service](#the-fit-service)). |

A refusal (the first five) completes without an `artifact`; `none_found` and
`none_new` complete with one. No kind emits a `*_failed` event: a failure is the
`error` frame, and its copy is static, so an exception's text never reaches a
client.

Each job leaves one log line, such as
`Topics build | run=… topics=… added=… assigned=… deferred=…`, and a refused
one `Topics build refused for '<collection>' | code=…`, with `found=… needed=…`
for a sample too small to fit. The fit adds
`Topics fit | n=… dim=… topics=… outliers=… fit_s=…`, and the labels
`Topic labels | written=… fallback=…`. These lines carry counts, codes and ids,
never chunk text, labels, keywords or vectors.

## The ingest stage

With `TOPICS_ENABLED` and `TOPICS_ALIGN_ON_INGEST` on (both default), every
ingest job runs an align as one of its last stages, after entity resolution
and before the collection summary: it places the collection's pending chunks,
the job's new ones among them, in the active run. It reports
`ingestion_progress` frames of the form
`Aligning topics: {done}/{total} chunks processed`, which the ingest card shows
as an **Aligning topics** bar. It never waits for the lock:

- **Another topics job holds the collection.** The stage is skipped with the
  warning `topics_deferred`, and the new chunks stay pending until the next
  align.
- **The stage fails.** The warning `topics_align_failed`; the ingest still
  completes.
- **No active run, or the dense vectors changed size.** The stage does
  nothing; the second case logs `code=dimension_changed`.

Its warnings show on the ingest job's card, like the ingest's own. A cancelled
ingest stops in this stage as in any other.

## In the app

The **Topics** tab, the chat's topic focus and the Inspector's **Topics**
column appear only while `GET /config` reports `topics_enabled`. Each screen is
described in [ui-guide.md](ui-guide.md).

- **The Topics tab** (Analysis) shows the active run's coverage, its topics
  largest first, and the actions that build, extend and align it, plus a CSV
  download. Which job outcome it shows, and how its CSV keeps a spreadsheet
  from reading a cell as a formula, are in
  [ui-guide.md](ui-guide.md#analysis-srcroutesanalysistsx).
- **The topic focus** narrows the chat and the search panel to one topic. A
  topic row's **Ask in chat** opens a new chat with it, and the chat's first
  turn stores it with the session (`topic_focus` on `/query` and
  `/stream_query`), so each chat keeps its own topic across reloads until it is
  removed or the chat is deleted; its states are in
  [ui-guide.md](ui-guide.md#chat-srcrouteschattsx). It is text-only: image
  points carry no topic keys, so a focused search returns no image hits and a
  focused answer has no image lane.
- **The Inspector's Topics column** shows each document's topics in the active
  run, from the documents listing's `topics` field.

## In agent turns

An agent turn on the `all` or `documents` target is offered a `topics` tool
while the collection's active run has topics, the turn is not scoped, its
filters compile for Qdrant, and the context window has room for the tool.
`topics()` lists the run's topics largest first, each with its label, three
keywords, size and share; `topics(topic=T<n>)` shows that topic's five passages
nearest its centre, numbered like a search's. A turn reads the listing once: a
second call gets a short note. Under the turn's filters the listing counts only
the chunks they select. The counts a listing showed reach the response
validator as topic checks, as keyword counts do. See
[retrieval-and-agents.md](retrieval-and-agents.md#topics).

## Configuration

| Variable | Default | Effect |
|---|---|---|
| `TOPICS_ENABLED` | `true` | Topics routes, jobs, the ingest stage, the agent tool and the SPA's topics screens. |
| `TOPICS_ALIGN_ON_INGEST` | `true` | Run the ingest stage. |
| `TOPICS_LABELS` | `true` | Have the chat model label each topic. |
| `TOPICS_FIT_MAX_POINTS` | `100000` | Most chunks one fit samples. |
| `TOPICS_MIN_TEXT_CHARS` | `40` | Shortest prepared text that joins a fit. |
| `TOPICS_SEED` | `42` | Sampling and fit seed. |
| `TOPICS_MIN_TOPIC_SIZE` | `0` | Smallest topic; `0` sizes it from the sample. |
| `TOPICS_MAX_TOPICS` | `0` | Topic-reduction target; `0` keeps every topic. |
| `TOPICS_ADMIT_PERCENTILE` | `5` | Percentile of a topic's member cosines that sets its floor. |
| `TOPICS_MERGE_MIN_SIMILARITY` | `0.7` | Centre cosine at which a discovered topic counts as a repeat. |
| `TOPICS_API_BASE` | `http://vllm-router:4000` | Fit service base URL; `topics-only`: `http://topics-only:8000`. |
| `TOPICS_API_KEY` | *unset* | Bearer key; the router requires one. |
| `TOPICS_TIMEOUT` | `3600` | Seconds one fit request may take. |
| `TOPICS_DIR` | `~/docint/topics` | Where runs are stored. |
| `DOCINT_TOPICS_CONCURRENCY` | `1` | Topics jobs at once, process-wide. |

Bounds and fallbacks are in
[configuration.md](configuration.md#topics--topicsconfig), and `TOPICS_DIR`,
with its compose value, in its Paths table.

## Limitations

- **A CLI run does not refresh the API's Inspector cache.** The Inspector's
  topic tallies come from the API process's cached documents listing, which a
  job run through the API drops when it writes. `uv run topics` and
  `make topics` run in a process of their own, so after one the column keeps
  its old tallies until the API process drops that cache for another reason,
  such as the collection's next ingest or delete.
- **A CLI rebuild sends no frames to open tabs.** An open chat keeps showing its
  topic focus as applied, and keeps sending the old run's value, until it reads
  the overview again; the Topics tab shows neither the run's progress nor its
  outcome.
- **First and last seen only widen.** Every job that places chunks in a topic
  widens its range to the earliest and latest `reference_metadata.timestamp`
  (or `reference_metadata.posting_timestamp`) among them, and nothing narrows
  it. A topic whose chunks carry neither has none.
- **A rebuild ends old topic filters.** A rebuild mints a new run id, writes
  every eligible chunk under it and deletes every other run, so a rule naming
  an older run's value matches nothing. Once it has read the new run, the SPA
  stops sending such a focus and says so; any other client has to read the
  overview again.
- **Unassigned chunks are counted, not listed.** Topic `-1` has no evidence,
  documents or chunks route (404). The overview counts the unassigned and the
  pending chunks.
