# Data extracts

A **data extract** is the written form of what docint ingested: full
transcripts with timestamps and speakers, keyframe descriptions with the
second they were sampled from, image captions and the text read out of the
pixels, document text, and the social posting each artifact hangs off.

It exists because that content is otherwise only inside Qdrant payloads.
Text is easy enough to re-obtain from the original file; a transcript is not.
Before this, an analyst who needed one for a report had to push the media
through [Nextext](https://github.com/nos-tromo/Nextext) separately — a round
trip whose output knows nothing of the manifest → postings joins docint made
when it linked that clip to its post.

Extracts complement the [Report Builder](reports.md) rather than replacing
it. A report is hand-picked evidence; an extract is everything, unfiltered.

The bundle's PDF is written as **that report's appendix**, not as a second
document with its own conventions: it carries the report's footer disclaimer,
its case-file number in the running header, its operator line, a contents
block, and — the load-bearing part — the report's own provenance rows, so a
posting cited in the report and the same posting in the appendix name the
account identically. Every entry is numbered `A.1`, `A.2` …, which is how a
finding cites one.

## What you get

A ZIP laid out one folder per source:

```
mydocs-extract-20260102-0304/
  README.md                  index of every source, with its folder
  extract.md                 all sources in one document
  extract.pdf                the same, paginated (see the caps below)
  documents/report.pdf-a1b2c3d4/
    extract.md               the document's text, then its figures
    figures/report_page1_<id>.jpg   each figure as its stored thumbnail
  media/clip.mp4-f1e2d3c4/
    extract.md               transcript and keyframes together
    clip.transcript.txt      Nextext's own banner-fenced layout
    keyframes/clip_frame_000_01-12.jpg
  postings/examplenet/example-account/20260102-1111aaaa/
    extract.md               the post, its pictures, its clips
    post-clip.transcript.txt one per clip on the post
    media/Bild März.jpg
```

Every file is named after the file it came out of, as the export shipped it —
case, spaces and non-ASCII included. A content hash identified the bytes and
nothing else, which is no help to an analyst who knows the attachment by name.
Only what a path or a filesystem cannot take is removed (separators, control
characters, the set Windows rejects), a folder keeps its short hash suffix
because two sources may legitimately share a name, and two pictures colliding
inside one posting are disambiguated rather than one overwriting the other.
Three things have no name of their own and are named for their source: a
document figure (`<document>_page<N>_<id>`), a video keyframe
(`<clip>_frame_<index>_<mm-ss>`), and the thumbnail's extension, which is the
stored format rather than the original's — the bytes in the bundle are JPEG.

Postings are listed **newest first**. Documents, clips and loose pictures carry
no date of their own — an ingest date would order the export rather than the
evidence — so they keep their name order, and an undated posting sorts after
the dated ones rather than being buried among them.

Figures are the **stored 768px thumbnails**, never a re-fetch of the
original: an extract must be renderable from the index alone, with no source
volume mounted. The transcript layout in `transcript.txt` matches Nextext's
byte for byte, so a reader who has seen one recognises the other.

In the PDF a figure sits **beside** the words describing it rather than above
them, and a transcript is a table (time, speaker, text) rather than a run of
timestamped lines — one column per field means a reader can scan a speaker,
and a stamp in its own cell cannot be reordered into the words by an
Arabic line's bidi run. Right-to-left text is marked as such explicitly,
because WeasyPrint honours neither `dir="auto"` nor `unicode-bidi`.

The vision tagger's **keyword lists are not extracted**. They are retrieval
machinery, and beside a caption that already says what a picture shows they
read as noise; the caption and the text read out of the pixels are both kept.

Headings follow `RESPONSE_LANGUAGE`, like the report exports.

## Filing an appendix under a report

The case file and operator printed on the PDF come from the **active report**:
the SPA sends them with the build, and the extracts panel names the report the
next build will be filed under. With no active report both are simply absent,
for the reason the Report screen leaves its own operator empty rather than
guessing — an appendix naming a different operator than its report would be
worse than one naming none.

Over HTTP they are `reference_number` and `operator`, on the `POST` body and
as query parameters on the per-source download. The CLI takes
`--reference-number` and `--operator`, which is how an offline build gets them
with no report to inherit from.

## Building one

**From the SPA.** The Inspector has an *Data extracts* panel: **Build
extract** queues one for the active collection, progress appears in the
panel, and finished bundles are listed underneath with a download link. Each
document row also carries a download action for that source alone.

**Over HTTP.** See [api-reference.md](api-reference.md):

| Route | What it does |
|---|---|
| `POST /collections/{name}/extracts` | Queue a build (202 + `job_id`, 409 while one is in flight). Body may carry `{"target": "<id>"}`, `reference_number` and `operator`. |
| `GET /collections/{name}/extracts` | List stored bundles, newest first. |
| `GET /collections/{name}/extracts/{id}/download` | Download one bundle. |
| `DELETE /collections/{name}/extracts/{id}` | Delete one bundle. |
| `GET /collections/{name}/sources/{source_id}/extract.{md,pdf,zip}` | Render one source immediately. Takes `reference_number` and `operator` as query parameters. |

A collection build is a background job (`kind="extract"`) sharing the
owner-multiplexed stream at `GET /ingest/jobs/events`, framed as
`extract_started` / `extract_progress` / `extract_completed`. The terminal
frame carries the stored artifact.

**From the CLI**, on a host with no HTTP access to the backend:

```bash
make extract                      # prompts for the collection
make extract COLLECTION=mydocs
uv run extract mydocs --target a1b2c3d4 --no-pdf --out ./out
uv run extract mydocs --reference-number AZ-12/26 --operator "A. Analyst"
```

The CLI writes into `RESULTS_PATH` by default and reads Qdrant only — no
inference, so it is safe on an airgapped host.

## Addressing one source

`source_id` is whichever identity you have: a document's file hash, a media
file's content hash, a standalone image's id, or a posting uuid. The
Inspector's document table shows the file hash.

One shape is not small: a **postings table**'s file hash expands to every
post recorded in it. Above `EXTRACT_SYNC_MAX_UNITS` (default 50) the
synchronous route answers **413** rather than rendering for minutes on the
request; the SPA turns that into a targeted background build, so the same
click still gets you the bundle.

## Limits and retention

| Variable | Default | Effect |
|---|---|---|
| `EXTRACT_DIR` | `~/docint/extracts` | Where bundles are stored. Compose pins it onto the `pipeline-storage` volume. |
| `EXTRACT_RETENTION_DAYS` | `7` | Age at which a stored bundle is pruned. |
| `EXTRACT_MAX_PER_COLLECTION` | `5` | Bundles kept per collection. |
| `EXTRACT_PDF_MAX_UNITS` | `200` | Above this the combined PDF is skipped. |
| `EXTRACT_PDF_MAX_FIGURES` | `400` | Same, counted in figures. |
| `EXTRACT_SYNC_MAX_UNITS` | `50` | Units a per-source download may render inline. |
| `DOCINT_EXTRACT_CONCURRENCY` | `1` | Concurrent builds. |

The PDF caps are about memory: WeasyPrint holds the whole document plus every
decoded image resident, and a figure-heavy collection is a multi-gigabyte
render. When a cap trips, the Markdown files and the figures are still
complete and the README says why the PDF is missing — the bundle is never
silently short.

Stored bundles share their collection's lifecycle: deleting a collection
deletes its extracts, like its `_images` and `_entities` companions.

## What an extract does not do

- **It does not re-read anything.** No inference, no OCR, no transcription —
  only what ingestion already stored. A clip ingested before keyframe
  timestamps existed extracts without them; see
  [migrations.md](migrations.md).
- **It does not rename what was never named.** A social keyframe ingested
  before the clip's name was stamped on it has no file name to show, and a
  re-ingest cannot add one to its transcript segments (the pipeline skips a
  file hash it already holds), so such a collection has to be ingested afresh
  to gain them. Segments naming the transient `<clip>.nextext.jsonl` the
  transcript was parsed from are the one case that *is* recovered on read: the
  suffix is stripped, because that file is deleted during the ingest and
  naming it sends a reader after something that never existed.
- **It does not read an account out of thin air.** A chat-style export (the
  `messages` schema) carries no account-id or handle column, so the handle is
  recovered from the permalink's own path — and only for hosts where that path
  segment *is* the account. The numeric id in such a URL identifies the
  posting, not the account, and is never printed as one.
- **It does not re-order a document it cannot order.** Chunks read in page,
  then character-offset order. A collection that stamped neither is emitted
  in storage order and says so in the output, rather than passing that off as
  the document's own reading order.
- **It is not a durable archive.** Bundles are pruned. Download what you need.
