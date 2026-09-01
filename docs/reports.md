# Curated reports (Report Builder)

The collection-wide exports in
[api-reference.md](api-reference.md#collection-csv-exports) are all-or-nothing.
For an investigative case you usually want a *curated* document — only the chat
answers, entity findings, and hate-speech findings that matter, with the
duplicate chunks a single entity drags in collapsed. The **Report Builder**
(the **Report** tab in the SPA) is that workflow.

## Building a report

- An **"+ Report"** control sits on every chat answer, entity finding, and
  hate-speech finding. Clicking it snapshots that one artifact into the active
  report (auto-creating an "Untitled report" the first time). Re-adding the same
  chunk is a no-op — findings are deduped by `chunk_id`.
- **"Add all" takes a whole section at once.** The Entities and Hate speech
  headers carry a second control beside their CSV download that adds *every*
  finding in the open section to the active report in one action — the whole
  hate-speech set, or every chunk mentioning the selected entity. It is meant
  for the case the per-artifact control makes tedious: a corpus with hundreds
  of relevant findings.
  - **"All" means every match, not the rows on screen.** The findings tables
    page in 50 rows at a time behind a "Load more", so the control walks the
    remaining pages itself before adding — a report built from whatever had
    been scrolled into view would be a silent sample.
  - Above 100 findings it asks first, and it refuses a set larger than the
    deployment's cap (`REPORT_BATCH_MAX_ITEMS`, default 2000, advertised by
    `GET /config`) rather than adding a partial one; narrow the selection —
    pick a more specific entity — and repeat. It walks one finding past the cap
    on purpose, so an oversize section is refused outright instead of being
    quietly trimmed to a capful.
  - Findings the report already holds are skipped, not duplicated, so the
    control is safe to press again after adding a few by hand. The outcome is
    stated beside it ("12 added, 3 already in report").
  - **A batch carries the translations you already made.** Translations are
    held app-wide for the session, keyed by the chunk's own text, so "Add all"
    freezes one into every snapshot whose text you translated — including rows
    scrolled out of view. It never translates anything itself: a finding you
    never translated is added untranslated. Use **Translate all** first (see
    below) — and if you added findings before translating them, run it and then
    press "Add all" again: the second pass backfills the translations into the
    snapshots already in the report.
- **Translate all** sits beside it in the same header and translates every
  finding the section's filter matches — again the whole set, not the rows
  paged in. It is a foreground run of one call per distinct chunk text: the
  button shows how far it has got, stays clickable to stop, and can be pressed
  again to pick up the remainder (anything already translated this session is
  never re-sent). Above 100 findings it asks first, since a large section takes
  minutes. If the translation model is unreachable the run stops after three
  consecutive failures rather than working through the whole section to report
  the same outage. Nothing is stored by translating: the translations live in
  the browser for the session, and only reach the server when a finding they
  belong to is added to a report.
- The **Report** view lists your reports and, for the active one, shows the
  picked artifacts grouped by type with per-item notes, reordering, and removal.
- Reports are **owner-scoped** and persisted server-side in the same SQLite
  store as chat sessions; each item is **snapshotted at add-time**, so later
  re-ingestion of the collection never changes a finished report.
- **Visual evidence travels with the reference.** When an added artifact points
  at an image — a chat answer citing a figure or photo, a finding on an image
  document, a video keyframe — the server freezes a small **thumbnail** into
  the snapshot (as a self-contained data URI, no live collection needed). The
  Report tab shows it under the item, and the Markdown/HTML/PDF exports render
  it inline beside the text ("Image evidence" / "Video keyframe"). Thumbnails
  are generated at ingestion; collections ingested before this shipped gain
  them on re-ingest of the same files (video keyframes need a fresh transcript
  run — see `docint/core/ingest/media_transcribe.py`).
  A chat answer's images render as a strip of **captioned figures** beneath
  its source list, each captioned with the number the answer cites (`[2]`), so
  a reader can tell which figure the text means; a finding shows its one figure
  inside the finding table.
- **A report belongs to one collection.** Switching the active collection
  releases the active report, so the next "+ Report" click starts one for the
  collection you are actually working in — a report's document overview and its
  evidence always describe the same collection.
- **Findings from pictures show the picture in the Analysis tab too.** An
  entity or hate-speech finding whose chunk was read out of an image (a
  screenshot, a photographed page, a video keyframe) renders the source image
  beside it; clicking it opens the full-size preview. That view is live rather
  than frozen — nothing is being exported there.
- **The frozen evidence is zoomable, and it is all inside the file.** Thumbnails
  are generated at 768px (~355 dpi at the size the exports print a figure), so a
  reader can zoom into a PDF page and still read what a poster or a slide says.
  Every export embeds them as data URIs — an exported HTML or PDF references
  nothing outside itself and keeps working after the collection is deleted — and
  clicking a figure in the Report tab enlarges it from those same bytes. Roughly
  26 KB per image inside a report snapshot; a 20-image report gains about half a
  megabyte.
  Collections ingested before this gain the larger thumbnails on the next
  re-ingest of the same files, which also removes a duplicate copy earlier
  versions stored alongside each point.

## Translated findings carry into the snapshot

Translating a finding before adding it to a report carries that translation
into the report's snapshot, so exports (Markdown, HTML, PDF, CSV, JSON) show
it as an additive labeled block or column next to the original — e.g.
"Machine translation (→ Deutsch)" when the active locale is German. The
translation overlay itself is described in
[ui-guide.md](ui-guide.md#on-demand-translation-of-source-content).

A finding added before it was translated is not stuck that way. Re-running
"Add all" over the section backfills the translation into the stored snapshot:
findings the report already holds are still skipped, except where the report's
copy has no translation and the new one does. The merge is strictly additive —
only the translation is written, and a translation already in the report is
never replaced, since it is the one you saw when you added the finding. The
outcome line counts those separately ("1 added, 12 translations added, 40
already in report").

## Exporting

Export a finished report in five formats (also available directly over HTTP —
see [api-reference.md](api-reference.md#reports)). The examples below assume
the API is reachable on port 8000; under Docker the backend publishes no host
port, so run them from inside the network or through the gateway:

```bash
curl -O "http://localhost:8000/reports/1/export.pdf"   # paginated case-file PDF (WeasyPrint)
curl -O "http://localhost:8000/reports/1/export.md"    # combined Markdown
curl    "http://localhost:8000/reports/1/export.html"  # self-contained HTML (also the PDF source)
curl -O "http://localhost:8000/reports/1/export.json"  # structured selection
curl -O "http://localhost:8000/reports/1/export.zip"   # per-type CSV bundle (reuses csv_stream.py schemas)
```

Every export leads with the **summaries**, then chat answers, entity findings,
and hate-speech findings; entity and hate-speech findings carry their source
**reference metadata** (network, author, timestamp, …) alongside the chunk. The
report name is the single headline and the subheader stays on one line
(collection · creation date · operator).

The PDF is rendered server-side by WeasyPrint into a real paginated document: a
running header carrying the case file (*Aktenzeichen*) in the upper-right
corner, page numbers and an "AI-generated — verify before further processing"
disclaimer in the footer of every page, findings kept whole across page breaks,
and Noto fonts for multi-script text. It needs WeasyPrint's native libraries,
which the backend image installs; if they are absent the `.pdf` route returns
503 while every other format keeps working.

## Further reading

- [api-reference.md](api-reference.md#reports) — every `/reports*` route.
- [ui-guide.md](ui-guide.md#report-srcroutesreporttsx) — the Report screen.
- [api-reference.md](api-reference.md#collection-csv-exports) — the
  all-or-nothing collection exports this workflow complements.
