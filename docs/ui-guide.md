# UI guide

The user interface is a **React single-page app** (Vite + TypeScript) that
talks to the FastAPI backend over JSON and SSE. It lives in `frontend/`,
is built with pnpm, and in production is served by an nginx sidecar that
reverse-proxies API routes to the backend (see [deployment.md](deployment.md)).
In development it runs on the Vite dev server at `http://localhost:5173`,
which proxies API calls to the backend on `:8000`.

> The previous Streamlit UI has been removed. Everything below describes
> the React app under `frontend/src/`.

## Stack

| Concern | Choice |
|---|---|
| Build / dev server | Vite (`pnpm dev` → `:5173`, `pnpm build` → `dist/`) |
| Language | TypeScript (strict) + React |
| Routing | `react-router-dom` v7 (`src/routes/Router.tsx`) |
| Server state | TanStack Query (`src/api/queryClient.ts`) |
| Client state | Zustand (`src/stores/`) |
| Styling | Tailwind CSS v4 + `@infra/ui` semantic tokens (light default, OS-preference/`AppHeader`-toggled dark) |
| Tests | Vitest + Testing Library (happy-dom) |

`package.json` scripts: `dev`, `build` (`tsc -b && vite build`), `preview`,
`test` (`vitest run`), `test:watch`, `lint` (eslint), `typecheck`.

## Entry & layout

- `index.html` → `src/main.tsx` mounts `<App>` into `#root`.
- `src/App.tsx` wraps the tree in the TanStack Query provider and the
  React Router `BrowserRouter`, then renders `src/routes/Router.tsx`.
- `src/layout/Shell.tsx` is the persistent layout: `@infra/ui`'s `AppHeader`
  (federation portal link + light/dark/system theme toggle) on top, then a
  sidebar + main-area row below; `src/layout/Sidebar.tsx` holds the
  navigation, the collection selector, and the session list.

## Routes

| Path | Component | Screen |
|---|---|---|
| `/` | `src/routes/Dashboard.tsx` | Dashboard |
| `/chat`, `/chat/:sessionId` | `src/routes/Chat.tsx` | Chat |
| `/ingest` | `src/routes/Ingest.tsx` | Ingest |
| `/analysis` | `src/routes/Analysis.tsx` | Analysis |
| `/inspector` | `src/routes/Inspector.tsx` | Inspector |
| `/report` | `src/routes/Report.tsx` | Report |

### Dashboard (`src/routes/Dashboard.tsx`)

KPI cards (backend status, collection / document / session counts), a
top-entities chart (`src/components/dashboard/TopEntitiesChart.tsx`) with
top-k / min-mention filters, and a recent-sessions list. Hooks:
`useCollections`, `useDocumentsCount`, `useNerStats`, `useSessions`.

While collection retention is on ([retention.md](retention.md)), an
**Automatic deletion** card (`src/components/dashboard/RetentionCard.tsx`)
spells out the rule and lists every collection with its deletion date, soonest
first — the eight soonest, with the rest counted — flagging those due within
30 days. The collection in use is always freshly active, so the rows that
matter are the others, which is why the card sits here rather than beside the
active collection. In the sidebar, a line under the collection picker says how
many *other* collections are due within 30 days and links here; it is one line
rather than a marker on each option because the picker's panel sizes to its
longest label and a date suffix pushed it past the sidebar's edge. Both read
`GET /collections/retention` through `useCollectionsRetention`
(`src/hooks/useRetention.ts`), which asks nothing while `/config` reports
retention `off`. Dates are formatted by `src/lib/formatDate.ts` in UTC, the
day the server acts on.

### Chat (`src/routes/Chat.tsx`)

The primary surface. Streams the answer token-by-token from
`POST /stream_query` over SSE, loads/saves session history
(`/chat/:sessionId`), and builds metadata filters (MIME, date range,
hate-speech-only, custom field/operator/value rules) via
`src/components/chat/FilterBuilder.tsx`. Each exchange renders as a
`ChatTurn` with `Citation`s, a `ValidationBanner` for answer/source
mismatches, and an optional `GraphDebugPanel`. Cancellation uses an
`AbortController`.

The answer is Markdown: `grounded_qa.txt` on a plain turn, and
`agent_system.txt` on an agent turn, ask the model for lists, bold, tables and
topic headings where structure helps, and for plain paragraphs otherwise.
`components/common/MarkdownProse.tsx` renders it, shared with the Analysis
summary. A table scrolls inside its own box, citation markers such as `[3]` stay
plain text, and raw HTML in an answer is shown as text, never parsed.

The metadata-filter disclosure, the retrieval-mode toggle and the
**Answer from** control hold the right edge of the Chat header row
(`src/components/chat/ChatControls.tsx`): they
govern what any answer retrieves against, so they belong to the chat rather
than to the search panel, which owns only the query field and the hits. Each
hit is a tile that scopes the chat when clicked, and the panel's collapsed
rail is a bare toggle that tints while a scope is live. Icon-only controls draw
from `@infra/ui` or, for what no other app could use,
`src/components/common/icons.tsx`; never a text glyph, whose shape is
whatever font the viewer's browser happens to pick.

**Answer from** chooses which evidence the next answer may come from:
Everything (the default), Documents, or Images. It is one icon button that
steps through the three, drawn as stacked layers, a page, or a picture — the
setting has three values rather than two, but it is changed rarely, and a
labelled dropdown in a row of icon controls reads as a different kind of
control altogether. Hovering it names the current target in full ("Answer
from: Images"), the way the two toggles beside it name their states. The
choice matters because it changes what an answer is *made of* — under Images
the model is shown the stored pictures themselves, not only their
descriptions. Choosing Images
also grows the filter panel a group of its own (kind of imagery, clip or file
name, and a time range written the way a player shows it, `m:ss`); those
presets stay hidden under the other targets, where a keyframe-time rule would
match nothing and silently empty the answer. A visual turn that could not
load its pictures says so above the sources, the same way a skipped rerank
does.

The choice belongs to one chat. It is not remembered across reloads. New chat
returns it to Everything, even from a chat with no turns or a stateless one, as
does a collection switch away from a started chat and opening Chat with no
session; opening a saved session leaves the current choice as it is. A turn
answered from Documents or Images alone says so with a quiet chip beside
"Assistant" ("Answered from documents only" / "Answered from images only";
"Nur aus Dokumenten beantwortet" / "Nur aus Bildern beantwortet" in German),
and the chat's transcript download writes the same line after that answer. The
chip is drawn from the target stored with the turn, so it survives a reload.
Everything shows no chip, nor does a turn answered from a hand-picked scope
(the scope outranks the target) or one saved before the target was recorded.

**Ask in chat** on an Analysis topic opens a new chat with that topic as its
**topic focus**, shown as a chip at the inner end of the header's controls
(`src/components/chat/TopicFocusChip.tsx`): the topic's number and
label (`T3 · <label>`) over a line saying what the next turn does with it, and
a control that removes it. The focus adds one metadata rule, `docint_topic`
equal to `<run id>:<topic id>` ([topics.md](topics.md#payload-keys-and-indexes)), to
every search of the search panel, its CSV export included, and to chat turns
answered from Everything or Documents, which then answer from the topic's
chunks alone, without their wider sections. The chip has four states; when
several apply, the first wins:

- **stale**: the focused run is no longer the collection's active run, for
  example after a rebuild. Nothing is sent, since the old value matches no
  chunk;
- **scoped**: chunks are pinned, so an answer comes from them alone and ignores
  every filter, and the rule applies to searches only;
- **visual**: the Images target. Its turns get no topic rule, while searches
  keep it;
- **applied**: the chip is tinted, and the rule rides on searches and turns
  alike.

The focus belongs to the chat it was asked in. Until that chat's first turn it
is only a pick (`src/stores/topicFocus.ts`); the first turn sends it as
`topic_focus`, and the server keeps it on the session that turn creates. From
then on the chat's topic is the session's (`useChatTopicFocus`, read from
`GET /sessions/{id}/topic`), so reopening the chat after a reload, or in
another browser, brings it back, and every other chat keeps its own topic or
none. The chip's remove control takes it off the chat for good
(`DELETE /sessions/{id}/topic`); deleting the chat, or its collection, removes
it too. **New chat** starts without a topic.

Topic focus is text-only: image points carry no topic keys, so a focused search
returns no image hits and a focused answer has no image lane. A pick is keyed
by collection and owner, so it shows and applies only in the collection it was
picked in; an admin who switches to another owner's collection of the same name
does not carry it over. Until the chat's topic and the active run have been
read the chip is hidden, and searches and turns both wait for those reads:
**Send** stays disabled, Enter sends nothing, and once something is typed a
note under the composer says the chat's topic is being read. A read that fails
holds nothing: the search or turn goes out without the rule, and the chip stays
hidden. A topics job that ends
while the chat is open refetches the run, so a rebuild finished from the app
marks the focus stale at once; one run through the CLI sends no such signal
(see [topics.md](topics.md#limitations)).

Most turns are **agent turns**: the model searches the collection itself before
it answers (see [retrieval-and-agents.md](retrieval-and-agents.md#agent-turns)).
While it works, the turn lists each step as it happens:

- a search, with its query and how many new passages it found;
- an image search, with the visual cues it looked for and how many new images
  it found;
- the keyframes of a clip, or one of them ("Viewed keyframe #3 of …");
- a keyword check, with its words, the field when that is not the text, the
  exact number of chunks that contain them, how many chunks the keyword index
  could not check, and how many new passages it found ("Keyword check “…” in
  Author: 3 matches (2 chunks not checked), 1 new"). When the count failed, the
  step says "count unavailable" in place of the number. The trail does not yet
  show that a check ran while the collection was being written: the answer says
  so, and the validation banner flags an answer that leaves it out;
- a passage read again ("Read passage [2]");
- a listing of the collection's topics ("Listed the topics"), or one topic's
  passages, with how many were new ("Read topic T3: 5 passages");
- a note ("Note: …"): text the model wrote before it decided to search
  again. Whatever of that text had already streamed into the answer moves into
  the trail, and the answer starts afresh;
- "Started from n passages", on a scoped turn: the pinned selection it
  answers from;
- "Carried over n passages from earlier answers", on a follow-up: the passages
  its earlier answers cite, which it starts out holding as its first sources.
  The row opens the trail, ahead of any search, and stays in it once the turn
  is done: in the disclosure below with the turn's other steps, or, when the
  turn took no other step, alone as a line under the answer.

When the turn finishes, its steps fold behind a **Searches behind this
answer** disclosure where the live list stood. Every agent turn with a step
besides its pinned selection and its carried passages gets one, lookups
included. A follow-up that only carried passages took no other step, so its row
stands as a line under the answer, with no disclosure. A turn whose stream
failed keeps the steps it showed, in the same form, above the error.

A muted note says when a turn stopped searching at one of its limits: the
round limit, the time for the turn, the room for evidence in the model's
context, or searches that found nothing new. A limit is how research ends,
not a failure: at the smallest context window a turn often stops for room,
above all when pictures fill it. An amber notice is kept for a turn that may
rest on less than it could: the model could not run its own searches, so the
answer rests on a standard search, or not every search could be completed. An
answer the model's length limit cut off gets an amber notice too, saying it may
be incomplete. Each maps one `agent.reason` token to its own copy.

A question that asks for a judgement — what could be improved, whether a
method convinces — gets an **assessment**. The answer opens with what the
sources say, cited as usual, and continues under a **Model assessment** label
with the model's own judgement, which may draw on general knowledge and is not
checked against the sources. A note under the answer says so, and the
validation banner speaks of the source-backed part only. An assessment the
model left empty says so in an amber notice. A turn cut off after its text
began, by an error or a dropped connection, opens with a note that it did not
finish and may be incomplete; the note travels with the text into the
transcript export. An agent turn that had shown a step adds that the text may
contain the model's own judgement without its label. A turn that failed cannot
be added to a report.

A **Search in** picker under the query field chooses what the keywords
match: Text (the chunk body, the default), Author, Network or UUID. The
result is always chunks, so "everything this author wrote" is an ordinary
search whose tiles scope the chat like any other. Filtering by filename is
not a search — use the chat's metadata filters for that.

**Author** deliberately spans everything that names a person — display name,
vanity handle, numeric id, and the copies of all three that an image or
transcript inherits from the posting it came from. An investigator holding
any one of those identifiers can paste it without first working out which
kind it is. Names match on prefixes (`krieg` finds `Krieger`); ids match
exactly, because they are numbers rather than prose.

**UUID** is the sole identifier of a single posting artifact. Paste one and
the hits are that posting plus every image, keyframe and transcript segment
derived from it; dashed or undashed both work. It matches exactly — a uuid
is never a prefix of anything.

The CSV export beside the counts honours the field, and stays available with
a blank query, where it exports the whole filtered collection.

### Ingest (`src/routes/Ingest.tsx`)

Drag-and-drop upload (`src/components/ingest/Dropzone.tsx`) into a selected
or newly-created collection. Streams the multipart upload plus SSE progress
events (`src/components/ingest/IngestionStatus.tsx`) for per-file and
per-stage status. A selection goes up in small batches (at most 100 files and
64 MiB per request); a batch that stops making progress for two minutes is
aborted and its unsaved files are sent again, up to twice — so a request that
never completes can no longer freeze the run. Each retry shows as a warning in
the run's card until that batch goes through or is skipped; a skipped batch
keeps its own warning. A run's warnings always sit inside its card.
Every request of a run (the staged check, each batch and retry, the finalize)
goes to the owner the run started under, even if an admin switches to another
user's collection while it uploads.
The entity-extraction, hate-speech, collection-summary and topics
options are `@infra/ui` `ToggleButton`s (lit = on), seeded once per mount
from `GET /config/ingest-defaults`. **Summary** gates the collection-summary
rebuild that runs when the ingest finishes; it maps the whole collection, so
on a long batch it is the longest part of the run and switching it off is
what makes the job end at the last document. The summary can still be built
later, on demand, from the Analysis screen's Summary tab. **Topics** appears
only while the deployment has topics switched on (`topics_enabled` on
`GET /config`): lit, the run builds the collection's topics, or places its
chunks in the existing ones and looks for new topics among those that fit
none; unlit, the new chunks are only placed in the existing topics
([topics.md](topics.md#the-ingest-stage)). The topics work runs beside the
summary rather than after it, and the run's card gives it one **Topics** bar
that names the step it is on (reading the sample, fitting, labelling,
assigning, aligning), pulses through a step without a total, and reads
**Done** once the step is over while the summary may still be running.

A picked file is not added when a different file in the selection already has
its name (its path inside a picked folder, or its bare name otherwise), and a
notice lists it. The server stores a collection's uploads under that name, so
the second file would overwrite the first. Picking the same file again (same
name, size and modification time) is ignored without a notice.

While a run is live the card also carries a bar per preprocessing stage —
PDFs read, scanned pages recognized, images described, audio and video
transcribed, keyframes described. That work starts as each file is saved, so
the bars appear during the upload, long before a job exists, and carry on
through it; they are polled from `GET /ingest/staged` rather than parsed out
of the job's messages. The staged card above the form draws the same bars,
which is what a browser that died mid-upload comes back to. Transcription has
only that one bar: the job adds a *remaining* audio-and-video bar solely for
clips it has to send to Nextext itself, because the pool left them without a
transcript.

Below the form, every ingest job the caller owns is listed newest first, one
card each (`src/components/ingest/IngestJobList.tsx`): queued runs waiting on
a worker slot, runs in progress, finished runs with their warnings, and
interrupted runs offering a re-queue. Starting a second run therefore never
hides the first. Each card carries its own Dismiss once the run has ended;
"Clear finished" above the list dismisses them all at once and leaves
anything still running. A completed run selects the collection it filled, so
with several finishing in a row the last one wins. The list and its live
progress belong to the owner selected in the sidebar: an admin viewing another
user's collections sees that user's jobs, and each job is checked, re-run,
dismissed and selected in the namespace it was queued in.

### Analysis (`src/routes/Analysis.tsx`)

Four tabs: **NER**, **Hate Speech** (`HateSpeechTable`), **Summary**
(`SummaryPanel`) and **Topics** (`TopicsPanel`), the last only while
`GET /config` reports `topics_enabled`. Pre-warms the NER aggregate when
opened.

The NER and Hate Speech tabs list what was found in images as well as in
text: the words printed in a PDF figure, an image attached to a posting or a
video keyframe, with the image's caption also read for entities (see
[Ingestion — Images](ingestion.md#images--imagespy)). Such a row names the
file the words came from: the PDF a figure was cut out of, the attached
image itself, or the clip a keyframe belongs to. Every such row shows the
picture beside it, a keyframe and a PDF figure included; clicking it
enlarges a still from its own file and a keyframe or figure from the stored
image. A hate-speech row judged from an image's words and description also
carries a **From an image** pill. Either tab shows
an image's text as labelled parts — **Text in the image** (the words printed
in it), **Image description** and **Tags** — rather than one passage, so it
is clear which words the picture says and which are docint's description.

The **Summary** tab reads before it builds. On open — and whenever the
selected collection changes — it probes `GET /summarize`, which only ever
reads: a stored summary renders straight away beside a **Refresh** icon,
and `204` (nothing stored) leaves a labelled **Create** button instead. A
build is minutes of map-reduce, so it starts only when an operator presses
one of those two, never because a tab was opened.

Creating and refreshing go through `POST /summarize`, which answers `202`
with a `job_id`; the panel follows it on the shared ingest-job SSE stream
and shows a progress bar (mapped units / total units) while the tree
summarizer runs, then renders the result with its coverage diagnostics once
`summary_completed` arrives. A refresh leaves the previous summary on
screen throughout — blanking the panel for the length of a rebuild would
take away what the operator already had.

The **NER** tab opens with a **Table / Graph** view toggle (only one is shown
at a time):

- **Table** — an `EntitySelect` category + entity picker. The category filter
  re-filters the entity dropdown and pre-selects that category's top entity.
- **Graph** — an interactive, force-directed entity graph rendered by the
  shared `@infra/ui` `ForceGraph` primitive (the same component chorus's
  reactive graph exploration uses; see chorus ADR 0016), fed by
  `GET /collections/ner/graph`. `EntityGraph` (`src/components/analysis/
  EntityGraph.tsx`) is a thin adapter: `src/lib/entityGraphElements.ts` maps
  the NER graph payload onto `ForceGraph`'s `{nodes, edges}` shape and bridges
  docint's single-entity `${text}::${type}` selection key onto the
  primitive's node-id selection API. Nodes are draggable (with collision),
  the canvas zooms (wheel) and pans (background drag), and clicking a node
  selects that entity. `ForceGraph` itself renders the controls row (**Min
  edges** stepper, **Edge length** slider, **Zoom**, **Fit**, **Reset**) and
  the maximize overlay, and adds native `<title>` tooltips, fit-to-view, and
  marquee/shift-click multi-select (docint's single-entity findings panel
  just follows the most recently selected node). A small node-count control
  (`GraphTopKControl`, parent-owned state) sits above the graph. Selecting one
  or more nodes shows a **Remove node**/**Remove {n} nodes** button
  (Backspace/Delete also works); removal is view-only local state in
  `EntityGraph` — the underlying NER data is untouched, and removed nodes
  reappear on the next fetch or top-K change. **Export
  JSON**/**Export GraphML**/**Export HTML** buttons (shown once the graph is
  non-empty) serialize the current view state — respecting node removals —
  via `@infra/ui`'s `toGraphJson`/`toGraphML`/`toGraphHtml` + `downloadText`;
  this is client-side only, with no backend round-trip.

The type legend is computed from the full node set, while the min-edges filter only affects what is drawn—so legend entries can name types whose nodes are currently filtered out; this is a deliberate divergence from the previous local renderer, accepted because filter state is component-internal.

Either selection surface drives the shared **findings table**
(`EntityFindingsTable` → `EntityFinding` rows): one chunk per row, with all
locator/reference metadata flattened into a single Metadata column and an
inline "Add to report" control. `HateSpeechTable` follows the same one-row-per-
finding table shape.

Both tables carry the same three section-wide controls in their header, in
order: **Translate all**, **Add all findings to report**, **Export CSV**. Each
acts on every finding the section's filter matches — the tables page 50 rows at
a time, so all three walk the remaining cursor pages themselves rather than act
on what happens to be on screen.

The **Topics** tab (`src/components/analysis/TopicsPanel.tsx`) shows the active
collection's topic run ([topics.md](topics.md)). A caption counts its topics
and its coverage (chunks assigned of the total, unassigned, pending) beside
its icon actions: **Restart analysis**, which asks first, since chat filters on
the current topics stop matching; **Find new topics**, while some chunks are
unassigned; **Align pending chunks**, while some are pending; the report
toggle (`AddToReportButton`, as on the Summary tab), which adds the whole run —
every topic's label, counts, keywords and summary — to the active report as a
single item, and takes it out again; and **Download CSV**. A collection without
a run shows a **Start analysis** button instead.
While a job runs, a line names its stage, with a bar where the stage has a
total. A banner suggests a rebuild when the collection is now embedded with
another model than the run was built with.

Each topic is a row (`TopicRow.tsx`), largest first: its number (`T3`), its
label, its size and share of the collection's chunks, a **New** badge for a
topic a later discover added, its first five keywords and the dates its chunks
span. Opening a row, one at a time, shows the topic's **Summary** (four to five
sentences written when the topic was added), its
**Evidence** (the five chunks nearest its centre), its **Documents** (the
source files it draws on, at most 50, each with its chunk count) and **All
passages**, 50 at a time; the three are read only while the row is open. **Ask
in chat** opens a new chat focused on the topic (see
[Chat](#chat-srcrouteschattsx)); the chat keeps it once its first question is
sent.

The tab shows warnings or errors for the collection's newest topics job
only when this tab started it or
watched it live. A job counts as watched live once any of its frames arrives
1000 ms or more after its started frame, and stays so for the page session
(`liveJobs` in `src/stores/ingestJobs.ts`). A replayed history arrives within
milliseconds, so after a reload a run that finished elsewhere does not show its
old outcome again; the overview reflects that run all the same. Warnings render
from their codes, never from the backend's message; a collection too small to
fit is told how many chunks with enough text it has and how many topics need,
when the warning carries both counts. The CSV holds one row per
topic (`id,label,size,share,keywords,first_seen,last_seen,origin`, `share` as a
fraction, keywords joined by `; `), and a text cell starting with `=`, `+`, `-`,
`@`, a tab or a carriage return is written with a leading apostrophe, so a
spreadsheet reads it as text (`src/lib/csv.ts`).

### Inspector (`src/routes/Inspector.tsx`)

A paginated document table (`src/components/inspector/DocumentTable.tsx`)
over the active collection, plus a per-session ZIP export
(`SessionZipButton`). The summary strip above the table
(`DocumentSummary`) reads collection-wide aggregates from
`GET /collections/documents/summary` — so the document/node totals and the
file-type / entity-type breakdown stay accurate regardless of how many pages
the table has lazily loaded (the counts are not derived from the loaded rows).

Between the two sits the **Data extracts** panel
(`src/components/inspector/ExtractsPanel.tsx`): one control builds a written
extract of the collection, and the bundles already stored for it are listed
with their size, contents and a download link. Progress comes from the shared
job store that `Shell`'s single owner-multiplexed stream already feeds, so a
build survives navigating away and a reload re-attaches to it like an ingest
run. Each document row also carries a download action for that source alone
(`SourceExtractAction`); it fetches rather than links, because a postings
table's hash expands to every post in it and the server answers 413 — the same
click then queues a targeted build. See [extracts.md](extracts.md).

While the collection has an active topic run, the document table gains a
**Topics** column between Entities and Hash: up to three badges, largest first,
each naming a topic (`T3` and its label, cut to 24 characters and whole in the
tooltip) with the document's chunk count in it, then `+N` for the rest. Values
of another run, and the run's unassigned chunks, name no topic of the active
run and are left out; a document with none shows `—`. The tallies come from the
documents listing's `topics` field
([api-reference.md](api-reference.md#get-collectionsdocuments)), and the column
refetches when a topics job on the collection ends: any job the page's job
stream reports, which a CLI run is not
([topics.md](topics.md#limitations)).

### Report (`src/routes/Report.tsx`)

The Report Builder surface. Lists the reports of the active collection, and
only those (with no collection selected it lists none and offers no new one),
and, for the active one, shows the picked artifacts grouped by type with per-item notes, reordering, and
removal, plus the five export formats. Each type is a foldable section whose
contents peek out while shut; the control is the `@infra/ui`
`DisclosureButton` at the bar's right edge, not the bar itself. The sections
run summaries, topics, chat answers, entity findings, hate-speech findings,
the order the exports use. Artifacts are added from elsewhere in
the app: an **"+ Report"** control sits on every chat answer, entity finding,
and hate-speech finding, and on the Analysis screen's summary and topics; a
topics item is titled by its topic count and previews the first labels. Switching the active collection releases the active
report, so a report and its evidence always describe the same collection; every
report request names the active collection and the server refuses any other, so
a report kept active from elsewhere (another tab, an earlier session) answers
404 and is dropped by `useReport` rather than written into. Reports created
before every report named its collection are listed apart, under **Reports
without a collection** (`UnassignedReports`), each with a **Move to …**
action that places it in the active collection and opens it;
with no collection selected they are listed without one.

The PDF is the one export that is not a plain link. A large report renders for
longer than the gateway waits on a single request, so choosing **PDF** queues a
background render (`POST /reports/{id}/pdf`, on the same job machinery as the
Inspector's data extracts) and a compact status line beside the export menu
follows it: queued while another export holds the render slot, then rendering
with the page the layout pass has reached, then finishing — or a failure. While
a render is under way the entry cannot start another. Progress comes from the
shared job stream (`src/hooks/useReportPdf.ts`); the status route
(`GET /reports/{id}/pdf/status`) is the authority on which job is the report's
and on whether the stored PDF still matches it. The render this tab asked for
downloads by itself once it is ready — exactly once, even when the stream
replays the job after a reconnect — and a stored PDF that matches the report is
linked straight from the menu. Editing the report outdates that PDF, so the next
**PDF** renders it again. The other formats stay plain download links.

The full workflow — snapshot semantics, frozen image evidence, and what each
export contains — is in [reports.md](reports.md).

## Localization

The single env var `RESPONSE_LANGUAGE` (values `en` | `de`, default `en`)
controls the *entire* app — both backend and SPA chrome — with one knob:

- **Backend**: prompts, `ui_strings` in reports, and export captions (PDF
  headers, CSV column names). Unknown values silently fall back to `en`.
- **SPA**: The React interface — buttons, labels, navigation, form hints, and
  error messages — flows from a typed locale catalog (`frontend/src/i18n/`)
  with `en` and `de` as canonical languages, maintained in parity with each new
  feature. The `useT()` hook and `LanguageProvider` read `RESPONSE_LANGUAGE`
  from `GET /config`.

Keys are dot-namespaced by screen (`common.*`, `chat.*`, `ingest.*`, …) and
interpolate via `{name}` placeholders. JSON output schemas, intent labels,
enum values, and API field names stay English in every locale — they are
protocol, not prose. See
[configuration.md](configuration.md#response-language--languageconfig).

### On-demand translation of source content

Chat source citations, entity findings, and hate-speech findings each show a
hover/focus-revealed **Translate** control. Clicking it fetches an on-demand
machine translation into the operator's active locale (`RESPONSE_LANGUAGE`)
and swaps it in for the original in place — a "Translation" label marks the
swapped view, and a second click ("Show original") brings the original back;
the original is always one click away, never discarded. Long chunks stay
clamped to four lines behind a "Show more" toggle in either view. This is a
display-time overlay only: nothing ingested or stored is ever translated.
For a finding judged from an image only the text printed in it is translated:
the description and tags are written in `RESPONSE_LANGUAGE` already, so they
stay in view, and an image with no printed text has no Translate control.
Text that is already in `RESPONSE_LANGUAGE` is not translated. The same goes
for text that is only names, numbers or URLs. The original stays in view under
a "No translation needed" note, and nothing is carried into a report. The model
makes that call. A model that copies foreign text instead of translating it is
read the same way, so that text also shows the note.

**Translate all**, in each Analysis section's header, does the same for every
finding the section matches instead of one at a time — a corpus in a language
nobody on the team reads is not made readable one hover at a time. It runs in
the foreground, one call per distinct chunk text with three in flight, showing
how far it has got and staying clickable to stop; pressing it again picks up
the remainder, since nothing already translated this session is re-sent. Text
found to be in `RESPONSE_LANGUAGE` already is counted separately ("already in
English"), is never sent again, and does not count toward the failures. Three
consecutive failures end the run, so an unreachable model is reported in
seconds rather than after the whole section. It stores nothing — the
translations live in the browser for the session.

Translating a finding before adding it to a report carries that translation
into the report's snapshot, and re-running "Add all" backfills translations
into findings already added — see [reports.md](reports.md).

Translation reuses the same chat model as the rest of docint over the same
router endpoint — there is no dedicated translation runtime and no
`TRANSLATE_API_BASE` to configure. Set `TRANSLATE_MODEL` in `.env` to use a
different model than chat's `TEXT_MODEL`; it defaults to `TEXT_MODEL`.
Airgap-safe: no new container and no new network egress target. A
target-language override (translating into a language other than the active
locale) is not yet supported.

## State

- **`src/stores/ui.ts`** (`useUiStore`, Zustand) — selected collection,
  current session id, and preview modal. There is no merge-mode control or
  store field: the UI always requests merged (resolved) entities, while the
  API's `entity_merge_mode` parameter remains for the backend contract.
  Persisted to `localStorage`: the active collection + owner, current
  session id, and the graph node-count override.
- **`src/stores/chatFilters.ts`** (`useChatFiltersStore`) — query mode,
  retrieval mode, the retrieval target ("Answer from") with its visual-target
  presets, the reasoning toggle (off by default; sent as `reasoning`
  on every chat request), and the metadata-filter builder state;
  `buildPayload()` serialises the filters for requests. Persisted to
  `localStorage`, except the retrieval target, which belongs to one chat and
  starts each new one at `all`.
- **`src/stores/topicFocus.ts`** (`useTopicFocusStore`) — the topic **Ask in
  chat** picked for a chat not started yet. Not persisted: once the chat's
  first turn has run, its topic is the one the server keeps with the session.
- Server state (collections, documents, NER, sessions) is owned by
  TanStack Query hooks under `src/hooks/` (`useCollections`, `useNer`,
  `useSessions`, `useDocuments`). A collection-scoped key carries the owner
  as well as the collection name (`useActiveCollection`): two owners can name
  a collection the same, and an admin switching between them changes only the
  owner.

## API layer

All HTTP lives under `src/api/`:

- `client.ts` — `apiGet` / `apiPost` / `apiDelete`, an `ApiError` type, and
  a base URL from `VITE_API_BASE_URL` (defaults to relative, so the dev
  proxy / nginx handles routing).
- `sse.ts`, `upload.ts` — SSE stream parsers (`streamSse`, `streamUpload`)
  for token streaming and multipart upload progress.
- `chat.ts`, `collections.ts`, `sessions.ts`, `ingest.ts`, `analysis.ts` —
  typed endpoint wrappers; `types.ts` holds the shared types.
- `queryClient.ts` — the TanStack Query client (30s stale time, no retry on
  4xx, no refetch on window focus).

Dev proxy targets are declared in `frontend/vite.config.ts`
(`/collections`, `/sessions`, `/sources`, `/query`, `/stream_query`,
`/summarize`, `/ingest`, `/agent` → `http://localhost:8000`); the
production equivalents live in `frontend/nginx/default.conf`.

## Tests

Vitest specs sit next to their subjects under `frontend/src` (for example
`src/api/sse.test.ts`, `src/stores/ui.test.ts`, `src/routes/Chat.test.tsx`,
`src/layout/Sidebar.test.tsx`). Run them with:

```bash
cd frontend
pnpm test          # run once
pnpm test:watch    # watch mode
```

## Adding a screen

1. Add a route component under `src/routes/` and register its path in
   `src/routes/Router.tsx`.
2. Add a navigation entry in `src/layout/Sidebar.tsx`.
3. Put shared widgets under `src/components/<area>/` and data access in a
   `src/hooks/use*.ts` hook backed by an `src/api/*.ts` wrapper.
4. Add a Vitest spec next to the new files.
