# UI guide

The Streamlit UI is a thin client over the FastAPI backend. It is
launched via `uv run docint` (alias for `docint.app:run` in
`docint/app.py`) and renders five pages plus a persistent sidebar.

## Entry point

`docint/app.py` wires together:

- `set_offline_env()` — enables Hugging Face offline mode if
  `DOCINT_OFFLINE` is truthy.
- `init_logger()` — loguru setup (stderr + rotating file).
- `configure_page()` and `apply_custom_css()` — page config and CSS
  overrides from `docint/ui/theme.py`.
- `init_session_state()` — session-state defaults from
  `docint/ui/state.py`.
- `render_sidebar()` — persistent sidebar.
- A page registry (`_PAGE_RENDERERS`) that maps the active page name
  (`st.session_state.current_page`) to a renderer callable.
- `render_footer()` — app footer.

The page registry:

```python
_PAGE_RENDERERS = {
    "Dashboard": render_dashboard,
    "Chat": render_chat,
    "Ingest": render_ingestion,
    "Analysis": render_analysis,
    "Inspector": render_inspector,
}
```

## Sidebar — `docint/ui/sidebar.py`

- **Page nav** — buttons that call `navigate_to()` on the session
  state to switch between the five pages.
- **Collection picker** — lists collections via `/collections/list`
  and activates the selected one via `/collections/select`.
- **Session picker** — lists active/past sessions via
  `/sessions/list`, lets the user switch or delete a session.
- **Backend health indicator** — a small badge showing whether the
  backend is reachable.
- **Refresh controls** — manually refresh collections, sessions, and
  the NER cache.
- **User guides** — short inline cheat-sheets.

## Page 1 — Dashboard (`docint/ui/dashboard.py`)

The landing page. Shows:

- KPI cards: backend status, number of collections, number of
  documents in the active collection, number of sessions.
- A collection selector duplicated from the sidebar for convenience.
- A "recent sessions" list powered by `/sessions/list`.
- A document frequency chart (file-type distribution).
- An NER overview (top entities, top entity types, top relations)
  when a collection is selected and NER is available.

Rendering helpers come from `docint/ui/components.py`.

## Page 2 — Chat (`docint/ui/chat.py`)

The primary interaction surface. Features:

- **Streaming Q&A** — wraps `POST /stream_query` or
  `POST /agent/chat/stream` depending on mode.
- **Query modes** — `answer`, `entity_occurrence`, or
  `entity_occurrence_multi` (matches the `query_mode` field on
  `QueryIn`).
- **Retrieval mode toggle** — `session` (default) or `stateless`,
  wired to `QueryIn.retrieval_mode`.
- **Metadata filters UI** — add `MetadataFilterIn` objects by MIME
  type, date range, or arbitrary field/operator/value triples. The
  supported operators are the ones defined on the Pydantic model.
- **Source panel** — renders each citation with score, filename, page
  (if available), and a link to `/sources/preview` for inline preview.
- **Reasoning panel** — when the LLM exposes reasoning traces, they
  are rendered in a collapsible expander.
- **Validation banner** — surfaces
  `validation_mismatch` / `validation_reason` from the backend when
  `RESPONSE_VALIDATION_ENABLED=true`.
- **Graph debug** — if `GRAPHRAG_ENABLED=true`, the graph expansion
  trace returned by `/query` is rendered for debugging.

## Page 3 — Ingest (`docint/ui/ingest.py`)

- File uploader that hits `POST /ingest/upload` with a multipart
  request.
- Progress readout that follows the SSE events streamed back
  (`upload`, `processing`, `done`).
- Ingestion summary after completion.
- Error handling that normalises the SSE payloads into human-friendly
  messages.

## Page 4 — Analysis (`docint/ui/analysis.py`)

- **Collection summarisation** — calls `/summarize` or
  `/summarize/stream` and renders the summary plus `summary_diagnostics`.
- **NER export** — reads `/collections/ner` and
  `/collections/ner/stats`, renders entity occurrences and
  co-occurrence bar charts.
- **Hate-speech export** — reads `/collections/hate-speech` and
  renders flagged chunks.
- **CSV export** — lets the user download entities and hate-speech
  findings as CSV.
- Coverage diagnostics for summaries (shows which documents were
  covered vs. uncovered).

## Page 5 — Inspector (`docint/ui/inspector.py`)

- Document listing via `/collections/documents` rendered as a table.
- Source file downloads — builds ZIP archives from the files staged
  under `QDRANT_SRC_DIR` using the helpers in
  `docint/ui/components.py` (`build_source_files_zip`).
- Extracts the unique sources referenced from the currently selected
  session, so the user can download everything cited in a
  conversation.

## Shared helpers

`docint/ui/components.py` contains pure-render helpers that are reused
across pages:

- `format_score` — format retrieval scores for display.
- `normalize_entities` / `normalize_relations` — parse the NER payload
  into a consistent shape.
- `source_label` — format a source for display in a citation.
- `aggregate_ner` — compute co-occurrence / totals.
- `render_source_item`, `render_ner_overview` — Streamlit widgets.
- `build_source_files_zip` — archive staged source files into a
  downloadable ZIP.

## Theme

`docint/ui/theme.py` sets the Streamlit page config (`page_title`,
`page_icon`, `layout="wide"`) and applies custom CSS for:

- tighter metric padding,
- expander border styling,
- footer styling.

## State model

`docint/ui/state.py` centralises session-state defaults:

- `PAGES` / `PAGE_ICONS` — page labels and icons for the sidebar.
- `current_page`, `messages`, `selected_collection`, caches per-page,
  `backend_online` — the fields every renderer reads from
  `st.session_state`.
- `navigate_to()` — helper that updates `current_page` and triggers
  a rerun.

## Adding a new page

1. Create a `render_foo()` function in `docint/ui/foo.py`.
2. Import it in `docint/app.py` and register it in `_PAGE_RENDERERS`.
3. Add the page to `PAGES` / `PAGE_ICONS` in `docint/ui/state.py` and
   wire a navigation button in `docint/ui/sidebar.py`.
4. Write a test under `tests/test_ui_*.py` mirroring the pattern of
   existing page tests.
