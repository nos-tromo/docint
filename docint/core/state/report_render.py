"""Pure renderers turning a curated report into Markdown / HTML / PDF / JSON / CSV.

Every renderer reads **only** the report dict produced by
:meth:`docint.core.state.report_manager.ReportManager.get_report` (whose items
carry frozen JSON ``snapshot``s) — never Qdrant — which is what makes a finished
report immune to later re-ingestion. Section headings flow through
:func:`docint.utils.ui_strings.ui_string` (en/de); JSON keys and
``artifact_type`` values stay English (protocol, not prose).

The HTML renderer emits one self-contained, styled document used both for the
``.html`` on-screen export and as WeasyPrint's input for the real paginated
PDF, so layout is defined once.
"""

from __future__ import annotations

import html
import io
import json
import zipfile
from collections import OrderedDict
from datetime import datetime
from typing import Any

from docint.utils.env_cfg import language_endonym
from docint.utils.reference_metadata import BODY_TEXT_FIELDS, reference_metadata_items
from docint.utils.ui_strings import ui_string

# Artifact types (English protocol values, mirrored in the frontend).
ARTIFACT_CHAT = "chat_answer"
ARTIFACT_ENTITY = "entity_finding"
ARTIFACT_HATE = "hate_speech_finding"
ARTIFACT_SUMMARY = "summary"

# Render order: Summaries -> Chat -> Entities -> Hate-speech (summaries lead the
# document — they set the context for the findings that follow).
SECTION_ORDER: tuple[tuple[str, str], ...] = (
    (ARTIFACT_SUMMARY, "report_section_summaries"),
    (ARTIFACT_CHAT, "report_section_chat"),
    (ARTIFACT_ENTITY, "report_section_entities"),
    (ARTIFACT_HATE, "report_section_hate_speech"),
)

# Stable in-document anchor id per section, shared by the section headings and the
# table-of-contents links (the targets WeasyPrint resolves page numbers against).
SECTION_ANCHOR: dict[str, str] = {
    ARTIFACT_SUMMARY: "sec-summaries",
    ARTIFACT_CHAT: "sec-chat",
    ARTIFACT_ENTITY: "sec-entities",
    ARTIFACT_HATE: "sec-hate",
}

COLLECTION_OVERVIEW_ANCHOR = "sec-collection-overview"
COLLECTION_OVERVIEW_HEADING = "report_section_collection_overview"
_HASH_DISPLAY_CHARS = 12


def _overview_snapshot(report: dict[str, Any]) -> dict[str, Any] | None:
    """Return the overview snapshot iff the trailing section should render.

    Renders only when the report opts in (``show_collection_overview``) AND the
    snapshot has at least one document — an empty manifest reads as a bug, so it
    is omitted like an empty item-section.
    """
    if not report.get("show_collection_overview"):
        return None
    overview = report.get("collection_overview") or None
    if not overview or not (overview.get("documents") or []):
        return None
    return overview


def _overview_units(doc: dict[str, Any]) -> str:
    """Pages-or-rows cell for a manifest row ("—" when neither applies)."""
    pages = int(doc.get("page_count") or 0)
    if pages > 0:
        return str(pages)
    rows = int(doc.get("row_count") or 0)
    if rows > 0:
        return str(rows)
    return "—"


def _short_hash(value: Any) -> str:
    """Truncate a file hash to its display prefix ("—" when absent)."""
    text = str(value or "")
    return text[:_HASH_DISPLAY_CHARS] if text else "—"


def _overview_file_types(overview: dict[str, Any]) -> str:
    """Summarize the snapshot's file-type counts as "N Label, …" ("—" when none)."""
    parts = [f"{ft.get('count')} {ft.get('label')}" for ft in (overview.get("file_types") or [])]
    return ", ".join(parts) if parts else "—"


_CHUNK_MAX_CHARS = 1500

# CSV bundle column schemas for the chat/summary artifacts (entity & hate-speech
# reuse the canonical schemas in ``docint.utils.csv_stream``).
CHAT_ANSWER_COLUMNS: tuple[str, ...] = ("session_id", "turn_idx", "question", "answer", "sources")
SUMMARY_COLUMNS: tuple[str, ...] = ("collection", "summary")
COLLECTION_OVERVIEW_COLUMNS: tuple[str, ...] = ("filename", "type", "pages", "rows", "nodes", "hash")


class PdfEngineUnavailableError(RuntimeError):
    """Raised when the PDF engine (WeasyPrint + native libs) is unavailable."""


def _group_items(items: list[dict[str, Any]]) -> OrderedDict[str, list[dict[str, Any]]]:
    """Group items by artifact type, preserving each item's position order."""
    grouped: OrderedDict[str, list[dict[str, Any]]] = OrderedDict(
        (artifact_type, []) for artifact_type, _ in SECTION_ORDER
    )
    for item in items:
        grouped.setdefault(item.get("artifact_type", ""), []).append(item)
    return grouped


def _truncate(text: str, limit: int = _CHUNK_MAX_CHARS) -> str:
    """Trim long chunk text for readable report bodies."""
    text = (text or "").strip()
    if len(text) > limit:
        return text[:limit].rstrip() + " …"
    return text


def _translation_label(lang: str) -> str:
    """Build the machine-translation heading, suffixed with the target language's endonym.

    Shared by the Markdown (:func:`_md_translation_row`) and HTML
    (:func:`_html_translation_row`) renderers so the label stays identical
    across export formats.

    Args:
        lang (str): The raw ``target_lang`` code (e.g. ``"de"``), or ``""``.

    Returns:
        str: ``"Machine translation (→ Deutsch)"`` when ``lang`` is set
        (rendered via :func:`docint.utils.env_cfg.language_endonym`), or the
        bare heading when ``lang`` is empty.
    """
    heading = ui_string("report_label_machine_translation")
    return f"{heading} (→ {language_endonym(lang)})" if lang else heading


def _md_cell(value: Any) -> str:
    """Escape a value for use inside a single Markdown table cell.

    Pipes are escaped and newlines become ``<br>`` so verbatim evidence text
    (multi-line chunks, posting texts) cannot break the table grid.
    """
    text = str(value if value is not None else "").strip()
    return "<br>".join(text.replace("|", "\\|").splitlines())


def _md_translation_row(snap: dict[str, Any]) -> list[str]:
    """Markdown finding-table row for an optional machine-translation, or []."""
    tr = snap.get("translation") or {}
    text = _truncate(tr.get("text") or "")
    if not text:
        return []
    label = _translation_label(str(tr.get("target_lang") or "").strip())
    return [f"| {_md_cell(label)} | {_md_cell(text)} |"]


def _date_only(value: Any) -> str:
    """Reduce an ISO datetime to its calendar date (``YYYY-MM-DD``).

    The report dict carries ``created_at`` as an ISO timestamp; the subheader
    shows only the creation *date* so it stays on a single line. Falls back to
    the leading 10 characters when the value cannot be parsed.
    """
    text = str(value or "").strip()
    if not text:
        return ""
    try:
        return datetime.fromisoformat(text).date().isoformat()
    except ValueError:
        return text[:10]


def _location(snap: dict[str, Any]) -> str:
    """Render a 'page N' / 'row N' locator from a snapshot, or ''."""
    page = snap.get("page")
    row = snap.get("row")
    if page is not None:
        return f"{ui_string('report_label_page')} {page}"
    if row is not None:
        return f"{ui_string('report_label_row')} {row}"
    return ""


def _source_oneline(src: dict[str, Any], *, include_score: bool = True) -> str:
    """Compact single-line description of a chat citation/source.

    Args:
        src (dict[str, Any]): The citation/source dict.
        include_score (bool): Whether to append the ``[0.000]`` relevance score.
            The human-facing report renderers pass ``False`` (a bare score reads
            like debug output); the CSV/JSON data exports keep the default so the
            number stays available for downstream analysis.
    """
    name = src.get("filename") or src.get("source") or ""
    loc = _location(src)
    score = src.get("score")
    parts = [str(name)]
    if loc:
        parts.append(f"({loc})")
    if include_score and isinstance(score, (int, float)):
        parts.append(f"[{score:.3f}]")
    return " ".join(p for p in parts if p)


def _dedupe_entities(entities: list[dict[str, Any]]) -> list[tuple[str, str]]:
    """De-duplicate entity mentions case-insensitively, preserving first order.

    A single chunk often names the same surface form many times; the report
    shows each distinct ``(text, type)`` once.

    Args:
        entities (list[dict[str, Any]]): Raw entity dicts (``text`` / ``type``).

    Returns:
        list[tuple[str, str]]: Ordered ``(text, type)`` pairs in first-seen
        casing, one per distinct case-insensitive mention.
    """
    seen: set[tuple[str, str]] = set()
    out: list[tuple[str, str]] = []
    for e in entities:
        text = str(e.get("text") or "").strip()
        etype = str(e.get("type") or "").strip()
        if not text:
            continue
        key = (text.lower(), etype.lower())
        if key in seen:
            continue
        seen.add(key)
        out.append((text, etype))
    return out


# --------------------------------------------------------------------------- #
# JSON
# --------------------------------------------------------------------------- #
def render_json(report: dict[str, Any]) -> str:
    """Serialize the full report (including snapshots) as pretty JSON."""
    return json.dumps(report, ensure_ascii=False, indent=2)


# --------------------------------------------------------------------------- #
# Markdown
# --------------------------------------------------------------------------- #
def _md_reference_metadata_rows(snap: dict[str, Any]) -> list[str]:
    """Finding-table rows for the source's reference metadata (provenance), if any.

    Surfaces the stable citation fields (network, author, timestamp, …) captured
    at ingestion as label/value rows under a bold subheading row; the body-text
    fields are skipped because the chunk text is already shown in the top cell.
    """
    items = reference_metadata_items(snap, skip_keys=BODY_TEXT_FIELDS)
    if not items:
        return []
    rows = [f"| **{ui_string('report_label_reference_metadata')}** |  |"]
    rows += [f"| {_md_cell(label)} | {_md_cell(value)} |" for label, value in items]
    return rows


def _md_chat(snap: dict[str, Any], note: str | None) -> list[str]:
    lines = [
        f"### {ui_string('report_label_question')}: {snap.get('user_text', '').strip()}",
        "",
        f"**{ui_string('report_label_answer')}:** {snap.get('model_response', '').strip()}",
    ]
    sources = snap.get("sources") or []
    if sources:
        lines += ["", f"**{ui_string('report_label_sources')}:**"]
        lines += [f"- {_source_oneline(s, include_score=False)}" for s in sources]
    if note:
        lines += ["", f"*{ui_string('report_label_note')}: {note.strip()}*"]
    lines.append("")
    return lines


def _md_finding_table(snap: dict[str, Any], note: str | None, *, tag: str, body_rows: list[str]) -> list[str]:
    """Render one finding as a single two-column Markdown table.

    The GFM header row gives the tag and the verbatim chunk text their
    prominent top placement; everything else (reason, source, translation,
    reference metadata, note) sits below as label/value rows.
    """
    chunk = _truncate(snap.get("chunk_text") or "")
    lines = [
        f"| {_md_cell(tag)} | {_md_cell(chunk)} |",
        "| --- | --- |",
    ]
    lines += body_rows
    meta = " — ".join(p for p in [snap.get("filename") or "", _location(snap)] if p)
    if meta:
        lines.append(f"| {ui_string('report_label_source')} | {_md_cell(meta)} |")
    lines += _md_translation_row(snap)
    lines += _md_reference_metadata_rows(snap)
    if note:
        lines.append(f"| {ui_string('report_label_note')} | {_md_cell(note)} |")
    lines.append("")
    return lines


def _md_entity(snap: dict[str, Any], note: str | None) -> list[str]:
    body_rows: list[str] = []
    entities = _dedupe_entities(snap.get("entities") or [])
    if entities:
        rendered = ", ".join(f"{text} [{etype}]" if etype else text for text, etype in entities)
        body_rows.append(f"| {ui_string('report_label_entities')} | {_md_cell(rendered)} |")
    return _md_finding_table(snap, note, tag=str(snap.get("entity_label") or ""), body_rows=body_rows)


def _md_hate(snap: dict[str, Any], note: str | None) -> list[str]:
    category = snap.get("category") or ""
    confidence = snap.get("confidence") or ""
    body_rows: list[str] = []
    reason = snap.get("reason")
    if reason:
        body_rows.append(f"| {ui_string('report_label_reason')} | {_md_cell(reason)} |")
    return _md_finding_table(snap, note, tag=f"{category} ({confidence})".strip(), body_rows=body_rows)


def _md_summary(snap: dict[str, Any], note: str | None) -> list[str]:
    lines = [f"### {snap.get('collection') or ''}".rstrip(), "", (snap.get("text") or "").strip()]
    if note:
        lines += ["", f"*{ui_string('report_label_note')}: {note.strip()}*"]
    lines.append("")
    return lines


def _md_collection_overview(overview: dict[str, Any]) -> list[str]:
    """Markdown for the trailing document-overview section (strip + manifest table)."""
    strip = "  ·  ".join(
        [
            f"{ui_string('report_overview_documents')}: {overview.get('document_count', 0)}",
            f"{ui_string('report_overview_nodes')}: {overview.get('node_count', 0)}",
            f"{ui_string('report_overview_file_types')}: {_overview_file_types(overview)}",
            f"{ui_string('report_overview_entity_types')}: {len(overview.get('entity_types') or [])}",
        ]
    )
    lines = [f"## {ui_string(COLLECTION_OVERVIEW_HEADING)}", "", strip, ""]
    lines += [
        (
            f"| {ui_string('report_overview_col_document')} "
            f"| {ui_string('report_overview_col_type')} "
            f"| {ui_string('report_overview_col_units')} "
            f"| {ui_string('report_overview_col_hash')} |"
        ),
        "| --- | --- | ---: | --- |",
    ]
    for doc in overview.get("documents") or []:
        filename = str(doc.get("filename") or "").replace("|", "\\|")
        type_label = doc.get("type_label") or "—"
        units = _overview_units(doc)
        file_hash = _short_hash(doc.get("file_hash"))
        lines.append(f"| {filename} | {type_label} | {units} | {file_hash} |")
    lines.append("")
    return lines


_MD_DISPATCH = {
    ARTIFACT_CHAT: _md_chat,
    ARTIFACT_ENTITY: _md_entity,
    ARTIFACT_HATE: _md_hate,
    ARTIFACT_SUMMARY: _md_summary,
}


def _md_toc(grouped: OrderedDict[str, list[dict[str, Any]]], overview_present: bool) -> list[str]:
    """Render a Markdown contents list — section names only (Markdown has no pages)."""
    entries = [
        f"- {ui_string(heading_key)}" for artifact_type, heading_key in SECTION_ORDER if grouped.get(artifact_type)
    ]
    if overview_present:
        entries.append(f"- {ui_string(COLLECTION_OVERVIEW_HEADING)}")
    if not entries:
        return []
    return [f"## {ui_string('report_section_toc')}", "", *entries, ""]


def render_markdown(report: dict[str, Any]) -> str:
    """Render the report as a single Markdown document."""
    title = report.get("title") or "Report"
    lines = [f"# {title}", ""]

    # Case file (Aktenzeichen) on its own line — the Markdown analogue of the
    # PDF's running header; kept out of the subheader by design.
    if report.get("reference_number"):
        lines += [f"**{ui_string('report_label_reference')}:** {report['reference_number']}", ""]

    # Subheader: collection · creation date · operator, on a single line.
    meta_bits = []
    if report.get("collection_name"):
        meta_bits.append(f"{ui_string('report_label_collection')}: {report['collection_name']}")
    if report.get("created_at"):
        meta_bits.append(f"{ui_string('report_label_generated')}: {_date_only(report['created_at'])}")
    if report.get("operator"):
        meta_bits.append(f"{ui_string('report_label_operator')}: {report['operator']}")
    if meta_bits:
        lines += ["  ·  ".join(meta_bits), ""]

    grouped = _group_items(report.get("items") or [])
    overview = _overview_snapshot(report)
    if not any(grouped.values()) and overview is None:
        lines += [ui_string("report_empty"), ""]
        return "\n".join(lines)

    if report.get("show_toc"):
        lines += _md_toc(grouped, overview is not None)

    for artifact_type, heading_key in SECTION_ORDER:
        items = grouped.get(artifact_type) or []
        if not items:
            continue
        lines += [f"## {ui_string(heading_key)}", ""]
        renderer = _MD_DISPATCH[artifact_type]
        for item in items:
            lines += renderer(item.get("snapshot") or {}, item.get("note"))

    if overview is not None:
        lines += _md_collection_overview(overview)

    # Footer note: AI-generation caveat, after the content.
    lines += ["---", "", f"*{ui_string('report_disclaimer')}*", ""]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# HTML (also the PDF source)
# --------------------------------------------------------------------------- #
_HTML_STYLE = """
@page {
  size: A4;
  margin: 2.4cm 1.8cm 2cm;
  @top-right { content: element(refnum); }
  @bottom-left { content: element(disclaimer); }
  @bottom-right {
    content: "Page " counter(page) " / " counter(pages);
    font-family: 'Noto Sans', 'DejaVu Sans', 'Liberation Sans', Arial, sans-serif;
    font-size: 8pt; color: #888;
  }
}
* { box-sizing: border-box; }
body {
  font-family: 'Noto Sans', 'Noto Sans CJK SC', 'DejaVu Sans', 'Liberation Sans', Arial, sans-serif;
  font-size: 10.5pt; line-height: 1.45; color: #1a1a1a; margin: 0;
}
h1.report-title { font-size: 20pt; font-weight: 600; margin: 0 0 4pt; }
.report-meta { color: #666; font-size: 9pt; margin-bottom: 14pt; white-space: nowrap; }
/* Case file (top-right) + AI disclaimer (bottom-left) are lifted into the page
   margins by WeasyPrint, so they repeat on every page. Placed near the top of
   the body so the running element is current from page 1. On screen (no paged
   media) `position: running()` is ignored and they fall back to inline notes. */
.running-refnum, .running-disclaimer {
  font-family: 'Noto Sans', 'DejaVu Sans', 'Liberation Sans', Arial, sans-serif;
  font-size: 8pt; color: #888;
}
.running-refnum { position: running(refnum); }
.running-disclaimer { position: running(disclaimer); font-style: italic; }
h2.section {
  font-size: 13pt; font-weight: 600; border-bottom: 1px solid #333; padding-bottom: 3pt;
  margin: 22pt 0 8pt; break-after: avoid;
}
/* Contents (Inhaltsverzeichnis). Page numbers are emitted only in paged media
   (WeasyPrint renders @media print) via target-counter; on screen the entries are
   plain in-document anchors. */
.toc { margin: 12pt 0 18pt; break-after: avoid; }
.toc-head { font-weight: 600; font-size: 12pt; margin: 0 0 5pt; }
.toc ul { list-style: none; margin: 0; padding: 0; }
.toc li { margin: 1.5pt 0; }
.toc a { display: flex; justify-content: space-between; text-decoration: none; color: #1a1a1a; }
@media print {
  .toc a::after { content: target-counter(attr(href), page); color: #666; padding-left: 10pt; }
}
/* Flat layout: no boxed cards. Findings sit in open space, separated from one
   another by a single hairline rule between consecutive items. */
.item { margin: 0; padding: 0; }
/* Every item flows across page breaks — findings included. A finding table
   (full chunk text + entity badges) is routinely taller than a page, and any
   `break-inside: avoid` on it makes WeasyPrint push the whole block onto a
   fresh page: the section heading strands alone on an almost-empty page and
   a page-sized gap opens before the content. */
.item + .item { border-top: 1px solid #e6e6e6; margin-top: 12pt; padding-top: 12pt; }
.item-title { font-weight: 600; font-size: 11pt; margin: 0 0 2pt; }
/* One table per finding: shaded top row gives the tag + verbatim chunk text
   their prominent placement; every remaining field is a muted label/value row
   below. Verbatim evidence text keeps `pre-wrap` — never reflowed. */
table.finding { width: 100%; border-collapse: collapse; margin: 4pt 0; }
table.finding td { border: 1px solid #e6e6e6; padding: 3pt 6pt; vertical-align: top; }
/* No `break-inside: avoid` on finding rows: the chunk row and the entity-badge
   row can each approach a page in height, and an unbreakable row jumps whole to
   the next page, leaving the previous one half empty. Rows split mid-cell like
   ordinary table content instead. */
table.finding tr.f-top td { background: #f7f7f7; }
table.finding td.f-tag { width: 24%; font-weight: 600; font-size: 9.5pt; }
table.finding td.f-text { white-space: pre-wrap; font-size: 9.5pt; color: #222; }
table.finding td.f-key { width: 24%; font-weight: 600; color: #555; font-size: 8pt; }
table.finding td.f-val { white-space: pre-wrap; font-size: 8pt; color: #444; }
table.finding tr.f-subhead td {
  font-weight: 600; color: #444; font-size: 8pt; background: #fafafa; padding: 2pt 6pt;
}
/* Rendered Markdown prose (summaries, chat answers). */
.prose { margin: 2pt 0 4pt; }
.prose > :first-child { margin-top: 0; }
.prose > :last-child { margin-bottom: 0; }
.prose p { margin: 0 0 5pt; }
.prose ul, .prose ol { margin: 3pt 0 5pt; padding-left: 16pt; }
.prose li { margin: 1pt 0; }
.prose strong { font-weight: 600; }
.prose h1, .prose h2, .prose h3, .prose h4 { font-size: 11pt; font-weight: 600; margin: 7pt 0 3pt; }
.note { font-style: italic; color: #444; margin-top: 5pt; }
.label { font-weight: 600; color: #333; }
.badge {
  display: inline-block; padding: 1pt 6pt; border-radius: 3px;
  background: #f0f0f0; font-size: 8.5pt; margin: 0 4pt 2pt 0;
}
.badge .etype { color: #999; font-size: 7.5pt; }
ul.sources { margin: 4pt 0 0; padding-left: 16pt; font-size: 9pt; }
.empty { color: #888; font-style: italic; }
.overview-strip { color: #555; font-size: 9pt; margin: 4pt 0 8pt; }
table.manifest { width: 100%; border-collapse: collapse; font-size: 8.5pt; }
table.manifest th, table.manifest td {
  text-align: left; padding: 3pt 6pt; border-bottom: 1px solid #eee; vertical-align: top;
}
table.manifest th { font-weight: 600; color: #444; border-bottom: 1px solid #ccc; }
table.manifest td.num, table.manifest th.num { text-align: right; }
table.manifest td.hash { font-family: 'DejaVu Sans Mono', 'Liberation Mono', monospace; color: #666; }
table.manifest tr { break-inside: avoid; }
"""


def _esc(value: Any) -> str:
    """HTML-escape an arbitrary value."""
    return html.escape(str(value if value is not None else ""))


_MD_RENDERER: Any = None


def _render_markdown_html(text: str) -> str:
    """Render LLM-authored Markdown (summaries, chat answers) to safe HTML.

    Raw HTML in the source is escaped (``html=False``), so investigator-facing
    text can never inject markup into the report. Evidence chunk text is *not*
    routed here — it stays verbatim via :func:`_esc`. The renderer is built once
    and cached; the import is lazy so importing this module stays cheap.

    Args:
        text (str): The Markdown source.

    Returns:
        str: Rendered HTML, or ``""`` for empty input.
    """
    global _MD_RENDERER
    text = (text or "").strip()
    if not text:
        return ""
    if _MD_RENDERER is None:
        from markdown_it import MarkdownIt

        _MD_RENDERER = MarkdownIt("commonmark", {"html": False})
    return str(_MD_RENDERER.render(text))


def _html_note(note: str | None) -> str:
    if not note:
        return ""
    return f'<div class="note">{ui_string("report_label_note")}: {_esc(note)}</div>'


def _html_finding_row(label: str, value_html: str) -> str:
    """One label/value row of a finding table (value passed as ready HTML)."""
    return f'<tr><td class="f-key">{_esc(label)}</td><td class="f-val">{value_html}</td></tr>'


def _html_translation_row(snap: dict[str, Any]) -> str:
    """Finding-table row for an optional machine-translation, or ''."""
    tr = snap.get("translation") or {}
    text = _truncate(tr.get("text") or "")
    if not text:
        return ""
    label = _translation_label(str(tr.get("target_lang") or "").strip())
    return _html_finding_row(label, _esc(text))


def _html_reference_metadata_rows(snap: dict[str, Any]) -> str:
    """Finding-table rows for the source's reference metadata (provenance).

    Every captured field is kept (chain-of-custody), one muted label/value row
    per field under a bold subheading row.
    """
    items = reference_metadata_items(snap, skip_keys=BODY_TEXT_FIELDS)
    if not items:
        return ""
    head = f'<tr class="f-subhead"><td colspan="2">{ui_string("report_label_reference_metadata")}</td></tr>'
    return head + "".join(_html_finding_row(label, _esc(value)) for label, value in items)


def _html_finding_table(snap: dict[str, Any], note: str | None, *, tag_html: str, body_rows: str) -> str:
    """Render one finding as a single two-column table.

    The shaded top row gives the tag and the verbatim chunk text their
    prominent placement; everything else (reason, source, translation,
    reference metadata, note) sits below as label/value rows.
    """
    chunk = _truncate(snap.get("chunk_text") or "")
    rows = [f'<tr class="f-top"><td class="f-tag">{tag_html}</td><td class="f-text">{_esc(chunk)}</td></tr>']
    rows.append(body_rows)
    meta = " — ".join(p for p in [str(snap.get("filename") or ""), _location(snap)] if p)
    if meta:
        rows.append(_html_finding_row(ui_string("report_label_source"), _esc(meta)))
    rows.append(_html_translation_row(snap))
    rows.append(_html_reference_metadata_rows(snap))
    if note:
        rows.append(_html_finding_row(ui_string("report_label_note"), _esc(note)))
    return f'<table class="finding">{"".join(rows)}</table>'


def _html_chat(snap: dict[str, Any], note: str | None) -> str:
    parts = [
        f'<div class="item-title">{ui_string("report_label_question")}: {_esc(snap.get("user_text"))}</div>',
        f'<div class="label">{ui_string("report_label_answer")}:</div>',
        f'<div class="prose">{_render_markdown_html(snap.get("model_response") or "")}</div>',
    ]
    sources = snap.get("sources") or []
    if sources:
        items = "".join(f"<li>{_esc(_source_oneline(s, include_score=False))}</li>" for s in sources)
        parts.append(f'<div class="label">{ui_string("report_label_sources")}:</div><ul class="sources">{items}</ul>')
    parts.append(_html_note(note))
    return "".join(parts)


def _html_entity(snap: dict[str, Any], note: str | None) -> str:
    body_rows = ""
    entities = _dedupe_entities(snap.get("entities") or [])
    if entities:
        badges = "".join(
            f'<span class="badge">{_esc(text)}'
            + (f' <span class="etype">{_esc(etype)}</span>' if etype else "")
            + "</span>"
            for text, etype in entities
        )
        body_rows = _html_finding_row(ui_string("report_label_entities"), badges)
    return _html_finding_table(snap, note, tag_html=_esc(snap.get("entity_label")), body_rows=body_rows)


def _html_hate(snap: dict[str, Any], note: str | None) -> str:
    tag_html = (
        f'<span class="badge">{_esc(snap.get("category"))}</span>'
        f'<span class="badge">{_esc(snap.get("confidence"))}</span>'
    )
    body_rows = ""
    if snap.get("reason"):
        body_rows = _html_finding_row(ui_string("report_label_reason"), _esc(snap.get("reason")))
    return _html_finding_table(snap, note, tag_html=tag_html, body_rows=body_rows)


def _html_summary(snap: dict[str, Any], note: str | None) -> str:
    parts = [
        f'<div class="item-title">{_esc(snap.get("collection"))}</div>',
        f'<div class="prose">{_render_markdown_html(snap.get("text") or "")}</div>',
        _html_note(note),
    ]
    return "".join(parts)


def _html_collection_overview(overview: dict[str, Any]) -> str:
    """HTML for the trailing document-overview section (strip + manifest table)."""
    strip_items = [
        (ui_string("report_overview_documents"), overview.get("document_count", 0)),
        (ui_string("report_overview_nodes"), overview.get("node_count", 0)),
        (ui_string("report_overview_file_types"), _overview_file_types(overview)),
        (ui_string("report_overview_entity_types"), len(overview.get("entity_types") or [])),
    ]
    strip = "  ·  ".join(f"{_esc(label)}: {_esc(value)}" for label, value in strip_items)
    rows = "".join(
        "<tr>"
        f"<td>{_esc(doc.get('filename'))}</td>"
        f"<td>{_esc(doc.get('type_label') or '—')}</td>"
        f'<td class="num">{_esc(_overview_units(doc))}</td>'
        f'<td class="hash">{_esc(_short_hash(doc.get("file_hash")))}</td>'
        "</tr>"
        for doc in overview.get("documents") or []
    )
    return (
        f'<div class="overview-strip">{strip}</div>'
        '<table class="manifest"><thead><tr>'
        f"<th>{_esc(ui_string('report_overview_col_document'))}</th>"
        f"<th>{_esc(ui_string('report_overview_col_type'))}</th>"
        f'<th class="num">{_esc(ui_string("report_overview_col_units"))}</th>'
        f"<th>{_esc(ui_string('report_overview_col_hash'))}</th>"
        f"</tr></thead><tbody>{rows}</tbody></table>"
    )


_HTML_DISPATCH = {
    ARTIFACT_CHAT: _html_chat,
    ARTIFACT_ENTITY: _html_entity,
    ARTIFACT_HATE: _html_hate,
    ARTIFACT_SUMMARY: _html_summary,
}


def _html_toc(grouped: OrderedDict[str, list[dict[str, Any]]], overview_present: bool) -> str:
    """Render the contents block (Inhaltsverzeichnis) linking each present section.

    Lists only sections that have content, section-level only. Page numbers come
    from WeasyPrint's ``target-counter`` in paged media (see the ``@media print``
    stylesheet rule); on screen the entries are plain in-document anchors.
    """
    entries = [
        f'<li><a href="#{SECTION_ANCHOR[artifact_type]}">{_esc(ui_string(heading_key))}</a></li>'
        for artifact_type, heading_key in SECTION_ORDER
        if grouped.get(artifact_type)
    ]
    if overview_present:
        entries.append(
            f'<li><a href="#{COLLECTION_OVERVIEW_ANCHOR}">{_esc(ui_string(COLLECTION_OVERVIEW_HEADING))}</a></li>'
        )
    if not entries:
        return ""
    return (
        f'<nav class="toc"><div class="toc-head">{_esc(ui_string("report_section_toc"))}</div>'
        f"<ul>{''.join(entries)}</ul></nav>"
    )


def render_html(report: dict[str, Any]) -> str:
    """Render the report as a self-contained, styled HTML document.

    The same document is served as the ``.html`` export and fed to WeasyPrint
    for the PDF, so paged-media rules (``@page`` page numbers, the running
    title header, section-heading break control) live here once.
    """
    title = report.get("title") or "Report"
    locale = "en"
    try:
        from docint.utils.env_cfg import load_language_env

        locale = load_language_env().code
    except Exception:
        pass

    # Subheader: collection · creation date · operator (kept to one line via CSS).
    meta_bits = []
    if report.get("collection_name"):
        meta_bits.append(f"{ui_string('report_label_collection')}: {_esc(report['collection_name'])}")
    if report.get("created_at"):
        meta_bits.append(f"{ui_string('report_label_generated')}: {_esc(_date_only(report['created_at']))}")
    if report.get("operator"):
        meta_bits.append(f"{ui_string('report_label_operator')}: {_esc(report['operator'])}")
    meta_html = f'<div class="report-meta">{"  ·  ".join(meta_bits)}</div>' if meta_bits else ""

    body_parts = [f'<h1 class="report-title">{_esc(title)}</h1>', meta_html]

    # Case file → running top-right header (its only appearance), prefixed with a
    # short localized label ("File:" / "Az.:") so a bare number does not read as a
    # page artifact; AI disclaimer → running bottom-left footer. Both markers sit
    # near the top so the running element is current from the first page onward (a
    # marker placed last would only surface on the final page). With no case file
    # the running element is absent and the header stays empty.
    if report.get("reference_number"):
        body_parts.append(
            f'<div class="running-refnum">'
            f"{_esc(ui_string('report_label_reference_abbr'))}: {_esc(report['reference_number'])}</div>"
        )
    body_parts.append(f'<div class="running-disclaimer">{_esc(ui_string("report_disclaimer"))}</div>')

    grouped = _group_items(report.get("items") or [])
    overview = _overview_snapshot(report)
    if not any(grouped.values()) and overview is None:
        body_parts.append(f'<p class="empty">{_esc(ui_string("report_empty"))}</p>')
    else:
        if report.get("show_toc"):
            body_parts.append(_html_toc(grouped, overview is not None))
        for artifact_type, heading_key in SECTION_ORDER:
            items = grouped.get(artifact_type) or []
            if not items:
                continue
            anchor = SECTION_ANCHOR.get(artifact_type, "")
            body_parts.append(f'<h2 class="section" id="{anchor}">{_esc(ui_string(heading_key))}</h2>')
            renderer = _HTML_DISPATCH[artifact_type]
            for item in items:
                body_parts.append(f'<div class="item">{renderer(item.get("snapshot") or {}, item.get("note"))}</div>')
        if overview is not None:
            body_parts.append(
                f'<h2 class="section" id="{COLLECTION_OVERVIEW_ANCHOR}">'
                f"{_esc(ui_string(COLLECTION_OVERVIEW_HEADING))}</h2>"
            )
            body_parts.append(f'<div class="item">{_html_collection_overview(overview)}</div>')

    return (
        f'<!DOCTYPE html>\n<html lang="{_esc(locale)}">\n<head>\n'
        f'<meta charset="utf-8">\n<title>{_esc(title)}</title>\n'
        f"<style>{_HTML_STYLE}</style>\n</head>\n<body>\n"
        f"{''.join(p for p in body_parts if p)}\n</body>\n</html>\n"
    )


# --------------------------------------------------------------------------- #
# PDF (WeasyPrint, lazily imported + import-guarded)
# --------------------------------------------------------------------------- #
def _load_weasyprint() -> tuple[Any, Exception | None]:
    """Import WeasyPrint lazily; return (HTML class | None, error | None)."""
    try:
        from weasyprint import HTML

        return HTML, None
    except Exception as exc:  # ImportError, or OSError when native libs are absent
        return None, exc


def render_pdf(report: dict[str, Any]) -> bytes:
    """Render the report as a real paginated PDF via WeasyPrint.

    Args:
        report (dict[str, Any]): The report dict from ``get_report``.

    Returns:
        bytes: The PDF document.

    Raises:
        PdfEngineUnavailableError: When WeasyPrint or its native libraries are
            not installed (so the API can degrade to a 503 on the PDF route
            only, leaving the other export formats working).
    """
    html_cls, error = _load_weasyprint()
    if html_cls is None:
        raise PdfEngineUnavailableError(
            "PDF export requires WeasyPrint and its native libraries (Pango/cairo); "
            f"install them to enable PDF reports. Underlying error: {error}"
        )
    return bytes(html_cls(string=render_html(report)).write_pdf())


# --------------------------------------------------------------------------- #
# CSV bundle (ZIP)
# --------------------------------------------------------------------------- #
def _chat_csv_row(snap: dict[str, Any]) -> dict[str, Any]:
    sources = snap.get("sources") or []
    return {
        "session_id": snap.get("session_id") or "",
        "turn_idx": snap.get("turn_idx") if snap.get("turn_idx") is not None else "",
        "question": snap.get("user_text") or "",
        "answer": snap.get("model_response") or "",
        "sources": "; ".join(_source_oneline(s) for s in sources),
    }


def _summary_csv_row(snap: dict[str, Any]) -> dict[str, Any]:
    return {"collection": snap.get("collection") or "", "summary": snap.get("text") or ""}


def _overview_csv_row(doc: dict[str, Any]) -> dict[str, Any]:
    """CSV row for one document-overview manifest entry (full, untruncated hash).

    Numeric count cells (``pages``/``rows``/``nodes``) render a real ``0`` when
    the count is zero and blank only when the count is absent (``None``). The
    ``row_count: 0`` vs. ``row_count: None`` distinction is defensive only —
    ``rag.list_documents`` deletes ``max_rows`` whenever it is 0, so a real
    snapshot never actually carries ``row_count: 0``; this just keeps the cell
    correct (rather than collapsing to blank) if that upstream behavior ever
    changes.
    """
    return {
        "filename": doc.get("filename") or "",
        "type": doc.get("type_label") or "",
        "pages": doc.get("page_count") if doc.get("page_count") is not None else "",
        "rows": doc.get("row_count") if doc.get("row_count") is not None else "",
        "nodes": doc.get("node_count") if doc.get("node_count") is not None else "",
        "hash": doc.get("file_hash") or "",  # full hash — evidentiary integrity
    }


def report_csv_bundle(report: dict[str, Any]) -> bytes:
    """Build a ZIP of per-type CSVs containing only the report's selected rows.

    Entity and hate-speech CSVs reuse the canonical schemas/row builders from
    :mod:`docint.utils.csv_stream` so they match the existing collection
    exports column-for-column; chat answers and summaries get report-local
    schemas. When the trailing document-overview section is present (see
    :func:`_overview_snapshot`), an additional ``collection-overview.csv``
    manifest is included, carrying the *full* file hash (unlike the truncated
    display copy in the Markdown/HTML renderers) for evidentiary integrity.
    """
    from docint.utils.csv_stream import (
        HATE_SPEECH_COLUMNS,
        NER_SOURCE_COLUMNS,
        hate_speech_row,
        ner_source_row,
        stream_csv,
    )

    grouped = _group_items(report.get("items") or [])
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        entities = grouped.get(ARTIFACT_ENTITY) or []
        if entities:
            rows = (
                ner_source_row(i.get("snapshot") or {}, entity_label=(i.get("snapshot") or {}).get("entity_label", ""))
                for i in entities
            )
            zf.writestr("entity-findings.csv", b"".join(stream_csv(rows, NER_SOURCE_COLUMNS)))

        hate = grouped.get(ARTIFACT_HATE) or []
        if hate:
            rows = (hate_speech_row(i.get("snapshot") or {}) for i in hate)
            zf.writestr("hate-speech.csv", b"".join(stream_csv(rows, HATE_SPEECH_COLUMNS)))

        chat = grouped.get(ARTIFACT_CHAT) or []
        if chat:
            rows = (_chat_csv_row(i.get("snapshot") or {}) for i in chat)
            zf.writestr("chat-answers.csv", b"".join(stream_csv(rows, CHAT_ANSWER_COLUMNS)))

        summaries = grouped.get(ARTIFACT_SUMMARY) or []
        if summaries:
            rows = (_summary_csv_row(i.get("snapshot") or {}) for i in summaries)
            zf.writestr("summaries.csv", b"".join(stream_csv(rows, SUMMARY_COLUMNS)))

        overview = _overview_snapshot(report)
        if overview is not None:
            rows = (_overview_csv_row(d) for d in overview.get("documents") or [])
            zf.writestr("collection-overview.csv", b"".join(stream_csv(rows, COLLECTION_OVERVIEW_COLUMNS)))

        if not any(grouped.values()) and overview is None:
            zf.writestr("README.txt", "This report has no items yet.\n")

    buffer.seek(0)
    return buffer.getvalue()
