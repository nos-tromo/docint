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

_CHUNK_MAX_CHARS = 1500

# CSV bundle column schemas for the chat/summary artifacts (entity & hate-speech
# reuse the canonical schemas in ``docint.utils.csv_stream``).
CHAT_ANSWER_COLUMNS: tuple[str, ...] = ("session_id", "turn_idx", "question", "answer", "sources")
SUMMARY_COLUMNS: tuple[str, ...] = ("collection", "summary")


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
def _md_reference_metadata(snap: dict[str, Any]) -> list[str]:
    """Render the source's reference metadata (provenance) rows, if any.

    Surfaces the stable citation fields (network, author, timestamp, …) captured
    at ingestion; the body-text fields are skipped because the chunk text is
    already shown above.
    """
    items = reference_metadata_items(snap, skip_keys=BODY_TEXT_FIELDS)
    if not items:
        return []
    lines = ["", f"**{ui_string('report_label_reference_metadata')}:**"]
    lines += [f"- {label}: {value}" for label, value in items]
    return lines


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


def _md_entity(snap: dict[str, Any], note: str | None) -> list[str]:
    label = snap.get("entity_label") or ""
    meta = " — ".join(p for p in [snap.get("filename") or "", _location(snap)] if p)
    lines = [f"### {label}".rstrip()]
    if meta:
        lines.append(f"*{ui_string('report_label_source')}: {meta}*")
    chunk = _truncate(snap.get("chunk_text") or "")
    if chunk:
        lines += ["", "> " + "\n> ".join(chunk.splitlines())]
    entities = _dedupe_entities(snap.get("entities") or [])
    if entities:
        rendered = ", ".join(f"{text} [{etype}]" if etype else text for text, etype in entities)
        lines += ["", f"**{ui_string('report_label_entities')}:** {rendered}"]
    lines += _md_reference_metadata(snap)
    if note:
        lines += ["", f"*{ui_string('report_label_note')}: {note.strip()}*"]
    lines.append("")
    return lines


def _md_hate(snap: dict[str, Any], note: str | None) -> list[str]:
    category = snap.get("category") or ""
    confidence = snap.get("confidence") or ""
    lines = [f"### {category} ({confidence})".strip()]
    reason = snap.get("reason")
    if reason:
        lines.append(f"**{ui_string('report_label_reason')}:** {reason.strip()}")
    meta = " — ".join(p for p in [snap.get("filename") or "", _location(snap)] if p)
    if meta:
        lines.append(f"*{ui_string('report_label_source')}: {meta}*")
    chunk = _truncate(snap.get("chunk_text") or "")
    if chunk:
        lines += ["", "> " + "\n> ".join(chunk.splitlines())]
    lines += _md_reference_metadata(snap)
    if note:
        lines += ["", f"*{ui_string('report_label_note')}: {note.strip()}*"]
    lines.append("")
    return lines


def _md_summary(snap: dict[str, Any], note: str | None) -> list[str]:
    lines = [f"### {snap.get('collection') or ''}".rstrip(), "", (snap.get("text") or "").strip()]
    if note:
        lines += ["", f"*{ui_string('report_label_note')}: {note.strip()}*"]
    lines.append("")
    return lines


_MD_DISPATCH = {
    ARTIFACT_CHAT: _md_chat,
    ARTIFACT_ENTITY: _md_entity,
    ARTIFACT_HATE: _md_hate,
    ARTIFACT_SUMMARY: _md_summary,
}


def _md_toc(grouped: OrderedDict[str, list[dict[str, Any]]]) -> list[str]:
    """Render a Markdown contents list — section names only (Markdown has no pages)."""
    entries = [
        f"- {ui_string(heading_key)}" for artifact_type, heading_key in SECTION_ORDER if grouped.get(artifact_type)
    ]
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
    if not any(grouped.values()):
        lines += [ui_string("report_empty"), ""]
        return "\n".join(lines)

    if report.get("show_toc"):
        lines += _md_toc(grouped)

    for artifact_type, heading_key in SECTION_ORDER:
        items = grouped.get(artifact_type) or []
        if not items:
            continue
        lines += [f"## {ui_string(heading_key)}", ""]
        renderer = _MD_DISPATCH[artifact_type]
        for item in items:
            lines += renderer(item.get("snapshot") or {}, item.get("note"))

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
/* Keep a compact finding (entity / hate-speech card) intact across a page break.
   Prose items (summaries, chat answers) deliberately OMIT this: they are often
   taller than a page, and `break-inside: avoid` would push the whole block onto a
   fresh page — stranding its section heading on an almost-empty page (orphaned
   heading) and leaving a large gap. Prose flows; only cards avoid in-page breaks. */
.item--card { break-inside: avoid; }
.item + .item { border-top: 1px solid #e6e6e6; margin-top: 12pt; padding-top: 12pt; }
.item-title { font-weight: 600; font-size: 11pt; margin: 0 0 2pt; }
.item-meta { color: #777; font-size: 8.5pt; margin: 0 0 5pt; }
/* Verbatim evidence text (social-media posts, document chunks) — kept exact. */
.chunk {
  white-space: pre-wrap; background: #fafafa; border-left: 2px solid #ddd;
  padding: 5pt 8pt; font-size: 9pt; color: #333; margin: 5pt 0;
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
/* Provenance: every field kept, but laid out as a compact, muted two-column
   key/value grid so the identifiers do not dominate the page. */
.rm-head { font-weight: 600; color: #444; font-size: 8.5pt; margin-top: 6pt; }
.refmeta { columns: 2; column-gap: 18pt; margin: 2pt 0 0; font-size: 8pt; color: #666; }
.refmeta .rm-row { break-inside: avoid; margin: 0 0 1pt; }
.refmeta .rm-key { font-weight: 600; color: #555; }
.empty { color: #888; font-style: italic; }
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


def _html_chunk(snap: dict[str, Any]) -> str:
    chunk = _truncate(snap.get("chunk_text") or "")
    return f'<div class="chunk">{_esc(chunk)}</div>' if chunk else ""


def _html_reference_metadata(snap: dict[str, Any]) -> str:
    """Render the source's reference metadata (provenance) as a compact block.

    Every captured field is kept (chain-of-custody), laid out as a tight, muted
    two-column key/value grid so the identifiers no longer dominate the page.
    """
    items = reference_metadata_items(snap, skip_keys=BODY_TEXT_FIELDS)
    if not items:
        return ""
    rows = "".join(
        f'<div class="rm-row"><span class="rm-key">{_esc(label)}:</span> {_esc(value)}</div>' for label, value in items
    )
    return (
        f'<div class="rm-head">{ui_string("report_label_reference_metadata")}:</div><div class="refmeta">{rows}</div>'
    )


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
    meta = " — ".join(p for p in [str(snap.get("filename") or ""), _location(snap)] if p)
    parts = [f'<div class="item-title">{_esc(snap.get("entity_label"))}</div>']
    if meta:
        parts.append(f'<div class="item-meta">{ui_string("report_label_source")}: {_esc(meta)}</div>')
    parts.append(_html_chunk(snap))
    entities = _dedupe_entities(snap.get("entities") or [])
    if entities:
        badges = "".join(
            f'<span class="badge">{_esc(text)}'
            + (f' <span class="etype">{_esc(etype)}</span>' if etype else "")
            + "</span>"
            for text, etype in entities
        )
        parts.append(f'<div><span class="label">{ui_string("report_label_entities")}:</span> {badges}</div>')
    parts.append(_html_reference_metadata(snap))
    parts.append(_html_note(note))
    return "".join(parts)


def _html_hate(snap: dict[str, Any], note: str | None) -> str:
    parts = [
        f'<div class="item-title"><span class="badge">{_esc(snap.get("category"))}</span>'
        f'<span class="badge">{_esc(snap.get("confidence"))}</span></div>'
    ]
    if snap.get("reason"):
        parts.append(
            f'<div><span class="label">{ui_string("report_label_reason")}:</span> {_esc(snap.get("reason"))}</div>'
        )
    meta = " — ".join(p for p in [str(snap.get("filename") or ""), _location(snap)] if p)
    if meta:
        parts.append(f'<div class="item-meta">{ui_string("report_label_source")}: {_esc(meta)}</div>')
    parts.append(_html_chunk(snap))
    parts.append(_html_reference_metadata(snap))
    parts.append(_html_note(note))
    return "".join(parts)


def _html_summary(snap: dict[str, Any], note: str | None) -> str:
    parts = [
        f'<div class="item-title">{_esc(snap.get("collection"))}</div>',
        f'<div class="prose">{_render_markdown_html(snap.get("text") or "")}</div>',
        _html_note(note),
    ]
    return "".join(parts)


_HTML_DISPATCH = {
    ARTIFACT_CHAT: _html_chat,
    ARTIFACT_ENTITY: _html_entity,
    ARTIFACT_HATE: _html_hate,
    ARTIFACT_SUMMARY: _html_summary,
}

# Compact "card" findings (entity / hate-speech) get `break-inside: avoid` so a
# single finding is never split across pages; the prose artifacts (summary, chat
# answer) flow instead — see the `.item--card` CSS note on orphaned headings.
_CARD_ARTIFACTS = frozenset({ARTIFACT_ENTITY, ARTIFACT_HATE})


def _html_toc(grouped: OrderedDict[str, list[dict[str, Any]]]) -> str:
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
    title header, ``break-inside: avoid`` on cards) live here once.
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
    if not any(grouped.values()):
        body_parts.append(f'<p class="empty">{_esc(ui_string("report_empty"))}</p>')
    else:
        if report.get("show_toc"):
            body_parts.append(_html_toc(grouped))
        for artifact_type, heading_key in SECTION_ORDER:
            items = grouped.get(artifact_type) or []
            if not items:
                continue
            anchor = SECTION_ANCHOR.get(artifact_type, "")
            body_parts.append(f'<h2 class="section" id="{anchor}">{_esc(ui_string(heading_key))}</h2>')
            renderer = _HTML_DISPATCH[artifact_type]
            item_class = "item item--card" if artifact_type in _CARD_ARTIFACTS else "item"
            for item in items:
                body_parts.append(
                    f'<div class="{item_class}">{renderer(item.get("snapshot") or {}, item.get("note"))}</div>'
                )

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


def report_csv_bundle(report: dict[str, Any]) -> bytes:
    """Build a ZIP of per-type CSVs containing only the report's selected rows.

    Entity and hate-speech CSVs reuse the canonical schemas/row builders from
    :mod:`docint.utils.csv_stream` so they match the existing collection
    exports column-for-column; chat answers and summaries get report-local
    schemas.
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

        if not any(grouped.values()):
            zf.writestr("README.txt", "This report has no items yet.\n")

    buffer.seek(0)
    return buffer.getvalue()
