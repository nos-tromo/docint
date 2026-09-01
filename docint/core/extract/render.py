"""Render extract units as Markdown, transcript text and HTML.

Pure: units in, strings out. Two renderers rather than one, following the
report module — Markdown is what an analyst edits, HTML is what WeasyPrint
paginates, and evidence text must stay verbatim in both (escaped in HTML,
never re-parsed as markup).

Figures are addressed two ways. Inside a bundle they are files on disk and
the caller passes a path map; on their own (a single-source download, the
HTML, the PDF) they are inlined as data URIs so the document references
nothing outside itself.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from docint.core.extract.units import DocumentUnit, Figure, ImageUnit, MediaUnit, PostingUnit, Segment, Unit
from docint.core.state.report_render import _HTML_STYLE, _esc
from docint.utils.ui_strings import ui_string

__all__ = [
    "extract_html",
    "format_clock",
    "format_short",
    "index_markdown",
    "transcript_txt",
    "unit_markdown",
]

#: Nextext's transcript banner, copied so a docint transcript and a Nextext
#: one are the same file to a reader.
_TXT_RULE = "=" * 40

#: Section heading per unit kind.
_KIND_HEADING = {
    "document": "extract_heading_documents",
    "media": "extract_heading_media",
    "posting": "extract_heading_postings",
    "image": "extract_heading_images",
}

_EXTRA_STYLE = """
figure.extract-figure { width: 75mm; }
pre.evidence { white-space: pre-wrap; font-family: inherit; font-size: 10pt; margin: 0 0 8pt; }
h2.unit { font-size: 13pt; font-weight: 600; border-bottom: 1px solid #333; padding-bottom: 3pt;
          margin: 20pt 0 6pt; break-after: avoid; }
h3.part { font-size: 11pt; font-weight: 600; margin: 12pt 0 4pt; break-after: avoid; }
table.meta { border-collapse: collapse; margin: 0 0 8pt; font-size: 9.5pt; }
table.meta td { border: 1px solid #ddd; padding: 2pt 6pt; vertical-align: top; }
table.meta td.k { color: #555; white-space: nowrap; }
.figure-meta { font-size: 9pt; color: #444; margin: 0 0 10pt; }
.note { font-size: 9pt; color: #8a6d00; margin: 0 0 8pt; }
"""


def format_clock(seconds: float | None) -> str:
    """Format seconds as ``hh:mm:ss``.

    Args:
        seconds (float | None): Offset into a clip.

    Returns:
        str: The stamp, or ``""`` when there is no offset.
    """
    if seconds is None:
        return ""
    total = int(seconds)
    return f"{total // 3600:02d}:{(total % 3600) // 60:02d}:{total % 60:02d}"


def format_short(seconds: float | None) -> str:
    """Format seconds as ``mm:ss``, the way a frame caption reads.

    Args:
        seconds (float | None): Offset into a clip.

    Returns:
        str: The stamp, or ``""`` when there is no offset.
    """
    if seconds is None:
        return ""
    total = int(seconds)
    return f"{total // 60:02d}:{total % 60:02d}"


def _stamps(segment: Segment) -> tuple[str, str]:
    """Return a segment's start/end stamps, computing them when unstored."""
    start = segment.start_ts or format_clock(segment.start_seconds)
    end = segment.end_ts or format_clock(segment.end_seconds)
    return start, end


def transcript_txt(segments: Sequence[Segment]) -> str:
    """Render a transcript in Nextext's own banner-fenced block layout.

    Args:
        segments (Sequence[Segment]): The clip's segments, in order.

    Returns:
        str: The rendered blocks with a single trailing newline; ``""`` when
            there are no segments, so a silent clip yields no file at all.
    """
    blocks: list[str] = []
    for segment in segments:
        start, end = _stamps(segment)
        header = f"[{start} - {end}]"
        if segment.speaker:
            header = f"{header}  {segment.speaker}"
        blocks.append(f"{_TXT_RULE}\n{header}\n{_TXT_RULE}\n{segment.text}")
    return "\n\n".join(blocks) + "\n" if blocks else ""


def _figure_src(figure: Figure, paths: Mapping[str, str] | None) -> str:
    """Return the image reference for a figure: a bundle path or a data URI."""
    if paths is not None:
        return paths.get(figure.image_id, "")
    if not figure.thumbnail_b64:
        return ""
    return f"data:{figure.thumbnail_mime or 'image/jpeg'};base64,{figure.thumbnail_b64}"


def _figure_label(figure: Figure, position: int) -> str:
    """Name a figure by its time, its page or its position."""
    if figure.kind == "keyframe":
        stamp = format_short(figure.time_sec)
        return f"[{stamp}]" if stamp else f"#{(figure.index if figure.index is not None else position) + 1}"
    if figure.page_number is not None:
        return f"{ui_string('report_label_page')} {figure.page_number}"
    return f"#{position + 1}"


def _figure_facts(figure: Figure) -> list[tuple[str, str]]:
    """Return a figure's readable fields, the printed text first.

    What a picture *says* comes before what it *shows*: the printed words are
    what someone typed, the caption is what a model inferred.
    """
    facts: list[tuple[str, str]] = []
    if figure.ocr_text:
        facts.append((ui_string("extract_label_ocr_text"), figure.ocr_text))
    if figure.description:
        facts.append((ui_string("extract_label_description"), figure.description))
    if figure.tags:
        facts.append((ui_string("extract_label_tags"), ", ".join(figure.tags)))
    return facts


# --------------------------------------------------------------------------- #
# Markdown
# --------------------------------------------------------------------------- #
def _md_figures(figures: Sequence[Figure], heading: str, paths: Mapping[str, str] | None) -> list[str]:
    """Render a figure list as Markdown."""
    if not figures:
        return []
    lines = [f"## {heading}", ""]
    for position, figure in enumerate(figures):
        lines.append(f"**{_figure_label(figure, position)}**")
        lines.append("")
        src = _figure_src(figure, paths)
        if src:
            lines += [f"![{figure.image_id}]({src})", ""]
        for label, value in _figure_facts(figure):
            lines += [f"{label}: {value}", ""]
    return lines


def _md_transcript(unit: MediaUnit) -> list[str]:
    """Render a clip's transcript as timestamped Markdown lines."""
    if not unit.segments:
        return [f"*{ui_string('extract_note_no_transcript')}*", ""]
    lines = [f"## {ui_string('extract_label_transcript')}", ""]
    for segment in unit.segments:
        start, end = _stamps(segment)
        speaker = f" {segment.speaker}:" if segment.speaker else ""
        lines += [f"[{start} - {end}]{speaker} {segment.text}", ""]
    return lines


def _md_meta(rows: Sequence[tuple[str, str]]) -> list[str]:
    """Render a small key/value table."""
    present = [(label, value) for label, value in rows if value]
    if not present:
        return []
    lines = ["| | |", "|---|---|"]
    lines += [f"| {label} | {value.replace('|', chr(92) + '|')} |" for label, value in present]
    lines.append("")
    return lines


def unit_markdown(unit: Unit, figure_paths: Mapping[str, str] | None = None) -> str:
    """Render one unit as a standalone Markdown document.

    Args:
        unit (Unit): The document, clip, posting or image to render.
        figure_paths (Mapping[str, str] | None): Image id to bundle-relative
            path. Omit to inline every figure as a data URI.

    Returns:
        str: The Markdown document, ending in a newline.
    """
    lines = [f"# {unit.title}", ""]

    if isinstance(unit, DocumentUnit):
        lines += _md_meta(
            [
                (ui_string("extract_label_document"), unit.file_name),
                (ui_string("report_label_source"), unit.mimetype),
            ]
        )
        if unit.approximate_order:
            lines += [f"*{ui_string('extract_note_order_approximate')}*", ""]
        if unit.chunks:
            lines += [f"## {ui_string('extract_label_text')}", ""]
            for chunk in unit.chunks:
                lines += [chunk.text, ""]
        lines += _md_figures(unit.figures, ui_string("extract_label_figures"), figure_paths)

    elif isinstance(unit, MediaUnit):
        lines += _md_meta([(ui_string("extract_label_clip"), unit.file_name)])
        lines += _md_transcript(unit)
        lines += _md_figures(unit.keyframes, ui_string("extract_label_keyframes"), figure_paths)

    elif isinstance(unit, PostingUnit):
        ref = unit.reference
        lines += _md_meta(
            [
                (ui_string("report_label_posting"), str(ref.get("network") or "")),
                (ui_string("report_label_account"), str(ref.get("author") or ref.get("author_id") or "")),
                (ui_string("report_label_generated"), str(ref.get("timestamp") or "")),
                (ui_string("report_label_source"), str(ref.get("url") or "")),
            ]
        )
        if unit.text:
            lines += [f"## {ui_string('report_label_posting_text')}", "", unit.text, ""]
        lines += _md_figures(unit.images, ui_string("extract_heading_images"), figure_paths)
        for clip in unit.media:
            lines += [f"## {ui_string('extract_label_clip')}: {clip.title}", ""]
            lines += [line for line in _md_transcript(clip) if not line.startswith("## ")]
            lines += _md_figures(clip.keyframes, ui_string("extract_label_keyframes"), figure_paths)

    elif isinstance(unit, ImageUnit):
        lines += _md_meta([(ui_string("extract_heading_images"), unit.file_name)])
        if unit.figure is not None:
            src = _figure_src(unit.figure, figure_paths)
            if src:
                lines += [f"![{unit.figure.image_id}]({src})", ""]
            for label, value in _figure_facts(unit.figure):
                lines += [f"{label}: {value}", ""]
        elif unit.caption:
            lines += [unit.caption, ""]

    return "\n".join(lines).rstrip("\n") + "\n"


def index_markdown(
    units: Sequence[Unit],
    *,
    collection: str,
    created_at: str,
    paths: Mapping[str, str],
    pdf_skipped: bool = False,
) -> str:
    """Render the bundle's README: what is in it and where each unit sits.

    Args:
        units (Sequence[Unit]): Every unit in the extract.
        collection (str): The collection's logical name.
        created_at (str): ISO timestamp of the build.
        paths (Mapping[str, str]): Unit key to its folder inside the bundle.
        pdf_skipped (bool): Whether the combined PDF was left out.

    Returns:
        str: The README document.
    """
    lines = [
        f"# {ui_string('extract_title')}: {collection}",
        "",
        f"{ui_string('report_label_generated')}: {created_at[:19].replace('T', ' ')}",
        "",
        f"*{ui_string('extract_disclaimer')}*",
        "",
    ]
    if pdf_skipped:
        lines += [f"> {ui_string('extract_note_pdf_skipped')}", ""]
    if not units:
        lines += [ui_string("extract_empty"), ""]
        return "\n".join(lines)

    lines += [f"## {ui_string('extract_label_contents')}", ""]
    current = ""
    for unit in units:
        heading = ui_string(_KIND_HEADING[unit.kind])
        if heading != current:
            lines += ["", f"### {heading}", ""]
            current = heading
        path = paths.get(unit.key, "")
        counts = _unit_counts(unit)
        suffix = f" — {counts}" if counts else ""
        lines.append(f"- `{path}` — {unit.title}{suffix}" if path else f"- {unit.title}{suffix}")
    lines.append("")
    return "\n".join(lines)


def _unit_counts(unit: Unit) -> str:
    """Summarize a unit's size for the index line."""
    parts: list[str] = []
    if isinstance(unit, DocumentUnit) and unit.chunks:
        parts.append(f"{len(unit.chunks)} x {ui_string('extract_label_text')}")
    if isinstance(unit, MediaUnit) and unit.segments:
        parts.append(f"{len(unit.segments)} x {ui_string('extract_label_segments')}")
    if isinstance(unit, PostingUnit):
        segments = sum(len(clip.segments) for clip in unit.media)
        if segments:
            parts.append(f"{segments} x {ui_string('extract_label_segments')}")
    figures = len(unit.figures)
    if figures:
        parts.append(f"{figures} x {ui_string('extract_label_figures')}")
    return ", ".join(parts)


# --------------------------------------------------------------------------- #
# HTML (also the PDF source)
# --------------------------------------------------------------------------- #
def _html_figure(figure: Figure, position: int) -> str:
    """Render one figure with its caption and readable fields."""
    src = _figure_src(figure, None)
    label = _figure_label(figure, position)
    parts = []
    if src:
        parts.append(
            f'<figure class="evidence extract-figure"><img src="{_esc(src)}" alt="{_esc(figure.image_id)}">'
            f"<figcaption>{_esc(label)}</figcaption></figure>"
        )
    else:
        parts.append(f"<p><strong>{_esc(label)}</strong></p>")
    for name, value in _figure_facts(figure):
        parts.append(f'<p class="figure-meta"><strong>{_esc(name)}:</strong> {_esc(value)}</p>')
    return "".join(parts)


def _html_meta(rows: Sequence[tuple[str, str]]) -> str:
    """Render a small key/value table."""
    present = [(label, value) for label, value in rows if value]
    if not present:
        return ""
    cells = "".join(f'<tr><td class="k">{_esc(label)}</td><td>{_esc(value)}</td></tr>' for label, value in present)
    return f'<table class="meta">{cells}</table>'


def _html_transcript(unit: MediaUnit) -> str:
    """Render a clip's transcript as escaped, preformatted evidence."""
    if not unit.segments:
        return f'<p class="note">{_esc(ui_string("extract_note_no_transcript"))}</p>'
    lines = []
    for segment in unit.segments:
        start, end = _stamps(segment)
        speaker = f" {segment.speaker}:" if segment.speaker else ""
        lines.append(f"[{start} - {end}]{speaker} {segment.text}")
    return f'<pre class="evidence">{_esc(chr(10).join(lines))}</pre>'


def _html_unit(unit: Unit) -> str:
    """Render one unit as an HTML section."""
    parts = [f'<h2 class="unit">{_esc(unit.title)}</h2>']

    def _figures(figures: Sequence[Figure], heading: str) -> None:
        if figures:
            parts.append(f'<h3 class="part">{_esc(heading)}</h3>')
            parts.extend(_html_figure(figure, position) for position, figure in enumerate(figures))

    if isinstance(unit, DocumentUnit):
        parts.append(_html_meta([(ui_string("report_label_source"), unit.mimetype)]))
        if unit.approximate_order:
            parts.append(f'<p class="note">{_esc(ui_string("extract_note_order_approximate"))}</p>')
        if unit.chunks:
            parts.append(f'<h3 class="part">{_esc(ui_string("extract_label_text"))}</h3>')
            parts.append(f'<pre class="evidence">{_esc((chr(10) * 2).join(c.text for c in unit.chunks))}</pre>')
        _figures(unit.figures, ui_string("extract_label_figures"))
    elif isinstance(unit, MediaUnit):
        parts.append(f'<h3 class="part">{_esc(ui_string("extract_label_transcript"))}</h3>')
        parts.append(_html_transcript(unit))
        _figures(unit.keyframes, ui_string("extract_label_keyframes"))
    elif isinstance(unit, PostingUnit):
        ref = unit.reference
        parts.append(
            _html_meta(
                [
                    (ui_string("report_label_posting"), str(ref.get("network") or "")),
                    (ui_string("report_label_account"), str(ref.get("author") or ref.get("author_id") or "")),
                    (ui_string("report_label_generated"), str(ref.get("timestamp") or "")),
                    (ui_string("report_label_source"), str(ref.get("url") or "")),
                ]
            )
        )
        if unit.text:
            parts.append(f'<pre class="evidence">{_esc(unit.text)}</pre>')
        _figures(unit.images, ui_string("extract_heading_images"))
        for clip in unit.media:
            parts.append(f'<h3 class="part">{_esc(ui_string("extract_label_clip"))}: {_esc(clip.title)}</h3>')
            parts.append(_html_transcript(clip))
            _figures(clip.keyframes, ui_string("extract_label_keyframes"))
    elif isinstance(unit, ImageUnit):
        if unit.figure is not None:
            parts.append(_html_figure(unit.figure, 0))
        elif unit.caption:
            parts.append(f'<pre class="evidence">{_esc(unit.caption)}</pre>')

    return "".join(parts)


def extract_html(units: Sequence[Unit], *, collection: str, created_at: str, title: str | None = None) -> str:
    """Render an extract as a self-contained, styled HTML document.

    The same document is served as the ``.html`` file and fed to WeasyPrint
    for the PDF, so the paged-media rules live in one place — the report's
    own stylesheet, plus the few extract-specific rules above.

    Args:
        units (Sequence[Unit]): The units to render, in bundle order.
        collection (str): The collection's logical name.
        created_at (str): ISO timestamp of the build.
        title (str | None): Document title; defaults to the localized one.

    Returns:
        str: A complete HTML document.
    """
    heading = title or f"{ui_string('extract_title')}: {collection}"
    meta = "  ·  ".join(
        [
            f"{ui_string('report_label_collection')}: {collection}",
            f"{ui_string('report_label_generated')}: {created_at[:10]}",
        ]
    )
    body: list[str] = [
        f'<h1 class="report-title">{_esc(heading)}</h1>',
        f'<div class="report-meta">{_esc(meta)}</div>',
        f'<div class="running-disclaimer">{_esc(ui_string("extract_disclaimer"))}</div>',
    ]
    current = ""
    for unit in units:
        section = ui_string(_KIND_HEADING[unit.kind])
        if section != current:
            body.append(f'<h2 class="section">{_esc(section)}</h2>')
            current = section
        body.append(_html_unit(unit))
    if not units:
        body.append(f'<p class="empty">{_esc(ui_string("extract_empty"))}</p>')
    return (
        "<!DOCTYPE html>\n"
        f'<html><head><meta charset="utf-8"><title>{_esc(heading)}</title>'
        f"<style>{_HTML_STYLE}{_EXTRA_STYLE}</style></head><body>"
        f"{''.join(body)}</body></html>"
    )
