"""Render extract units as Markdown, transcript text and HTML.

Pure: units in, strings out. Two renderers rather than one, following the
report module — Markdown is what an analyst edits, HTML is what WeasyPrint
paginates, and evidence text must stay verbatim in both (escaped in HTML,
never re-parsed as markup).

The PDF is **the curated report's appendix**, not a second document with its
own conventions: it carries the report's footer disclaimer, its case-file
header, its operator line, its contents block, and — the load-bearing part —
its provenance rows, built by the report's own :func:`_provenance_rows`. A
posting rendered here and the same posting cited in the report must name the
account identically, so there is exactly one place that decides how an account
reads.

Figures are addressed two ways. Inside a bundle they are files on disk and
the caller passes a path map; on their own (a single-source download, the
HTML, the PDF) they are inlined as data URIs so the document references
nothing outside itself.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Mapping, Sequence

from docint.core.extract.units import DocumentUnit, Figure, ImageUnit, MediaUnit, PostingUnit, Segment, Unit
from docint.core.state.report_render import _HTML_STYLE, _esc, _provenance_rows
from docint.utils.ui_strings import ui_string

__all__ = [
    "appendix_numbers",
    "extract_html",
    "format_clock",
    "format_short",
    "index_markdown",
    "text_direction",
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

#: In-document anchor per unit kind, linked from the contents block. Mirrors
#: the report's ``SECTION_ANCHOR``, which is what WeasyPrint resolves page
#: numbers against.
_KIND_ANCHOR = {
    "document": "sec-extract-documents",
    "media": "sec-extract-media",
    "posting": "sec-extract-postings",
    "image": "sec-extract-images",
}

#: Prefix of an appendix entry's number, so a curated finding can cite "A.5".
_APPENDIX_PREFIX = "A"

_EXTRA_STYLE = """
pre.evidence { white-space: pre-wrap; font-family: inherit; font-size: 10pt; margin: 0 0 8pt; }
h2.unit { font-size: 13pt; font-weight: 600; border-bottom: 1px solid #333; padding-bottom: 3pt;
          margin: 20pt 0 6pt; break-after: avoid; }
h3.part { font-size: 11pt; font-weight: 600; margin: 12pt 0 4pt; break-after: avoid; }
/* Provenance key/value, sized like the report's finding table so the appendix
   and the report it belongs to print the same block. `anywhere`, not
   `break-word`: WeasyPrint excludes break-word from min-content measurement,
   so an unbroken URL would widen the column past the page margin. */
table.meta { width: 100%; border-collapse: collapse; margin: 0 0 8pt; font-size: 9.5pt; }
table.meta td { border: 1px solid #e6e6e6; padding: 2pt 6pt; vertical-align: top; overflow-wrap: anywhere; }
table.meta td.k { width: 16%; font-weight: 600; color: #555; font-size: 8pt; }
/* Figures: one row per figure, the picture beside the words that describe it —
   stacked, the text left the page's right half empty for every figure. Row-level
   `break-inside: avoid` is right here (unlike the report's finding rows, which
   approach a page in height): an <img> is monolithic in WeasyPrint, so without it
   a row starting low on the page puts its text here and its picture overleaf. */
table.figures { width: 100%; border-collapse: collapse; margin: 0 0 8pt; }
table.figures td { border: 1px solid #e6e6e6; padding: 3pt 6pt; vertical-align: top; }
table.figures tr { break-inside: avoid; }
table.figures td.fig { width: 64mm; }
table.figures figure.evidence { display: block; width: auto; margin: 0; }
table.figures figure.evidence img { max-width: 60mm; max-height: 70mm; }
table.figures p.fig-label { font-weight: 600; font-size: 9.5pt; margin: 0 0 3pt; }
/* Transcript: a table, so the stamp cannot be reordered into the words by bidi
   and a reader can scan down one speaker. <thead> repeats on every page. */
table.transcript { width: 100%; border-collapse: collapse; margin: 0 0 8pt; font-size: 9.5pt; }
table.transcript th, table.transcript td {
  border: 1px solid #e6e6e6; padding: 2pt 6pt; vertical-align: top; text-align: left;
}
table.transcript th { background: #f7f7f7; font-weight: 600; font-size: 8.5pt; color: #444; white-space: nowrap; }
table.transcript td.t-time { white-space: nowrap; font-size: 8.5pt; color: #555; }
table.transcript td.t-speaker { white-space: nowrap; color: #555; }
table.transcript td.t-text { white-space: pre-wrap; overflow-wrap: anywhere; }
[dir="rtl"] { text-align: right; }
.figure-meta { font-size: 9pt; color: #444; margin: 0 0 6pt; overflow-wrap: anywhere; }
.note { font-size: 9pt; color: #8a6d00; margin: 0 0 8pt; }
"""


def text_direction(text: str) -> str:
    """Return ``"rtl"`` when ``text`` reads right-to-left, else ``"ltr"``.

    Decided here rather than in CSS because WeasyPrint ignores ``dir="auto"``
    and ``unicode-bidi`` — only an explicit ``dir="ltr"``/``dir="rtl"`` reaches
    its line breaker. The rule is the first *strong* character's bidi class,
    which is what the Unicode algorithm itself uses for a paragraph.

    Args:
        text (str): The text about to be rendered.

    Returns:
        str: ``"rtl"`` or ``"ltr"``.
    """
    for char in text:
        if unicodedata.bidirectional(char) in {"R", "AL"}:
            return "rtl"
        if unicodedata.bidirectional(char) == "L":
            return "ltr"
    return "ltr"


def appendix_numbers(units: Sequence[Unit]) -> dict[str, str]:
    """Number the units ``A.1``, ``A.2`` … in the order they are rendered.

    A curated report cites its appendix by number, so the number belongs to the
    position in the document, not to the unit — the same posting extracted on
    its own is ``A.1`` there. Keyed by ``unit.key``, which is unique per unit.

    Args:
        units (Sequence[Unit]): The units, in bundle order.

    Returns:
        dict[str, str]: Unit key to its appendix number.
    """
    return {unit.key: f"{_APPENDIX_PREFIX}.{position}" for position, unit in enumerate(units, start=1)}


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
    """Name a figure by its time, its page, its file or its position."""
    if figure.kind == "keyframe":
        stamp = format_short(figure.time_sec)
        if stamp:
            return f"{ui_string('report_label_video_keyframe')} {stamp}"
        index = figure.index if figure.index is not None else position
        return f"{ui_string('report_label_video_keyframe')} #{index + 1}"
    if figure.page_number is not None:
        return f"{ui_string('report_label_page')} {figure.page_number}"
    return figure.file_name or f"#{position + 1}"


def _figure_facts(figure: Figure) -> list[tuple[str, str]]:
    """Return a figure's readable fields, the printed text first.

    What a picture *says* comes before what it *shows*: the printed words are
    what someone typed, the caption is what a model inferred. The tagger's
    keyword list is deliberately absent — it is retrieval machinery, and beside
    a caption that already says what the picture shows it reads as noise.
    """
    facts: list[tuple[str, str]] = []
    if figure.ocr_text:
        facts.append((ui_string("extract_label_ocr_text"), figure.ocr_text))
    if figure.description:
        facts.append((ui_string("extract_label_description"), figure.description))
    return facts


def _posting_rows(unit: PostingUnit) -> list[tuple[str, str]]:
    """Return a posting's provenance rows, exactly as the report prints them.

    Built by the report's own :func:`_provenance_rows` off a snapshot-shaped
    dict, so the appendix and the report name a source, a posting and an
    account identically — including the handle and account ID this used to
    drop. The unit's reference metadata is the same dict a report snapshot
    carries, so no translation is needed beyond the file name and row.

    Args:
        unit (PostingUnit): The posting to describe.

    Returns:
        list[tuple[str, str]]: ``(label, value)`` rows, values stripped.
    """
    snapshot = {"reference_metadata": unit.reference, "filename": unit.file_name, "row": unit.row}
    return [(label, value.strip()) for label, value in _provenance_rows(snapshot) if value.strip()]


def _unit_heading(unit: Unit, numbers: Mapping[str, str] | None) -> str:
    """Prefix a unit's title with its appendix number, when it has one."""
    number = (numbers or {}).get(unit.key)
    return f"{number}  {unit.title}" if number else unit.title


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
    """Render a small key/value table.

    A value may hold a newline (the report's Posting row puts the URL on its
    own line); a Markdown table cell cannot, so those become a line break.
    """
    present = [(label, value) for label, value in rows if value]
    if not present:
        return []
    lines = ["| | |", "|---|---|"]
    lines += [f"| {label} | {value.replace('|', chr(92) + '|').replace(chr(10), '<br>')} |" for label, value in present]
    lines.append("")
    return lines


def unit_markdown(
    unit: Unit,
    figure_paths: Mapping[str, str] | None = None,
    *,
    numbers: Mapping[str, str] | None = None,
) -> str:
    """Render one unit as a standalone Markdown document.

    Args:
        unit (Unit): The document, clip, posting or image to render.
        figure_paths (Mapping[str, str] | None): Image id to bundle-relative
            path. Omit to inline every figure as a data URI.
        numbers (Mapping[str, str] | None): Unit key to appendix number, from
            :func:`appendix_numbers`. Omit to render unnumbered.

    Returns:
        str: The Markdown document, ending in a newline.
    """
    lines = [f"# {_unit_heading(unit, numbers)}", ""]

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
        lines += _md_meta(_posting_rows(unit))
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
    reference_number: str | None = None,
    operator: str | None = None,
    numbers: Mapping[str, str] | None = None,
) -> str:
    """Render the bundle's README: what is in it and where each unit sits.

    Args:
        units (Sequence[Unit]): Every unit in the extract.
        collection (str): The collection's logical name.
        created_at (str): ISO timestamp of the build.
        paths (Mapping[str, str]): Unit key to its folder inside the bundle.
        pdf_skipped (bool): Whether the combined PDF was left out.
        reference_number (str | None): Case file this appendix belongs to.
        operator (str | None): Who built it.
        numbers (Mapping[str, str] | None): Unit key to appendix number.

    Returns:
        str: The README document.
    """
    lines = [
        f"# {ui_string('extract_title')}: {collection}",
        "",
        f"{ui_string('report_label_generated')}: {created_at[:19].replace('T', ' ')}",
        "",
    ]
    if reference_number:
        lines += [f"**{ui_string('report_label_reference')}:** {reference_number}", ""]
    if operator:
        lines += [f"**{ui_string('report_label_operator')}:** {operator}", ""]
    lines += [f"*{ui_string('extract_disclaimer')}*", ""]
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
        title = _unit_heading(unit, numbers)
        lines.append(f"- `{path}` — {title}{suffix}" if path else f"- {title}{suffix}")
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
def _html_figure_row(figure: Figure, position: int) -> str:
    """Render one figure as a table row: the picture beside its description."""
    src = _figure_src(figure, None)
    label = _figure_label(figure, position)
    picture = (
        f'<figure class="evidence"><img src="{_esc(src)}" alt="{_esc(figure.image_id)}"></figure>'
        if src
        else f'<p class="note">{_esc(ui_string("extract_note_no_image"))}</p>'
    )
    facts = [
        f'<p class="figure-meta" dir="{text_direction(value)}"><strong>{_esc(name)}:</strong> {_esc(value)}</p>'
        for name, value in _figure_facts(figure)
    ]
    return f'<tr><td class="fig">{picture}</td><td><p class="fig-label">{_esc(label)}</p>{"".join(facts)}</td></tr>'


def _html_figures(figures: Sequence[Figure]) -> str:
    """Render a figure list as one table, a row per figure."""
    if not figures:
        return ""
    rows = "".join(_html_figure_row(figure, position) for position, figure in enumerate(figures))
    return f'<table class="figures">{rows}</table>'


def _html_meta(rows: Sequence[tuple[str, str]]) -> str:
    """Render a key/value table. Newlines inside a value are kept as breaks."""
    present = [(label, value) for label, value in rows if value]
    if not present:
        return ""
    cells = "".join(
        f'<tr><td class="k">{_esc(label)}</td><td>{_esc(value).replace(chr(10), "<br>")}</td></tr>'
        for label, value in present
    )
    return f'<table class="meta">{cells}</table>'


def _html_evidence(text: str) -> str:
    """Render verbatim evidence text, right-to-left when that is what it is."""
    return f'<pre class="evidence" dir="{text_direction(text)}">{_esc(text)}</pre>'


def _html_transcript(unit: MediaUnit) -> str:
    """Render a clip's transcript as a table: time, speaker, spoken words.

    A table rather than preformatted lines for two reasons. A reader scans one
    speaker or one moment down a column; and a stamp in its own cell cannot be
    reordered into the words by the bidi algorithm, which is what displaced it
    on an Arabic transcript. The speaker column is omitted when the transcript
    was never diarized, rather than printed empty.
    """
    if not unit.segments:
        return f'<p class="note">{_esc(ui_string("extract_note_no_transcript"))}</p>'
    with_speaker = any(segment.speaker for segment in unit.segments)
    head = [f"<th>{_esc(ui_string('extract_label_time'))}</th>"]
    if with_speaker:
        head.append(f"<th>{_esc(ui_string('report_label_speaker'))}</th>")
    head.append(f"<th>{_esc(ui_string('extract_label_text'))}</th>")

    rows = []
    for segment in unit.segments:
        start, end = _stamps(segment)
        cells = [f'<td class="t-time">{_esc(f"{start} - {end}" if end else start)}</td>']
        if with_speaker:
            cells.append(f'<td class="t-speaker">{_esc(segment.speaker)}</td>')
        cells.append(f'<td class="t-text" dir="{text_direction(segment.text)}">{_esc(segment.text)}</td>')
        rows.append(f"<tr>{''.join(cells)}</tr>")
    return f'<table class="transcript"><thead><tr>{"".join(head)}</tr></thead><tbody>{"".join(rows)}</tbody></table>'


def _html_unit(unit: Unit, numbers: Mapping[str, str] | None = None) -> str:
    """Render one unit as an HTML section."""
    parts = [f'<h2 class="unit">{_esc(_unit_heading(unit, numbers))}</h2>']

    def _figures(figures: Sequence[Figure], heading: str) -> None:
        if figures:
            parts.append(f'<h3 class="part">{_esc(heading)}</h3>')
            parts.append(_html_figures(figures))

    if isinstance(unit, DocumentUnit):
        parts.append(_html_meta([(ui_string("report_label_source"), unit.mimetype)]))
        if unit.approximate_order:
            parts.append(f'<p class="note">{_esc(ui_string("extract_note_order_approximate"))}</p>')
        if unit.chunks:
            parts.append(f'<h3 class="part">{_esc(ui_string("extract_label_text"))}</h3>')
            parts.append(_html_evidence((chr(10) * 2).join(c.text for c in unit.chunks)))
        _figures(unit.figures, ui_string("extract_label_figures"))
    elif isinstance(unit, MediaUnit):
        parts.append(f'<h3 class="part">{_esc(ui_string("extract_label_transcript"))}</h3>')
        parts.append(_html_transcript(unit))
        _figures(unit.keyframes, ui_string("extract_label_keyframes"))
    elif isinstance(unit, PostingUnit):
        parts.append(_html_meta(_posting_rows(unit)))
        if unit.text:
            parts.append(_html_evidence(unit.text))
        _figures(unit.images, ui_string("extract_heading_images"))
        for clip in unit.media:
            parts.append(f'<h3 class="part">{_esc(ui_string("extract_label_clip"))}: {_esc(clip.title)}</h3>')
            parts.append(_html_transcript(clip))
            _figures(clip.keyframes, ui_string("extract_label_keyframes"))
    elif isinstance(unit, ImageUnit):
        if unit.figure is not None:
            parts.append(_html_figures([unit.figure]))
        elif unit.caption:
            parts.append(_html_evidence(unit.caption))

    return "".join(parts)


def _html_toc(units: Sequence[Unit]) -> str:
    """Render the contents block: one entry per section present, with its page.

    Section-level like the report's, and for the same reason — an appendix of
    several hundred units would otherwise open with pages of links. Page
    numbers come from WeasyPrint's ``target-counter`` in paged media; on screen
    the entries are plain anchors.
    """
    seen: list[str] = []
    for unit in units:
        if unit.kind not in seen:
            seen.append(unit.kind)
    if not seen:
        return ""
    entries = "".join(
        f'<li><a href="#{_KIND_ANCHOR[kind]}">{_esc(ui_string(_KIND_HEADING[kind]))}</a></li>' for kind in seen
    )
    return (
        f'<nav class="toc"><div class="toc-head">{_esc(ui_string("report_section_toc"))}</div><ul>{entries}</ul></nav>'
    )


def extract_html(
    units: Sequence[Unit],
    *,
    collection: str,
    created_at: str,
    title: str | None = None,
    reference_number: str | None = None,
    operator: str | None = None,
    numbers: Mapping[str, str] | None = None,
) -> str:
    """Render an extract as a self-contained, styled HTML document.

    The same document is served as the ``.html`` file and fed to WeasyPrint
    for the PDF, so the paged-media rules live in one place — the report's
    own stylesheet, plus the few extract-specific rules above. The chrome is
    the report's too: the same running footer disclaimer, the same top-right
    case-file element, the same operator line, so the appendix and the report
    it belongs to read as one deliverable.

    Args:
        units (Sequence[Unit]): The units to render, in bundle order.
        collection (str): The collection's logical name.
        created_at (str): ISO timestamp of the build.
        title (str | None): Document title; defaults to the localized one.
        reference_number (str | None): Case file, shown top-right on every page.
        operator (str | None): Who built it, shown in the meta line.
        numbers (Mapping[str, str] | None): Unit key to appendix number.

    Returns:
        str: A complete HTML document.
    """
    heading = title or f"{ui_string('extract_title')}: {collection}"
    meta_bits = [
        f"{ui_string('report_label_collection')}: {collection}",
        f"{ui_string('report_label_generated')}: {created_at[:10]}",
    ]
    if operator:
        meta_bits.append(f"{ui_string('report_label_operator')}: {operator}")
    body: list[str] = [
        f'<h1 class="report-title">{_esc(heading)}</h1>',
        f'<div class="report-meta">{_esc("  ·  ".join(meta_bits))}</div>',
    ]
    # Both markers sit near the top so the running element is current from the
    # first page onward — one placed last would only surface on the final page.
    if reference_number:
        body.append(
            f'<div class="running-refnum">'
            f"{_esc(ui_string('report_label_reference_abbr'))}: {_esc(reference_number)}</div>"
        )
    body.append(f'<div class="running-disclaimer">{_esc(ui_string("report_disclaimer"))}</div>')
    body.append(_html_toc(units))

    current = ""
    for unit in units:
        section = ui_string(_KIND_HEADING[unit.kind])
        if section != current:
            body.append(f'<h2 class="section" id="{_KIND_ANCHOR[unit.kind]}">{_esc(section)}</h2>')
            current = section
        body.append(_html_unit(unit, numbers))
    if not units:
        body.append(f'<p class="empty">{_esc(ui_string("extract_empty"))}</p>')
    return (
        "<!DOCTYPE html>\n"
        f'<html><head><meta charset="utf-8"><title>{_esc(heading)}</title>'
        f"<style>{_HTML_STYLE}{_EXTRA_STYLE}</style></head><body>"
        f"{''.join(body)}</body></html>"
    )
