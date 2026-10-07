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
import re
import unicodedata
import zipfile
from collections import OrderedDict
from datetime import UTC, datetime
from typing import Any

from docint.core.state.pdf_progress import ProgressCallback, progress_scope
from docint.utils.env_cfg import language_endonym
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


#: What a frozen thumbnail may be: a base64 raster image, the only thing the
#: thumbnail pipeline produces. Nothing a renderer embeds can then name another
#: resource or break out of a Markdown image's parentheses.
_THUMBNAIL_DATA_URI = re.compile(r"data:image/(?:jpeg|png|webp|gif);base64,[A-Za-z0-9+/]+={0,2}")


def _thumbnail_view(container: dict[str, Any]) -> tuple[str, str] | None:
    """Validate a container's frozen thumbnail into ``(data_uri, label)``.

    Snapshots are caller-supplied JSON, so the validation is load-bearing:
    only a base64 raster image (:data:`_THUMBNAIL_DATA_URI`) may ever reach an
    ``<img src>`` or a Markdown image — anything else (a ``javascript:`` URI, a
    remote URL, an SVG, which can reference further files) renders nothing.
    Shared by the Markdown and HTML renderers so the label stays identical
    across export formats.

    Args:
        container (dict[str, Any]): A finding snapshot or a chat source dict
            that may carry a ``thumbnail`` object.

    Returns:
        tuple[str, str] | None: ``(data_uri, localized label)``, or ``None``
        when there is no renderable thumbnail.
    """
    thumb = container.get("thumbnail")
    if not isinstance(thumb, dict):
        return None
    data_uri = thumb.get("data_uri")
    if not isinstance(data_uri, str) or not _THUMBNAIL_DATA_URI.fullmatch(data_uri):
        return None
    label_key = (
        "report_label_video_keyframe" if thumb.get("kind") == "video_keyframe" else "report_label_image_evidence"
    )
    return data_uri, ui_string(label_key)


def _evidence_caption(src: dict[str, Any]) -> str:
    """Caption tying one figure to its entry in the source list.

    A chat answer can cite several images at once, and side by side they are
    indistinguishable — the caption carries the same ``[n] filename`` the list
    entry does, so a reader can tell which figure the answer meant. Findings
    hold exactly one figure and name their source in the provenance rows, so
    they pass no caption.

    Args:
        src (dict[str, Any]): A chat source dict.

    Returns:
        str: The caption, or ``""`` when the source names nothing.
    """
    name = str(src.get("filename") or src.get("source") or "").strip()
    index = src.get("citation_index")
    if isinstance(index, int) and not isinstance(index, bool):
        return f"[{index}] {name}".strip()
    return name


def _numbered_source_oneline(src: dict[str, Any]) -> str:
    """``_source_oneline`` prefixed with the citation number the generator saw."""
    line = _source_oneline(src, include_score=False)
    index = src.get("citation_index")
    if isinstance(index, int) and not isinstance(index, bool):
        return f"[{index}] {line}"
    return line


def _md_thumbnail_row(snap: dict[str, Any]) -> list[str]:
    """Markdown finding-table row for an optional frozen thumbnail, or []."""
    view = _thumbnail_view(snap)
    if view is None:
        return []
    data_uri, label = view
    return [f"| {_md_cell(label)} | ![{_md_cell(label)}]({data_uri}) |"]


def _md_translation_row(snap: dict[str, Any]) -> list[str]:
    """Markdown finding-table row for an optional machine-translation, or []."""
    tr = snap.get("translation") or {}
    text = _truncate(tr.get("text") or "")
    if not text:
        return []
    label = _translation_label(str(tr.get("target_lang") or "").strip())
    return [f"| {_md_cell(label)} | {_md_cell(text)} |"]


def _image_parts(snap: dict[str, Any]) -> tuple[str, str, str]:
    """An image finding's printed words, description and tags, each truncated.

    All three are empty for a text finding, and for a snapshot frozen before
    rows carried an image's parts apart; both render ``chunk_text`` instead.

    Args:
        snap (dict[str, Any]): A finding snapshot.

    Returns:
        tuple[str, str, str]: ``(printed words, description, tags)``.
    """
    tags = snap.get("image_tags")
    tag_text = ", ".join(str(tag).strip() for tag in tags if str(tag).strip()) if isinstance(tags, list) else ""
    return (
        _truncate(str(snap.get("ocr_text") or "")),
        _truncate(str(snap.get("image_description") or "")),
        tag_text,
    )


def _md_image_rows(snap: dict[str, Any]) -> list[str]:
    """Markdown rows for an image's printed words, their translation, its description and tags."""
    printed, description, tags = _image_parts(snap)
    lines: list[str] = []
    if printed:
        lines.append(f"| {_md_cell(ui_string('image_label_text'))} | {_md_cell(printed)} |")
        lines += _md_translation_row(snap)
    if description:
        lines.append(f"| {_md_cell(ui_string('image_label_description'))} | {_md_cell(description)} |")
    if tags:
        lines.append(f"| {_md_cell(ui_string('image_label_tags'))} | {_md_cell(tags)} |")
    return lines


def _parse_timestamp(value: Any) -> datetime | None:
    """Parse an extended ISO-8601 date or datetime, or ``None`` when it is not one.

    Only the extended form (``YYYY-MM-DD…``) is accepted: ``fromisoformat`` also
    reads the basic form, which would turn an epoch number into a date.

    Args:
        value (Any): The raw value, typically a posting's ``timestamp``.

    Returns:
        datetime | None: The parsed value, naive when it carries no offset.
    """
    text = str(value or "").strip()
    if len(text) < 10 or text[4:5] != "-":
        return None
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def _utc_offset_label(parsed: datetime) -> str:
    """Name a parsed timestamp's offset as ``UTC±HH:MM[:SS]`` (``UTC`` at zero), or ``""`` when naive."""
    offset = parsed.utcoffset()
    if offset is None:
        return ""
    total = int(offset.total_seconds())
    if total == 0:
        return "UTC"
    hours, rest = divmod(abs(total), 3600)
    minutes, seconds = divmod(rest, 60)
    label = f"UTC{'+' if total > 0 else '-'}{hours:02d}:{minutes:02d}"
    return f"{label}:{seconds:02d}" if seconds else label


def _format_timestamp(value: Any) -> str:
    """Render a timestamp in the report's locale, keeping its seconds and its own offset.

    The value is never converted to another zone: a posting time is evidence,
    and the offset it was exported with is part of it. Anything that is not an
    ISO timestamp is returned verbatim.

    Args:
        value (Any): The raw timestamp.

    Returns:
        str: E.g. ``"23.09.2026 20:31:52 (UTC+02:00)"`` under ``de``.
    """
    text = str(value or "").strip()
    parsed = _parse_timestamp(text)
    if parsed is None:
        return text
    if len(text) == 10:
        return parsed.strftime(ui_string("report_date_format"))
    pattern = ui_string("report_datetime_format")
    if text[16:17] != ":":  # no seconds in the source: print none rather than invent ":00"
        pattern = pattern.replace(":%S", "")
    shown = parsed.strftime(pattern)
    offset = _utc_offset_label(parsed)
    return f"{shown} ({offset})" if offset else shown


def _format_date(value: Any) -> str:
    """Reduce an ISO datetime to its calendar date in the report's locale.

    The report dict carries ``created_at`` as an ISO timestamp; the subheader
    shows only the creation *date* so it stays on a single line. Falls back to
    the leading 10 characters when the value cannot be parsed.
    """
    text = str(value or "").strip()
    parsed = _parse_timestamp(text)
    return parsed.strftime(ui_string("report_date_format")) if parsed is not None else text[:10]


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


# Pipeline-internal "network" values: wiring, not provenance. A transcript
# segment's real network is its parent posting's `posting_network`; the bare
# `network: nextext` stamp never reaches an investigator-facing report.
_INTERNAL_NETWORKS: frozenset[str] = frozenset({"nextext"})

# Posting fields coalesced across the two snapshot shapes (see _posting_view).
_POSTING_KEYS: tuple[str, ...] = ("network", "author", "author_id", "vanity", "timestamp", "url", "text", "id")


def _ref_meta(snap: dict[str, Any]) -> dict[str, Any]:
    """Return the snapshot's ``reference_metadata`` dict (``{}`` when absent)."""
    raw = snap.get("reference_metadata")
    return raw if isinstance(raw, dict) else {}


def _posting_view(rm: dict[str, Any]) -> tuple[dict[str, str], bool]:
    """Coalesce the two posting shapes into one view of the (parent) posting.

    A social-table row carries its posting fields bare (``network``/``author``/
    ``timestamp``/…); a media-derived artifact (transcript segment, keyframe)
    carries them prefixed (``posting_network``/…) while its bare ``network`` is
    the internal pipeline stamp and its bare ``timestamp`` is the in-media
    offset. ``posting_*`` wins; bare fields are used only when the artifact
    itself *is* the posting (i.e. its network is not pipeline-internal).

    Args:
        rm (dict[str, Any]): The snapshot's ``reference_metadata``.

    Returns:
        tuple[dict[str, str], bool]: The coalesced posting fields (empty-string
        for absent values) and whether the artifact's bare network is internal.
    """
    internal = str(rm.get("network") or "").strip().lower() in _INTERNAL_NETWORKS

    def pick(key: str) -> str:
        value = rm.get(f"posting_{key}")
        if value is None and not internal:
            value = rm.get(key)
        return str(value).strip() if value is not None else ""

    return {key: pick(key) for key in _POSTING_KEYS}, internal


def _posting_text(snap: dict[str, Any]) -> str:
    """The (parent) posting's own text, shown next to the referenced chunk.

    Empty when it would merely repeat the chunk: a finding on the posting
    itself carries the posting's text *as* its chunk, so the row only earns
    its place for media-derived artifacts (transcript segments, keyframes)
    whose chunk differs from the parent posting's text.
    """
    post, _ = _posting_view(_ref_meta(snap))
    text = _truncate(post["text"])
    chunk = _truncate(str(snap.get("chunk_text") or ""))
    if " ".join(text.split()) == " ".join(chunk.split()):
        return ""
    return text


def _provenance_rows(snap: dict[str, Any]) -> list[tuple[str, str]]:
    """Build the finding's metadata block as grouped ``(label, value)`` rows.

    Three rows at most, one per logical key, in fixed order — file reference,
    then posting, then account — each answering one investigator question and
    never repeating a value another row already carries:

    * **Source** — which file, and where in it (page/row, or the in-media
      timestamp for transcript segments; the original media file is preferred
      over the derived transcript artifact). Language/speaker ride along.
    * **Posting** — network, posting time (:func:`_format_timestamp`), posting
      ID, URL. The media ID appears only when it differs from the posting ID.
    * **Account** — author display name with handle and account ID inline.

    Pipeline-internal fields (``network: nextext``, ``type``, UUIDs,
    ``text_id``) are deliberately dropped: the full snapshot stays available in
    the JSON/CSV data exports; the report is informative, not exhaustive.
    """
    rm = _ref_meta(snap)
    post, internal = _posting_view(rm)
    rows: list[tuple[str, str]] = []

    # Source: file + position (+ language/speaker).
    file_ref = str(rm.get("source_file") or snap.get("filename") or "").strip()
    bits = [file_ref] if file_ref else []
    media_ts = str(rm.get("timestamp") or "").strip() if internal else ""
    location = media_ts or _location(snap)
    if location:
        bits.append(location)
    language = str(rm.get("language") or rm.get("detected_language") or "").strip()
    if language:
        bits.append(f"{ui_string('report_label_language')}: {language}")
    speaker = str(rm.get("speaker") or "").strip()
    if speaker:
        bits.append(f"{ui_string('report_label_speaker')}: {speaker}")
    if bits:
        rows.append((ui_string("report_label_source"), " · ".join(bits)))

    # Posting: network · timestamp · ID, with the URL on its own line.
    bits = [b for b in (post["network"], _format_timestamp(post["timestamp"])) if b]
    if post["id"]:
        bits.append(f"ID {post['id']}")
    media_id = str(rm.get("media_id") or "").strip()
    if media_id and media_id != post["id"]:
        bits.append(f"{ui_string('report_label_media_id')} {media_id}")
    posting = " · ".join(bits)
    if post["url"]:
        posting = f"{posting}\n{post['url']}" if posting else post["url"]
    if posting:
        rows.append((ui_string("report_label_posting"), posting))

    # Account: display name (handle · ID).
    details = []
    if post["vanity"]:
        details.append("@" + post["vanity"].lstrip("@"))
    if post["author_id"]:
        details.append(f"ID {post['author_id']}")
    account = post["author"]
    if details:
        joined = " · ".join(details)
        account = f"{account} ({joined})" if account else joined
    if account:
        rows.append((ui_string("report_label_account"), account))

    return rows


def _collapse_entity_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge entity findings that reference the same chunk into one block.

    An operator adding one chunk several times (once per entity of interest)
    should get a single reference block whose header names every entity label;
    the mention lists and notes are merged. Keyed by ``chunk_id`` (falling back
    to the file/locator/text tuple for old snapshots without one); original
    item dicts and snapshots are never mutated.
    """
    merged: OrderedDict[Any, dict[str, Any]] = OrderedDict()
    labels: dict[Any, list[str]] = {}
    notes: dict[Any, list[str]] = {}
    for item in items:
        snap = dict(item.get("snapshot") or {})
        key = snap.get("chunk_id") or (snap.get("filename"), snap.get("page"), snap.get("row"), snap.get("chunk_text"))
        label = str(snap.get("entity_label") or "").strip()
        note = str(item.get("note") or "").strip()
        if key not in merged:
            merged[key] = {**item, "snapshot": snap}
            labels[key] = [label] if label else []
            notes[key] = [note] if note else []
            continue
        target = merged[key]["snapshot"]
        if label and label not in labels[key]:
            labels[key].append(label)
        if note and note not in notes[key]:
            notes[key].append(note)
        target["entities"] = list(target.get("entities") or []) + list(snap.get("entities") or [])
        if not target.get("translation") and snap.get("translation"):
            target["translation"] = snap["translation"]
    for key, item in merged.items():
        item["snapshot"]["entity_label"] = " · ".join(labels[key])
        item["note"] = " · ".join(notes[key]) or None
    return list(merged.values())


def _newest_first_key(item: dict[str, Any]) -> tuple[bool, float]:
    """Sort key placing an item by the posting time its finding shows, newest first, undated last.

    The time is the one the Posting row prints (:func:`_posting_view`), so the
    order a reader sees and the dates they read agree. A naive value is read
    as UTC so that it still compares with offset-carrying ones.
    """
    post, _ = _posting_view(_ref_meta(item.get("snapshot") or {}))
    parsed = _parse_timestamp(post["timestamp"])
    if parsed is None:
        return (True, 0.0)
    return (False, -(parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)).timestamp())


def _section_items(artifact_type: str, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return the items one section renders, in the order it renders them.

    Entity findings on one chunk collapse into a single block. Hate-speech
    findings read newest first by posting time, as the extract appendix lists
    postings — a bulk add lands them in whatever order the findings table paged
    them in — and findings without a date follow in their stored order. Every
    other section keeps the order the investigator set. The JSON and CSV
    exports keep the stored order.
    """
    if artifact_type == ARTIFACT_ENTITY:
        return _collapse_entity_items(items)
    if artifact_type == ARTIFACT_HATE:
        return sorted(items, key=_newest_first_key)
    return items


def _enum_label(prefix: str, raw: Any) -> str:
    """Localized display label for a protocol enum value, or the value itself when unknown.

    Mirrors ``frontend/src/lib/hateCategoryLabel.ts``: the stored value stays
    English protocol; only what a reader sees is translated, and a value with no
    label (a future category) shows as stored.
    """
    value = str(raw or "").strip()
    if not value:
        return ""
    try:
        return ui_string(f"{prefix}{value.lower()}")
    except KeyError:
        return value


def _hate_tags(snap: dict[str, Any]) -> tuple[str, str]:
    """A hate-speech finding's category label and its labelled confidence (``""`` when absent)."""
    category = _enum_label("hate_category_", snap.get("category"))
    confidence = _enum_label("hate_confidence_", snap.get("confidence"))
    return category, f"{ui_string('report_label_confidence')}: {confidence}" if confidence else ""


# --------------------------------------------------------------------------- #
# JSON
# --------------------------------------------------------------------------- #
def render_json(report: dict[str, Any]) -> str:
    """Serialize the full report (including snapshots) as pretty JSON."""
    return json.dumps(report, ensure_ascii=False, indent=2)


# --------------------------------------------------------------------------- #
# Markdown
# --------------------------------------------------------------------------- #
def _md_chat(snap: dict[str, Any], note: str | None) -> list[str]:
    lines = [
        f"### {ui_string('report_label_question')}: {snap.get('user_text', '').strip()}",
        "",
        f"**{ui_string('report_label_answer')}:** {snap.get('model_response', '').strip()}",
    ]
    sources = snap.get("sources") or []
    if sources:
        lines += ["", f"**{ui_string('report_label_sources')}:**"]
        lines += [f"- {_numbered_source_oneline(s)}" for s in sources]
        for src in sources:
            if not isinstance(src, dict):
                continue
            view = _thumbnail_view(src)
            if view is None:
                continue
            data_uri, label = view
            caption = _evidence_caption(src)
            lines += ["", f"![{_md_cell(caption or label)}]({data_uri})"]
            if caption:
                lines.append(f"*{_md_cell(caption)}*")
    if note:
        lines += ["", f"*{ui_string('report_label_note')}: {note.strip()}*"]
    lines.append("")
    return lines


def _is_image_finding(snap: dict[str, Any]) -> bool:
    """Whether a finding was judged from a picture: it carries the picture or its parts."""
    return _thumbnail_view(snap) is not None or any(_image_parts(snap))


def _md_finding_table(snap: dict[str, Any], note: str | None, *, tag: str, detail_rows: list[str]) -> list[str]:
    """Render one finding as a single two-column Markdown table.

    The GFM header row gives the tag and the verbatim chunk text their
    prominent top placement. Content stays together below it — translation,
    then the parent posting's text — followed by the type-specific rows
    (entities / reason) and the grouped provenance block (source → posting →
    account, see :func:`_provenance_rows`). An image finding leads with its
    picture, replaces the chunk text with labelled rows for its printed words
    (their translation directly under them), description and tags, and puts
    the reason before the posting's text, as the HTML does.
    """
    printed, description, tags = _image_parts(snap)
    chunk = "" if printed or description or tags else _truncate(snap.get("chunk_text") or "")
    lines = [
        f"| {_md_cell(tag)} | {_md_cell(chunk)} |",
        "| --- | --- |",
    ]
    lines += _md_thumbnail_row(snap)
    lines += _md_image_rows(snap)
    if not printed:
        lines += _md_translation_row(snap)
    posting_text = _posting_text(snap)
    posting_rows = (
        [f"| {_md_cell(ui_string('report_label_posting_text'))} | {_md_cell(posting_text)} |"] if posting_text else []
    )
    lines += [*detail_rows, *posting_rows] if _is_image_finding(snap) else [*posting_rows, *detail_rows]
    lines += [f"| {_md_cell(label)} | {_md_cell(value)} |" for label, value in _provenance_rows(snap)]
    if note:
        lines.append(f"| {ui_string('report_label_note')} | {_md_cell(note)} |")
    lines.append("")
    return lines


def _md_entity(snap: dict[str, Any], note: str | None, number: int) -> list[str]:
    detail_rows: list[str] = []
    entities = _dedupe_entities(snap.get("entities") or [])
    if entities:
        rendered = ", ".join(f"{text} [{etype}]" if etype else text for text, etype in entities)
        detail_rows.append(f"| {ui_string('report_label_entities')} | {_md_cell(rendered)} |")
    tag = " · ".join(part for part in (f"#{number}", str(snap.get("entity_label") or "").strip()) if part)
    return _md_finding_table(snap, note, tag=tag, detail_rows=detail_rows)


def _md_hate(snap: dict[str, Any], note: str | None, number: int) -> list[str]:
    detail_rows: list[str] = []
    reason = snap.get("reason")
    if reason:
        detail_rows.append(f"| {ui_string('report_label_reason')} | {_md_cell(reason)} |")
    tag = " · ".join(part for part in (f"#{number}", *_hate_tags(snap)) if part)
    return _md_finding_table(snap, note, tag=tag, detail_rows=detail_rows)


def _summary_title(snap: dict[str, Any], collection: str) -> str:
    """A summary's own title — its collection — unless the report's subheader already names it."""
    title = str(snap.get("collection") or "").strip()
    return "" if title == collection else title


def _md_summary(snap: dict[str, Any], note: str | None, collection: str) -> list[str]:
    title = _summary_title(snap, collection)
    lines = [f"### {title}", ""] if title else []
    lines.append((snap.get("text") or "").strip())
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


#: Sections whose items are numbered findings: each carries ``#n`` and the
#: contents entry counts them.
_NUMBERED_SECTIONS: frozenset[str] = frozenset({ARTIFACT_ENTITY, ARTIFACT_HATE})

Section = tuple[str, str, list[dict[str, Any]]]


def _report_sections(report: dict[str, Any]) -> list[Section]:
    """The report's non-empty sections as ``(artifact type, heading key, items)``, in render order.

    Built once per render so the contents block counts exactly the findings
    the body numbers (entity findings on one chunk collapse into one).
    """
    grouped = _group_items(report.get("items") or [])
    return [
        (artifact_type, heading_key, _section_items(artifact_type, grouped[artifact_type]))
        for artifact_type, heading_key in SECTION_ORDER
        if grouped.get(artifact_type)
    ]


def _toc_label(artifact_type: str, heading_key: str, count: int) -> str:
    """A contents entry: the section heading, with its finding count for numbered sections."""
    heading = ui_string(heading_key)
    return f"{heading} ({count})" if artifact_type in _NUMBERED_SECTIONS else heading


def _md_item(artifact_type: str, snap: dict[str, Any], note: str | None, number: int, collection: str) -> list[str]:
    """Markdown for one item of a section, ``number`` being its 1-based place there."""
    if artifact_type == ARTIFACT_CHAT:
        return _md_chat(snap, note)
    if artifact_type == ARTIFACT_ENTITY:
        return _md_entity(snap, note, number)
    if artifact_type == ARTIFACT_HATE:
        return _md_hate(snap, note, number)
    return _md_summary(snap, note, collection)


def _md_toc(sections: list[Section], overview_present: bool) -> list[str]:
    """Render a Markdown contents list — section names only (Markdown has no pages)."""
    entries = [f"- {_toc_label(artifact_type, key, len(items))}" for artifact_type, key, items in sections]
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
        meta_bits.append(f"{ui_string('report_label_generated')}: {_format_date(report['created_at'])}")
    if report.get("operator"):
        meta_bits.append(f"{ui_string('report_label_operator')}: {report['operator']}")
    if meta_bits:
        lines += ["  ·  ".join(meta_bits), ""]

    sections = _report_sections(report)
    overview = _overview_snapshot(report)
    if not sections and overview is None:
        lines += [ui_string("report_empty"), ""]
        return "\n".join(lines)

    if report.get("show_toc"):
        lines += _md_toc(sections, overview is not None)

    collection = str(report.get("collection_name") or "").strip()
    for artifact_type, heading_key, items in sections:
        lines += [f"## {ui_string(heading_key)}", ""]
        for number, item in enumerate(items, start=1):
            lines += _md_item(artifact_type, item.get("snapshot") or {}, item.get("note"), number, collection)

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
    content: "__PAGE_LABEL__ " counter(page) " / " counter(pages);
    font-family: 'Noto Sans', 'DejaVu Sans', 'Liberation Sans', Arial, sans-serif;
    font-size: 8pt; color: #888;
  }
}
* { box-sizing: border-box; }
/* Emoji are left to fontconfig's fallback (the image installs Noto Color
   Emoji). Naming the emoji font here would make Pango draw digits, `#` and `*`
   from it too — they carry the Unicode emoji property — printing every number
   as spaced-out keycap glyphs. */
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
/* A forced break keeps the margin after it, so the chapter would start lower than a page's content. */
h2.section.page-start { break-before: page; margin-top: 0; }
/* Contents (Inhaltsverzeichnis). Page numbers are emitted only in paged media
   (WeasyPrint renders @media print) via target-counter; on screen the entries are
   plain in-document anchors. */
.toc { margin: 12pt 0 18pt; break-after: avoid; }
.toc-head { font-weight: 600; font-size: 12pt; margin: 0 0 5pt; }
.toc ul { list-style: none; margin: 0; padding: 0; }
.toc li { margin: 1.5pt 0; }
/* Floated, not flex: WeasyPrint never fills a target-counter inside a flex item (it prints 0). */
.toc a { display: block; text-decoration: none; color: #1a1a1a; }
@media print {
  .toc a::after { content: target-counter(attr(href), page); float: right; color: #666; padding-left: 10pt; }
}
/* Every item flows across page breaks — findings included. A finding table
   (full chunk text + entity badges) is routinely taller than a page, and a
   `break-inside: avoid` on the whole item makes WeasyPrint push it onto a fresh
   page: the section heading strands alone on an almost-empty page and a
   page-sized gap opens before the content. Rows decide instead (see below). */
.item { margin: 0; padding: 0; }
/* Findings are boxed tables and need only space between them; prose items
   (chat answers, summaries) are separated by a hairline rule. A rule above a
   boxed finding printed alone at the top of a page whenever a break fell
   between two findings. */
.item + .item { margin-top: 12pt; }
.item-prose + .item-prose { border-top: 1px solid #e6e6e6; padding-top: 12pt; }
.item-title { font-weight: 600; font-size: 11pt; margin: 0 0 2pt; }
/* One table per finding, ordered top-to-bottom: a full-width shaded header
   band carries the number and tags (the finding's title bar), the evidence
   follows at full width (fewer wrapped lines than a squeezed column), and
   every remaining field is a muted label/value row with a slim label column.
   Verbatim evidence text keeps `pre-wrap` — never reflowed. */
/* Fixed layout: auto layout sizes columns by measuring every cell, and text
   that breaks `anywhere` is measured per character (quadratic per cell). */
table.finding { width: 100%; border-collapse: collapse; margin: 4pt 0; table-layout: fixed; }
table.finding col.f-key { width: 16%; }
table.finding td { border: 1px solid #e6e6e6; padding: 3pt 6pt; vertical-align: top; }
/* The band is the table's <thead>: WeasyPrint never leaves it at the foot of a
   page without a row under it, and repeats it over the rest of a finding that
   continues overleaf, so every page says which finding it is showing. */
table.finding tr.f-head td { background: #f7f7f7; font-weight: 600; font-size: 9.5pt; }
.f-num { color: #777; margin-right: 6pt; }
/* Short rows move whole: a label never ends a page while its value starts the
   next. "Short" is estimated per row (`_KEEP_MAX_PT`); evidence text, the entity
   badges and any long value carry no class and split like ordinary text — an
   unbreakable row that tall would jump to the next page and leave the previous
   one half empty. */
table.finding tr.f-keep, table.finding tr.f-media { break-inside: avoid; }
table.finding td.f-text { white-space: pre-wrap; overflow-wrap: anywhere; font-size: 9.5pt; color: #222; }
/* Only a word too long for the slim label column (`Bildbeschreibung`) may hyphenate. */
table.finding td.f-key { font-weight: 600; color: #555; font-size: 8pt; hyphens: auto; hyphenate-limit-chars: 13 4 4; }
table.finding td.f-val { white-space: pre-wrap; overflow-wrap: anywhere; font-size: 8pt; color: #444; }
/* A finding judged from a picture leads with it: the figure on the left, its
   printed words, description and tags in a column beside it. A nested
   fixed-layout table, not a float: WeasyPrint sets a float's neighbouring block
   below it, not beside it. The text breaks `break-word`, not `anywhere`, so
   nothing here is measured per character. */
table.finding table.media-grid { width: 100%; border-collapse: collapse; table-layout: fixed; margin: 0; }
table.finding table.media-grid td { border: 0; padding: 0; vertical-align: top; }
table.finding table.media-grid col.m-fig { width: 66mm; }
table.finding table.media-grid td.m-fig { padding-right: 4mm; }
.m-label { font-weight: 600; color: #555; font-size: 8pt; margin: 5pt 0 1pt; }
.m-label:first-child { margin-top: 0; }
.m-text, .m-val { white-space: pre-wrap; overflow-wrap: break-word; }
.m-text { font-size: 9.5pt; color: #222; }
.m-val { font-size: 8pt; color: #444; }
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
  display: inline-block; padding: 1pt 5pt; border-radius: 3px;
  background: #f0f0f0; font-size: 8.5pt; margin: 0 3pt 2pt 0;
}
.badge.conf-high { background: #f6dede; color: #7a1717; }
.badge.conf-low { color: #666; }
.badge .etype { color: #999; font-size: 7.5pt; }
ul.sources { margin: 4pt 0 0; padding-left: 16pt; font-size: 9pt; }
/* Evidence figures. Inline-block, not flex: WeasyPrint's flex support is
   partial, and a strip of figures on a shared baseline is exactly what
   inline-block already does. Fixed width so several captions align. */
.evidence-strip { margin: 4pt 0 0; }
figure.evidence { display: inline-block; vertical-align: top; width: 55mm; margin: 0 6pt 4pt 0; }
figure.evidence img { display: block; max-width: 55mm; max-height: 70mm; border: 1px solid #ddd; }
figure.evidence figcaption {
  font-size: 7.5pt; color: #555; margin-top: 1.5pt; line-height: 1.25;
  /* break-word, not anywhere: a filename longer than the figure is wide has to
     break mid-word or it would overflow into its neighbour. */
  overflow-wrap: break-word;
}
table.finding figure.evidence { margin: 0; }
/* A finding's own picture, read at up to 62 by 90 mm: a portrait story or a
   screenshot stays legible, and a landscape frame keeps the full column. */
figure.evidence.lead { display: block; width: auto; }
figure.evidence.lead img { max-width: 62mm; max-height: 90mm; }
.empty { color: #888; font-style: italic; }
.overview-strip { color: #555; font-size: 9pt; margin: 4pt 0 8pt; }
/* Dense on purpose: the manifest lists every document of the collection, often
   hundreds of single-line rows. */
table.manifest { width: 100%; border-collapse: collapse; font-size: 7.5pt; line-height: 1.25; }
table.manifest th, table.manifest td {
  text-align: left; padding: 1.5pt 4pt; border-bottom: 1px solid #eee; vertical-align: top;
}
table.manifest th { font-weight: 600; color: #444; border-bottom: 1px solid #ccc; }
table.manifest td.num, table.manifest th.num { text-align: right; }
table.manifest td.hash { font-family: 'DejaVu Sans Mono', 'Liberation Mono', monospace; color: #666; }
table.manifest tr { break-inside: avoid; }
"""


def _html_style() -> str:
    """Return the shared stylesheet with its page-number label in the report's language.

    Shared with the extract renderer, whose appendix prints the same footer.
    """
    label = ui_string("report_label_page").replace("\\", "\\\\").replace('"', '\\"')
    return _HTML_STYLE.replace("__PAGE_LABEL__", label)


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


def _display_width(line: str) -> int:
    """How many narrow character cells a line takes: a wide one (CJK, most emoji) counts as two."""
    return sum(2 if unicodedata.east_asian_width(char) in "WF" else 1 for char in line)


def _text_height_pt(text: str, cells_per_line: int, line_pt: float) -> float:
    """Estimate, pessimistically, how tall ``text`` sets in a column ``cells_per_line`` narrow characters wide.

    Every line the text breaks itself counts, which is what makes a short text
    of many lines (a list, a poem, OCR of a poster) tall.
    """
    lines = text.splitlines() or [""]
    return line_pt * sum(max(1, -(-_display_width(line) // cells_per_line)) for line in lines)


# A detail value (8pt) in the value column beside the slim label column, and the
# tallest such row that still moves to the next page whole. A kept row that does
# not fit the rest of a page leaves that much blank space behind it, and one
# taller than a page cannot be kept at all — WeasyPrint pushes it to a fresh page
# and splits it there anyway, printing that page without the finding's band.
_VALUE_LINE = (72, 12.0)
_KEEP_MAX_PT = 150.0


def _html_finding_row(label: str, value_html: str, *, keep: bool) -> str:
    """One label/value row of a finding table (value passed as ready HTML).

    Args:
        label (str): The row's label.
        value_html (str): The value, already escaped.
        keep (bool): Whether the row moves to the next page whole rather than
            splitting.

    Returns:
        str: The row markup.
    """
    row_class = ' class="f-keep"' if keep else ""
    return f'<tr{row_class}><td class="f-key">{_esc(label)}</td><td class="f-val">{value_html}</td></tr>'


def _html_text_row(label: str, text: str) -> str:
    """A label/value row of plain text, kept whole only while it is short (:data:`_KEEP_MAX_PT`).

    So a label never ends a page while its value starts the next, and a long
    value — a posting of many lines, its translation — still splits rather than
    leaving a page half blank.
    """
    return _html_finding_row(label, _esc(text), keep=_text_height_pt(text, *_VALUE_LINE) <= _KEEP_MAX_PT)


def _html_evidence_row(label: str, text: str) -> str:
    """One labelled row of verbatim evidence text, set like the chunk text (and split like it)."""
    return f'<tr><td class="f-key">{_esc(label)}</td><td class="f-text">{_esc(text)}</td></tr>'


def _html_evidence_figure(data_uri: str, label: str, caption: str = "", *, lead: bool = False) -> str:
    """One captioned evidence figure (data URI already validated by ``_thumbnail_view``).

    A ``figure`` rather than a bare ``img`` so the image and the words naming
    it move together across a page break, and so several of them line up on a
    shared baseline in the chat strip instead of hanging off whatever height
    each happens to have.

    Args:
        data_uri (str): The validated inline image.
        label (str): Localized evidence label, used as the alt text.
        caption (str): Optional visible caption.
        lead (bool): Whether this is a finding's own picture, set larger at the
            head of its finding (see :func:`_html_media_rows`).

    Returns:
        str: The figure markup.
    """
    figcaption = f"<figcaption>{_esc(caption)}</figcaption>" if caption else ""
    figure_class = "evidence lead" if lead else "evidence"
    return f'<figure class="{figure_class}"><img src="{_esc(data_uri)}" alt="{_esc(label)}">{figcaption}</figure>'


def _translation_part(snap: dict[str, Any]) -> tuple[str, str] | None:
    """A snapshot's machine translation as ``(label, text)``, or ``None`` without one."""
    tr = snap.get("translation") or {}
    text = _truncate(tr.get("text") or "")
    if not text:
        return None
    return _translation_label(str(tr.get("target_lang") or "").strip()), text


def _html_translation_row(snap: dict[str, Any]) -> str:
    """Finding-table row for an optional machine-translation, or ''."""
    part = _translation_part(snap)
    return _html_text_row(*part) if part else ""


def _html_image_rows(snap: dict[str, Any]) -> str:
    """Rows for an image's printed words, their translation, its description and tags.

    Used for a snapshot that carries an image's parts but no picture. The
    printed words are evidence, so they keep the chunk text's verbatim style
    beside their label; the description and tags are machine-written detail,
    set like the reason.
    """
    printed, description, tags = _image_parts(snap)
    rows: list[str] = []
    if printed:
        rows.append(_html_evidence_row(ui_string("image_label_text"), printed))
        rows.append(_html_translation_row(snap))
    if description:
        rows.append(_html_text_row(ui_string("image_label_description"), description))
    if tags:
        rows.append(_html_text_row(ui_string("image_label_tags"), tags))
    return "".join(rows)


def _media_parts(snap: dict[str, Any]) -> list[tuple[str, str, bool]]:
    """The text a finding's picture carries beside it, as ``(label, text, is_evidence)``.

    In reading order: the printed words with their translation directly under
    them, the description, the tags — and the translation last when nothing is
    printed in the picture, since it then translates the finding's whole text.
    A snapshot frozen before rows carried the parts apart shows its chunk text,
    unlabelled, as the full-width chunk row would.
    """
    printed, description, tags = _image_parts(snap)
    translation = _translation_part(snap)
    translated = [(translation[0], translation[1], False)] if translation else []
    if not (printed or description or tags):
        chunk = _truncate(snap.get("chunk_text") or "")
        return ([("", chunk, True)] if chunk else []) + translated
    parts: list[tuple[str, str, bool]] = []
    if printed:
        parts += [(ui_string("image_label_text"), printed, True), *translated]
    if description:
        parts.append((ui_string("image_label_description"), description, False))
    if tags:
        parts.append((ui_string("image_label_tags"), tags, False))
    if not printed:
        parts += translated
    return parts


# How much text fits beside a lead figure, estimated pessimistically: narrow
# character cells per line and line height for evidence (9.5pt) and detail (8pt)
# text in the ~100mm column beside a 62mm figure, plus a line per part's label.
_BESIDE_EVIDENCE_LINE = (46, 14.0)
_BESIDE_DETAIL_LINE = (56, 12.0)
_BESIDE_LABEL_PT = 17.0
# The media row is kept whole, and a kept row taller than a page cannot be: it
# is pushed to a fresh page and split there anyway, which prints the finding's
# band alone on the page before or drops it from the page the finding starts
# on. Text that might not fit beside the figure goes below it instead.
_BESIDE_MAX_PT = 520.0


def _beside_height_pt(parts: list[tuple[str, str, bool]]) -> float:
    """Estimate the height of the text column beside a lead figure, in points."""
    total = 0.0
    for label, text, evidence in parts:
        cells, line_pt = _BESIDE_EVIDENCE_LINE if evidence else _BESIDE_DETAIL_LINE
        total += _text_height_pt(text, cells, line_pt) + (_BESIDE_LABEL_PT if label else 0.0)
    return total


def _html_media_rows(snap: dict[str, Any], view: tuple[str, str]) -> list[str]:
    """Rows leading a finding judged from a picture: the picture, and its text beside it.

    The picture is the evidence, so it opens the finding at a readable size,
    captioned with its kind (image or video keyframe); its printed words,
    description and tags share its height in a column beside it — a portrait
    story and its short lines of overlaid text take one block, not two. The
    row is kept whole, so the picture never sits on one page and its words on
    the next. When the text could outgrow a page, the picture keeps a row of
    its own and the text follows as ordinary rows that may split.

    Args:
        snap (dict[str, Any]): The finding snapshot.
        view (tuple[str, str]): Its validated ``(data_uri, label)`` thumbnail view.

    Returns:
        list[str]: The rows' markup.
    """
    data_uri, label = view
    figure = _html_evidence_figure(data_uri, label, label, lead=True)
    parts = _media_parts(snap)
    if _beside_height_pt(parts) <= _BESIDE_MAX_PT:
        beside = "".join(
            (f'<div class="m-label">{_esc(part_label)}</div>' if part_label else "")
            + f'<div class="{"m-text" if evidence else "m-val"}">{_esc(text)}</div>'
            for part_label, text, evidence in parts
        )
        return [
            '<tr class="f-media"><td colspan="2"><table class="media-grid">'
            '<colgroup><col class="m-fig"><col></colgroup>'
            f'<tr><td class="m-fig">{figure}</td><td class="m-parts">{beside}</td></tr></table></td></tr>'
        ]
    rows = [f'<tr class="f-media"><td colspan="2">{figure}</td></tr>']
    for part_label, text, evidence in parts:
        if not evidence:
            rows.append(_html_text_row(part_label, text))
        elif part_label:
            rows.append(_html_evidence_row(part_label, text))
        else:
            rows.append(f'<tr><td colspan="2" class="f-text">{_esc(text)}</td></tr>')
    return rows


def _html_finding_table(snap: dict[str, Any], note: str | None, *, band_html: str, detail_rows: list[str]) -> str:
    """Render one finding as a single table.

    A full-width shaded header band carries the number and tags (the finding's
    title bar), the verbatim chunk text follows at full width, and content
    stays together: the translation and the parent posting's text sit directly
    under the chunk. The type-specific rows (entities / reason) follow, then
    the grouped provenance block (source → posting → account, see
    :func:`_provenance_rows`). The chunk row is omitted when there is no chunk.

    A finding judged from a picture opens with it instead
    (:func:`_html_media_rows`), and its reason comes before the posting's
    text: the picture and its words already show what was judged. A snapshot
    carrying the picture's parts but not the picture lists them as labelled
    rows.
    """
    view = _thumbnail_view(snap)
    rows: list[str] = []
    if view is not None:
        rows += _html_media_rows(snap, view)
    else:
        printed, description, tags = _image_parts(snap)
        chunk = "" if printed or description or tags else _truncate(snap.get("chunk_text") or "")
        if chunk:
            rows.append(f'<tr><td colspan="2" class="f-text">{_esc(chunk)}</td></tr>')
        rows.append(_html_image_rows(snap))
        if not printed:
            rows.append(_html_translation_row(snap))
    posting_text = _posting_text(snap)
    posting_rows = [_html_text_row(ui_string("report_label_posting_text"), posting_text)] if posting_text else []
    rows += [*detail_rows, *posting_rows] if _is_image_finding(snap) else [*posting_rows, *detail_rows]
    rows.extend(_html_text_row(label, value) for label, value in _provenance_rows(snap))
    if note:
        rows.append(_html_text_row(ui_string("report_label_note"), note))
    return (
        '<table class="finding"><colgroup><col class="f-key"><col></colgroup>'
        f'<thead><tr class="f-head"><td colspan="2">{band_html}</td></tr></thead>'
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def _html_chat(snap: dict[str, Any], note: str | None) -> str:
    parts = [
        f'<div class="item-title">{ui_string("report_label_question")}: {_esc(snap.get("user_text"))}</div>',
        f'<div class="label">{ui_string("report_label_answer")}:</div>',
        f'<div class="prose">{_render_markdown_html(snap.get("model_response") or "")}</div>',
    ]
    sources = snap.get("sources") or []
    if sources:
        items = "".join(f"<li>{_esc(_numbered_source_oneline(s))}</li>" for s in sources)
        parts.append(f'<div class="label">{ui_string("report_label_sources")}:</div><ul class="sources">{items}</ul>')
        figures = [
            _html_evidence_figure(view[0], view[1], _evidence_caption(s))
            for s, view in ((s, _thumbnail_view(s)) for s in sources if isinstance(s, dict))
            if view is not None
        ]
        if figures:
            parts.append(f'<div class="evidence-strip">{"".join(figures)}</div>')
    parts.append(_html_note(note))
    return "".join(parts)


def _html_band(number: int, tags_html: str) -> str:
    """A finding's header band: its number in the section, then its tags."""
    return f'<span class="f-num">#{number}</span>{tags_html}'


def _html_entity(snap: dict[str, Any], note: str | None, number: int) -> str:
    detail_rows: list[str] = []
    entities = _dedupe_entities(snap.get("entities") or [])
    if entities:
        badges = "".join(
            f'<span class="badge">{_esc(text)}'
            + (f' <span class="etype">{_esc(etype)}</span>' if etype else "")
            + "</span>"
            for text, etype in entities
        )
        detail_rows.append(_html_finding_row(ui_string("report_label_entities"), badges, keep=False))
    band = _html_band(number, _esc(snap.get("entity_label")))
    return _html_finding_table(snap, note, band_html=band, detail_rows=detail_rows)


#: Confidence values that get their own badge tint (the prompt's fixed enum).
_CONFIDENCE_LEVELS: frozenset[str] = frozenset({"high", "medium", "low"})


def _html_hate(snap: dict[str, Any], note: str | None, number: int) -> str:
    category, confidence = _hate_tags(snap)
    level = str(snap.get("confidence") or "").strip().lower()
    tint = f" conf-{level}" if level in _CONFIDENCE_LEVELS else ""
    tags_html = (f'<span class="badge">{_esc(category)}</span>' if category else "") + (
        f'<span class="badge{tint}">{_esc(confidence)}</span>' if confidence else ""
    )
    detail_rows: list[str] = []
    if snap.get("reason"):
        detail_rows.append(_html_text_row(ui_string("report_label_reason"), str(snap.get("reason"))))
    return _html_finding_table(snap, note, band_html=_html_band(number, tags_html), detail_rows=detail_rows)


def _html_summary(snap: dict[str, Any], note: str | None, collection: str) -> str:
    title = _summary_title(snap, collection)
    parts = [
        f'<div class="item-title">{_esc(title)}</div>' if title else "",
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


def _html_item(artifact_type: str, snap: dict[str, Any], note: str | None, number: int, collection: str) -> str:
    """HTML for one item of a section, ``number`` being its 1-based place there."""
    if artifact_type == ARTIFACT_CHAT:
        return _html_chat(snap, note)
    if artifact_type == ARTIFACT_ENTITY:
        return _html_entity(snap, note, number)
    if artifact_type == ARTIFACT_HATE:
        return _html_hate(snap, note, number)
    return _html_summary(snap, note, collection)


def _html_toc(sections: list[Section], overview_present: bool) -> str:
    """Render the contents block (Inhaltsverzeichnis) linking each present section.

    Lists only sections that have content, section-level only, numbered
    sections with their finding count. Page numbers come from WeasyPrint's
    ``target-counter`` in paged media (see the ``@media print`` stylesheet
    rule); on screen the entries are plain in-document anchors.
    """
    entries = [
        f'<li><a href="#{SECTION_ANCHOR[artifact_type]}">{_esc(_toc_label(artifact_type, key, len(items)))}</a></li>'
        for artifact_type, key, items in sections
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
        meta_bits.append(f"{ui_string('report_label_generated')}: {_esc(_format_date(report['created_at']))}")
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

    sections = _report_sections(report)
    overview = _overview_snapshot(report)
    if not sections and overview is None:
        body_parts.append(f'<p class="empty">{_esc(ui_string("report_empty"))}</p>')
    else:
        if report.get("show_toc"):
            body_parts.append(_html_toc(sections, overview is not None))
        collection = str(report.get("collection_name") or "").strip()
        for artifact_type, heading_key, items in sections:
            anchor = SECTION_ANCHOR.get(artifact_type, "")
            body_parts.append(f'<h2 class="section" id="{anchor}">{_esc(ui_string(heading_key))}</h2>')
            item_class = "item" if artifact_type in _NUMBERED_SECTIONS else "item item-prose"
            for number, item in enumerate(items, start=1):
                rendered = _html_item(artifact_type, item.get("snapshot") or {}, item.get("note"), number, collection)
                body_parts.append(f'<div class="{item_class}">{rendered}</div>')
        if overview is not None:
            # Its own page after the findings; alone, it stays under the title.
            heading_class = "section page-start" if sections else "section"
            body_parts.append(
                f'<h2 class="{heading_class}" id="{COLLECTION_OVERVIEW_ANCHOR}">'
                f"{_esc(ui_string(COLLECTION_OVERVIEW_HEADING))}</h2>"
            )
            body_parts.append(f'<div class="item">{_html_collection_overview(overview)}</div>')

    return (
        f'<!DOCTYPE html>\n<html lang="{_esc(locale)}">\n<head>\n'
        f'<meta charset="utf-8">\n<title>{_esc(title)}</title>\n'
        f"<style>{_html_style()}</style>\n</head>\n<body>\n"
        f"{''.join(p for p in body_parts if p)}\n</body>\n</html>\n"
    )


# --------------------------------------------------------------------------- #
# PDF (WeasyPrint, lazily imported + import-guarded)
# --------------------------------------------------------------------------- #
def _load_weasyprint() -> tuple[Any, Exception | None]:
    """Import WeasyPrint lazily; return (document factory | None, error | None).

    The factory takes ``string=`` like ``weasyprint.HTML`` and lets the document
    fetch nothing but ``data:`` URIs. Every export embeds what it shows, so any
    other URL in one — an SVG thumbnail in a caller-supplied snapshot naming a
    file on the server, a Markdown image in an LLM-written answer — is refused
    instead of being read off the server's disk or the network. A fetcher is
    made per document: it keeps per-request state, and renders run in threads.
    """
    try:
        from weasyprint import HTML
        from weasyprint.urls import URLFetcher
    except Exception as exc:  # ImportError, or OSError when native libs are absent
        return None, exc

    def document(string: str) -> Any:
        return HTML(string=string, url_fetcher=URLFetcher(allowed_protocols={"data"}))

    return document, None


def pdf_engine_available() -> bool:
    """Whether WeasyPrint and its native libraries load, so a PDF can be rendered.

    Returns:
        bool: ``True`` when :func:`html_to_pdf` can run.
    """
    html_cls, _error = _load_weasyprint()
    return html_cls is not None


def html_to_pdf(document: str, progress: ProgressCallback | None = None) -> bytes:
    """Paginate a self-contained HTML document with WeasyPrint.

    Args:
        document (str): The complete HTML, styles and images inlined.
        progress (ProgressCallback | None): Told each step of the render, from
            this thread; an exception it raises stops the render.

    Returns:
        bytes: The PDF document.

    Raises:
        PdfEngineUnavailableError: When WeasyPrint or its native libraries are
            not installed, so a route can degrade to a 503 on the PDF format
            alone and leave the others working.
    """
    html_cls, error = _load_weasyprint()
    if html_cls is None:
        raise PdfEngineUnavailableError(
            "PDF export requires WeasyPrint and its native libraries (Pango/cairo); "
            f"install them to enable PDF reports. Underlying error: {error}"
        )
    with progress_scope(progress):
        return bytes(html_cls(string=document).write_pdf())


def render_pdf(report: dict[str, Any], progress: ProgressCallback | None = None) -> bytes:
    """Render the report as a real paginated PDF via WeasyPrint.

    Args:
        report (dict[str, Any]): The report dict from ``get_report``.
        progress (ProgressCallback | None): See :func:`html_to_pdf`.

    Returns:
        bytes: The PDF document.

    Raises:
        PdfEngineUnavailableError: See :func:`html_to_pdf`.
    """
    return html_to_pdf(render_html(report), progress)


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
