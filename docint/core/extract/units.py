"""Partition a collection's raw points into the units an extract renders.

Pure and dependency-free: no Qdrant, no RAG, no FastAPI. Callers stream
``(point_id, payload)`` pairs in from the main collection and its ``_images``
companion, and get back one unit per document, media clip, social posting or
standalone image, each carrying its text, transcript and figures in reading
order.

Three joins are load-bearing here, because the payloads do not agree on one
identity:

- A **document** joins its figures on ``source_doc_id`` == its ``file_hash``.
- A **social** artifact joins on ``posting_uuid``, never ``file_hash``: a
  social transcript segment's ``file_hash`` is that of a transient JSONL the
  ingest deleted, so two clips on one posting would otherwise merge.
- A **standalone image** joins on ``image_id``, which is the file's own
  content hash, so its caption node and its CLIP point are one unit.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from docint.core.summary.units import _reference_metadata, is_social_payload, payload_text

__all__ = [
    "Chunk",
    "DocumentUnit",
    "Figure",
    "ImageUnit",
    "MediaUnit",
    "PostingUnit",
    "Segment",
    "Unit",
    "handle_from_url",
    "partition",
    "resolve_target",
]

#: Networks whose posting URL is ``https://<host>/<handle>/status/<id>``, so a
#: handle can be read back out of a link when the export never carried one.
_HANDLE_URL_HOSTS = frozenset({"x.com", "twitter.com", "mobile.twitter.com"})
#: Path segments that are a route, never an account.
_HANDLE_URL_RESERVED = frozenset({"i", "status", "home", "search", "hashtag", "intent"})

#: Payload key marking a transcript segment node.
DOC_KIND_FIELD = "docint_doc_kind"
TRANSCRIPT_SEGMENT = "transcript_segment"
#: Hierarchy tag whose coarse parents duplicate their children's text.
HIER_TYPE_FIELD = "docint_hier_type"
HIER_COARSE = "coarse"


@dataclass(frozen=True)
class Figure:
    """One stored image: a document figure, a video keyframe or a picture.

    Attributes:
        image_id (str): Content hash of the image bytes.
        kind (str): ``"figure"``, ``"keyframe"`` or ``"image"``.
        file_name (str): Source file the image came from, when known.
        page_number (int | None): Page the figure was extracted from.
        time_sec (float | None): Seconds into the clip a keyframe came from.
        index (int | None): Sampling position of a keyframe.
        description (str): The vision tagger's caption.
        tags (tuple[str, ...]): The vision tagger's tags.
        ocr_text (str): Text read out of the pixels.
        thumbnail_b64 (str): Stored thumbnail, base64.
        thumbnail_mime (str): Thumbnail media type.
    """

    image_id: str
    kind: str
    file_name: str = ""
    page_number: int | None = None
    time_sec: float | None = None
    index: int | None = None
    description: str = ""
    tags: tuple[str, ...] = ()
    ocr_text: str = ""
    thumbnail_b64: str = ""
    thumbnail_mime: str = "image/jpeg"


@dataclass(frozen=True)
class Segment:
    """One transcript segment.

    Attributes:
        sentence_index (int): Position in the transcript.
        start_seconds (float | None): Segment start.
        end_seconds (float | None): Segment end.
        start_ts (str): Preformatted start stamp from Nextext.
        end_ts (str): Preformatted end stamp from Nextext.
        speaker (str): Diarized speaker label, when present.
        language (str): Transcript language code.
        text (str): The spoken words.
    """

    sentence_index: int
    start_seconds: float | None
    end_seconds: float | None
    start_ts: str
    end_ts: str
    speaker: str
    language: str
    text: str


@dataclass(frozen=True)
class Chunk:
    """One chunk of document text.

    Attributes:
        point_id (str): Qdrant point id, the final ordering tie-break.
        page (int | None): Page the chunk came from.
        text (str): The chunk's text.
    """

    point_id: str
    page: int | None
    text: str


@dataclass
class DocumentUnit:
    """A document and the figures extracted from it."""

    key: str
    file_name: str
    mimetype: str = ""
    chunks: list[Chunk] = field(default_factory=list)
    figures: list[Figure] = field(default_factory=list)
    approximate_order: bool = False
    kind: str = "document"

    @property
    def title(self) -> str:
        """Display name for this unit."""
        return self.file_name or self.key


@dataclass
class MediaUnit:
    """One audio/video clip: its transcript and its keyframes."""

    key: str
    file_name: str = ""
    segments: list[Segment] = field(default_factory=list)
    keyframes: list[Figure] = field(default_factory=list)
    kind: str = "media"

    @property
    def title(self) -> str:
        """Display name for this unit."""
        return self.file_name or self.key

    @property
    def figures(self) -> list[Figure]:
        """Every image this unit renders."""
        return self.keyframes


@dataclass
class PostingUnit:
    """A social posting with every artifact linked to it."""

    key: str
    reference: dict[str, Any] = field(default_factory=dict)
    text: str = ""
    file_name: str = ""
    row: int | None = None
    images: list[Figure] = field(default_factory=list)
    media: list[MediaUnit] = field(default_factory=list)
    source_hashes: set[str] = field(default_factory=set)
    kind: str = "posting"

    @property
    def title(self) -> str:
        """Display name for this unit: the account, then when it posted.

        An account's postings are otherwise a run of identical headings — the
        example collection opens with eight of them — and the timestamp is the
        one field that always distinguishes two posts by the same author.
        """
        author = str(self.reference.get("author") or self.reference.get("author_id") or "").strip()
        stamp = str(self.reference.get("timestamp") or "").strip()[:16].replace("T", " ")
        if author and stamp:
            return f"{author} · {stamp}"
        return author or stamp or self.key

    @property
    def figures(self) -> list[Figure]:
        """Every image this unit renders, its clips' frames included."""
        return [*self.images, *(frame for clip in self.media for frame in clip.keyframes)]


@dataclass
class ImageUnit:
    """A standalone image file."""

    key: str
    file_name: str = ""
    caption: str = ""
    figure: Figure | None = None
    kind: str = "image"

    @property
    def title(self) -> str:
        """Display name for this unit."""
        return self.file_name or self.key

    @property
    def figures(self) -> list[Figure]:
        """Every image this unit renders."""
        return [self.figure] if self.figure is not None else []


Unit = DocumentUnit | MediaUnit | PostingUnit | ImageUnit

#: Sort rank per kind, so a bundle always lists documents, then clips, then
#: postings, then loose pictures.
_KIND_RANK = {"document": 0, "media": 1, "posting": 2, "image": 3}


def _as_int(value: Any) -> int | None:
    """Coerce a payload value to ``int``, or ``None`` when it is not one."""
    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_float(value: Any) -> float | None:
    """Coerce a payload value to ``float``, or ``None`` when it is not one."""
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _text(value: Any) -> str:
    """Return a stripped string for any payload value."""
    return str(value).strip() if value is not None else ""


def _node_data(payload: dict[str, Any]) -> dict[str, Any]:
    """Return the parsed llama-index node blob, or ``{}``."""
    raw = payload.get("_node_content")
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
        except (TypeError, ValueError):
            return {}
        if isinstance(parsed, dict):
            return parsed
    return {}


def _first(payload: dict[str, Any], *keys: str) -> str:
    """Return the first non-empty string among ``keys``."""
    for key in keys:
        value = _text(payload.get(key))
        if value:
            return value
    return ""


def _figure(payload: dict[str, Any], kind: str) -> Figure:
    """Build a :class:`Figure` from an ``_images`` companion payload."""
    tags = payload.get("llm_tags")
    return Figure(
        image_id=_text(payload.get("image_id")),
        kind=kind,
        file_name=_first(payload, "file_name", "filename", "source_file") or _basename(payload.get("source_path")),
        page_number=_as_int(payload.get("page_number")),
        time_sec=_as_float(payload.get("keyframe_time_sec")),
        index=_as_int(payload.get("keyframe_index")),
        description=_text(payload.get("llm_description")),
        tags=tuple(_text(tag) for tag in tags if _text(tag)) if isinstance(tags, list) else (),
        ocr_text=_text(payload.get("ocr_text")),
        thumbnail_b64=_text(payload.get("thumbnail_b64")),
        thumbnail_mime=_text(payload.get("thumbnail_mime")) or "image/jpeg",
    )


def _basename(value: Any) -> str:
    """Return the file name part of a path-ish payload value."""
    text = _text(value)
    return text.rsplit("/", 1)[-1] if text else ""


def handle_from_url(url: str) -> str:
    """Return the account handle a posting URL names, or ``""``.

    A chat-style export (the ``messages`` schema: ``Chat ID``/``Sender``/
    ``Text``) has no account-id or handle column at all, so the only place the
    account is identified is the permalink —
    ``https://x.com/<handle>/status/<id>``. Reading it back is what lets the
    appendix name an account the way the curated report does, on collections
    already ingested and with no pipeline change.

    Deliberately narrow: only hosts whose first path segment *is* the account
    (:data:`_HANDLE_URL_HOSTS`), and never a known route segment. The numeric
    id in such a URL identifies the *posting*, not the account, so it is never
    surfaced as one.

    Args:
        url (str): The posting's URL.

    Returns:
        str: The handle without its ``@``, or ``""`` when the URL names none.
    """
    text = _text(url)
    if "://" not in text:
        return ""
    _scheme, _, rest = text.partition("://")
    host, _, path = rest.partition("/")
    host = host.split("@")[-1].split(":")[0].lower().removeprefix("www.")
    if host not in _HANDLE_URL_HOSTS or not path:
        return ""
    handle = path.split("?")[0].split("#")[0].split("/")[0].strip()
    return "" if handle.lower() in _HANDLE_URL_RESERVED else handle


def _segment(payload: dict[str, Any]) -> Segment:
    """Build a :class:`Segment` from a transcript-segment payload."""
    return Segment(
        sentence_index=_as_int(payload.get("sentence_index")) or 0,
        start_seconds=_as_float(payload.get("start_seconds")),
        end_seconds=_as_float(payload.get("end_seconds")),
        start_ts=_text(payload.get("start_ts")),
        end_ts=_text(payload.get("end_ts")),
        speaker=_text(payload.get("speaker")),
        language=_first(payload, "whisper_language", "language", "detected_language"),
        text=payload_text(payload),
    )


def _figure_kind(payload: dict[str, Any]) -> str:
    """Classify an ``_images`` point as a keyframe, a figure or a picture.

    A social keyframe and a social image share ``source_type: social_media``
    (the linker stamps it on both), so the discriminator is the nested
    reference metadata's own ``type``.
    """
    source_type = _text(payload.get("source_type"))
    if source_type in {"social_media_keyframe", "video_keyframe"}:
        return "keyframe"
    if _text(_reference_metadata(payload).get("type")) == "keyframe":
        return "keyframe"
    if source_type == "document":
        return "figure"
    return "image"


def _is_transcript(payload: dict[str, Any]) -> bool:
    """Return whether a main-collection payload is a transcript segment."""
    return _text(payload.get(DOC_KIND_FIELD)) == TRANSCRIPT_SEGMENT


def _posting_uuid(payload: dict[str, Any]) -> str:
    """Return the posting a payload belongs to, by any of its spellings."""
    ref = _reference_metadata(payload)
    return _text(payload.get("posting_uuid")) or _text(ref.get("posting_uuid")) or _text(ref.get("uuid"))


def _chunk_order(point_id: str, payload: dict[str, Any]) -> tuple[int, int, int, str]:
    """Return a document chunk's reading-order key.

    Order is ``(page, character offset, re-split part, point id)``. Only
    ``page`` is stamped flat; the offset lives inside the serialized node, and
    a chunk with neither is ordered by point id alone — which the unit reports
    as approximate rather than passing off as the document's own order.
    """
    node = _node_data(payload)
    page = _as_int(payload.get("page")) or _as_int(payload.get("page_number")) or 0
    offset = _as_int(node.get("start_char_idx")) or 0
    part = _as_int(payload.get("split_part_index")) or 0
    return (page, offset, part, point_id)


def _row_index(payload: dict[str, Any]) -> int | None:
    """Return the table row a payload came from, by either of its spellings."""
    row = _as_int(payload.get("row"))
    if row is not None:
        return row
    table = payload.get("table")
    return _as_int(table.get("row_index")) if isinstance(table, dict) else None


def _has_order_signal(payload: dict[str, Any]) -> bool:
    """Return whether a chunk carries a page or a character offset."""
    if payload.get("page") is not None or payload.get("page_number") is not None:
        return True
    return _node_data(payload).get("start_char_idx") is not None


@dataclass
class _Accumulator:
    """Mutable buckets filled while scanning points, keyed by unit identity."""

    documents: dict[str, DocumentUnit] = field(default_factory=dict)
    media: dict[str, MediaUnit] = field(default_factory=dict)
    postings: dict[str, PostingUnit] = field(default_factory=dict)
    images: dict[str, ImageUnit] = field(default_factory=dict)
    chunk_rows: dict[str, list[tuple[tuple[int, int, int, str], Chunk, bool]]] = field(default_factory=dict)
    segment_rows: dict[tuple[str, str], list[Segment]] = field(default_factory=dict)

    def posting(self, uuid: str) -> PostingUnit:
        """Return the posting unit for ``uuid``, creating it if needed."""
        return self.postings.setdefault(uuid, PostingUnit(key=uuid))

    def clip(self, owner: str, media_key: str) -> MediaUnit:
        """Return the clip ``media_key`` of ``owner`` (a posting, or standalone)."""
        if not owner:
            return self.media.setdefault(media_key, MediaUnit(key=media_key))
        posting = self.posting(owner)
        for clip in posting.media:
            if clip.key == media_key:
                return clip
        clip = MediaUnit(key=media_key)
        posting.media.append(clip)
        return clip


def _ingest_main_point(point_id: str, payload: dict[str, Any], acc: _Accumulator) -> None:
    """Route one main-collection point into its unit bucket."""
    if _text(payload.get(HIER_TYPE_FIELD)) == HIER_COARSE:
        return

    if _is_transcript(payload):
        uuid = _posting_uuid(payload)
        media_key = _text(payload.get("media_id")) or _first(payload, "media_file_hash", "file_hash")
        if not media_key:
            return
        clip = acc.clip(uuid, media_key)
        if not clip.file_name:
            clip.file_name = _first(payload, "source_file", "file_name", "filename")
        acc.segment_rows.setdefault((uuid, media_key), []).append(_segment(payload))
        return

    ref = _reference_metadata(payload)
    uuid = _text(ref.get("uuid"))
    if uuid and (is_social_payload(payload) or _text(ref.get("type")) == "posting"):
        posting = acc.posting(uuid)
        posting.reference = {**ref, **posting.reference} if posting.reference else dict(ref)
        posting.text = posting.text or _text(ref.get("text")) or payload_text(payload)
        posting.file_name = posting.file_name or _first(payload, "file_name", "filename")
        if posting.row is None:
            posting.row = _row_index(payload)
        if not _text(posting.reference.get("vanity")):
            handle = handle_from_url(_text(posting.reference.get("url")))
            if handle:
                posting.reference["vanity"] = handle
        file_hash = _text(payload.get("file_hash"))
        if file_hash:
            posting.source_hashes.add(file_hash)
        return

    image_id = _text(payload.get("image_id"))
    if image_id:
        unit = acc.images.setdefault(image_id, ImageUnit(key=image_id))
        unit.file_name = unit.file_name or _first(payload, "file_name", "filename")
        unit.caption = unit.caption or payload_text(payload)
        return

    key = _text(payload.get("file_hash")) or _first(payload, "file_name", "filename")
    if not key:
        return
    unit = acc.documents.setdefault(key, DocumentUnit(key=key, file_name=""))
    unit.file_name = unit.file_name or _first(payload, "file_name", "filename")
    unit.mimetype = unit.mimetype or _first(payload, "mimetype", "file_type", "mime_type")
    chunk = Chunk(point_id=point_id, page=_as_int(payload.get("page")), text=payload_text(payload))
    acc.chunk_rows.setdefault(key, []).append((_chunk_order(point_id, payload), chunk, _has_order_signal(payload)))


def _ingest_image_point(payload: dict[str, Any], acc: _Accumulator) -> None:
    """Route one ``_images`` companion point into its unit bucket."""
    kind = _figure_kind(payload)
    figure = _figure(payload, kind)
    if not figure.image_id:
        return

    uuid = _posting_uuid(payload)
    if uuid:
        posting = acc.posting(uuid)
        if kind == "keyframe":
            media_key = _text(payload.get("media_id")) or figure.file_name or uuid
            acc.clip(uuid, media_key).keyframes.append(figure)
        else:
            posting.images.append(figure)
        return

    source_doc_id = _text(payload.get("source_doc_id"))
    source_type = _text(payload.get("source_type"))
    if kind == "keyframe" and source_doc_id:
        clip = acc.clip("", source_doc_id)
        clip.file_name = clip.file_name or figure.file_name
        clip.keyframes.append(figure)
        return
    if source_type == "document" and source_doc_id:
        unit = acc.documents.setdefault(source_doc_id, DocumentUnit(key=source_doc_id, file_name=figure.file_name))
        unit.figures.append(figure)
        return

    key = figure.image_id
    unit_img = acc.images.setdefault(key, ImageUnit(key=key))
    unit_img.file_name = unit_img.file_name or figure.file_name
    unit_img.figure = figure


def _finalize(acc: _Accumulator) -> list[Unit]:
    """Order every bucket's members and return the units, sorted."""
    for key, rows in acc.chunk_rows.items():
        unit = acc.documents[key]
        unit.chunks = [chunk for _order, chunk, _signal in sorted(rows, key=lambda row: row[0])]
        unit.approximate_order = not any(signal for _order, _chunk, signal in rows)

    for (uuid, media_key), segments in acc.segment_rows.items():
        clip = acc.clip(uuid, media_key)
        clip.segments = sorted(segments, key=lambda seg: (seg.sentence_index, seg.start_seconds or 0.0))

    def _frame_order(frame: Figure) -> tuple[int, float, int, str]:
        return (
            0 if frame.time_sec is not None else 1,
            frame.time_sec if frame.time_sec is not None else 0.0,
            frame.index if frame.index is not None else 0,
            frame.image_id,
        )

    for clip in acc.media.values():
        clip.keyframes.sort(key=_frame_order)
    for posting in acc.postings.values():
        posting.images.sort(key=lambda figure: figure.image_id)
        posting.media.sort(key=lambda clip: clip.key)
        for clip in posting.media:
            clip.keyframes.sort(key=_frame_order)
    for document in acc.documents.values():
        document.figures.sort(key=lambda figure: (figure.page_number or 0, figure.image_id))

    units: list[Unit] = [*acc.documents.values(), *acc.media.values(), *acc.postings.values(), *acc.images.values()]
    units.sort(key=lambda unit: (_KIND_RANK[unit.kind], unit.title.lower(), unit.key))
    return units


def partition(
    main_points: Iterable[tuple[str, dict[str, Any]]],
    image_points: Iterable[tuple[str, dict[str, Any]]],
) -> list[Unit]:
    """Group a collection's points into the units an extract renders.

    Args:
        main_points (Iterable[tuple[str, dict[str, Any]]]): ``(point_id,
            payload)`` pairs from the collection itself.
        image_points (Iterable[tuple[str, dict[str, Any]]]): The same from its
            ``_images`` companion.

    Returns:
        list[Unit]: Units in a deterministic order — documents, clips,
            postings, then standalone images — regardless of input order.
    """
    acc = _Accumulator()
    for point_id, payload in main_points:
        if isinstance(payload, dict):
            _ingest_main_point(str(point_id), payload, acc)
    for _point_id, payload in image_points:
        if isinstance(payload, dict):
            _ingest_image_point(payload, acc)
    return _finalize(acc)


def resolve_target(units: Sequence[Unit], source_id: str) -> list[Unit]:
    """Return the units a per-source extract of ``source_id`` covers.

    A postings table is not one document: its file hash expands to every
    posting recorded in it, which is why a synchronous per-source download
    needs a unit cap.

    Args:
        units (Sequence[Unit]): Every unit in the collection.
        source_id (str): A file hash, a media hash, an image id or a posting
            uuid.

    Returns:
        list[Unit]: The matching units, empty when nothing matches.
    """
    target = source_id.strip()
    if not target:
        return []
    direct = [unit for unit in units if unit.key == target]
    if direct:
        return direct
    return [unit for unit in units if isinstance(unit, PostingUnit) and target in unit.source_hashes]
