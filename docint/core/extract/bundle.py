"""Assemble rendered units into an extract bundle.

Pure: units and bytes in, a ZIP out. Nothing here touches Qdrant, the
filesystem or the network, so the layout is fully testable.

Two rules shape the layout. Figures are written once as files and referenced
relatively, because inlining a 26 KB data URI per figure into every Markdown
copy triples a bundle for nothing. And every path segment is slugged and
suffixed with a hash: a folder name derives from a handle or a filename, both
of which are untrusted input, and two sources may legitimately share one name.
"""

from __future__ import annotations

import io
import re
import unicodedata
import zipfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime

from loguru import logger

from docint.core.extract.render import (
    appendix_numbers,
    extract_html,
    format_short,
    index_markdown,
    transcript_txt,
    unit_markdown,
)
from docint.core.extract.units import DocumentUnit, Figure, ImageUnit, MediaUnit, PostingUnit, Unit
from docint.utils.env_cfg import ExtractConfig

__all__ = ["BundleResult", "build_bundle", "build_single"]

#: Folder each unit kind lives under, inside the bundle root.
_KIND_DIR = {"document": "documents", "media": "media", "posting": "postings", "image": "images"}
#: Media type per single-source format.
_MEDIA_TYPES = {
    "md": "text/markdown; charset=utf-8",
    "pdf": "application/pdf",
    "zip": "application/zip",
}
#: Fixed ZIP timestamp so two builds of the same data are byte-identical.
_ZIP_EPOCH = (1980, 1, 1, 0, 0, 0)
_SLUG_MAX = 48
_SLUG_STRIP = re.compile(r"[^a-z0-9]+")


@dataclass
class BundleResult:
    """A rendered bundle and what went into it.

    Attributes:
        zip_bytes (bytes): The archive.
        counts (dict[str, int]): Units per kind plus a figure total.
        pdf_skipped (bool): Whether the combined PDF was left out.
    """

    zip_bytes: bytes
    counts: dict[str, int] = field(default_factory=dict)
    pdf_skipped: bool = False


def slug(text: str, *, fallback: str = "item") -> str:
    """Reduce arbitrary text to a safe, lowercase path segment.

    Args:
        text (str): Untrusted source text (a filename, a handle, a network).
        fallback (str): Used when nothing survives the reduction.

    Returns:
        str: ASCII, lowercase, hyphen-separated, length-capped.
    """
    folded = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii").lower()
    cleaned = _SLUG_STRIP.sub("-", folded).strip("-")
    return cleaned[:_SLUG_MAX].strip("-") or fallback


def _stem(name: str, fallback: str) -> str:
    """Slug a filename without its extension."""
    return slug(name.rsplit(".", 1)[0] if "." in name else name, fallback=fallback)


def _unit_dir(unit: Unit) -> str:
    """Return the bundle-relative folder for one unit.

    A posting files under ``network/author/date-uuid`` so a reader can walk an
    account's activity; everything else under ``kind/name-hash``.
    """
    short = slug(unit.key, fallback="id")[:8] or "id"
    if isinstance(unit, PostingUnit):
        ref = unit.reference
        network = slug(str(ref.get("network") or ""), fallback="network")
        author = slug(str(ref.get("author") or ref.get("author_id") or ""), fallback="account")
        stamp = _SLUG_STRIP.sub("", str(ref.get("timestamp") or ""))[:8] or "undated"
        return f"postings/{network}/{author}/{stamp}-{short}"
    return f"{_KIND_DIR[unit.kind]}/{_stem(unit.title, unit.kind)}-{short}"


def _figure_name(figure: Figure, position: int) -> str:
    """Return the file name a figure is written under, inside its folder."""
    extension = "png" if "png" in (figure.thumbnail_mime or "") else "jpg"
    if figure.kind == "keyframe":
        stamp = format_short(figure.time_sec).replace(":", "-")
        index = figure.index if figure.index is not None else position
        suffix = f"_{stamp}" if stamp else ""
        return f"keyframes/frame_{index:03d}{suffix}.{extension}"
    return f"figures/{slug(figure.image_id, fallback='figure')}.{extension}"


def _figure_files(unit: Unit) -> list[tuple[Figure, str]]:
    """Pair every figure of a unit with its path inside the unit's folder."""
    pairs: list[tuple[Figure, str]] = []
    if isinstance(unit, MediaUnit):
        pairs += [(figure, _figure_name(figure, i)) for i, figure in enumerate(unit.keyframes)]
    elif isinstance(unit, PostingUnit):
        pairs += [(figure, f"media/{slug(figure.image_id, fallback='image')}.jpg") for figure in unit.images]
        for clip in unit.media:
            pairs += [(figure, _figure_name(figure, i)) for i, figure in enumerate(clip.keyframes)]
    elif isinstance(unit, DocumentUnit):
        pairs += [(figure, _figure_name(figure, i)) for i, figure in enumerate(unit.figures)]
    elif isinstance(unit, ImageUnit) and unit.figure is not None:
        pairs.append((unit.figure, _figure_name(unit.figure, 0)))
    return pairs


def _decode(figure: Figure) -> bytes | None:
    """Decode a figure's stored thumbnail, or ``None`` when it has none."""
    if not figure.thumbnail_b64:
        return None
    import base64
    import binascii

    try:
        return base64.b64decode(figure.thumbnail_b64, validate=True)
    except (binascii.Error, ValueError) as exc:
        logger.warning("Skipping unreadable thumbnail for image '{}': {}", figure.image_id, exc)
        return None


def _transcripts(unit: Unit) -> list[tuple[str, str]]:
    """Return ``(relative path, text)`` for every transcript a unit holds."""
    if isinstance(unit, MediaUnit):
        text = transcript_txt(unit.segments)
        return [("transcript.txt", text)] if text else []
    if isinstance(unit, PostingUnit):
        files: list[tuple[str, str]] = []
        for clip in unit.media:
            text = transcript_txt(clip.segments)
            if not text:
                continue
            name = "transcript.txt" if len(unit.media) == 1 else f"transcript-{_stem(clip.title, 'clip')}.txt"
            files.append((name, text))
        return files
    return []


def _counts(units: Sequence[Unit]) -> dict[str, int]:
    """Count units per kind plus the total number of figures."""
    counts = {"documents": 0, "media": 0, "postings": 0, "images": 0}
    key = {"document": "documents", "media": "media", "posting": "postings", "image": "images"}
    for unit in units:
        counts[key[unit.kind]] += 1
    counts["figures"] = sum(len(unit.figures) for unit in units)
    return counts


def _write(archive: zipfile.ZipFile, name: str, payload: bytes) -> None:
    """Add one member with a fixed timestamp, keeping builds reproducible."""
    info = zipfile.ZipInfo(filename=name, date_time=_ZIP_EPOCH)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o644 << 16
    archive.writestr(info, payload)


def _render_pdf(
    units: Sequence[Unit],
    *,
    collection: str,
    created_at: str,
    cfg: ExtractConfig,
    pdf: Callable[[str], bytes] | None,
    figures: int,
    reference_number: str | None = None,
    operator: str | None = None,
    numbers: Mapping[str, str] | None = None,
) -> bytes | None:
    """Render the combined PDF, or ``None`` when it is capped or unavailable.

    A failure here never fails the bundle: the per-unit Markdown and the
    figures are the deliverable, and the PDF is a convenience on top.
    """
    if pdf is None or len(units) > cfg.pdf_max_units or figures > cfg.pdf_max_figures:
        return None
    try:
        return pdf(
            extract_html(
                units,
                collection=collection,
                created_at=created_at,
                reference_number=reference_number,
                operator=operator,
                numbers=numbers,
            )
        )
    except Exception as exc:
        logger.warning("Extract PDF skipped: {}", exc)
        return None


def build_bundle(
    units: Sequence[Unit],
    *,
    collection: str,
    cfg: ExtractConfig,
    pdf: Callable[[str], bytes] | None,
    now: datetime,
    progress: Callable[[int, int], None] | None = None,
    reference_number: str | None = None,
    operator: str | None = None,
) -> BundleResult:
    """Assemble an extract into a ZIP.

    Args:
        units (Sequence[Unit]): The units to write, in bundle order.
        collection (str): The collection's logical name.
        cfg (ExtractConfig): Caps governing the combined PDF.
        pdf (Callable[[str], bytes] | None): HTML-to-PDF engine, or ``None``
            to write no PDF at all.
        now (datetime): Build time, used in the root folder name.
        progress (Callable[[int, int], None] | None): Called with
            ``(rendered, total)`` after each unit.
        reference_number (str | None): Case file this appendix belongs to.
        operator (str | None): Who built it.

    Returns:
        BundleResult: The archive, its counts, and whether the PDF was skipped.
    """
    created_at = now.isoformat()
    root = f"{slug(collection, fallback='collection')}-extract-{now:%Y%m%d-%H%M}"
    paths = {unit.key: _unit_dir(unit) for unit in units}
    numbers = appendix_numbers(units)

    buffer = io.BytesIO()
    combined: list[str] = []
    total = len(units)
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for position, unit in enumerate(units, start=1):
            folder = paths[unit.key]
            figure_files = _figure_files(unit)
            body = unit_markdown(unit, {figure.image_id: path for figure, path in figure_files}, numbers=numbers)
            _write(archive, f"{root}/{folder}/extract.md", body.encode("utf-8"))
            combined.append(unit_markdown(unit, numbers=numbers))
            for name, text in _transcripts(unit):
                _write(archive, f"{root}/{folder}/{name}", text.encode("utf-8"))
            for figure, path in figure_files:
                payload = _decode(figure)
                if payload is not None:
                    _write(archive, f"{root}/{folder}/{path}", payload)
            if progress is not None:
                progress(position, total)

        figures = sum(len(unit.figures) for unit in units)
        document = _render_pdf(
            units,
            collection=collection,
            created_at=created_at,
            cfg=cfg,
            pdf=pdf,
            figures=figures,
            reference_number=reference_number,
            operator=operator,
            numbers=numbers,
        )
        pdf_skipped = pdf is not None and document is None
        if document is not None:
            _write(archive, f"{root}/extract.pdf", document)
        _write(archive, f"{root}/extract.md", "\n\n---\n\n".join(combined).encode("utf-8"))
        readme = index_markdown(
            units,
            collection=collection,
            created_at=created_at,
            paths=paths,
            pdf_skipped=pdf_skipped,
            reference_number=reference_number,
            operator=operator,
            numbers=numbers,
        )
        _write(archive, f"{root}/README.md", readme.encode("utf-8"))

    return BundleResult(zip_bytes=buffer.getvalue(), counts=_counts(units), pdf_skipped=pdf_skipped)


def build_single(
    units: Sequence[Unit],
    fmt: str,
    *,
    collection: str,
    now: datetime,
    pdf: Callable[[str], bytes] | None = None,
    reference_number: str | None = None,
    operator: str | None = None,
) -> tuple[bytes, str]:
    """Render one source's extract for an immediate download.

    Args:
        units (Sequence[Unit]): The units the source resolved to.
        fmt (str): ``"md"``, ``"pdf"`` or ``"zip"``.
        collection (str): The collection's logical name.
        now (datetime): Build time.
        pdf (Callable[[str], bytes] | None): Engine, required for ``"pdf"``.
        reference_number (str | None): Case file this appendix belongs to.
        operator (str | None): Who built it.

    Returns:
        tuple[bytes, str]: ``(payload, media type)``.

    Raises:
        ValueError: On an unknown format, or a PDF request with no engine.
    """
    if fmt not in _MEDIA_TYPES:
        raise ValueError(f"Unsupported extract format: {fmt!r}")
    # A single-source download is its own document, so it numbers from A.1.
    numbers = appendix_numbers(units)
    if fmt == "md":
        body = "\n\n---\n\n".join(unit_markdown(unit, numbers=numbers) for unit in units)
        return body.encode("utf-8"), _MEDIA_TYPES["md"]
    if fmt == "pdf":
        if pdf is None:
            raise ValueError("PDF export requires an engine.")
        html = extract_html(
            units,
            collection=collection,
            created_at=now.isoformat(),
            reference_number=reference_number,
            operator=operator,
            numbers=numbers,
        )
        return pdf(html), _MEDIA_TYPES["pdf"]
    cfg = ExtractConfig(retention_days=1, max_per_collection=1, pdf_max_units=0, pdf_max_figures=0, sync_max_units=1)
    bundle = build_bundle(
        units,
        collection=collection,
        cfg=cfg,
        pdf=None,
        now=now,
        reference_number=reference_number,
        operator=operator,
    )
    return bundle.zip_bytes, _MEDIA_TYPES["zip"]
