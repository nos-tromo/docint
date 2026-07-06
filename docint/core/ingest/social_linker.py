"""Join a social export's postings table to its media manifest + files.

Pure join logic lives here (counter stripping, set-membership matching,
flat basename file resolution). Routing of resolved media into the modality
pipelines (CLIP / Nextext) lives in :class:`SocialLinker` (Task 10).
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from loguru import logger

_COUNTER_SUFFIX = re.compile(r"_\d+$")
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".gif"}


@dataclass(frozen=True)
class MediaLink:
    """A media file resolved to its owning posting."""

    posting_uuid: str
    posting_id: str
    media_id: str
    path: Path


def strip_counter(media_id: str) -> str:
    """Return the ``Media ID`` with a single trailing ``_<digits>`` counter removed.

    Args:
        media_id (str): The media identifier, e.g. ``"<posting_id>_0"``.

    Returns:
        str: The candidate posting id (``media_id`` itself if no counter).
    """
    return _COUNTER_SUFFIX.sub("", str(media_id), count=1)


_CSV_DELIMITERS = (",", ";", "\t", "|")


def _sniff_delimiter(path: Path) -> str:
    """Detect a social export's CSV delimiter (often ';'); fall back to ','.

    Args:
        path (Path): The CSV file to inspect.

    Returns:
        str: The detected delimiter, or ``","`` when detection is inconclusive.
    """
    try:
        sample = path.read_text(encoding="utf-8-sig", errors="replace")[:8192]
    except OSError:
        return ","
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters="".join(_CSV_DELIMITERS))
        if dialect.delimiter in _CSV_DELIMITERS:
            return dialect.delimiter
    except csv.Error:
        pass
    counts = {d: sample.count(d) for d in _CSV_DELIMITERS}
    best = max(counts, key=lambda d: counts[d])
    return best if counts[best] else ","


def build_posting_index(postings_df: pd.DataFrame) -> dict[str, str]:
    """Return ``{Posting ID: UUID}`` from a postings DataFrame.

    Args:
        postings_df (pd.DataFrame): Table carrying ``Posting ID`` + ``UUID``.

    Returns:
        dict[str, str]: Mapping from posting id to posting UUID.
    """
    index: dict[str, str] = {}
    for _, row in postings_df.iterrows():
        posting_id = str(row.get("Posting ID") or "").strip()
        uuid = str(row.get("UUID") or "").strip()
        if posting_id and uuid:
            index[posting_id] = uuid
    return index


def resolve_media_rows(
    media_df: pd.DataFrame,
    posting_uuids: dict[str, str],
    media_dir: Path,
) -> list[MediaLink]:
    """Resolve manifest rows to MediaLinks by basename within a single directory.

    Media files must live directly in ``media_dir`` (the manifest's own folder).
    Only the basename of ``Exported media filename`` is used, matched
    case-insensitively against the files in ``media_dir`` — no recursion and no
    relative/absolute path handling, so resolution can never leave ``media_dir``.
    Orphan rows (no matching posting) and rows with no local file are counted and
    reported once, not logged per row (a full manifest may have tens of thousands).

    Args:
        media_df (pd.DataFrame): Manifest with ``Media ID`` + ``Exported media filename``.
        posting_uuids (dict[str, str]): ``Posting ID → UUID`` from the postings table.
        media_dir (Path): The single flat directory holding the media files.

    Returns:
        list[MediaLink]: One per row whose posting is known and file exists.
    """
    present = {p.name.lower(): p for p in media_dir.iterdir() if p.is_file()}
    links: list[MediaLink] = []
    orphan_skips = 0
    missing_skips = 0
    for _, row in media_df.iterrows():
        media_id = str(row.get("Media ID") or "").strip()
        if not media_id:
            continue
        posting_id = strip_counter(media_id)
        uuid = posting_uuids.get(posting_id)
        if uuid is None:
            orphan_skips += 1
            continue
        name = Path(str(row.get("Exported media filename") or "").strip().replace("\\", "/")).name
        path = present.get(name.lower()) if name else None
        if path is None:
            missing_skips += 1
            continue
        links.append(MediaLink(posting_uuid=uuid, posting_id=posting_id, media_id=media_id, path=path))
    if orphan_skips or missing_skips:
        # Aggregate rather than log per row: a full manifest dropped in with only a few
        # referenced files present would otherwise emit one line per row (tens of
        # thousands). A single summary keeps large drop-ins robust and quiet.
        logger.info(
            "Social linker: {} media linked, {} skipped ({} with no matching posting, "
            "{} with no local file) across {} manifest rows.",
            len(links),
            orphan_skips + missing_skips,
            orphan_skips,
            missing_skips,
            len(media_df),
        )
    return links


def is_image(path: Path) -> bool:
    """Return whether ``path`` has a still-image extension (vs. video/audio)."""
    return path.suffix.lower() in _IMAGE_EXTS


# ---------------------------------------------------------------------------
# Routing layer (SocialLinker + SocialLinkResult) — Task 10
# ---------------------------------------------------------------------------
from typing import Any  # noqa: E402

from llama_index.core import Document  # noqa: E402

from docint.core.ingest.images_service import ImageAsset, IngestContext  # noqa: E402
from docint.core.readers.json import CustomJSONReader  # noqa: E402
from docint.core.readers.tables import TableReader, is_media_manifest  # noqa: E402
from docint.utils.hashing import compute_file_hash  # noqa: E402

# Exact header set for the postings profile — derived from the single source of truth in
# TableReader so _find_tables stays in sync whenever the profile header list changes.
_POSTINGS_HEADERS: set[str] = next(
    (profile.normalized_headers for profile in TableReader.schema_profiles if profile.style == "postings"),
    set(),
)


@dataclass
class SocialLinkResult:
    """Outcome of a social-linker pass over one batch tree."""

    consumed_paths: set[Path] = field(default_factory=set)
    transcript_documents: list[Document] = field(default_factory=list)


@dataclass
class SocialLinker:
    """Join + route a social export's media, linking each artifact to its posting."""

    image_service: Any
    nextext_client: Any
    target_collection: str | None
    manifest: Any = None
    keyframe_dedup_cosine: float = 0.95

    def _find_tables(self, data_dir: Path) -> tuple[Path | None, Path | None]:
        """Locate the postings table and media manifest anywhere in the tree.

        Args:
            data_dir (Path): The batch tree root.

        Returns:
            tuple[Path | None, Path | None]: ``(postings_csv, media_csv)``.
        """
        postings: Path | None = None
        media: Path | None = None
        for path in sorted(data_dir.rglob("*.csv")):
            try:
                columns = pd.read_csv(path, sep=_sniff_delimiter(path), nrows=0, encoding="utf-8-sig").columns
            except Exception:
                continue
            normalized = {str(c).strip().casefold() for c in columns}
            if media is None and is_media_manifest(columns):
                media = path
            elif postings is None and normalized == _POSTINGS_HEADERS:
                postings = path
        return postings, media

    def run(self, data_dir: Path) -> SocialLinkResult:
        """Run the linker over ``data_dir``; no-op when it is not a social export.

        Args:
            data_dir (Path): The batch tree root.

        Returns:
            SocialLinkResult: Consumed paths + transcript Documents for the pipeline.
        """
        result = SocialLinkResult()
        postings_csv, media_csv = self._find_tables(data_dir)
        if postings_csv is None or media_csv is None:
            return result

        posting_uuids = build_posting_index(
            pd.read_csv(postings_csv, sep=_sniff_delimiter(postings_csv), dtype=str, encoding="utf-8-sig")
        )
        media_df = pd.read_csv(media_csv, sep=_sniff_delimiter(media_csv), dtype=str, encoding="utf-8-sig")
        links = resolve_media_rows(media_df, posting_uuids, media_csv.parent)

        result.consumed_paths.add(media_csv)
        context = IngestContext(source_collection=self.target_collection)
        for link in links:
            result.consumed_paths.add(link.path)
            extra = {"posting_id": link.posting_id, "media_id": link.media_id, "source_type": "social_media"}
            if is_image(link.path):
                self.image_service.ingest_image(
                    ImageAsset.from_path(
                        path=link.path,
                        source_type="social_media",
                        source_doc_id=link.posting_uuid,
                        extra_metadata={**extra, "posting_uuid": link.posting_uuid},
                    ),
                    context=context,
                )
            else:
                self._route_media_clip(link, context, result)
        return result

    def _route_media_clip(self, link: MediaLink, context: IngestContext, result: SocialLinkResult) -> None:
        """Send one video/audio file to Nextext; ingest transcript + keyframes.

        The transcript is cached in the ingest manifest keyed by the media
        file's content hash, so re-ingesting an unchanged file reuses it instead
        of re-running the (expensive) Nextext job. On a cache hit Nextext is not
        called at all — the keyframe points already persist in the ``_images``
        companion from the first run.

        Args:
            link (MediaLink): The resolved media file and its posting linkage.
            context (IngestContext): Collection-resolution context.
            result (SocialLinkResult): Accumulator for transcript Documents.
        """
        collection = self.target_collection or ""
        media_hash = compute_file_hash(link.path)
        cached = self.manifest.get_nextext_transcript(collection, media_hash) if self.manifest else None
        keyframes: list[bytes] = []
        if cached is not None:
            transcript: bytes | None = cached.encode("utf-8")
        else:
            outcome = self.nextext_client.process_media(link.path)
            transcript = outcome.transcript_jsonl
            keyframes = outcome.keyframes
            if transcript is not None and self.manifest is not None:
                self.manifest.cache_nextext_transcript(collection, media_hash, transcript.decode("utf-8"))
        extra = {"posting_id": link.posting_id, "media_id": link.media_id, "source_type": "social_media"}
        if keyframes:
            self.image_service.ingest_keyframe_set(
                keyframes,
                context=context,
                source_doc_id=link.posting_uuid,
                extra_metadata={**extra, "posting_uuid": link.posting_uuid},
                dedup_cosine=self.keyframe_dedup_cosine,
            )
        if transcript:
            self._ingest_transcript(link, transcript, result)

    def _ingest_transcript(self, link: MediaLink, transcript: bytes, result: SocialLinkResult) -> None:
        """Parse transcript JSONL into segment Documents stamped with the posting link.

        Writes a transient ``.nextext.jsonl`` next to the media file purely so
        ``CustomJSONReader`` (which reads from a path) can parse it; that
        transient file is marked consumed so the generic sweep ignores it. The
        durable cache of record is the ingest manifest (Task 9b), not this file.

        Args:
            link (MediaLink): The posting linkage to stamp.
            transcript (bytes): The transcript NDJSON bytes.
            result (SocialLinkResult): Accumulator for the produced Documents.
        """
        transient = link.path.parent / (link.path.name + ".nextext.jsonl")
        transient.write_bytes(transcript)
        result.consumed_paths.add(transient)
        docs = CustomJSONReader(is_jsonl=True).iter_documents(
            transient,
            extra_info={
                "posting_uuid": link.posting_uuid,
                "posting_id": link.posting_id,
                "media_id": link.media_id,
            },
        )
        result.transcript_documents.extend(docs)
