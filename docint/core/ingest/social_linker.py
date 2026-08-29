"""Join a social export's postings table to its media manifest + files.

Pure join logic lives here (counter stripping, set-membership matching, album
inference, basename file resolution across the batch tree). Routing of resolved
media into the modality pipelines (CLIP / Nextext) lives in
:class:`SocialLinker` (Task 10).
"""

from __future__ import annotations

import bisect
import csv
import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from loguru import logger

_COUNTER_SUFFIX = re.compile(r"_\d+$")
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".gif"}

#: Default slack allowed between a media row's timestamp and that of the posting
#: an album inference would attach it to. Deliberately tight: measured on a real
#: Telegram export every true album member sits 0-1 s from its posting, while a
#: neighbouring post is hours away, so a small window rejects nearly every
#: mis-attribution a pruned/partial export could otherwise produce.
_DEFAULT_ALBUM_TOLERANCE_S = 5.0

#: Shortest all-digit URL path segment treated as a network id. A shorter number
#: in a posting URL is a page, version or index — matching one would attach media
#: to an arbitrary posting.
_MIN_URL_ID_DIGITS = 8

#: All-digit path segments of a posting URL (``…/reel/<id>/``, ``…/video/<id>``).
_URL_ID_SEGMENT = re.compile(rf"/(\d{{{_MIN_URL_ID_DIGITS},}})(?=[/?#]|$)")

#: ``{Author ID: sorted [(message_no, Posting ID, timestamp)]}`` — see
#: :func:`build_posting_album_index`.
AlbumIndex = dict[str, list[tuple[int, str, pd.Timestamp]]]

#: ``{(network, author): sorted [(timestamp, Posting ID)]}`` — see
#: :func:`build_posting_time_index`.
TimeIndex = dict[tuple[str, str], list[tuple[pd.Timestamp, str]]]


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


def _derive_posting_id(network_id: str, media_id: str, posting_uuids: dict[str, str]) -> str | None:
    """Return the parent posting id for a media row, or ``None`` when unknown.

    Social exports encode the media→posting link differently, so try the known
    candidates in order and return the first that names a posting present in
    ``posting_uuids``:

    1. ``Network ID`` — a dedicated column that holds the parent ``Posting ID``
       (the common case; e.g. the AfD/Meta-style exports).
    2. the raw ``Media ID`` — some exports set it equal to the ``Posting ID``.
    3. ``strip_counter(Media ID)`` — for ``<Posting ID>_<counter>`` media ids.

    Args:
        network_id (str): The row's ``Network ID`` value (may be empty).
        media_id (str): The row's ``Media ID`` value.
        posting_uuids (dict[str, str]): Known ``Posting ID → UUID`` mapping.

    Returns:
        str | None: The matched posting id, or ``None`` if no candidate is known.
    """
    for candidate in (network_id, media_id, strip_counter(media_id)):
        if candidate and candidate in posting_uuids:
            return candidate
    return None


def build_posting_url_index(postings_df: pd.DataFrame) -> dict[str, str]:
    """Return ``{network id: Posting ID}`` derived from the postings' own URLs.

    Some exports key a posting by a crawler-minted UUID while the media manifest
    holds the bare network id, so the two can never meet through
    :func:`_derive_posting_id`. The posting's ``URL`` still carries that network
    id as a path segment (``…/reel/<id>/``, ``…/video/<id>``), which makes it an
    **exact** key rather than an inference — hence no timestamp corroboration and
    no enable flag.

    An id claimed by two different postings is dropped: linking against it would
    be a coin flip. Segments shorter than :data:`_MIN_URL_ID_DIGITS` are ignored,
    since a short number in a path is a page or version, not a network id.

    Args:
        postings_df (pd.DataFrame): Table carrying ``Posting ID`` + ``URL``.

    Returns:
        dict[str, str]: Mapping from URL-borne network id to posting id, empty
        when the table has no ``URL`` column.
    """
    if "URL" not in postings_df.columns:
        return {}
    index: dict[str, str] = {}
    collisions: set[str] = set()
    for _, row in postings_df.iterrows():
        posting_id = str(row.get("Posting ID") or "").strip()
        url = str(row.get("URL") or "").strip()
        if not posting_id or not url:
            continue
        for candidate in _URL_ID_SEGMENT.findall(url):
            if index.get(candidate, posting_id) != posting_id:
                collisions.add(candidate)
            index.setdefault(candidate, posting_id)
    for candidate in collisions:
        index.pop(candidate, None)
    return index


def _derive_posting_id_from_url(network_id: str, media_id: str, url_index: dict[str, str]) -> str | None:
    """Return the posting whose URL names this media row's id, or ``None``.

    Tries the same candidates in the same order as :func:`_derive_posting_id`,
    against the URL-derived index instead of the postings' own ids.

    Args:
        network_id (str): The row's ``Network ID`` value (may be empty).
        media_id (str): The row's ``Media ID`` value.
        url_index (dict[str, str]): Index from :func:`build_posting_url_index`.

    Returns:
        str | None: The matched posting id, or ``None`` when no candidate is known.
    """
    for candidate in (network_id, media_id, strip_counter(media_id)):
        if candidate and candidate in url_index:
            return url_index[candidate]
    return None


def build_posting_time_index(postings_df: pd.DataFrame) -> TimeIndex:
    """Return the per-channel timestamp index used to corroborate a key-less row.

    Keyed by ``(Network, Author)`` case-folded, because a *media* row carries the
    author's display name and network — not the ``Author ID`` the album index
    keys on. Values are sorted by timestamp so
    :func:`_infer_timestamp_posting_id` can bisect a window.

    Postings with no parseable timestamp, author or network are skipped: they
    cannot anchor a match, and indexing them would let an empty media field find
    an equally empty posting one.

    Args:
        postings_df (pd.DataFrame): Table carrying the postings export schema.

    Returns:
        TimeIndex: ``{(network, author): sorted [(timestamp, Posting ID)]}``,
        empty when the table lacks the columns the match needs.
    """
    if not {"Author", "Network"}.issubset(postings_df.columns):
        return {}
    stamps = _parse_timestamps(postings_df)
    if stamps is None:
        return {}
    channels: TimeIndex = {}
    for (_, row), stamp in zip(postings_df.iterrows(), stamps, strict=True):
        if pd.isna(stamp):
            continue
        posting_id = str(row.get("Posting ID") or "").strip()
        author = str(row.get("Author") or "").strip().casefold()
        network = str(row.get("Network") or "").strip().casefold()
        if not posting_id or not author or not network:
            continue
        channels.setdefault((network, author), []).append((stamp, posting_id))
    for entries in channels.values():
        entries.sort(key=lambda entry: entry[0])
    return channels


def _infer_timestamp_posting_id(
    network: str,
    author: str,
    media_stamp: pd.Timestamp | None,
    times: TimeIndex,
    tolerance_s: float,
) -> str | None:
    """Return the posting a key-less media row belongs to, or ``None``.

    The last resort for an export that carries no usable media→posting key at
    all: match within the same author's channel on timestamp agreement alone.
    Accepted **only when exactly one** posting falls inside the window — two
    candidates is ambiguity, and a coin flip presented to an investigator as
    provenance is worse than an unlinked row.

    Args:
        network (str): The media row's ``Network`` value.
        author (str): The media row's ``Author`` value.
        media_stamp (pd.Timestamp | None): The row's parsed ``Timestamp``.
        times (TimeIndex): Index from :func:`build_posting_time_index`.
        tolerance_s (float): Maximum allowed timestamp difference, in seconds.

    Returns:
        str | None: The parent ``Posting ID``, or ``None`` when the channel is
        unknown, no posting agrees, or more than one does.
    """
    if not times or media_stamp is None or pd.isna(media_stamp):
        return None
    entries = times.get((network.strip().casefold(), author.strip().casefold()))
    if not entries:
        return None
    window = pd.Timedelta(seconds=tolerance_s)
    stamps = [entry[0] for entry in entries]
    low = bisect.bisect_left(stamps, media_stamp - window)
    high = bisect.bisect_right(stamps, media_stamp + window)
    if high - low != 1:
        return None
    return entries[low][1]


def _split_channel(identifier: str, prefix: str) -> int | None:
    """Return the numeric message number ``identifier`` carries after ``prefix``.

    Args:
        identifier (str): A ``Posting ID`` or ``Media ID``.
        prefix (str): The channel id (a postings row's ``Author ID``).

    Returns:
        int | None: The message number, or ``None`` when ``identifier`` does not
        decompose as ``<prefix><digits>``.
    """
    if not prefix or not identifier.startswith(prefix):
        return None
    remainder = identifier[len(prefix) :]
    if not remainder or not remainder.isdigit():
        return None
    return int(remainder)


def build_posting_album_index(postings_df: pd.DataFrame) -> AlbumIndex:
    """Return the per-channel message-number index used to infer album membership.

    Some exports carry no media→posting foreign key at all: the manifest's
    ``Media ID`` / ``Network ID`` hold the media's *own* network message id. A
    multi-item post (a Telegram album / media group) is then N consecutive
    messages recorded as N media rows but a single posting, filed under the
    group's **last** message id — so the parent is the first posting whose
    message number is at or above the media's own.

    Only postings whose ``Posting ID`` decomposes as ``<Author ID><digits>``
    with a parseable ``Timestamp`` are indexed. That requirement is also the
    gate that keeps the inference inert on exports which *do* carry a key: a
    Meta-style ``Posting ID`` of ``<postingId>_<accountId>`` carries its
    ``Author ID`` as a suffix, so no channel is derived and the index is empty.

    Args:
        postings_df (pd.DataFrame): Table carrying the postings export schema.

    Returns:
        AlbumIndex: ``{Author ID: sorted [(message_no, Posting ID, timestamp)]}``,
        empty when the table does not decompose this way.
    """
    if "Author ID" not in postings_df.columns or "Timestamp" not in postings_df.columns:
        return {}
    stamps = _parse_timestamps(postings_df)
    if stamps is None:
        return {}
    channels: AlbumIndex = {}
    for (_, row), stamp in zip(postings_df.iterrows(), stamps, strict=True):
        if pd.isna(stamp):
            continue
        author_id = str(row.get("Author ID") or "").strip()
        posting_id = str(row.get("Posting ID") or "").strip()
        message_no = _split_channel(posting_id, author_id)
        if message_no is None:
            continue
        channels.setdefault(author_id, []).append((message_no, posting_id, stamp))
    for entries in channels.values():
        entries.sort(key=lambda entry: entry[0])
    return channels


def _infer_album_posting_id(
    media_id: str,
    media_stamp: pd.Timestamp | None,
    albums: AlbumIndex,
    tolerance_s: float,
) -> str | None:
    """Return the posting whose album ``media_id`` belongs to, or ``None``.

    Picks the first posting in the media's own channel whose message number is
    at or above the media's, then **requires the two timestamps to agree** within
    ``tolerance_s``. That corroboration is what makes the inference safe: when the
    owning posting is absent from the export the next one along is hours away and
    is rejected, leaving the row unlinked rather than mis-attributed.

    Args:
        media_id (str): The manifest row's ``Media ID``.
        media_stamp (pd.Timestamp | None): The row's parsed ``Timestamp``.
        albums (AlbumIndex): Index from :func:`build_posting_album_index`.
        tolerance_s (float): Maximum allowed timestamp difference, in seconds.

    Returns:
        str | None: The parent ``Posting ID``, or ``None`` when no channel
        matches, no posting sits above the media, or the timestamps disagree.
    """
    if not albums or media_stamp is None or pd.isna(media_stamp):
        return None
    channel: str | None = None
    message_no: int | None = None
    for prefix in albums:
        candidate_no = _split_channel(media_id, prefix)
        if candidate_no is None:
            continue
        # Longest matching channel wins, so a multi-account export cannot let a
        # short prefix shadow the account the media actually belongs to.
        if channel is None or len(prefix) > len(channel):
            channel, message_no = prefix, candidate_no
    if channel is None or message_no is None:
        return None
    entries = albums[channel]
    index = bisect.bisect_left([entry[0] for entry in entries], message_no)
    if index >= len(entries):
        return None
    _, posting_id, stamp = entries[index]
    if abs((stamp - media_stamp).total_seconds()) > tolerance_s:
        return None
    return posting_id


def _parse_timestamps(frame: pd.DataFrame) -> pd.Series | None:
    """Return ``frame``'s ``Timestamp`` column parsed to UTC, or ``None``.

    Unparseable values become ``NaT`` rather than raising — a malformed stamp
    costs that one row its album inference, never the whole ingest.

    Args:
        frame (pd.DataFrame): A postings or media table.

    Returns:
        pd.Series | None: UTC timestamps positionally aligned to ``frame``, or
        ``None`` when the table carries no ``Timestamp`` column.
    """
    if "Timestamp" not in frame.columns:
        return None
    return pd.to_datetime(frame["Timestamp"], utc=True, errors="coerce", format="mixed")


def build_file_index(root: Path) -> dict[str, list[Path]]:
    """Index every file under ``root`` (recursively) by lowercase basename.

    Args:
        root (Path): The batch tree root.

    Returns:
        dict[str, list[Path]]: ``{basename_lower: [paths]}``, sorted per key.
    """
    index: dict[str, list[Path]] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            index.setdefault(path.name.lower(), []).append(path)
    return index


def _pick_media_file(matches: list[Path], manifest_dir: Path) -> Path | None:
    """Return the single file a manifest basename names, or ``None`` if ambiguous.

    A manifest carries a basename and nothing else, so when the same name occurs
    in two subdirectories nothing in the data says which file is the evidence.
    A copy sitting directly beside the manifest breaks the tie; otherwise the row
    is refused rather than guessed at.

    Args:
        matches (list[Path]): Candidate files sharing the basename.
        manifest_dir (Path): Directory holding the media manifest.

    Returns:
        Path | None: The chosen file, or ``None`` when the name is ambiguous.
    """
    if len(matches) == 1:
        return matches[0]
    local = [path for path in matches if path.parent == manifest_dir]
    if len(local) == 1:
        return local[0]
    return None


def resolve_media_rows(
    media_df: pd.DataFrame,
    posting_uuids: dict[str, str],
    root: Path,
    *,
    manifest_dir: Path | None = None,
    url_index: dict[str, str] | None = None,
    albums: AlbumIndex | None = None,
    time_index: TimeIndex | None = None,
    album_tolerance_s: float = _DEFAULT_ALBUM_TOLERANCE_S,
) -> list[MediaLink]:
    """Resolve manifest rows to MediaLinks by basename anywhere under ``root``.

    Only the **basename** of ``Exported media filename`` is ever used, matched
    case-insensitively against the files under ``root``. The manifest's own path
    components are discarded, so an absolute path or a ``../`` traversal collapses
    to a name that is only ever looked for inside the batch tree — resolution
    provably cannot leave ``root``. That containment comes from matching basenames
    rather than from refusing to recurse, which is why subdirectories
    (``dir/photos/``, ``dir/videos/``) are safe to search.

    Each row's parent posting is resolved by the first of four paths that answers,
    strongest key first, so an export that already joins never reaches a fallback:

    1. the manifest's own key (:func:`_derive_posting_id`);
    2. the id carried in a posting's ``URL`` (:func:`_derive_posting_id_from_url`)
       — still an exact match, for exports whose ``Posting ID`` is a crawler UUID;
    3. album membership (:func:`_infer_album_posting_id`), timestamp-corroborated;
    4. the author's own timeline (:func:`_infer_timestamp_posting_id`), accepted
       only when exactly one posting agrees.

    Orphan rows, rows with no local file and rows whose basename is ambiguous are
    counted and reported once, not logged per row (a full manifest may have tens of
    thousands).

    Args:
        media_df (pd.DataFrame): Manifest with ``Media ID`` + ``Exported media filename``.
        posting_uuids (dict[str, str]): ``Posting ID → UUID`` from the postings table.
        root (Path): The batch tree root; every media file is looked up under it.
        manifest_dir (Path | None): Directory holding the manifest, used to break a
            duplicate-basename tie. Defaults to ``root``.
        url_index (dict[str, str] | None): Index from :func:`build_posting_url_index`.
            ``None`` (the default) disables URL-key matching entirely.
        albums (AlbumIndex | None): Index from :func:`build_posting_album_index`.
            ``None`` (the default) disables album inference entirely.
        time_index (TimeIndex | None): Index from :func:`build_posting_time_index`.
            ``None`` (the default) disables timestamp inference entirely.
        album_tolerance_s (float): Maximum timestamp disagreement, in seconds,
            allowed when accepting an inferred album or timestamp link.

    Returns:
        list[MediaLink]: One per row whose posting is known and file exists.
    """
    tie_break_dir = manifest_dir if manifest_dir is not None else root
    present = build_file_index(root)
    stamps = _parse_timestamps(media_df) if (albums or time_index) else None
    stamp_values: list[pd.Timestamp | None] = list(stamps) if stamps is not None else [None] * len(media_df)
    links: list[MediaLink] = []
    exact_links = 0
    url_links = 0
    album_links = 0
    timestamp_links = 0
    orphan_skips = 0
    missing_skips = 0
    ambiguous_skips = 0
    for (_, row), stamp in zip(media_df.iterrows(), stamp_values, strict=True):
        media_id = str(row.get("Media ID") or "").strip()
        if not media_id:
            continue
        network_id = str(row.get("Network ID") or "").strip()
        posting_id = _derive_posting_id(network_id, media_id, posting_uuids)
        matched_by = "key"
        if posting_id is None and url_index:
            posting_id = _derive_posting_id_from_url(network_id, media_id, url_index)
            matched_by = "url"
        if posting_id is None and albums:
            posting_id = _infer_album_posting_id(media_id, stamp, albums, album_tolerance_s)
            matched_by = "album"
        if posting_id is None and time_index:
            posting_id = _infer_timestamp_posting_id(
                str(row.get("Network") or ""),
                str(row.get("Author") or ""),
                stamp,
                time_index,
                album_tolerance_s,
            )
            matched_by = "timestamp"
        if posting_id is None or posting_id not in posting_uuids:
            orphan_skips += 1
            continue
        uuid = posting_uuids[posting_id]
        name = Path(str(row.get("Exported media filename") or "").strip().replace("\\", "/")).name
        matches: list[Path] = present.get(name.lower(), []) if name else []
        if not matches:
            missing_skips += 1
            continue
        path = _pick_media_file(matches, tie_break_dir)
        if path is None:
            ambiguous_skips += 1
            continue
        if matched_by == "url":
            url_links += 1
        elif matched_by == "album":
            album_links += 1
        elif matched_by == "timestamp":
            timestamp_links += 1
        else:
            exact_links += 1
        links.append(MediaLink(posting_uuid=uuid, posting_id=posting_id, media_id=media_id, path=path))
    if len(media_df):
        # Aggregate rather than log per row: a full manifest dropped in with only a few
        # referenced files present would otherwise emit one line per row (tens of
        # thousands). A single summary keeps large drop-ins robust and quiet.
        logger.info(
            "Social linker: {} media linked ({} by manifest key, {} by posting URL, {} by album inference, "
            "{} by timestamp), {} skipped "
            "({} with no matching posting, {} with no local file, {} with an ambiguous filename) "
            "across {} manifest rows.",
            len(links),
            exact_links,
            url_links,
            album_links,
            timestamp_links,
            orphan_skips + missing_skips + ambiguous_skips,
            orphan_skips,
            missing_skips,
            ambiguous_skips,
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
from docint.core.ingest.media_transcribe import MediaClip, MediaTranscriber  # noqa: E402
from docint.core.readers.tables import TableReader, is_media_manifest  # noqa: E402

# Exact header set for the postings profile — derived from the single source of truth in
# TableReader so _find_tables stays in sync whenever the profile header list changes.
_POSTINGS_HEADERS: set[str] = next(
    (profile.normalized_headers for profile in TableReader.schema_profiles if profile.style == "postings"),
    set(),
)

# Posting reference fields carried onto derived media artifacts, prefixed so they
# merge additively into an artifact's ``reference_metadata`` without clobbering
# the artifact's own fields (e.g. a transcript segment's ``network: nextext``).
_POSTING_REFERENCE_KEYS: dict[str, str] = {
    "network": "posting_network",
    "author": "posting_author",
    "author_id": "posting_author_id",
    "vanity": "posting_vanity",
    "timestamp": "posting_timestamp",
    "url": "posting_url",
    "text": "posting_text",
}


def build_posting_reference_index(postings_df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Return ``{Posting ID: prefixed posting reference fields}`` from a postings table.

    Reuses the :class:`TableReader` postings schema profile so the column
    mapping stays declared in exactly one place. Keys are prefixed via
    :data:`_POSTING_REFERENCE_KEYS` (``network`` → ``posting_network``, ...);
    empty / missing values are omitted.

    Args:
        postings_df (pd.DataFrame): Table carrying the postings export schema.

    Returns:
        dict[str, dict[str, Any]]: Mapping from posting id to the prefixed
        posting reference fields. Empty when the headers do not match the
        postings profile — derived artifacts then carry link ids only,
        matching the pre-enrichment behavior.
    """
    profile, normalized_map = TableReader._detect_schema_profile(postings_df.columns)
    if profile is None or profile.style != "postings":
        logger.warning(
            "Social linker: postings table does not match the postings profile; media artifacts keep link ids only."
        )
        return {}
    index: dict[str, dict[str, Any]] = {}
    for _, row in postings_df.iterrows():
        posting_id = str(row.get("Posting ID") or "").strip()
        if not posting_id:
            continue
        reference = TableReader._build_reference_metadata(
            profile=profile, row_dict=row.to_dict(), normalized_map=normalized_map
        )
        stamp: dict[str, Any] = {}
        for key, prefixed in _POSTING_REFERENCE_KEYS.items():
            value = reference.get(key)
            if value is None or (isinstance(value, float) and pd.isna(value)):
                continue
            text = str(value).strip()
            if text:
                stamp[prefixed] = text
        index[posting_id] = stamp
    return index


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
    nextext_max_concurrency: int = 4
    album_link_enabled: bool = True
    album_tolerance_s: float = _DEFAULT_ALBUM_TOLERANCE_S

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

        postings_df = pd.read_csv(postings_csv, sep=_sniff_delimiter(postings_csv), dtype=str, encoding="utf-8-sig")
        posting_uuids = build_posting_index(postings_df)
        posting_references = build_posting_reference_index(postings_df)
        media_df = pd.read_csv(media_csv, sep=_sniff_delimiter(media_csv), dtype=str, encoding="utf-8-sig")
        albums = build_posting_album_index(postings_df) if self.album_link_enabled else None
        # The URL key is an exact match, so it needs no enable flag; the timestamp
        # fallback is an inference and shares the album knobs.
        times = build_posting_time_index(postings_df) if self.album_link_enabled else None
        links = resolve_media_rows(
            media_df,
            posting_uuids,
            data_dir,
            manifest_dir=media_csv.parent,
            url_index=build_posting_url_index(postings_df),
            albums=albums,
            time_index=times,
            album_tolerance_s=self.album_tolerance_s,
        )

        result.consumed_paths.add(media_csv)
        context = IngestContext(source_collection=self.target_collection)
        clips: list[MediaClip] = []
        for link in links:
            posting_ref = posting_references.get(link.posting_id, {})
            link_ids = {
                "posting_uuid": link.posting_uuid,
                "posting_id": link.posting_id,
                "media_id": link.media_id,
            }
            if is_image(link.path):
                result.consumed_paths.add(link.path)
                self.image_service.ingest_image(
                    ImageAsset.from_path(
                        path=link.path,
                        source_type="social_media",
                        source_doc_id=link.posting_uuid,
                        extra_metadata={
                            **link_ids,
                            "source_type": "social_media",
                            **posting_ref,
                            "reference_metadata": {"type": "image", **link_ids, **posting_ref},
                        },
                    ),
                    context=context,
                )
            else:
                clips.append(
                    MediaClip(
                        path=link.path,
                        source_doc_id=link.posting_uuid,
                        keyframe_extra_metadata={
                            **link_ids,
                            "source_type": "social_media",
                            **posting_ref,
                            "reference_metadata": {"type": "keyframe", **link_ids, **posting_ref},
                        },
                        # Flat keys only: the transcript reader owns the segment's
                        # reference_metadata and merges these in additively.
                        transcript_extra_info={**link_ids, **posting_ref},
                    )
                )
        sub = MediaTranscriber(
            image_service=self.image_service,
            nextext_client=self.nextext_client,
            target_collection=self.target_collection,
            manifest=self.manifest,
            keyframe_dedup_cosine=self.keyframe_dedup_cosine,
            nextext_max_concurrency=self.nextext_max_concurrency,
        ).run(clips)
        result.consumed_paths |= sub.consumed_paths
        result.transcript_documents.extend(sub.transcript_documents)
        return result
