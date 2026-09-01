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

#: Long numeric runs in a permalink -- short ones are page numbers, indices and
#: the like, never a posting id.
_URL_ID_RE = re.compile(r"\d{6,}")

_COUNTER_SUFFIX = re.compile(r"_\d+$")
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".gif"}

#: Default slack allowed between a media row's timestamp and that of the posting
#: an album inference would attach it to. Deliberately tight: measured on a real
#: Telegram export every true album member sits 0-1 s from its posting, while a
#: neighbouring post is hours away, so a small window rejects nearly every
#: mis-attribution a pruned/partial export could otherwise produce.
_DEFAULT_ALBUM_TOLERANCE_S = 5.0

#: ``{Author ID: sorted [(message_no, Posting ID, timestamp)]}`` — see
#: :func:`build_posting_album_index`.
AlbumIndex = dict[str, list[tuple[int, str, pd.Timestamp]]]

#: ``{(network, author, timestamp): [Posting ID]}`` -- see
#: :func:`build_posting_stamp_index`.
StampIndex = dict[tuple[str, str, pd.Timestamp], list[str]]

#: ``{(network, text): [Posting ID]}`` -- see :func:`build_posting_text_index`.
TextIndex = dict[tuple[str, str], list[str]]

#: Manifest column carrying a media row's copy of its posting's text.
_MEDIA_TEXT_COLUMN = "Title"


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


def _network_id_candidates(row: pd.Series) -> set[str]:
    """Return the network-level ids a postings row advertises.

    ``Network Posting ID`` when the export fills it, plus any long numeric id in
    the permalink -- a reel-style posting leaves the column empty and carries its
    network id only in the ``URL``.

    Args:
        row (pd.Series): One postings row.

    Returns:
        set[str]: Candidate network-level ids, possibly empty.
    """
    candidates: set[str] = set()
    network_posting_id = str(row.get("Network Posting ID") or "").strip()
    if network_posting_id:
        candidates.add(network_posting_id)
        candidates.update(part for part in network_posting_id.split(":") if part)
    candidates.update(_URL_ID_RE.findall(str(row.get("URL") or "")))
    return candidates


def build_network_posting_index(postings_df: pd.DataFrame) -> dict[str, str]:
    """Return ``{network-level id: Posting ID}`` for postings whose own id is internal.

    Some exports mint an internal ``Posting ID`` -- a crawler UUID the media
    manifest never carries -- while the id the manifest *does* carry is the one
    the network itself uses, found in ``Network Posting ID`` or in the permalink.

    Two ids are refused rather than resolved to a guess: one that two postings
    both advertise (nothing says which owns a media row) and one that is an
    ``Author ID`` (an account appears in every one of its postings' URLs, so it
    would attach media to an arbitrary posting of that account).

    Args:
        postings_df (pd.DataFrame): Table carrying the postings export schema.

    Returns:
        dict[str, str]: Mapping from a network-level posting id to its ``Posting ID``.
    """
    owners: dict[str, set[str]] = {}
    account_ids: set[str] = set()
    for _, row in postings_df.iterrows():
        posting_id = str(row.get("Posting ID") or "").strip()
        account_ids.add(str(row.get("Author ID") or "").strip())
        if not posting_id:
            continue
        for candidate in _network_id_candidates(row):
            owners.setdefault(candidate, set()).add(posting_id)
    return {
        candidate: next(iter(postings))
        for candidate, postings in owners.items()
        if len(postings) == 1 and candidate not in account_ids
    }


def _derive_network_posting_id(network_id: str, media_id: str, network_index: dict[str, str]) -> str | None:
    """Return the posting a media row names by *network-level* id, or ``None``.

    Tried only after :func:`_derive_posting_id` has failed.

    Args:
        network_id (str): The row's ``Network ID`` value (may be empty).
        media_id (str): The row's ``Media ID`` value.
        network_index (dict[str, str]): Index from :func:`build_network_posting_index`.

    Returns:
        str | None: The matched ``Posting ID``, or ``None`` when nothing matches.
    """
    for candidate in (network_id, media_id, strip_counter(media_id)):
        if candidate and candidate in network_index:
            return network_index[candidate]
    return None


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


def build_posting_stamp_index(postings_df: pd.DataFrame) -> StampIndex:
    """Return the author-scoped timestamp index used for keyless media rows.

    The last resort for an export that carries no media->posting key at all and
    no message numbering to infer one from: a media row and its parent posting
    are stamped at the same instant, so the posting sharing a row's exact
    timestamp is its parent -- *provided there is only one*.

    Args:
        postings_df (pd.DataFrame): Table carrying the postings export schema.

    Returns:
        StampIndex: Posting ids grouped by ``(network, author, timestamp)``,
        empty when the table carries no parseable ``Timestamp``.
    """
    stamps = _parse_timestamps(postings_df)
    if stamps is None:
        return {}
    index: StampIndex = {}
    for (_, row), stamp in zip(postings_df.iterrows(), stamps, strict=True):
        posting_id = str(row.get("Posting ID") or "").strip()
        if not posting_id or pd.isna(stamp):
            continue
        key = (str(row.get("Network") or "").strip(), str(row.get("Author") or "").strip(), stamp)
        index.setdefault(key, []).append(posting_id)
    return index


def _infer_stamp_posting_id(row: pd.Series, media_stamp: pd.Timestamp | None, stamps: StampIndex) -> str | None:
    """Return the posting a keyless media row shares its instant with, or ``None``.

    Ambiguity is refused, never guessed at: two postings by one author at the
    same instant say nothing about which owns the media. So does an instant no
    posting shares, which is what a partial export looks like -- the parent is
    simply absent, and inventing one would be worse than leaving the row
    unlinked.

    Args:
        row (pd.Series): The manifest row.
        media_stamp (pd.Timestamp | None): The row's parsed ``Timestamp``.
        stamps (StampIndex): Index from :func:`build_posting_stamp_index`.

    Returns:
        str | None: The parent ``Posting ID``, or ``None`` when absent or ambiguous.
    """
    if media_stamp is None or pd.isna(media_stamp):
        return None
    key = (str(row.get("Network") or "").strip(), str(row.get("Author") or "").strip(), media_stamp)
    candidates = stamps.get(key, [])
    return candidates[0] if len(candidates) == 1 else None


def build_posting_text_index(postings_df: pd.DataFrame) -> TextIndex:
    """Return the network-scoped text index used for rows no author matches.

    The final rule, for the shape no author-scoped rule reaches: a shared post,
    whose manifest names the *original* author while the export's row is the
    *sharer's*. Both carry the text verbatim. Postings with no text are never
    indexed — an empty text would name every media-only post at once.

    Args:
        postings_df (pd.DataFrame): Table carrying the postings export schema.

    Returns:
        TextIndex: Posting ids grouped by ``(network, text)``, empty when the
        table carries no ``Text Content``.
    """
    if "Text Content" not in postings_df.columns:
        return {}
    index: TextIndex = {}
    for _, row in postings_df.iterrows():
        posting_id = str(row.get("Posting ID") or "").strip()
        text = str(row.get("Text Content") or "").strip()
        if not posting_id or not text:
            continue
        index.setdefault((str(row.get("Network") or "").strip(), text), []).append(posting_id)
    return index


def _infer_text_posting_id(row: pd.Series, texts: TextIndex) -> str | None:
    """Return the posting whose text a media row repeats verbatim, or ``None``.

    Equality is exact and case-sensitive — the rule's whole confidence is a
    *complete* text matching character for character. Ambiguity and absence are
    both refused, as in :func:`_infer_stamp_posting_id`.

    Args:
        row (pd.Series): The manifest row.
        texts (TextIndex): Index from :func:`build_posting_text_index`.

    Returns:
        str | None: The parent ``Posting ID``, or ``None`` when absent or ambiguous.
    """
    text = str(row.get(_MEDIA_TEXT_COLUMN) or "").strip()
    if not text:
        return None
    candidates = texts.get((str(row.get("Network") or "").strip(), text), [])
    return candidates[0] if len(candidates) == 1 else None


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
    network_index: dict[str, str] | None = None,
    albums: AlbumIndex | None = None,
    album_tolerance_s: float = _DEFAULT_ALBUM_TOLERANCE_S,
    stamps: StampIndex | None = None,
    texts: TextIndex | None = None,
) -> list[MediaLink]:
    """Resolve manifest rows to MediaLinks by basename anywhere under ``root``.

    Only the **basename** of ``Exported media filename`` is ever used, matched
    case-insensitively against the files under ``root``. The manifest's own path
    components are discarded, so an absolute path or a ``../`` traversal collapses
    to a name that is only ever looked for inside the batch tree — resolution
    provably cannot leave ``root``. That containment comes from matching basenames
    rather than from refusing to recurse, which is why subdirectories
    (``dir/photos/``, ``dir/videos/``) are safe to search.

    Each row's parent posting is taken from the manifest's own key when it names a
    known posting (see :func:`_derive_posting_id`); only when it does not is album
    membership inferred from ``albums``, and then only with timestamp agreement.

    Orphan rows, rows with no local file and rows whose basename is ambiguous are
    counted and reported once, not logged per row (a full manifest may have tens of
    thousands).

    Args:
        media_df (pd.DataFrame): Manifest with ``Media ID`` + ``Exported media filename``.
        posting_uuids (dict[str, str]): ``Posting ID → UUID`` from the postings table.
        root (Path): The batch tree root; every media file is looked up under it.
        manifest_dir (Path | None): Directory holding the manifest, used to break a
            duplicate-basename tie. Defaults to ``root``.
        network_index (dict[str, str] | None): Index from
            :func:`build_network_posting_index`, consulted only when the manifest's
            declared key names no known posting. ``None`` (the default) disables it.
        albums (AlbumIndex | None): Index from :func:`build_posting_album_index`.
            ``None`` (the default) disables album inference entirely.
        album_tolerance_s (float): Maximum timestamp disagreement, in seconds,
            allowed when accepting an inferred album link.
        stamps (StampIndex | None): Index from :func:`build_posting_stamp_index`,
            consulted for a row no key and no album ordering can reach.
            ``None`` (the default) disables it.
        texts (TextIndex | None): Index from :func:`build_posting_text_index`, the
            last resort for a row whose author names no posting (a shared post).
            ``None`` (the default) disables it.

    Returns:
        list[MediaLink]: One per row whose posting is known and file exists.
    """
    tie_break_dir = manifest_dir if manifest_dir is not None else root
    present = build_file_index(root)
    media_stamps = _parse_timestamps(media_df) if (albums or stamps) else None
    stamp_values: list[pd.Timestamp | None] = list(media_stamps) if media_stamps is not None else [None] * len(media_df)
    links: list[MediaLink] = []
    exact_links = 0
    network_links = 0
    album_links = 0
    stamp_links = 0
    text_links = 0
    orphan_skips = 0
    missing_skips = 0
    ambiguous_skips = 0
    for (_, row), stamp in zip(media_df.iterrows(), stamp_values, strict=True):
        media_id = str(row.get("Media ID") or "").strip()
        if not media_id:
            continue
        network_id = str(row.get("Network ID") or "").strip()
        posting_id = _derive_posting_id(network_id, media_id, posting_uuids)
        inferred = False
        by_network_id = False
        if posting_id is None and network_index:
            posting_id = _derive_network_posting_id(network_id, media_id, network_index)
            by_network_id = posting_id is not None
        if posting_id is None and albums:
            posting_id = _infer_album_posting_id(media_id, stamp, albums, album_tolerance_s)
            inferred = posting_id is not None
        by_stamp = False
        if posting_id is None and stamps:
            posting_id = _infer_stamp_posting_id(row, stamp, stamps)
            by_stamp = posting_id is not None
        by_text = False
        if posting_id is None and texts:
            posting_id = _infer_text_posting_id(row, texts)
            by_text = posting_id is not None
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
        if inferred:
            album_links += 1
        elif by_stamp:
            stamp_links += 1
        elif by_text:
            text_links += 1
        elif by_network_id:
            network_links += 1
        else:
            exact_links += 1
        links.append(MediaLink(posting_uuid=uuid, posting_id=posting_id, media_id=media_id, path=path))
    if len(media_df):
        # Aggregate rather than log per row: a full manifest dropped in with only a few
        # referenced files present would otherwise emit one line per row (tens of
        # thousands). A single summary keeps large drop-ins robust and quiet.
        logger.info(
            "Social linker: {} media linked ({} by manifest key, {} by network id, "
            "{} by album inference, {} by timestamp, {} by text match), {} skipped "
            "({} with no matching posting, {} with no local file, {} with an ambiguous filename) "
            "across {} manifest rows.",
            len(links),
            exact_links,
            network_links,
            album_links,
            stamp_links,
            text_links,
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


def _profile_headers(style: str) -> set[str]:
    """Return the normalized header set of a :class:`TableReader` schema profile.

    Derived from the single source of truth in ``TableReader`` so ``_find_tables``
    stays in sync whenever a profile's header list changes.

    Args:
        style (str): The profile's ``style`` (e.g. ``"postings"``, ``"messages"``).

    Returns:
        set[str]: The profile's normalized headers, empty when no profile matches.
    """
    return next(
        (profile.normalized_headers for profile in TableReader.schema_profiles if profile.style == style),
        set(),
    )


#: Exact header set of the postings profile — the canonical shape of a postings table.
_POSTINGS_HEADERS: set[str] = _profile_headers("postings")

#: Exact header set of the messages profile — the shape a chat-style export
#: carries its postings in, accepted as a substitute postings table.
_MESSAGES_HEADERS: set[str] = _profile_headers("messages")

#: Messages-profile columns renamed into the postings vocabulary the join rules read.
_MESSAGES_TO_POSTINGS: dict[str, str] = {
    "Chat ID": "Posting ID",
    "Sender": "Author",
    "Text": "Text Content",
}


def normalize_postings_frame(postings_df: pd.DataFrame) -> pd.DataFrame:
    """Return ``postings_df`` with a messages-schema table renamed into postings vocabulary.

    A chat-style export names the same things ``Chat ID`` / ``Sender`` /
    ``Text``; renaming once here leaves every rule and index builder untouched.
    Only an exact messages header-set match is rewritten, so a foreign table
    that merely carries a ``Text`` column is not reinterpreted.

    The rename destroys the profile match, so
    :func:`build_posting_reference_index` must run on the *original* frame.

    Args:
        postings_df (pd.DataFrame): The table found in the postings-table role.

    Returns:
        pd.DataFrame: A renamed copy for a messages table, else ``postings_df``.
    """
    lookup = {str(column).strip().casefold(): column for column in postings_df.columns}
    if set(lookup) != _MESSAGES_HEADERS:
        return postings_df
    renamed = {
        lookup[source.casefold()]: target
        for source, target in _MESSAGES_TO_POSTINGS.items()
        if source.casefold() in lookup
    }
    return postings_df.rename(columns=renamed)


# Schema profiles a posting's reference fields can be read from — the postings
# profile, and the messages profile a chat-style export carries instead.
_REFERENCE_PROFILE_STYLES: frozenset[str] = frozenset({"postings", "messages"})

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
    """Return ``{posting id: prefixed posting reference fields}`` from a postings table.

    Reuses the :class:`TableReader` schema profiles — postings, or the messages
    profile a chat-style export carries instead — so the column mapping stays
    declared in one place. Rows are keyed by the profile's own ``id_col``
    (``Posting ID`` / ``Chat ID``), the same column
    :func:`normalize_postings_frame` renames for the join. Keys are prefixed via
    :data:`_POSTING_REFERENCE_KEYS` (``network`` → ``posting_network``, ...);
    empty / missing values are omitted.

    Must be called on the *original* frame: the detection matches on the exact
    header set, which :func:`normalize_postings_frame` deliberately destroys.

    Args:
        postings_df (pd.DataFrame): Table carrying a social export schema.

    Returns:
        dict[str, dict[str, Any]]: Mapping from posting id to the prefixed
        posting reference fields. Empty when the headers match neither profile —
        derived artifacts then carry link ids only, matching the pre-enrichment
        behavior.
    """
    profile, normalized_map = TableReader._detect_schema_profile(postings_df.columns)
    if profile is None or profile.style not in _REFERENCE_PROFILE_STYLES:
        logger.warning(
            "Social linker: postings table matches neither the postings nor the messages profile; "
            "media artifacts keep link ids only."
        )
        return {}
    id_column = normalized_map.get(profile.id_col.strip().casefold(), profile.id_col)
    index: dict[str, dict[str, Any]] = {}
    for _, row in postings_df.iterrows():
        posting_id = str(row.get(id_column) or "").strip()
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
    timestamp_link_enabled: bool = True
    text_link_enabled: bool = True

    def _find_tables(self, data_dir: Path) -> tuple[Path | None, Path | None]:
        """Locate the postings table and media manifest anywhere in the tree.

        A messages-schema table stands in for the postings table when an export
        carries no real one (the X/Twitter shape). A postings table always wins
        when both are present — the substitute is a fallback, never a competitor —
        so precedence is resolved after the sweep rather than by filename order.

        Args:
            data_dir (Path): The batch tree root.

        Returns:
            tuple[Path | None, Path | None]: ``(postings_csv, media_csv)``.
        """
        postings_exact: Path | None = None
        postings_messages: Path | None = None
        media: Path | None = None
        for path in sorted(data_dir.rglob("*.csv")):
            try:
                columns = pd.read_csv(path, sep=_sniff_delimiter(path), nrows=0, encoding="utf-8-sig").columns
            except Exception:
                continue
            normalized = {str(c).strip().casefold() for c in columns}
            if media is None and is_media_manifest(columns):
                media = path
            elif postings_exact is None and normalized == _POSTINGS_HEADERS:
                postings_exact = path
            elif postings_messages is None and normalized == _MESSAGES_HEADERS:
                postings_messages = path
        return postings_exact or postings_messages, media

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
        # Order matters: the reference index detects the schema profile from the
        # original headers, which normalize_postings_frame deliberately rewrites.
        posting_references = build_posting_reference_index(postings_df)
        join_df = normalize_postings_frame(postings_df)
        posting_uuids = build_posting_index(join_df)
        media_df = pd.read_csv(media_csv, sep=_sniff_delimiter(media_csv), dtype=str, encoding="utf-8-sig")
        albums = build_posting_album_index(join_df) if self.album_link_enabled else None
        network_index = build_network_posting_index(join_df)
        stamps = build_posting_stamp_index(join_df) if self.timestamp_link_enabled else None
        texts = build_posting_text_index(join_df) if self.text_link_enabled else None
        links = resolve_media_rows(
            media_df,
            posting_uuids,
            data_dir,
            manifest_dir=media_csv.parent,
            network_index=network_index,
            albums=albums,
            album_tolerance_s=self.album_tolerance_s,
            stamps=stamps,
            texts=texts,
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
