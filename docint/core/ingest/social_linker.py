"""Link a ``me-dossier/1`` social export: posting and comment rows, media joined to their posts.

A crawler export holds one ``dossier.json`` per profile folder. It carries the
account's postings, the comments each one received, and its media, and the two
sides name each other explicitly: ``postings[].mediaIds`` and
``media[].attachedTo.postings``. The linker turns every posting and comment into
one row-shaped Document and routes each linked media file into the modality
pipelines (CLIP / Nextext), stamped with its posting's identity. The one link
the export still leaves out is a Telegram photo's, joined by the album rule
(:func:`build_posting_album_index`).
"""

from __future__ import annotations

import bisect
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from llama_index.core import Document
from loguru import logger

from docint.core.ingest.images_service import ImageAsset, IngestContext
from docint.core.ingest.media_transcribe import MediaClip, MediaTranscriber
from docint.core.ingest.preprocess import preprocess_key, run_all
from docint.core.jobs import JobCancelled
from docint.utils.hashing import compute_file_hash

#: The one export shape the linker reads.
DOSSIER_SCHEMA = "me-dossier/1"
DOSSIER_NAME = "dossier.json"
#: The crawl's own progress record, written beside every dossier.
PROGRESS_NAME = "progress.json"

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".gif"}

#: Default slack allowed between a photo's timestamp and that of the posting
#: the album rule would attach it to. Deliberately tight: measured on a real
#: Telegram export every true album member sits 0-1 s from its posting, while a
#: neighbouring post is hours away, so a small window rejects nearly every
#: mis-attribution a pruned/partial export could otherwise produce.
_DEFAULT_ALBUM_TOLERANCE_S = 5.0

#: ``{channel: sorted [(message_no, posting id, published)]}`` — see
#: :func:`build_posting_album_index`.
AlbumIndex = dict[str, list[tuple[int, str, datetime]]]

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


@dataclass(frozen=True)
class MediaLink:
    """A media file resolved to its owning posting."""

    posting_uuid: str
    posting_id: str
    media_id: str
    path: Path
    posting_ref: dict[str, str]


def is_image(path: Path) -> bool:
    """Return whether ``path`` has a still-image extension (vs. video/audio)."""
    return path.suffix.lower() in _IMAGE_EXTS


def build_file_index(root: Path) -> dict[str, list[Path]]:
    """Index every file under ``root`` (recursively) by lowercase basename.

    Args:
        root (Path): The tree to index.

    Returns:
        dict[str, list[Path]]: ``{basename_lower: [paths]}``, sorted per key.
    """
    index: dict[str, list[Path]] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            index.setdefault(path.name.lower(), []).append(path)
    return index


def _parse_time(value: Any) -> datetime | None:
    """Return an ISO-8601 stamp as an aware datetime, or ``None`` when it does not parse.

    A naive stamp is read as UTC, so any two parsed stamps compare.

    Args:
        value (Any): The raw ``publishedAt`` value.

    Returns:
        datetime | None: The stamp, timezone-aware, or ``None``.
    """
    try:
        stamp = datetime.fromisoformat(str(value))
    except ValueError:
        return None
    return stamp if stamp.tzinfo is not None else stamp.replace(tzinfo=UTC)


def _split_channel(identifier: str, prefix: str) -> int | None:
    """Return the numeric message number ``identifier`` carries after ``prefix``.

    Args:
        identifier (str): A posting's or photo's ``platformId``.
        prefix (str): The channel id (a posting's ``author.platformId``).

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


def build_posting_album_index(postings: list[dict[str, Any]]) -> AlbumIndex:
    """Return the per-channel message-number index used to infer album membership.

    Telegram names a photo only by its own message id, and a multi-photo post
    (an album) is N consecutive messages whose text sits on the **last** — so
    the owner of a photo no posting names is the first posting at or above its
    message number. A single-photo post is the same rule at distance zero: the
    photo carries its post's own id.

    Only postings whose ``platformId`` decomposes as
    ``<author.platformId><digits>`` with a parseable ``publishedAt`` are
    indexed. That requirement is also what keeps the rule inert elsewhere:
    Instagram's ``<postId>_<accountId>`` carries the account as a suffix and
    Facebook's ids are opaque, so neither yields a channel.

    Args:
        postings (list[dict[str, Any]]): The dossier's posting records.

    Returns:
        AlbumIndex: ``{channel: sorted [(message_no, posting id, published)]}``,
        empty when no posting decomposes this way.
    """
    channels: AlbumIndex = {}
    for posting in postings:
        channel = str((posting.get("author") or {}).get("platformId") or "")
        message_no = _split_channel(str(posting.get("platformId") or ""), channel)
        stamp = _parse_time(posting.get("publishedAt"))
        if message_no is None or stamp is None or not posting.get("id"):
            continue
        channels.setdefault(channel, []).append((message_no, str(posting["id"]), stamp))
    for entries in channels.values():
        entries.sort(key=lambda entry: entry[0])
    return channels


def _infer_album_posting_id(
    media_id: str,
    media_stamp: datetime | None,
    albums: AlbumIndex,
    tolerance_s: float,
) -> str | None:
    """Return the posting whose album ``media_id`` belongs to, or ``None``.

    Picks the first posting in the photo's own channel whose message number is
    at or above the photo's, then **requires the two timestamps to agree** within
    ``tolerance_s``. That corroboration is what makes the inference safe: when the
    owning posting is absent from the export the next one along is hours away and
    is rejected, leaving the photo unlinked rather than mis-attributed.

    Args:
        media_id (str): The photo's ``platformId``.
        media_stamp (datetime | None): The photo's parsed ``publishedAt``.
        albums (AlbumIndex): Index from :func:`build_posting_album_index`.
        tolerance_s (float): Maximum allowed timestamp difference, in seconds.

    Returns:
        str | None: The owning posting's id, or ``None`` when no channel
        matches, no posting sits above the photo, or the timestamps disagree.
    """
    if not albums or media_stamp is None:
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


def _reference(
    record: dict[str, Any],
    kind: str,
    *,
    anchor_text: str | None = None,
    parent_text: str | None = None,
) -> dict[str, Any]:
    """Return a posting's or comment's ``reference_metadata`` block.

    Only registry keys (``docint/utils/reference_metadata.py``), so the API,
    the SPA, reports, CSVs and search read a dossier row exactly as they read a
    social table row. Every id is the network's own (``platformId``).

    Args:
        record (dict[str, Any]): A posting or comment record.
        kind (str): ``posting`` or ``comment``.
        anchor_text (str | None): A comment's posting text.
        parent_text (str | None): The text of the comment a reply answers.

    Returns:
        dict[str, Any]: The reference fields.
    """
    author = record.get("author") or {}
    return {
        "network": record.get("network"),
        "type": kind,
        "uuid": record.get("id"),
        "url": record.get("url"),
        "timestamp": record.get("publishedAt"),
        "author": author.get("name"),
        "author_id": author.get("platformId"),
        "vanity": author.get("vanity"),
        "text": record.get("text"),
        "text_id": record.get("platformId"),
        "anchor_text": anchor_text,
        "parent_text": parent_text,
    }


def _posting_fields(posting: dict[str, Any]) -> dict[str, str]:
    """Return a posting's reference fields, prefixed for the artifacts cut from its media.

    Built from the same block the posting's own row carries, so a photo and the
    post it hangs off can never disagree. Empty values are omitted.

    Args:
        posting (dict[str, Any]): The posting record.

    Returns:
        dict[str, str]: ``{posting_network: …, posting_author: …, …}``.
    """
    reference = _reference(posting, "posting")
    fields: dict[str, str] = {}
    for key, prefixed in _POSTING_REFERENCE_KEYS.items():
        value = reference.get(key)
        text = "" if value is None else str(value).strip()
        if text:
            fields[prefixed] = text
    return fields


def _dossier_documents(dossier: dict[str, Any], path: Path, data_dir: Path) -> list[Document]:
    """Return one row-shaped Document per posting and per comment it received.

    The shape is a table row's — ``source: "table"`` plus ``table.style`` /
    ``row_index`` / ``n_rows`` — because node routing, social detection, the
    collection profile and extract posting units all key on it. Rows are
    numbered in file order, each posting followed by its comments; an empty
    text makes no row but keeps its number, as an empty table row did.

    Args:
        dossier (dict[str, Any]): The parsed dossier.
        path (Path): The dossier file.
        data_dir (Path): The batch root.

    Returns:
        list[Document]: The posting and comment rows, in file order.
    """
    rows: list[tuple[str, dict[str, Any]]] = []
    for posting in dossier.get("postings") or []:
        rows.append(("postings", _reference(posting, "posting")))
        comments = posting.get("comments") or []
        texts = {comment.get("id"): comment.get("text") for comment in comments}
        for comment in comments:
            parent_text = texts.get(comment.get("parentCommentId"))
            rows.append(
                ("comments", _reference(comment, "comment", anchor_text=posting.get("text"), parent_text=parent_text))
            )
    # Every profile's file is called dossier.json and the documents listing keys
    # by name, so a row names its file by its path within the batch.
    name = path.relative_to(data_dir).as_posix()
    file_hash = compute_file_hash(path)
    documents: list[Document] = []
    for row_index, (style, reference) in enumerate(rows):
        text = str(reference.get("text") or "").strip()
        if not text:
            continue
        metadata = {
            "file_path": str(path),
            "file_name": name,
            "filename": name,
            "file_hash": file_hash,
            "origin": {"filename": name, "filetype": "application/json"},
            "source": "table",
            "table": {"style": style, "row_index": row_index, "n_rows": len(rows)},
            "reference_metadata": reference,
        }
        documents.append(Document(text=text, metadata=metadata))
    return documents


def _dossier_links(
    dossier: dict[str, Any],
    folder: Path,
    *,
    album_tolerance_s: float | None,
    counts: dict[str, int],
) -> list[MediaLink]:
    """Join every media item to the postings that own it, resolved to its file.

    A link is explicit when either side names the other — ``posting.mediaIds``
    or ``media.attachedTo.postings``; only postings in this dossier count. A
    media item neither side names is attached by the album rule when
    ``album_tolerance_s`` is set, else it is left unlinked — and unclaimed, so
    the standalone passes still ingest it. Only the basename of ``media.file``
    is looked up, and only inside the dossier's own folder, so no path in the
    export can reach outside it.

    Args:
        dossier (dict[str, Any]): The parsed dossier.
        folder (Path): The dossier's profile folder.
        album_tolerance_s (float | None): The album rule's timestamp tolerance;
            ``None`` disables the rule.
        counts (dict[str, int]): Per-outcome counters, updated in place.

    Returns:
        list[MediaLink]: One per (media file, owning posting) pair.
    """
    postings = {str(posting["id"]): posting for posting in dossier.get("postings") or [] if posting.get("id")}
    owners: dict[str, dict[str, None]] = {}  # media id -> ordered set of posting ids
    for uuid, posting in postings.items():
        for media_id in posting.get("mediaIds") or []:
            owners.setdefault(str(media_id), {})[uuid] = None
    albums = build_posting_album_index(list(postings.values())) if album_tolerance_s is not None else {}
    files = build_file_index(folder)
    links: list[MediaLink] = []
    for item in dossier.get("media") or []:
        media_id = str(item.get("id") or "")
        for owner in (item.get("attachedTo") or {}).get("postings") or []:
            owners.setdefault(media_id, {})[str(owner.get("id"))] = None
        name = Path(str(item.get("file") or "").replace("\\", "/")).name
        if not name or item.get("status") == "missing":
            counts["missing"] += 1
            continue
        declared = owners.get(media_id, {})
        linked = [uuid for uuid in declared if uuid in postings]
        how = "explicit"
        # The rule only speaks for media the export names no owner for: one it
        # names an unexported owner for stays unlinked rather than re-homed.
        if not declared and album_tolerance_s is not None:
            inferred = _infer_album_posting_id(
                str(item.get("platformId") or ""), _parse_time(item.get("publishedAt")), albums, album_tolerance_s
            )
            if inferred:
                linked, how = [inferred], "album"
        if not linked:
            counts["unlinked"] += 1
            continue
        matches = files.get(name.lower(), [])
        if len(matches) != 1:
            counts["ambiguous" if matches else "no_file"] += 1
            continue
        counts[how] += 1
        for uuid in linked:
            posting = postings[uuid]
            links.append(
                MediaLink(
                    posting_uuid=uuid,
                    posting_id=str(posting.get("platformId") or ""),
                    media_id=str(item.get("platformId") or media_id),
                    path=matches[0],
                    posting_ref=_posting_fields(posting),
                )
            )
    return links


def _bookkeeping(path: Path, data_dir: Path) -> set[Path]:
    """Return the export's bookkeeping around a dossier, so no generic reader ingests it.

    That is the crawl's ``progress.json`` beside the dossier, and the
    underscore-prefixed run files in the export root — the folder holding the
    profile folders — when that folder is inside the batch.

    Args:
        path (Path): The dossier file.
        data_dir (Path): The batch root.

    Returns:
        set[Path]: The paths to claim.
    """
    progress = path.with_name(PROGRESS_NAME)
    claimed = {progress} if progress.is_file() else set()
    if path.parent != data_dir:
        claimed |= {entry for entry in path.parent.parent.iterdir() if entry.is_file() and entry.name.startswith("_")}
    return claimed


@dataclass
class SocialLinkResult:
    """Outcome of a social-linker pass over one batch tree."""

    consumed_paths: set[Path] = field(default_factory=set)
    documents: list[Document] = field(default_factory=list)


@dataclass
class SocialLinker:
    """Read a batch's ``me-dossier/1`` exports and route their media, linked to their postings."""

    image_service: Any
    nextext_client: Any
    target_collection: str | None
    manifest: Any = None
    keyframe_dedup_cosine: float = 0.95
    nextext_max_concurrency: int = 4
    album_link_enabled: bool = True
    album_tolerance_s: float = _DEFAULT_ALBUM_TOLERANCE_S
    # The preprocessing pool: images are linked through it across the export,
    # and one the pool is still storing loose is waited on first. ``None``
    # runs inline.
    pool: Any = None
    # The job's progress channel; an export of a few thousand images spends
    # this whole pass saying nothing otherwise.
    progress_callback: Callable[[str], None] | None = None

    def run(self, data_dir: Path) -> SocialLinkResult:
        """Read every ``dossier.json`` under ``data_dir``; no-op when there is none.

        Args:
            data_dir (Path): The batch tree root.

        Returns:
            SocialLinkResult: Claimed paths, plus the posting, comment and
            transcript Documents for the pipeline.
        """
        result = SocialLinkResult()
        counts = dict.fromkeys(
            (
                "dossiers",
                "unreadable",
                "postings",
                "comments",
                "explicit",
                "album",
                "missing",
                "no_file",
                "ambiguous",
                "unlinked",
                "messages",
                "own_comments",
            ),
            0,
        )
        tolerance = self.album_tolerance_s if self.album_link_enabled else None
        links: list[MediaLink] = []
        for path in sorted(data_dir.rglob(DOSSIER_NAME)):
            # Claimed before it is read: an unreadable or future-format dossier
            # must not reach the generic JSON reader either.
            result.consumed_paths |= {path, *_bookkeeping(path, data_dir)}
            try:
                dossier = json.loads(path.read_text(encoding="utf-8-sig"))
                schema = dossier.get("schema") if isinstance(dossier, dict) else None
                if schema != DOSSIER_SCHEMA:
                    raise ValueError(f"schema {schema!r} is not {DOSSIER_SCHEMA!r}")
                documents = _dossier_documents(dossier, path, data_dir)
                links += _dossier_links(dossier, path.parent, album_tolerance_s=tolerance, counts=counts)
            except Exception as exc:  # one bad dossier must not cost the rest of the export
                counts["unreadable"] += 1
                logger.warning("Social linker skipped '{}': {}", path.relative_to(data_dir).as_posix(), exc)
                continue
            postings = sum(1 for doc in documents if doc.metadata["table"]["style"] == "postings")
            counts["dossiers"] += 1
            counts["postings"] += postings
            counts["comments"] += len(documents) - postings
            counts["messages"] += len(dossier.get("messages") or [])
            counts["own_comments"] += len(dossier.get("ownComments") or [])
            result.documents.extend(documents)
        if not counts["dossiers"] and not counts["unreadable"]:
            return result
        skipped = counts["missing"] + counts["no_file"] + counts["ambiguous"] + counts["unlinked"]
        logger.info(
            "Social linker: {} dossiers ({} unreadable), {} postings and {} comments; {} media linked "
            "({} explicitly, {} by album inference), {} skipped ({} missing from the export, "
            "{} with no local file, {} with an ambiguous filename, {} with no posting).",
            counts["dossiers"],
            counts["unreadable"],
            counts["postings"],
            counts["comments"],
            counts["explicit"] + counts["album"],
            counts["explicit"],
            counts["album"],
            skipped,
            counts["missing"],
            counts["no_file"],
            counts["ambiguous"],
            counts["unlinked"],
        )
        if counts["messages"] or counts["own_comments"]:
            logger.warning(
                "Social linker: {} messages and {} own comments are not ingested; the export format "
                "has no sample of them yet.",
                counts["messages"],
                counts["own_comments"],
            )
        try:
            routed = self._route(links)
        except JobCancelled:
            raise
        except Exception as exc:  # the rows must not depend on the media stores being up
            logger.warning(
                "Social linker could not route media; postings kept, media left to the standalone passes: {}", exc
            )
            return result
        result.consumed_paths |= routed.consumed_paths
        result.documents.extend(routed.documents)
        return result

    def _route(self, links: list[MediaLink]) -> SocialLinkResult:
        """Route linked media into CLIP / Nextext, stamped with their posting's identity.

        Args:
            links (list[MediaLink]): The resolved media links.

        Returns:
            SocialLinkResult: The media paths claimed and the transcript Documents.
        """
        routed = SocialLinkResult()
        context = IngestContext(source_collection=self.target_collection)
        collection = self.target_collection or ""
        clips: list[MediaClip] = []
        image_jobs: list[tuple[str, Callable[[], Any]]] = []
        for link in links:
            posting_ref = link.posting_ref
            link_ids = {
                "posting_uuid": link.posting_uuid,
                "posting_id": link.posting_id,
                "media_id": link.media_id,
            }
            if is_image(link.path):
                routed.consumed_paths.add(link.path)
                asset = ImageAsset.from_path(
                    path=link.path,
                    source_type="social_media",
                    source_doc_id=link.posting_uuid,
                    extra_metadata={
                        **link_ids,
                        "source_type": "social_media",
                        **posting_ref,
                        "reference_metadata": {"type": "image", **link_ids, **posting_ref},
                    },
                )
                image_jobs.append(self._image_job(asset, context, collection, link.posting_uuid))
            else:
                # The clip's own name and hash, stamped on every artifact cut
                # from it: without them a keyframe names no file at all and a
                # segment names the transient JSONL it was parsed from, so an
                # extract and a report cannot say which attachment an artifact
                # came out of. `file_hash` is deliberately NOT set here -- it
                # is the transcript's own hash, and the pipeline dedupes
                # already-ingested files by it.
                media_hash = compute_file_hash(link.path)
                media_ref = {
                    "source_file": link.path.name,
                    "media_file_hash": media_hash,
                }
                clips.append(
                    MediaClip(
                        path=link.path,
                        source_doc_id=link.posting_uuid,
                        media_hash=media_hash,
                        keyframe_extra_metadata={
                            **link_ids,
                            "source_type": "social_media",
                            **media_ref,
                            "source_path": str(link.path),
                            **posting_ref,
                            "reference_metadata": {
                                "type": "keyframe",
                                **link_ids,
                                **media_ref,
                                **posting_ref,
                            },
                        },
                        # Flat keys only: the transcript reader owns the segment's
                        # reference_metadata and merges these in additively.
                        transcript_extra_info={
                            **link_ids,
                            **media_ref,
                            "file_name": link.path.name,
                            "filename": link.path.name,
                            "file_path": str(link.path),
                            **posting_ref,
                        },
                    )
                )
        run_all(self.pool, image_jobs, on_done=self._image_progress)
        sub = MediaTranscriber(
            image_service=self.image_service,
            nextext_client=self.nextext_client,
            target_collection=self.target_collection,
            manifest=self.manifest,
            keyframe_dedup_cosine=self.keyframe_dedup_cosine,
            nextext_max_concurrency=self.nextext_max_concurrency,
            pool=self.pool,
            progress_callback=self.progress_callback,
            preprocess_progress=self.pool.progress if self.pool is not None else None,
        ).run(clips)
        routed.consumed_paths |= sub.consumed_paths
        routed.documents.extend(sub.transcript_documents)
        return routed

    def _image_progress(self, done: int, total: int) -> None:
        """Report how many of the export's linked images have been stored."""
        if self.progress_callback:
            self.progress_callback(f"Linking images: {done}/{total} images linked")

    def _image_job(
        self, asset: ImageAsset, context: IngestContext, collection: str, posting_uuid: str
    ) -> tuple[str, Callable[[], Any]]:
        """One linked image as a keyed task: wait for any loose copy in flight, then link it.

        The key names the posting as well as the bytes, so it never collides
        with the pool's standalone task for the same file — which is joined,
        not replaced, since a loose point must still be *relinked* by the
        social call that follows. Joining from a pool worker is safe only
        because every ``image:`` task is submitted before any link task (by
        the upload hook or the job's prefetch) and the executor is FIFO, so
        the joined task is never queued behind the one waiting on it.

        Args:
            asset (ImageAsset): The social asset for the image.
            context (IngestContext): Collection-resolution context.
            collection (str): Physical collection name.
            posting_uuid (str): The posting the image belongs to.

        Returns:
            tuple[str, Callable[[], Any]]: ``(key, task)`` for ``run_all``.
        """
        if self.pool is None:
            return "", lambda: self.image_service.ingest_image(asset, context=context)
        file_hash = compute_file_hash(asset.image_path) if asset.image_path else ""
        pool = self.pool

        def task() -> Any:
            pool.join(preprocess_key("image", collection, file_hash))
            return self.image_service.ingest_image(asset, context=context)

        return preprocess_key("image-link", collection, f"{file_hash}:{posting_uuid}"), task
