"""Join a social export's postings table to its media manifest + files.

Pure join logic lives here (counter stripping, set-membership matching,
recursive file resolution). Routing of resolved media into the modality
pipelines (CLIP / Nextext) lives in :class:`SocialLinker` (Task 10).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
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


def build_file_index(root: Path) -> dict[str, list[Path]]:
    """Index every file under ``root`` (recursively) by lowercase basename.

    Args:
        root (Path): The batch tree root.

    Returns:
        dict[str, list[Path]]: ``{basename_lower: [paths]}`` (sorted per key).
    """
    index: dict[str, list[Path]] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            index.setdefault(path.name.lower(), []).append(path)
    return index


def _resolve_path(
    exported_filename: str,
    file_index: dict[str, list[Path]],
    tables_dir: Path,
) -> Path | None:
    """Resolve an ``Exported media filename`` to a file in the batch tree.

    Prefers a path relative to the tables folder when the value carries one;
    otherwise matches by basename across the tree, logging on collision.

    Args:
        exported_filename (str): The manifest's filename (basename or rel path).
        file_index (dict[str, list[Path]]): Recursive basename index.
        tables_dir (Path): Directory holding the manifest (relative-path anchor).

    Returns:
        Path | None: The resolved file, or ``None`` when missing.
    """
    name = str(exported_filename or "").strip().replace("\\", "/")
    if not name:
        return None
    if "/" in name:
        candidate = (tables_dir / name).resolve()
        if candidate.is_file():
            return candidate
    matches = file_index.get(Path(name).name.lower(), [])
    if not matches:
        return None
    if len(matches) > 1:
        logger.warning("Media filename {!r} matched {} files; using {}", name, len(matches), matches[0])
    return matches[0]


def resolve_media_rows(
    media_df: pd.DataFrame,
    posting_uuids: dict[str, str],
    file_index: dict[str, list[Path]],
    *,
    tables_dir: Path,
) -> list[MediaLink]:
    """Resolve manifest rows to ``MediaLink``s, skipping orphans and missing files.

    Args:
        media_df (pd.DataFrame): Manifest with ``Media ID`` + ``Exported media filename``.
        posting_uuids (dict[str, str]): ``Posting ID → UUID`` from the postings table.
        file_index (dict[str, list[Path]]): Recursive basename index.
        tables_dir (Path): Manifest directory (relative-path anchor).

    Returns:
        list[MediaLink]: One per row whose posting is known and file exists.
    """
    links: list[MediaLink] = []
    for _, row in media_df.iterrows():
        media_id = str(row.get("Media ID") or "").strip()
        if not media_id:
            continue
        posting_id = strip_counter(media_id)
        uuid = posting_uuids.get(posting_id)
        if uuid is None:
            logger.debug("Orphan media row {!r} (no posting {!r})", media_id, posting_id)
            continue
        path = _resolve_path(str(row.get("Exported media filename") or ""), file_index, tables_dir)
        if path is None:
            logger.warning("Media file for {!r} not found; skipping.", media_id)
            continue
        links.append(MediaLink(posting_uuid=uuid, posting_id=posting_id, media_id=media_id, path=path))
    return links


def is_image(path: Path) -> bool:
    """Return whether ``path`` has a still-image extension (vs. video/audio)."""
    return path.suffix.lower() in _IMAGE_EXTS
