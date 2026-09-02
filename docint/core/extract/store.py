"""On-disk store for rendered extracts.

A bundle outlives the job that built it: the job registry is in-memory and
evicts finished runs, so the list a client sees must come from here, never
from the registry.

One directory per physical collection, holding an archive and a JSON sidecar
per build. Both the collection name and the extract id are validated before
any path join — they arrive from a request, and a store that writes where it
is told is a directory traversal.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from loguru import logger

__all__ = ["ExtractRecord", "ExtractStore"]

#: One stored build, as listed over the API.
ExtractRecord = dict[str, Any]

_ID_PATTERN = re.compile(r"^[0-9]{8}-[0-9]{6}-[0-9a-f]{8}$")
_COLLECTION_PATTERN = re.compile(r"^[A-Za-z0-9._-]{1,128}$")


class ExtractStore:
    """Read and write extract bundles under one root directory."""

    def __init__(self, root: Path) -> None:
        """Create the store.

        Args:
            root (Path): Directory holding one subdirectory per collection.
        """
        self._root = Path(root)
        #: The root's normalized form with a trailing separator, so a prefix test
        #: cannot pass for a sibling whose name merely starts with the root's.
        self._root_prefix = os.path.join(os.path.realpath(self._root), "")

    def _contained(self, candidate: Path) -> Path:
        """Return a path once it is proven to resolve under the store root.

        The patterns above already reject every separator, so a valid name can
        never fail here. This is the containment proof itself: it normalizes the
        path and compares it against the root, which is what makes the guard
        independent of the caller and holds even where the root is reached
        through a symlink.

        The shape matters as much as the check. Normalizing and then testing the
        prefix on its own is what a static analyser can follow, so the condition
        stays a single ``startswith`` whose false branch raises. Widening it into
        a compound test leaves the surviving branch proving nothing, which is how
        an earlier attempt at this guard passed review and still read as unsafe.

        Args:
            candidate (Path): The path about to be read or written.

        Returns:
            Path: The normalized path.

        Raises:
            ValueError: When the path resolves outside the root.
        """
        resolved = os.path.realpath(candidate)
        if not resolved.startswith(self._root_prefix):
            raise ValueError(f"Extract path escapes the store root: {candidate!r}")
        return Path(resolved)

    def _dir(self, physical: str) -> Path:
        """Return a collection's directory, refusing an unsafe name.

        Args:
            physical (str): Owner-namespaced Qdrant collection name.

        Returns:
            Path: The collection's directory (not created).

        Raises:
            ValueError: When the name could escape the root.
        """
        if not _COLLECTION_PATTERN.match(physical or ""):
            raise ValueError(f"Unsafe collection name for the extract store: {physical!r}")
        return self._contained(self._root / physical)

    def path(self, physical: str, extract_id: str) -> Path:
        """Return the archive path for one build.

        Args:
            physical (str): Collection directory name.
            extract_id (str): The build's id.

        Returns:
            Path: The ``.zip`` path (which may not exist).

        Raises:
            ValueError: When either identifier is unsafe.
        """
        if not _ID_PATTERN.match(extract_id or ""):
            raise ValueError(f"Unsafe extract id: {extract_id!r}")
        return self._contained(self._dir(physical) / f"{extract_id}.zip")

    def write(self, physical: str, *, zip_bytes: bytes, meta: dict[str, Any], now: datetime) -> ExtractRecord:
        """Store one build and return its record.

        The archive is written to a temporary name and moved into place, so a
        crash mid-write cannot leave a half-archive that lists as complete.

        Args:
            physical (str): Collection directory name.
            zip_bytes (bytes): The archive.
            meta (dict[str, Any]): Extra sidecar fields — ``collection``,
                ``target``, ``counts``, ``pdf_skipped``.
            now (datetime): Build time.

        Returns:
            ExtractRecord: The stored record.
        """
        extract_id = f"{now:%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:8]}"
        directory = self._dir(physical)
        directory.mkdir(parents=True, exist_ok=True)
        archive = directory / f"{extract_id}.zip"
        temporary = directory / f".{extract_id}.zip.tmp"
        temporary.write_bytes(zip_bytes)
        os.replace(temporary, archive)

        collection = str(meta.get("collection") or physical)
        record: ExtractRecord = {
            "extract_id": extract_id,
            "collection": collection,
            "filename": f"{collection}-extract-{now:%Y%m%d-%H%M}.zip",
            "created_at": now.astimezone(UTC).isoformat(),
            "size": len(zip_bytes),
            **{key: value for key, value in meta.items() if key != "collection"},
        }
        (directory / f"{extract_id}.json").write_text(json.dumps(record), encoding="utf-8")
        return record

    def list(self, physical: str) -> list[ExtractRecord]:
        """Return a collection's stored builds, newest first.

        A sidecar whose archive is gone, or whose JSON will not parse, is
        skipped with a warning rather than failing the whole listing.

        Args:
            physical (str): Collection directory name.

        Returns:
            list[ExtractRecord]: The records.
        """
        try:
            directory = self._dir(physical)
        except ValueError:
            return []
        if not directory.is_dir():
            return []
        records: list[ExtractRecord] = []
        for sidecar in directory.glob("*.json"):
            if not (directory / f"{sidecar.stem}.zip").is_file():
                continue
            try:
                record = json.loads(sidecar.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                logger.warning("Skipping unreadable extract record {!r}: {}", sidecar.name, exc)
                continue
            if isinstance(record, dict):
                records.append(record)
        records.sort(key=lambda record: str(record.get("extract_id") or ""), reverse=True)
        return records

    def get(self, physical: str, extract_id: str) -> ExtractRecord | None:
        """Return one stored build, or ``None`` when it is unknown.

        Args:
            physical (str): Collection directory name.
            extract_id (str): The build's id.

        Returns:
            ExtractRecord | None: The record, ``None`` when absent or unsafe.
        """
        try:
            archive = self.path(physical, extract_id)
        except ValueError:
            return None
        if not archive.is_file():
            return None
        return next((record for record in self.list(physical) if record.get("extract_id") == extract_id), None)

    def delete(self, physical: str, extract_id: str) -> bool:
        """Remove one stored build.

        Args:
            physical (str): Collection directory name.
            extract_id (str): The build's id.

        Returns:
            bool: Whether an archive was actually removed.
        """
        try:
            archive = self.path(physical, extract_id)
        except ValueError:
            return False
        removed = archive.is_file()
        archive.unlink(missing_ok=True)
        archive.with_suffix(".json").unlink(missing_ok=True)
        return removed

    def prune(self, physical: str, *, retention_days: int, max_per_collection: int, now: datetime) -> int:
        """Drop builds past their retention or beyond the per-collection cap.

        Args:
            physical (str): Collection directory name.
            retention_days (int): Age after which a build is removed.
            max_per_collection (int): Newest builds to keep.
            now (datetime): Reference time.

        Returns:
            int: How many builds were removed.
        """
        records = self.list(physical)
        doomed = [record["extract_id"] for record in records[max_per_collection:]]
        cutoff = now.timestamp() - retention_days * 86_400
        for record in records[:max_per_collection]:
            created = str(record.get("created_at") or "")
            try:
                stamp = datetime.fromisoformat(created).timestamp()
            except ValueError:
                continue
            if stamp < cutoff:
                doomed.append(str(record["extract_id"]))
        return sum(1 for extract_id in doomed if self.delete(physical, extract_id))

    def delete_collection(self, physical: str) -> None:
        """Remove every build for a collection, as its deletion cascade.

        Args:
            physical (str): Collection directory name.
        """
        try:
            directory = self._dir(physical)
        except ValueError:
            return
        shutil.rmtree(directory, ignore_errors=True)
