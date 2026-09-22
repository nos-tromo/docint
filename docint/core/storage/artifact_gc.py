"""Finding and removing a deleted collection's PDF pipeline artifacts.

``PIPELINE_ARTIFACTS_DIR`` holds one directory per PDF, named by the file's
content hash: page text, chunks, tables and figures the page pipeline wrote.
It is a cache — rebuilt on demand — and shared between collections, since two
collections holding the same PDF share its hash. A directory is therefore
removed only when nothing else still refers to it. Removing too much costs a
re-read of that PDF; removing too little leaves a document's text on disk.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Iterable
from pathlib import Path

from loguru import logger


def recorded_sources(root: Path) -> dict[str, str]:
    """Map each artifact directory to the source file its manifest records.

    Args:
        root (Path): The artifacts root.

    Returns:
        dict[str, str]: Directory name (the file hash) to recorded source path;
            directories without a readable manifest are left out.
    """
    if not root.is_dir():
        return {}
    sources: dict[str, str] = {}
    for entry in root.iterdir():
        try:
            data = json.loads((entry / "manifest.json").read_text())
        except (OSError, ValueError):
            continue
        source = data.get("file_path") if isinstance(data, dict) else None
        if isinstance(source, str) and source:
            sources[entry.name] = source
    return sources


def _inside(path: str, directory: Path) -> bool:
    """Whether ``path`` names something inside ``directory``."""
    return Path(path).is_relative_to(directory)


def collection_artifacts(source_dir: Path, manifest_hashes: set[str], recorded: dict[str, str]) -> set[str]:
    """Return the artifact directories a collection accounts for.

    Its manifest's rows, whatever their status — the pipeline writes artifacts
    before a run records success — plus directories whose manifest names a file
    inside the collection's source directory, which is how an upload that was
    preprocessed but never ingested is found.

    Args:
        source_dir (Path): The collection's source directory.
        manifest_hashes (set[str]): Hashes its ingest manifest records.
        recorded (dict[str, str]): From :func:`recorded_sources`.

    Returns:
        set[str]: Artifact directory names.
    """
    return set(manifest_hashes) | {name for name, source in recorded.items() if _inside(source, source_dir)}


def still_referenced(
    *,
    recorded: dict[str, str],
    other_source_dirs: Iterable[Path],
    other_manifest_hashes: set[str],
    inflight: set[str],
) -> set[str]:
    """Return the artifact directories something else still needs.

    Args:
        recorded (dict[str, str]): From :func:`recorded_sources`.
        other_source_dirs (Iterable[Path]): Every other collection's source directory.
        other_manifest_hashes (set[str]): Hashes every other collection's manifest records.
        inflight (set[str]): Hashes a preprocessing worker is reading.

    Returns:
        set[str]: Artifact directory names to keep.
    """
    others = list(other_source_dirs)
    used_by_files = {name for name, source in recorded.items() if any(_inside(source, d) for d in others)}
    return set(other_manifest_hashes) | set(inflight) | used_by_files


def remove_artifacts(root: Path, names: Iterable[str]) -> int:
    """Remove artifact directories by name; never anything outside ``root``.

    Args:
        root (Path): The artifacts root.
        names (Iterable[str]): Directory names (file hashes).

    Returns:
        int: How many directories are gone.
    """
    removed = 0
    for name in names:
        if not name or name in {".", ".."} or Path(name).name != name:
            logger.warning("Refusing to remove artifact directory named '{}'.", name)
            continue
        directory = root / name
        if not directory.is_dir():
            continue
        shutil.rmtree(directory, ignore_errors=True)
        if directory.exists():
            logger.warning("Could not remove artifact directory '{}'.", name)
        else:
            removed += 1
    return removed
