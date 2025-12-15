from __future__ import annotations

import shutil
from pathlib import Path

from loguru import logger


def stage_sources_to_qdrant(
    data_dir: Path, collection: str, qdrant_sources: Path
) -> Path:
    """
    Ensure source files reside under the shared sources root (separate from Qdrant's internal collections).

    Files/directories from ``data_dir`` are copied into ``qdrant_sources/<collection>``.
    If ``data_dir`` already points to that target, no copy occurs.

    Args:
        data_dir (Path): Path to the original data directory.
        collection (str): Qdrant collection name.
        qdrant_sources (Path): Root path dedicated to source files (outside Qdrant collections).

    Returns:
        Path: The path to the staged sources directory.
    """
    target = qdrant_sources / collection
    source = data_dir if data_dir.is_dir() else data_dir.parent

    try:
        if source.resolve() == target.resolve():
            return target
    except Exception:
        # If resolve fails (e.g., permissions), continue with copy logic.
        logger.debug(
            "Falling back to copy for sources staging: {} -> {}", source, target
        )

    target.mkdir(parents=True, exist_ok=True)

    for entry in source.iterdir():
        dest = target / entry.name
        if entry.is_file():
            shutil.copy2(entry, dest)
        elif entry.is_dir():
            shutil.copytree(entry, dest, dirs_exist_ok=True)

    return target
