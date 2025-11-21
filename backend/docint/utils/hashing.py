from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from loguru import logger

_DEFAULT_CHUNK_SIZE = 1024 * 1024  # 1 MiB


def compute_file_hash(
    path: str | Path,
    algorithm: str = "sha256",
    *,
    chunk_size: int = _DEFAULT_CHUNK_SIZE,
) -> str:
    """
    Compute the hexadecimal digest for a file.

    Args:
        path: File system path to hash.
        algorithm: Hash algorithm supported by :mod:`hashlib`.
        chunk_size: Size of the read buffer when streaming the file.

    Returns:
        Hexadecimal string representing the digest.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the specified hash algorithm is unsupported.
    """

    file_path = Path(path) if not isinstance(path, Path) else path
    if not file_path.is_file():
        logger.error(f"FileNotFoundError: File not found for hashing: {file_path}")
        raise FileNotFoundError(f"File not found for hashing: {file_path}")

    try:
        hasher = hashlib.new(algorithm)
    except ValueError as exc:  # pragma: no cover - defensive guard
        logger.error(f"ValueError: Unsupported hash algorithm: {algorithm}")
        raise ValueError(f"Unsupported hash algorithm: {algorithm}") from exc

    with file_path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def ensure_file_hash(
    metadata: dict[str, Any] | None,
    *,
    path: str | Path | None = None,
    file_hash: str | None = None,
    algorithm: str = "sha256",
) -> str:
    """
    Attach a ``file_hash`` field to metadata.

    Args:
        metadata: Metadata dictionary to mutate in place.
        path: Optional file path to hash when ``file_hash`` is not supplied.
        file_hash: Optional pre-computed hash digest to reuse.
        algorithm: Hash algorithm to use when computing from ``path``.

    Returns:
        The hex digest stored in the metadata.

    Raises:
        ValueError: If neither ``file_hash`` nor ``path`` is provided.
    """
    if metadata is None:
        metadata = {}

    if file_hash is None:
        if path is None:
            logger.error("ValueError: path is required when file_hash is not provided")
            raise ValueError("path is required when file_hash is not provided")
        file_hash = compute_file_hash(path, algorithm=algorithm)

    metadata["file_hash"] = file_hash

    return file_hash
