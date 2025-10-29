import mimetypes
from pathlib import Path

import magic
from loguru import logger

from docint.utils.logging_cfg import setup_logging

setup_logging()

_MAGIC = magic.Magic(mime=True)


def _guess_from_extension(path: Path) -> str | None:
    """
    Best-effort MIME guess based on the file extension.

    Args:
        path (Path): The path to the file.

    Returns:
        str | None: The guessed MIME type or None if it cannot be guessed.
    """

    guessed, _ = mimetypes.guess_type(str(path))
    return guessed


def get_mimetype(file_path: str | Path) -> str:
    """
    Get the MIME type of a file.

    Args:
        file_path (str | Path): The path to the file.

    Returns:
        str: The MIME type of the file.
    """

    path = Path(file_path)

    guessed = _guess_from_extension(path)
    if guessed:
        return guessed

    try:
        return _MAGIC.from_file(str(path))
    except Exception:
        # Fall back to a generic binary MIME type if libmagic cannot inspect the file.
        logger.warning(
            "Exception: Failed to determine MIME type for {}. Falling back to application/octet-stream",
            file_path,
        )
        return "application/octet-stream"
