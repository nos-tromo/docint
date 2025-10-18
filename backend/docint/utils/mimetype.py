from pathlib import Path
import mimetypes

import magic


_MAGIC = magic.Magic(mime=True)


def _guess_from_extension(path: Path) -> str | None:
    """Best-effort MIME guess based on the file extension."""

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
        return "application/octet-stream"
