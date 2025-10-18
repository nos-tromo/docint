from pathlib import Path

import magic


def get_mimetype(file_path: str | Path) -> str:
    """
    Get the MIME type of a file.

    Args:
        file_path (str | Path): The path to the file.

    Returns:
        str: The MIME type of the file.
    """
    mime = magic.Magic(mime=True)
    return mime.from_file(file_path)
