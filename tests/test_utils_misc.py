import hashlib
import json
from pathlib import Path

import pytest

from docint.utils.clean_text import basic_clean
from docint.utils.hashing import compute_file_hash, ensure_file_hash
from docint.utils.logging_cfg import _resolve_log_path, setup_logging


def test_basic_clean_normalizes_whitespace() -> None:
    """
    Test that basic_clean normalizes whitespace and newlines.
    """
    text = "Line 1\r\n\r\nLine 2\n\n\nLine 3   \n"
    cleaned = basic_clean(text)
    assert cleaned == "Line 1\nLine 2\nLine 3"


def test_compute_file_hash(tmp_path: Path) -> None:
    """
    Test that compute_file_hash correctly calculates the SHA256 hash of a file.

    Args:
        tmp_path (Path): The temporary path fixture.
    """
    file = tmp_path / "sample.txt"
    file.write_text("content")
    expected = hashlib.sha256(b"content").hexdigest()
    assert compute_file_hash(file) == expected


def test_compute_file_hash_missing(tmp_path: Path) -> None:
    """
    Test that compute_file_hash raises FileNotFoundError for missing files.

    Args:
        tmp_path (Path): The temporary path fixture.
    """
    with pytest.raises(FileNotFoundError):
        compute_file_hash(tmp_path / "missing.txt")


def test_ensure_file_hash_mutates_metadata(tmp_path: Path) -> None:
    """
    Test that ensure_file_hash adds the file hash to the metadata dictionary.

    Args:
        tmp_path (Path): The temporary path fixture.
    """
    file = tmp_path / "data.json"
    file.write_text(json.dumps({"key": "value"}))
    metadata: dict[str, object] = {}
    digest = ensure_file_hash(metadata, path=file)
    assert metadata["file_hash"] == digest


def test_ensure_file_hash_requires_inputs() -> None:
    """
    Test that ensure_file_hash raises ValueError if neither path nor file_hash is provided.
    """
    with pytest.raises(ValueError):
        ensure_file_hash({}, path=None, file_hash=None)


def test_resolve_log_path_and_setup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Test that _resolve_log_path resolves the log path and setup_logging creates the log file.

    Args:
        tmp_path (Path): The temporary path fixture.
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
    """
    target = tmp_path / "logs"
    path = _resolve_log_path(target)
    assert path.name == "docint.log"
    logfile = setup_logging(target, level="INFO", rotation="1 MB", retention=1)
    assert logfile.exists()
