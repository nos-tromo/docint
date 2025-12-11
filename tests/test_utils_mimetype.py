from pathlib import Path

import pytest

from docint.utils import mimetype


def test_guess_from_extension(tmp_path: Path) -> None:
    """
    Test that _guess_from_extension correctly identifies the mimetype from the file extension.

    Args:
        tmp_path (Path): The temporary path fixture.
    """
    file = tmp_path / "sample.txt"
    file.write_text("content")
    assert mimetype._guess_from_extension(file) == "text/plain"


def test_get_mimetype_falls_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Test that get_mimetype falls back to a default when magic detection fails.

    Args:
        tmp_path (Path): The temporary path fixture.
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
    """
    file = tmp_path / "binary.bin"
    file.write_bytes(b"\x00\x01")

    class DummyMagic:
        def from_file(self, _: str) -> str:
            raise RuntimeError("fail")

    monkeypatch.setattr(mimetype, "_MAGIC", DummyMagic())
    assert mimetype.get_mimetype(file) == "application/octet-stream"
