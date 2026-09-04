"""Tests that a source preview resolves an image or keyframe through the companion.

An image has no point in the main collection: its hash names its own content
(``image_id``) or the clip it was cut from (``media_file_hash``), and only the
``_images`` companion knows the path either was read from. Scrolling the main
collection alone 404'd the preview of every picture an answer cited.
"""

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import docint.core.api as api_module


def test_a_companion_hash_resolves_to_the_file_it_was_read_from(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The companion is consulted once the main collection turns up nothing."""
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"\x00")
    scrolled: list[str] = []

    def scroll(collection_name: str, **_kw: Any) -> tuple[list[Any], None]:
        """Answer only for the companion, recording what was asked."""
        scrolled.append(collection_name)
        if collection_name.endswith("_images"):
            return [
                SimpleNamespace(payload={"image_id": "frame-1", "media_file_hash": "clip-h", "source_path": str(clip)})
            ], None
        return [], None

    monkeypatch.setattr(api_module, "rag", SimpleNamespace(qdrant_client=SimpleNamespace(scroll=scroll)))
    monkeypatch.setattr(api_module, "_resolve_data_dir", lambda: tmp_path / "none")
    monkeypatch.setattr(api_module, "_resolve_qdrant_src_dir", lambda: tmp_path / "none")

    assert api_module._resolve_source_file_path("uabc__docs", "clip-h") == clip
    assert scrolled == ["uabc__docs", "uabc__docs", "uabc__docs_images"]
