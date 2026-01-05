from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from llama_index.core import Document

from docint.core.readers.audio import AudioReader


def _make_audio_reader(
    monkeypatch: pytest.MonkeyPatch,
    segments: list[dict[str, Any]] | None,
    text: str = "",
) -> AudioReader:
    """
    Create a stubbed AudioReader for testing.

    Args:
        monkeypatch (pytest.MonkeyPatch): The pytest monkeypatch fixture.
        segments (list[dict[str, Any]] | None): List of segments to return.
        text (str, optional): Text to return as the transcription. Defaults to "".

    Returns:
        AudioReader: A stubbed AudioReader instance for testing.
    """
    reader = AudioReader(device="cpu")

    # Stub model loading and audio loading to avoid external dependencies
    monkeypatch.setattr(reader, "_load_model", lambda: None)
    monkeypatch.setattr(reader, "_load_audio", lambda _: None)

    def fake_transcribe(_audio, _model) -> dict[str, Any]:
        """
        Fake transcription method.

        Args:
            _audio (_type_): Ignored.
            _model (_type_): Ignored.

        Returns:
            dict[str, Any]: The fake transcription result.
        """
        return {"segments": segments, "text": text}

    monkeypatch.setattr(reader, "_transcribe_audio", fake_transcribe)
    return reader


def test_audio_reader_builds_segmented_transcript(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """
    Test that AudioReader builds a segmented transcript correctly.

    Args:
        monkeypatch (pytest.MonkeyPatch): The pytest monkeypatch fixture.
        tmp_path (Path): Temporary directory provided by pytest.
    """
    segments = [
        {"start": 1.2, "end": 3.8, "text": "Hello"},
        {"start": 4.0, "end": 6.0, "text": "world."},
    ]
    reader = _make_audio_reader(monkeypatch, segments)
    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"fake")

    docs = reader.load_data(audio_path)

    assert len(docs) == 1
    doc = docs[0]
    assert isinstance(doc, Document)
    assert doc.text == "Hello world."
    assert doc.metadata["start_ts"] == "00:00:01"
    assert doc.metadata["end_ts"] == "00:00:06"
    assert doc.metadata["sentence_index"] == 0
    assert doc.metadata["source"] == "audio"
    assert doc.metadata["file_hash"]


def test_audio_reader_falls_back_to_text(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """
    Test that AudioReader falls back to full text when no segments are provided.

    Args:
        monkeypatch (pytest.MonkeyPatch): The pytest monkeypatch fixture.
        tmp_path (Path): Temporary directory provided by pytest.
    """
    reader = _make_audio_reader(monkeypatch, segments=None, text="Just text")
    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"fake")

    docs = reader.load_data(audio_path)
    assert len(docs) == 1
    assert docs[0].text == "Just text"
