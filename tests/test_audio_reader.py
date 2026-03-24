"""Tests for the audio reader (Whisper-based transcription)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from llama_index.core import Document

import docint.core.readers.audio as audio_module
import docint.utils.env_cfg as env_cfg
from docint.core.readers.audio import AudioReader
from docint.utils.env_cfg import load_whisper_env


def _write_whisper_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    profile_body: str = "",
) -> None:
    """Write a temporary TOML config for whisper-related tests."""

    config_path = tmp_path / "config.toml"
    config_path.write_text(
        (
            'active_profile = "test"\n\n'
            "[profiles.test.shared]\n\n"
            "[profiles.test.backend]\n\n"
            "[profiles.test.frontend]\n\n"
            "[profiles.test.worker]\n"
        )
        + profile_body,
        encoding="utf-8",
    )
    monkeypatch.setattr(env_cfg, "DEFAULT_CONFIG_PATH", config_path)
    monkeypatch.setattr(env_cfg, "_ACTIVE_PROFILE", None)
    monkeypatch.setattr(env_cfg, "_ACTIVE_ROLE", None)
    monkeypatch.delenv("DOCINT_PROFILE", raising=False)


def _make_audio_reader(
    monkeypatch: pytest.MonkeyPatch,
    segments: list[dict[str, Any]] | None,
    text: str = "",
) -> AudioReader:
    """Create a stubbed AudioReader for testing.

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
    monkeypatch.setattr(reader, "_detect_language", lambda _audio, _model: None)

    def fake_transcribe(_audio, _model, **kwargs) -> dict[str, Any]:
        """Fake transcription method.

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
    """Test that AudioReader builds a segmented transcript correctly.

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
    """Test that AudioReader falls back to full text when no segments are provided.

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


def test_load_whisper_env_invalid_task_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Invalid Whisper task values should fail clearly.

    Args:
        monkeypatch: The pytest monkeypatch fixture.
    """

    _write_whisper_config(
        tmp_path,
        monkeypatch,
        profile_body=(
            '\n[profiles.test.shared.whisper]\nmax_workers = 0\ntask = "summarize"\n'
        ),
    )

    with pytest.raises(ValueError, match="Unsupported whisper task"):
        load_whisper_env()


def test_audio_reader_translate_non_english_uses_translate_task(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Translate mode should use Whisper translation for non-English audio.

    Args:
        monkeypatch: The pytest monkeypatch fixture.
        tmp_path: Temporary directory provided by pytest.
    """

    _write_whisper_config(
        tmp_path,
        monkeypatch,
        profile_body=('\n[profiles.test.shared.whisper]\ntask = "translate"\n'),
    )
    reader = AudioReader(device="cpu")
    monkeypatch.setattr(reader, "_load_model", lambda: None)
    monkeypatch.setattr(reader, "_load_audio", lambda _: None)
    monkeypatch.setattr(reader, "_detect_language", lambda _audio, _model: "es")

    captured: dict[str, Any] = {}

    def fake_transcribe(_audio, _model, **kwargs) -> dict[str, Any]:
        captured.update(kwargs)
        return {"segments": None, "text": "Translated text"}

    monkeypatch.setattr(reader, "_transcribe_audio", fake_transcribe)
    audio_path = tmp_path / "spanish.wav"
    audio_path.write_bytes(b"fake")

    docs = reader.load_data(audio_path)

    assert docs[0].text == "Translated text"
    assert captured["task"] == "translate"
    assert captured["language"] == "es"
    assert docs[0].metadata["whisper_task"] == "translate"
    assert docs[0].metadata["whisper_language"] == "es"


def test_audio_reader_translate_english_forces_transcribe(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Translate mode should still transcribe English audio.

    Args:
        monkeypatch: The pytest monkeypatch fixture.
        tmp_path: Temporary directory provided by pytest.
    """

    _write_whisper_config(
        tmp_path,
        monkeypatch,
        profile_body=('\n[profiles.test.shared.whisper]\ntask = "translate"\n'),
    )
    reader = AudioReader(device="cpu")
    monkeypatch.setattr(reader, "_load_model", lambda: None)
    monkeypatch.setattr(reader, "_load_audio", lambda _: None)
    monkeypatch.setattr(reader, "_detect_language", lambda _audio, _model: "en")

    captured: dict[str, Any] = {}

    def fake_transcribe(_audio, _model, **kwargs) -> dict[str, Any]:
        captured.update(kwargs)
        return {"segments": None, "text": "English transcript"}

    monkeypatch.setattr(reader, "_transcribe_audio", fake_transcribe)
    audio_path = tmp_path / "english.wav"
    audio_path.write_bytes(b"fake")

    docs = reader.load_data(audio_path)

    assert docs[0].text == "English transcript"
    assert captured["task"] == "transcribe"
    assert captured["language"] == "en"
    assert docs[0].metadata["whisper_task"] == "transcribe"
    assert docs[0].metadata["whisper_language"] == "en"


def test_audio_reader_load_batch_data_parallel_preserves_order(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Parallel batch loading should preserve input ordering.

    Args:
        monkeypatch: The pytest monkeypatch fixture.
        tmp_path: Temporary directory provided by pytest.
    """

    _write_whisper_config(
        tmp_path,
        monkeypatch,
        profile_body=("\n[profiles.test.shared.whisper]\nmax_workers = 2\n"),
    )
    reader = AudioReader(device="cpu")
    first = tmp_path / "a.wav"
    second = tmp_path / "b.wav"
    third = tmp_path / "c.wav"
    first.write_bytes(b"a")
    second.write_bytes(b"b")
    third.write_bytes(b"c")

    class FakeFuture:
        def __init__(self, result: dict[str, Any]) -> None:
            self._result = result

        def result(self) -> dict[str, Any]:
            return self._result

    class FakeExecutor:
        def __init__(
            self,
            max_workers: int,
            mp_context: Any,
            initializer,
            initargs: tuple[Any, ...],
        ) -> None:
            self._initializer = initializer
            self._initargs = initargs

        def __enter__(self) -> FakeExecutor:
            self._initializer(*self._initargs)
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

        def submit(self, fn, job):
            return FakeFuture(fn(job))

    monkeypatch.setattr(audio_module, "ProcessPoolExecutor", FakeExecutor)
    monkeypatch.setattr(
        audio_module, "as_completed", lambda futures: list(futures)[::-1]
    )

    info_logs: list[str] = []
    warning_logs: list[str] = []

    class FakeLogger:
        def info(self, message: str, *args: Any) -> None:
            info_logs.append(message.format(*args))

        def warning(self, message: str, *args: Any) -> None:
            warning_logs.append(message.format(*args))

    monkeypatch.setattr(audio_module, "logger", FakeLogger())

    batches = reader.load_batch_data([first, second, third])

    assert [docs[0].text for docs in batches] == [
        "transcribed",
        "transcribed",
        "transcribed",
    ]
    assert [docs[0].metadata["file_name"] for docs in batches] == [
        "a.wav",
        "b.wav",
        "c.wav",
    ]
    assert any(
        "Starting batch transcription for 3 file(s) with 2 worker(s)." in msg
        for msg in info_logs
    )
    start_logs = [msg for msg in info_logs if "(parallel)" in msg]
    assert str(first) in start_logs[0]
    assert str(second) in start_logs[1]
    assert str(third) in start_logs[2]
    finished_logs = [msg for msg in info_logs if "[AudioReader] Finished " in msg]
    assert str(second) in finished_logs[0]
    assert str(third) in finished_logs[1]
    assert str(first) in finished_logs[2]
    assert info_logs.index(start_logs[2]) > info_logs.index(finished_logs[0])
    assert any(
        "Finished batch transcription: 3/3 file(s) succeeded." in msg
        for msg in info_logs
    )
    assert warning_logs == []


def test_transcribe_audio_job_reports_file_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Worker jobs should return an error payload instead of raising.

    Args:
        monkeypatch: The pytest monkeypatch fixture.
    """

    monkeypatch.setattr(
        audio_module.whisper,
        "load_audio",
        lambda file: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    audio_module._init_whisper_worker("turbo", "cpu", "transcribe")

    payload = audio_module._transcribe_audio_job(("bad.wav", "hash"))

    assert payload["file_path"] == "bad.wav"
    assert payload["file_hash"] == "hash"
    assert payload["result"] is None
    assert payload["error"] == "boom"
