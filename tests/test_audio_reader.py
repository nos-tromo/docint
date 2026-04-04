"""Tests for the audio reader (Whisper-based transcription)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import whisper
from llama_index.core import Document
from numpy import floating
from numpy.typing import NDArray

import docint.core.readers.audio as audio_module
from docint.core.readers.audio import AudioReader
from docint.utils.env_cfg import load_whisper_env


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


def _install_vllm_audio_client(
    monkeypatch: pytest.MonkeyPatch,
    *,
    transcription_response: dict[str, Any],
    translation_response: dict[str, Any] | None = None,
    translation_error: Exception | None = None,
) -> list[tuple[str, dict[str, Any]]]:
    """Install a fake OpenAI-compatible audio client for vLLM tests.

    Args:
        monkeypatch: The pytest monkeypatch fixture.
        transcription_response: Response returned by transcriptions.
        translation_response: Optional response returned by translations.
        translation_error: Optional exception raised by translations.

    Returns:
        Captured API call names and keyword arguments.
    """

    calls: list[tuple[str, dict[str, Any]]] = []

    class FakeTranscriptions:
        """Fake transcriptions endpoint."""

        def create(self, **kwargs: Any) -> dict[str, Any]:
            """Create a transcription.

            Returns:
                dict[str, Any]: The fake transcription result.
            """
            calls.append(("transcriptions", kwargs))
            return transcription_response

    class FakeTranslations:
        def create(self, **kwargs: Any) -> dict[str, Any]:
            """Create a translation.

            Returns:
                dict[str, Any]: The fake translation result.
            """
            calls.append(("translations", kwargs))
            if translation_error is not None:
                raise translation_error
            if translation_response is None:
                raise AssertionError("translation_response must be provided")
            return translation_response

    class FakeClient:
        """Fake OpenAI-compatible client with audio endpoints."""

        def __init__(self, **kwargs: Any) -> None:
            """Initialize the fake client.

            Args:
                **kwargs: Arbitrary keyword arguments.
            """
            self.kwargs = kwargs
            self.audio = type(
                "FakeAudioNamespace",
                (),
                {
                    "transcriptions": FakeTranscriptions(),
                    "translations": FakeTranslations(),
                },
            )()

    monkeypatch.setattr(audio_module, "_OpenAIClient", FakeClient)
    return calls


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


def test_load_whisper_env_invalid_values_fall_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Invalid Whisper env values should fall back to safe defaults.

    Args:
        monkeypatch: The pytest monkeypatch fixture.
    """

    monkeypatch.setenv("WHISPER_MAX_WORKERS", "0")
    monkeypatch.setenv("WHISPER_TASK", "summarize")

    cfg = load_whisper_env()

    assert cfg.max_workers == 1
    assert cfg.task == "transcribe"


def test_audio_reader_translate_non_english_uses_translate_task(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Translate mode should use Whisper translation for non-English audio.

    Args:
        monkeypatch: The pytest monkeypatch fixture.
        tmp_path: Temporary directory provided by pytest.
    """

    monkeypatch.setenv("WHISPER_TASK", "translate")
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

    monkeypatch.setenv("WHISPER_TASK", "translate")
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


def test_audio_reader_vllm_transcribe_uses_provider_backend(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """vLLM provider should use the OpenAI-compatible transcriptions endpoint.

    Args:
        monkeypatch: The pytest monkeypatch fixture.
        tmp_path: Temporary directory provided by pytest.
    """

    monkeypatch.setenv("INFERENCE_PROVIDER", "vllm")
    calls = _install_vllm_audio_client(
        monkeypatch,
        transcription_response={
            "language": "en",
            "segments": [
                {"start": 1.0, "end": 2.5, "text": "Hello"},
                {"start": 2.5, "end": 4.0, "text": "provider."},
            ],
            "text": "Hello provider.",
        },
    )
    audio_path = tmp_path / "provider.wav"
    audio_path.write_bytes(b"fake")

    docs = AudioReader(device="cpu").load_data(audio_path)

    assert len(docs) == 1
    assert docs[0].text == "Hello provider."
    assert docs[0].metadata["start_ts"] == "00:00:01"
    assert docs[0].metadata["end_ts"] == "00:00:04"
    assert docs[0].metadata["whisper_task"] == "transcribe"
    assert docs[0].metadata["whisper_language"] == "en"
    assert [name for name, _ in calls] == ["transcriptions"]
    assert calls[0][1]["response_format"] == "verbose_json"
    assert calls[0][1]["timestamp_granularities"] == ["segment"]


def test_audio_reader_openai_transcribe_uses_provider_backend(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """OpenAI provider should use the transcriptions endpoint.

    Args:
        monkeypatch: The pytest monkeypatch fixture.
        tmp_path: Temporary directory provided by pytest.
    """

    monkeypatch.setenv("INFERENCE_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_BASE", "https://api.openai.com/v1")
    calls = _install_vllm_audio_client(
        monkeypatch,
        transcription_response={
            "language": "en",
            "segments": [
                {"start": 0.0, "end": 1.4, "text": "OpenAI"},
                {"start": 1.4, "end": 2.8, "text": "transcript."},
            ],
            "text": "OpenAI transcript.",
        },
    )
    audio_path = tmp_path / "openai.wav"
    audio_path.write_bytes(b"fake")

    docs = AudioReader(device="cpu").load_data(audio_path)

    assert len(docs) == 1
    assert docs[0].text == "OpenAI transcript."
    assert docs[0].metadata["whisper_task"] == "transcribe"
    assert docs[0].metadata["whisper_language"] == "en"
    assert [name for name, _ in calls] == ["transcriptions"]
    assert calls[0][1]["response_format"] == "verbose_json"
    assert calls[0][1]["timestamp_granularities"] == ["segment"]


def test_audio_reader_vllm_translate_non_english_uses_translation_endpoint(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """vLLM provider should translate non-English audio when configured.

    Args:
        monkeypatch: The pytest monkeypatch fixture.
        tmp_path: Temporary directory provided by pytest.
    """

    monkeypatch.setenv("INFERENCE_PROVIDER", "vllm")
    monkeypatch.setenv("WHISPER_TASK", "translate")
    monkeypatch.setenv("WHISPER_MODEL", "openai/whisper-large-v3")
    calls = _install_vllm_audio_client(
        monkeypatch,
        transcription_response={
            "language": "es",
            "segments": [{"start": 0.0, "end": 1.0, "text": "Hola"}],
            "text": "Hola",
        },
        translation_response={
            "segments": [
                {"start": 0.0, "end": 1.2, "text": "Hello"},
                {"start": 1.2, "end": 2.0, "text": "world."},
            ],
            "text": "Hello world.",
        },
    )
    audio_path = tmp_path / "spanish-provider.wav"
    audio_path.write_bytes(b"fake")

    docs = AudioReader(device="cpu").load_data(audio_path)

    assert docs[0].text == "Hello world."
    assert docs[0].metadata["whisper_task"] == "translate"
    assert docs[0].metadata["whisper_language"] == "es"
    assert docs[0].metadata["start_ts"] == "00:00:00"
    assert docs[0].metadata["end_ts"] == "00:00:02"
    assert [name for name, _ in calls] == ["transcriptions", "translations"]


def test_audio_reader_openai_translate_non_english_uses_translation_endpoint(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """OpenAI provider should use the translations endpoint when needed.

    Args:
        monkeypatch: The pytest monkeypatch fixture.
        tmp_path: Temporary directory provided by pytest.
    """

    monkeypatch.setenv("INFERENCE_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_BASE", "https://api.openai.com/v1")
    monkeypatch.setenv("WHISPER_TASK", "translate")
    monkeypatch.setenv("WHISPER_MODEL", "whisper-1")
    calls = _install_vllm_audio_client(
        monkeypatch,
        transcription_response={
            "language": "de",
            "segments": [{"start": 0.0, "end": 1.0, "text": "Hallo"}],
            "text": "Hallo",
        },
        translation_response={
            "segments": [
                {"start": 0.0, "end": 1.0, "text": "Hello"},
                {"start": 1.0, "end": 2.0, "text": "there."},
            ],
            "text": "Hello there.",
        },
    )
    audio_path = tmp_path / "openai-translate.wav"
    audio_path.write_bytes(b"fake")

    docs = AudioReader(device="cpu").load_data(audio_path)

    assert len(docs) == 1
    assert docs[0].text == "Hello there."
    assert docs[0].metadata["whisper_task"] == "translate"
    assert docs[0].metadata["whisper_language"] == "de"
    assert [name for name, _ in calls] == ["transcriptions", "translations"]


def test_audio_reader_vllm_translate_unsupported_model_falls_back(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Unsupported provider translation models should fall back to transcription.

    Args:
        monkeypatch: The pytest monkeypatch fixture.
        tmp_path: Temporary directory provided by pytest.
    """

    monkeypatch.setenv("INFERENCE_PROVIDER", "vllm")
    monkeypatch.setenv("WHISPER_TASK", "translate")
    monkeypatch.setenv("WHISPER_MODEL", "turbo")
    calls = _install_vllm_audio_client(
        monkeypatch,
        transcription_response={
            "language": "fr",
            "segments": None,
            "text": "Bonjour",
        },
    )
    audio_path = tmp_path / "fallback.wav"
    audio_path.write_bytes(b"fake")

    docs = AudioReader(device="cpu").load_data(audio_path)

    assert docs[0].text == "Bonjour"
    assert docs[0].metadata["whisper_task"] == "transcribe"
    assert docs[0].metadata["whisper_language"] == "fr"
    assert [name for name, _ in calls] == ["transcriptions"]


def test_audio_reader_load_batch_data_parallel_preserves_order(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Parallel batch loading should preserve input ordering.

    Args:
        monkeypatch: The pytest monkeypatch fixture.
        tmp_path: Temporary directory provided by pytest.
    """

    monkeypatch.setenv("WHISPER_MAX_WORKERS", "2")
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


# ---------------------------------------------------------------------------
# _wav_for_provider conversion tests
# ---------------------------------------------------------------------------


def test_wav_for_provider_passthrough_for_native_suffix(tmp_path: Path) -> None:
    """Files with libsndfile-native suffixes should be yielded as-is.

    Args:
        tmp_path: Temporary directory provided by pytest.
    """
    wav = tmp_path / "clip.wav"
    wav.write_bytes(b"\x00")

    backend = audio_module.OpenAICompatibleAudioBackend
    with backend._wav_for_provider(wav) as result:
        assert result == wav


def test_wav_for_provider_converts_webm(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """WebM files should be converted to WAV via ffmpeg before yielding.

    Args:
        monkeypatch: The pytest monkeypatch fixture.
        tmp_path: Temporary directory provided by pytest.
    """
    webm = tmp_path / "clip.webm"
    webm.write_bytes(b"\x00")
    converted_path: Path | None = None

    def fake_run(cmd, **kwargs):  # noqa: ANN001, ANN003
        """Fake subprocess.run that writes a dummy WAV file."""
        # cmd[-1] is the output path
        Path(cmd[-1]).write_bytes(b"RIFF")

    monkeypatch.setattr(audio_module.subprocess, "run", fake_run)

    backend = audio_module.OpenAICompatibleAudioBackend
    with backend._wav_for_provider(webm) as result:
        converted_path = result
        assert result.suffix == ".wav"
        assert result != webm
        assert result.exists()

    # Temp file should be cleaned up after exiting the context
    assert converted_path is not None
    assert not converted_path.exists()


def test_wav_for_provider_cleans_up_on_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Temporary WAV file should be removed even when an error occurs.

    Args:
        monkeypatch: The pytest monkeypatch fixture.
        tmp_path: Temporary directory provided by pytest.
    """
    mp3 = tmp_path / "clip.mp3"
    mp3.write_bytes(b"\x00")
    converted_path: Path | None = None

    def fake_run(cmd, **kwargs):  # noqa: ANN001, ANN003
        Path(cmd[-1]).write_bytes(b"RIFF")

    monkeypatch.setattr(audio_module.subprocess, "run", fake_run)

    backend = audio_module.OpenAICompatibleAudioBackend
    with pytest.raises(RuntimeError, match="deliberate"):
        with backend._wav_for_provider(mp3) as result:
            converted_path = result
            raise RuntimeError("deliberate")

    assert converted_path is not None
    assert not converted_path.exists()


def test_vllm_transcribe_converts_webm_before_sending(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The provider transcription should convert WebM files before the API call.

    Args:
        monkeypatch: The pytest monkeypatch fixture.
        tmp_path: Temporary directory provided by pytest.
    """
    monkeypatch.setenv("INFERENCE_PROVIDER", "vllm")

    def fake_run(cmd, **kwargs):  # noqa: ANN001, ANN003
        Path(cmd[-1]).write_bytes(b"RIFF")

    monkeypatch.setattr(audio_module.subprocess, "run", fake_run)

    calls = _install_vllm_audio_client(
        monkeypatch,
        transcription_response={
            "language": "en",
            "segments": [{"start": 0.0, "end": 1.0, "text": "Test."}],
            "text": "Test.",
        },
    )

    # Patch the create method to capture the file name
    original_calls = calls

    audio_path = tmp_path / "talk.webm"
    audio_path.write_bytes(b"fake-webm")

    docs = AudioReader(device="cpu").load_data(audio_path)

    assert len(docs) == 1
    assert docs[0].text == "Test."
    # Verify the file sent to the API had a .wav suffix
    assert len(original_calls) == 1
    sent_file = original_calls[0][1]["file"]
    assert sent_file.name.endswith(".wav")


# ---------------------------------------------------------------------------
# WHISPER_SRC_LANGUAGE override tests
# ---------------------------------------------------------------------------


def test_src_language_skips_detection_and_passes_language(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """When WHISPER_SRC_LANGUAGE is set, detection is skipped and the language is forwarded.

    Args:
        monkeypatch: The pytest monkeypatch fixture.
        tmp_path: Temporary directory provided by pytest.
    """
    monkeypatch.setenv("WHISPER_SRC_LANGUAGE", "fr")
    reader = AudioReader(device="cpu")
    monkeypatch.setattr(reader, "_load_model", lambda: None)
    monkeypatch.setattr(reader, "_load_audio", lambda _: None)

    detect_called = False

    def _detect_should_not_be_called(
        _audio: NDArray[floating[Any]], _model: whisper.Whisper
    ) -> str | None:  # noqa: ANN001, ANN202
        """This function should not be called when WHISPER_SRC_LANGUAGE is set.

        Args:
            _audio (NDArray[floating[Any]]): The audio data.
            _model (whisper.Whisper): The Whisper model.

        Returns:
            str | None: Always returns None.
        """
        nonlocal detect_called
        detect_called = True
        return None

    monkeypatch.setattr(reader, "_detect_language", _detect_should_not_be_called)

    captured: dict[str, Any] = {}

    def fake_transcribe(
        _audio: NDArray[floating[Any]], _model: whisper.Whisper, **kwargs
    ) -> dict[str, Any]:  # noqa: ANN003
        """Fake transcribe method that captures the language argument passed to it.

        Args:
            _audio (NDArray[floating[Any]]): The audio data.
            _model (whisper.Whisper): The Whisper model.

        Returns:
            dict[str, Any]: The transcription result.
        """
        captured.update(kwargs)
        return {"segments": None, "text": "Bonjour."}

    monkeypatch.setattr(reader, "_transcribe_audio", fake_transcribe)

    audio_path = tmp_path / "french.wav"
    audio_path.write_bytes(b"fake")
    docs = reader.load_data(audio_path)

    assert not detect_called
    assert captured["language"] == "fr"
    assert docs[0].text == "Bonjour."


def test_src_language_worker_init_sets_global(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_init_whisper_worker should propagate src_language to the module global.

    Args:
        monkeypatch: The pytest monkeypatch fixture.
    """
    audio_module._init_whisper_worker("turbo", "cpu", "transcribe", src_language="de")
    assert audio_module._WORKER_SRC_LANGUAGE == "de"

    # Reset
    audio_module._init_whisper_worker("turbo", "cpu", "transcribe")
    assert audio_module._WORKER_SRC_LANGUAGE is None
