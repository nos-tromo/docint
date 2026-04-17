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
from docint.core.readers.audio import _audio_has_speech as _real_audio_has_speech
from docint.utils.env_cfg import load_whisper_env


@pytest.fixture(autouse=True)
def _bypass_vad_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    """Short-circuit the VAD/RMS pre-filter for every test in this module.

    ``AudioReader._transcribe_file_payload`` calls
    :func:`docint.core.readers.audio._audio_has_speech` before any Whisper
    inference runs. In production that guard loads the audio into a NumPy
    array and consults Silero VAD. The existing transcription tests stub
    ``_load_audio`` to return ``None`` (and the conftest-level whisper stub
    returns an opaque string) — neither of which is a valid input to the
    real guard, so without this fixture the guard would crash every test
    in the suite.

    The fixture therefore forces :func:`_audio_has_speech` to report
    ``(True, None)`` for every test. Tests that need to exercise the guard
    (the block below this fixture) re-monkeypatch
    ``audio_module._audio_has_speech`` explicitly, and the unit tests on
    the guard itself invoke the original implementation via the
    module-level :data:`_real_audio_has_speech` reference bound at import
    time (before this fixture runs).

    Args:
        monkeypatch (pytest.MonkeyPatch): Provided by pytest; the attribute
            replacement is undone automatically at test teardown.
    """

    monkeypatch.setattr(
        audio_module,
        "_audio_has_speech",
        lambda _audio: (True, None),
    )


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


# ---------------------------------------------------------------------------
# VAD guard + no_speech_prob filter
# ---------------------------------------------------------------------------


class _CapturingLogger:
    """In-memory stand-in for :mod:`loguru.logger` used by the VAD tests.

    Production code in ``docint.core.readers.audio`` formats messages with
    loguru's ``{}`` placeholder style (e.g. ``logger.warning("[x] {}",
    reason)``). Tests substitute this class for ``audio_module.logger`` so
    they can assert on the fully-formatted messages that would reach the
    log sink without needing a real loguru handler.

    ``info``/``debug`` writes and ``warning``/``error`` writes are kept in
    separate buckets so tests can assert that a particular message arrived
    at the correct severity.

    Attributes:
        info_logs (list[str]): Fully-formatted messages received by
            :meth:`info` and :meth:`debug`, in call order.
        warning_logs (list[str]): Fully-formatted messages received by
            :meth:`warning` and :meth:`error`, in call order.
    """

    def __init__(self) -> None:
        """Initialize empty capture buckets."""

        self.info_logs: list[str] = []
        self.warning_logs: list[str] = []

    def info(self, message: str, *args: Any) -> None:
        """Capture an ``info``-level log line.

        Args:
            message (str): Loguru-style template string (``{}`` placeholders).
            *args (Any): Positional arguments interpolated into *message*.
        """

        self.info_logs.append(message.format(*args))

    def warning(self, message: str, *args: Any) -> None:
        """Capture a ``warning``-level log line.

        Args:
            message (str): Loguru-style template string (``{}`` placeholders).
            *args (Any): Positional arguments interpolated into *message*.
        """

        self.warning_logs.append(message.format(*args))

    def error(self, message: str, *args: Any) -> None:
        """Capture an ``error``-level log line into :attr:`warning_logs`.

        Errors are grouped with warnings because the VAD-related call sites
        only emit ``warning`` in production; this method exists purely so
        that accidental ``logger.error`` calls do not raise ``AttributeError``
        during a test.

        Args:
            message (str): Loguru-style template string (``{}`` placeholders).
            *args (Any): Positional arguments interpolated into *message*.
        """

        self.warning_logs.append(message.format(*args))

    def debug(self, message: str, *args: Any) -> None:
        """Capture a ``debug``-level log line into :attr:`info_logs`.

        Args:
            message (str): Loguru-style template string (``{}`` placeholders).
            *args (Any): Positional arguments interpolated into *message*.
        """

        self.info_logs.append(message.format(*args))


def test_load_data_skips_silent_audio_with_warning(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """RMS-silent audio must short-circuit ingestion with a warning.

    End-to-end check for the first layer of the pre-Whisper guard. The
    guard is monkeypatched to report ``(False, "Audio RMS below silence
    threshold")``, simulating a digitally silent file (which in production
    would produce Whisper hallucinations). The test asserts that:

    * :meth:`AudioReader.load_data` returns ``[]`` — ingestion skipped, no
      documents emitted downstream;
    * a ``logger.warning`` is issued whose formatted message contains both
      the skip reason and the input file path, so operators can identify
      which file was rejected and why.

    The fake segment handed to ``_make_audio_reader`` is intentionally
    non-empty — if the guard silently failed, the reader would happily
    transcribe that segment and the ``docs == []`` assertion would catch
    it.

    Args:
        monkeypatch (pytest.MonkeyPatch): Overrides ``_audio_has_speech``
            and ``logger`` on ``audio_module`` for this test only.
        tmp_path (Path): Per-test temporary directory used to synthesize a
            sentinel audio path; no real audio is read.
    """

    reader = _make_audio_reader(
        monkeypatch, segments=[{"start": 0.0, "end": 1.0, "text": "unused"}]
    )
    monkeypatch.setattr(
        audio_module,
        "_audio_has_speech",
        lambda _audio: (False, "Audio RMS below silence threshold"),
    )
    fake_logger = _CapturingLogger()
    monkeypatch.setattr(audio_module, "logger", fake_logger)

    audio_path = tmp_path / "silent.wav"
    audio_path.write_bytes(b"fake")

    docs = reader.load_data(audio_path)

    assert docs == []
    assert any(
        "Audio RMS below silence threshold" in msg and str(audio_path) in msg
        for msg in fake_logger.warning_logs
    ), fake_logger.warning_logs


def test_load_data_skips_non_speech_audio_with_warning(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Noise-only audio (VAD negative) must be skipped with a warning.

    Companion to
    :func:`test_load_data_skips_silent_audio_with_warning` that covers the
    second layer of the pre-Whisper guard: audio which carries energy but
    does not contain detectable human speech (background music, traffic,
    white noise, tape hiss, etc.). ``_audio_has_speech`` is stubbed to
    report ``(False, "VAD detected no speech")``; the test asserts the
    reader returns ``[]`` and a warning mentioning both the VAD reason and
    the input path is logged.

    Args:
        monkeypatch (pytest.MonkeyPatch): Overrides ``_audio_has_speech``
            and ``logger`` on ``audio_module`` for this test only.
        tmp_path (Path): Per-test temporary directory used to synthesize a
            sentinel audio path.
    """

    reader = _make_audio_reader(
        monkeypatch, segments=[{"start": 0.0, "end": 1.0, "text": "unused"}]
    )
    monkeypatch.setattr(
        audio_module,
        "_audio_has_speech",
        lambda _audio: (False, "VAD detected no speech"),
    )
    fake_logger = _CapturingLogger()
    monkeypatch.setattr(audio_module, "logger", fake_logger)

    audio_path = tmp_path / "noise.wav"
    audio_path.write_bytes(b"fake")

    docs = reader.load_data(audio_path)

    assert docs == []
    assert any(
        "VAD detected no speech" in msg and str(audio_path) in msg
        for msg in fake_logger.warning_logs
    ), fake_logger.warning_logs


def test_filter_no_speech_segments_drops_high_prob_entries() -> None:
    """Verify the post-transcription ``no_speech_prob`` filter contract.

    Exercises every branch of
    :func:`docint.core.readers.audio._filter_no_speech_segments`:

    * ``no_speech_prob = 0.1`` — well below threshold, kept.
    * ``no_speech_prob = 0.9`` — well above threshold, dropped.
    * ``no_speech_prob`` missing — treated as ``0.0`` per :meth:`dict.get`
      default, kept.
    * ``no_speech_prob = 0.61`` — strictly above the 0.6 threshold, dropped.
    * ``no_speech_prob = 0.6`` — equal to the threshold, kept (the check
      is ``<=`` so the threshold itself is inclusive).

    The test relies on list ordering being preserved so the remaining
    segments can be checked by text.
    """

    segments = [
        {"start": 0.0, "end": 1.0, "text": "keep low", "no_speech_prob": 0.1},
        {"start": 1.0, "end": 2.0, "text": "drop high", "no_speech_prob": 0.9},
        {"start": 2.0, "end": 3.0, "text": "keep missing"},
        {"start": 3.0, "end": 4.0, "text": "drop boundary", "no_speech_prob": 0.61},
        {"start": 4.0, "end": 5.0, "text": "keep boundary", "no_speech_prob": 0.6},
    ]

    filtered = audio_module._filter_no_speech_segments(segments)

    assert [seg["text"] for seg in filtered] == [
        "keep low",
        "keep missing",
        "keep boundary",
    ]


def test_filter_no_speech_segments_returns_input_when_nothing_dropped() -> None:
    """Identity optimization must hold when no segment exceeds the threshold.

    To avoid needless allocation on the common happy path (nothing
    hallucinated), :func:`_filter_no_speech_segments` is expected to return
    the *same* list object it received when every segment passes. The
    assertion uses ``is`` — pointer identity, not value equality — to pin
    that contract down so a future refactor cannot silently regress it.
    """

    segments = [
        {"start": 0.0, "end": 1.0, "text": "a", "no_speech_prob": 0.1},
        {"start": 1.0, "end": 2.0, "text": "b"},
    ]

    assert audio_module._filter_no_speech_segments(segments) is segments


def test_load_data_applies_no_speech_filter_end_to_end(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The ``no_speech_prob`` filter must be wired into the full read path.

    A unit test on :func:`_filter_no_speech_segments` alone is not enough:
    a future refactor could easily drop the call site inside
    ``_transcribe_file_payload`` without any unit test failing. This test
    feeds a mixed segment list — two clean segments bracketing one
    hallucinated segment with ``no_speech_prob=0.95`` — through
    :meth:`AudioReader.load_data` and asserts that the emitted documents
    contain the clean text but not the hallucinated text.

    The assertion is deliberately loose (substring checks over the
    concatenated document text) so it is robust against changes in the
    segment→sentence merging logic.

    Args:
        monkeypatch (pytest.MonkeyPatch): Standard pytest fixture; used
            only via :func:`_make_audio_reader` helper.
        tmp_path (Path): Per-test temporary directory used to synthesize a
            sentinel audio path; no real audio is read.
    """

    segments = [
        {"start": 0.0, "end": 1.0, "text": "Hello", "no_speech_prob": 0.1},
        {
            "start": 1.0,
            "end": 2.0,
            "text": "hallucinated.",
            "no_speech_prob": 0.95,
        },
        {"start": 2.0, "end": 3.0, "text": "world.", "no_speech_prob": 0.2},
    ]
    reader = _make_audio_reader(monkeypatch, segments)
    audio_path = tmp_path / "clip.wav"
    audio_path.write_bytes(b"fake")

    docs = reader.load_data(audio_path)

    combined_text = " ".join(doc.text for doc in docs)
    assert "Hello" in combined_text
    assert "world." in combined_text
    assert "hallucinated" not in combined_text


def test_audio_has_speech_flags_digital_silence() -> None:
    """Digital silence must fail the RMS layer before VAD is consulted.

    The first layer of :func:`_audio_has_speech` is a cheap RMS energy
    check that rejects waveforms whose root-mean-square is below
    :data:`SILENCE_RMS_THRESHOLD` (0.01). The motivation is twofold:

    * it runs in microseconds, making silent files free to reject; and
    * Silero VAD is not loaded — useful when the model cache is cold and
      ``torch.hub.load`` would otherwise hit the network.

    This test uses one second (16 000 samples) of ``0.0`` float32 samples,
    which has an RMS of exactly zero — squarely below the threshold. The
    call goes through :data:`_real_audio_has_speech` (the import-time
    reference) to bypass the autouse fixture that otherwise short-circuits
    the guard.
    """

    import numpy as np

    silent = np.zeros(16000, dtype=np.float32)
    ok, reason = _real_audio_has_speech(silent)
    assert ok is False
    assert reason == "Audio RMS below silence threshold"


def test_audio_has_speech_skips_when_vad_reports_no_speech(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """High-energy non-speech audio must be rejected by the VAD layer.

    Exercises the second layer of :func:`_audio_has_speech`. The waveform
    is one second of ``0.5`` float32 samples — RMS = 0.5, which easily
    clears the 0.01 silence threshold and forces control to fall through
    to :func:`_detect_speech_vad`. That helper is monkeypatched to return
    ``False`` (simulating "no speech detected" from Silero), so the guard
    must report ``(False, "VAD detected no speech")``.

    In production this branch catches audio that carries energy (noise,
    music, tone) but has no human voice — the class of files most prone
    to Whisper hallucinations.

    Args:
        monkeypatch (pytest.MonkeyPatch): Used to stub
            ``audio_module._detect_speech_vad``. The stub is undone at
            teardown so the surrounding suite is unaffected.
    """

    import numpy as np

    monkeypatch.setattr(audio_module, "_detect_speech_vad", lambda _audio: False)
    loud = np.ones(16000, dtype=np.float32) * 0.5
    ok, reason = _real_audio_has_speech(loud)
    assert ok is False
    assert reason == "VAD detected no speech"


def test_audio_has_speech_passes_when_vad_reports_speech(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Audio that passes both guard layers must yield ``(True, None)``.

    The happy path: the waveform has enough energy to clear the RMS
    threshold *and* Silero VAD confirms speech. This is the case on every
    legitimate audio file and must result in ``(True, None)`` so the
    caller knows to proceed with transcription and that no skip reason
    needs to be logged.

    Args:
        monkeypatch (pytest.MonkeyPatch): Used to stub
            ``audio_module._detect_speech_vad`` to the positive response.
    """

    import numpy as np

    monkeypatch.setattr(audio_module, "_detect_speech_vad", lambda _audio: True)
    loud = np.ones(16000, dtype=np.float32) * 0.5
    ok, reason = _real_audio_has_speech(loud)
    assert ok is True
    assert reason is None


def test_audio_has_speech_passes_when_vad_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing Silero VAD model must degrade gracefully, not block ingestion.

    ``_detect_speech_vad`` returns ``None`` when the Silero model could
    not be loaded (offline with a cold cache, corrupted download, etc.).
    In that scenario the guard must trust the RMS layer and let the file
    through — the alternative would be silent, hard-to-debug ingestion
    failures on operator machines that never ran ``uv run load-models``.

    The test stubs :func:`_detect_speech_vad` to return ``None`` and
    supplies a waveform with an RMS comfortably above the silence
    threshold; the guard must therefore report ``(True, None)``.

    Args:
        monkeypatch (pytest.MonkeyPatch): Used to stub
            ``audio_module._detect_speech_vad`` to simulate a VAD load
            failure.
    """

    import numpy as np

    monkeypatch.setattr(audio_module, "_detect_speech_vad", lambda _audio: None)
    loud = np.ones(16000, dtype=np.float32) * 0.5
    ok, reason = _real_audio_has_speech(loud)
    assert ok is True
    assert reason is None


def test_transcribe_audio_job_skips_non_speech_audio(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The multiprocessing worker must honor the same guard as the main path.

    ``AudioReader.load_batch_data`` can dispatch to a
    :class:`concurrent.futures.ProcessPoolExecutor` where each file is
    transcribed inside :func:`_transcribe_audio_job` — a separate code
    path from :meth:`_transcribe_file_payload`. Without an explicit check
    here, a future refactor could easily leave the worker path
    un-guarded, silently letting every noise-only file through when batch
    mode is on.

    The test invokes the worker directly (not via the reader) with the
    guard stubbed to reject, then asserts on the payload schema used by
    ``_payload_to_documents``:

    * ``skipped=True`` + ``skip_reason`` is the structured signal callers
      use to distinguish "skipped on purpose" from "crashed";
    * ``result={"segments": []}`` yields zero documents downstream;
    * ``error=None`` — skips are not failures, so the error field must be
      cleared;
    * a formatted warning naming both the skip reason and the file path
      reached the logger.

    ``_get_worker_model`` is additionally stubbed as a safety net: the
    guard runs *before* the model would be consulted, so this stub should
    be unused, but stubbing it guarantees the test cannot accidentally
    call real Whisper even if the guard ever regresses.

    Args:
        monkeypatch (pytest.MonkeyPatch): Overrides
            ``_audio_has_speech``, ``logger``, and ``_get_worker_model``
            on ``audio_module``; the worker globals are reset implicitly
            by the subsequent call to ``_init_whisper_worker``.
    """

    monkeypatch.setattr(
        audio_module,
        "_audio_has_speech",
        lambda _audio: (False, "VAD detected no speech"),
    )
    fake_logger = _CapturingLogger()
    monkeypatch.setattr(audio_module, "logger", fake_logger)
    audio_module._init_whisper_worker("turbo", "cpu", "transcribe")
    # _get_worker_model should never run because the guard trips first, but
    # stub it anyway so the test never touches real Whisper.
    monkeypatch.setattr(audio_module, "_get_worker_model", lambda: object())

    payload = audio_module._transcribe_audio_job(("/tmp/fake.wav", "abc123"))

    assert payload["skipped"] is True
    assert payload["skip_reason"] == "VAD detected no speech"
    assert payload["result"] == {"segments": []}
    assert payload["error"] is None
    assert any(
        "VAD detected no speech" in msg and "/tmp/fake.wav" in msg
        for msg in fake_logger.warning_logs
    ), fake_logger.warning_logs
