from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from multiprocessing import get_context
import subprocess
import tempfile
from typing import Any, Generator, Literal, Sequence, cast

import whisper
from llama_index.core import Document
from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import MediaResource
from loguru import logger
from numpy import floating
from numpy.typing import NDArray
from openai import OpenAI as _OpenAIClient

from docint.utils.env_cfg import load_model_env, load_openai_env, load_whisper_env
from docint.utils.hashing import compute_file_hash, ensure_file_hash
from docint.utils.mimetype import get_mimetype

WhisperTask = Literal["transcribe", "translate"]
ProviderAudioInference = Literal["openai", "vllm"]
_ENGLISH_LANGUAGE_CODES = {"en", "eng", "english"}
_CPU_DEVICE_NAMES = {"cuda", "mps", "cpu"}

# Suffixes that libsndfile can decode natively.  Files whose extension is
# *not* in this set must be converted to WAV before being sent to a vLLM
# provider (which internally uses ``librosa`` → ``soundfile``).
_LIBSNDFILE_EXTENSIONS = {
    ".aif",
    ".aiff",
    ".au",
    ".avr",
    ".caf",
    ".flac",
    ".htk",
    ".mat",
    ".mpc",
    ".ogg",
    ".paf",
    ".pvf",
    ".raw",
    ".rf64",
    ".sd2",
    ".sds",
    ".sf",
    ".svx",
    ".voc",
    ".w64",
    ".wav",
    ".wavex",
    ".xi",
}
_WORKER_MODEL: whisper.Whisper | None = None
_WORKER_MODEL_ID: str | None = None
_WORKER_DEVICE: str | None = None
_WORKER_SRC_LANGUAGE: str | None = None
_WORKER_TASK: WhisperTask = "transcribe"


def _normalize_language_code(language: str | None) -> str | None:
    """Normalize a Whisper language value to a lowercase string.

    Args:
        language (str | None): The Whisper language value to normalize.

    Returns:
        str | None: The normalized language code, or ``None`` when the input is
        empty or missing.
    """

    if language is None:
        return None
    normalized = language.strip().lower()
    return normalized or None


def _is_english_language(language: str | None) -> bool:
    """Return True when Whisper identifies the language as English.

    Args:
        language (str | None): The Whisper language value to evaluate.

    Returns:
        bool: True when the normalized language value maps to English.
    """

    normalized = _normalize_language_code(language)
    return normalized in _ENGLISH_LANGUAGE_CODES


def _coerce_whisper_task(value: Any) -> WhisperTask:
    """Coerce a dynamic task value into a supported Whisper task.

    Args:
        value (Any): The candidate task value.

    Returns:
        WhisperTask: ``"transcribe"`` or ``"translate"``, defaulting to
        ``"transcribe"`` for unsupported values.
    """

    return value if value in {"transcribe", "translate"} else "transcribe"


def _select_whisper_task(
    configured_task: WhisperTask, detected_language: str | None
) -> WhisperTask:
    """Resolve the effective Whisper task for a single file.

    Args:
        configured_task (WhisperTask): The configured default Whisper task.
        detected_language (str | None): The detected source language, if available.

    Returns:
        WhisperTask: The task that should be applied to the current file.
    """

    if configured_task != "translate":
        return "transcribe"
    if detected_language is None or _is_english_language(detected_language):
        return "transcribe"
    return "translate"


def _detect_language_with_model(
    audio: NDArray[floating[Any]],
    model: whisper.Whisper,
    device: str | None,
) -> str | None:
    """Detect language using the loaded Whisper model when available.

    Args:
        audio (NDArray[floating[Any]]): The audio sample to analyze.
        model (whisper.Whisper): The loaded Whisper model.
        device (str | None): The inference device used by the model.

    Returns:
        str | None: The detected language code, or ``None`` when detection is
        unavailable.
    """

    if not hasattr(model, "detect_language"):
        return None
    if not hasattr(whisper, "pad_or_trim") or not hasattr(
        whisper, "log_mel_spectrogram"
    ):
        return None

    mel = whisper.log_mel_spectrogram(whisper.pad_or_trim(audio))
    mel_to = getattr(mel, "to", None)
    if callable(mel_to):
        target_device = device or "cpu"
        mel = mel_to(target_device)

    _, probabilities = model.detect_language(mel)
    if not isinstance(probabilities, dict) or not probabilities:
        return None
    probabilities = cast(dict[str, float], probabilities)

    best_language = max(probabilities, key=lambda code: probabilities[code])
    return _normalize_language_code(
        best_language if isinstance(best_language, str) else None
    )


def _run_whisper_transcribe(
    audio: NDArray[floating[Any]],
    model: whisper.Whisper,
    *,
    task: WhisperTask,
    language: str | None = None,
) -> dict[str, Any]:
    """Run Whisper transcription with explicitly typed keyword arguments.

    Args:
        audio (NDArray[floating[Any]]): The audio sample to transcribe.
        model (whisper.Whisper): The loaded Whisper model.
        task (WhisperTask): The Whisper task to execute.
        language (str | None): The detected language code, if available.

    Returns:
        dict[str, Any]: The transcription result returned by Whisper.
    """

    if language is None:
        return cast(
            dict[str, Any],
            whisper.transcribe(
                model=model,
                audio=audio,
                task=task,
            ),
        )

    return cast(
        dict[str, Any],
        whisper.transcribe(
            model=model,
            audio=audio,
            task=task,
            language=language,
        ),
    )


def _init_whisper_worker(
    model_id: str,
    device: str | None,
    configured_task: WhisperTask,
    src_language: str | None = None,
) -> None:
    """Initialize module-level worker state for Whisper multiprocessing.

    Args:
        model_id (str): The Whisper model identifier to load in workers.
        device (str | None): The inference device to use in workers.
        configured_task (WhisperTask): The configured default Whisper task.
        src_language (str | None): Optional source language override.
    """

    global \
        _WORKER_DEVICE, \
        _WORKER_MODEL, \
        _WORKER_MODEL_ID, \
        _WORKER_SRC_LANGUAGE, \
        _WORKER_TASK
    _WORKER_MODEL = None
    _WORKER_MODEL_ID = model_id
    _WORKER_DEVICE = device
    _WORKER_SRC_LANGUAGE = _normalize_language_code(src_language)
    _WORKER_TASK = configured_task


def _get_worker_model() -> whisper.Whisper:
    """Load and cache a Whisper model inside a worker process.

    Returns:
        whisper.Whisper: The cached Whisper model for the current worker.

    Raises:
        ValueError: If the worker model ID has not been initialized.
    """

    global _WORKER_MODEL
    if _WORKER_MODEL is None:
        if _WORKER_MODEL_ID is None:
            raise ValueError("Whisper worker model ID is not configured.")
        _WORKER_MODEL = whisper.load_model(_WORKER_MODEL_ID, device=_WORKER_DEVICE)
    return _WORKER_MODEL


def _transcribe_audio_job(job: tuple[str, str | None]) -> dict[str, Any]:
    """Transcribe a single audio file inside a worker process.

    Args:
        job (tuple[str, str | None]): The audio file path and optional file hash.

    Returns:
        dict[str, Any]: A normalized payload containing the transcription result
        or the captured error for the file.
    """

    file_path_str, file_hash = job
    try:
        model = _get_worker_model()
        audio = whisper.load_audio(file=file_path_str)
        detected_language: str | None = _WORKER_SRC_LANGUAGE
        if detected_language is None and _WORKER_TASK == "translate":
            detected_language = _detect_language_with_model(
                audio=audio,
                model=model,
                device=_WORKER_DEVICE,
            )

        selected_task = _select_whisper_task(_WORKER_TASK, detected_language)
        result = _run_whisper_transcribe(
            audio=audio,
            model=model,
            task=selected_task,
            language=detected_language or _WORKER_SRC_LANGUAGE,
        )
        return {
            "detected_language": detected_language,
            "error": None,
            "file_hash": file_hash,
            "file_path": file_path_str,
            "result": result,
            "selected_task": selected_task,
        }
    except Exception as exc:
        return {
            "detected_language": None,
            "error": str(exc),
            "file_hash": file_hash,
            "file_path": file_path_str,
            "result": None,
            "selected_task": "transcribe",
        }


class OpenAICompatibleAudioBackend:
    """Provider-backed audio transcription helper for OpenAI-compatible APIs."""

    @staticmethod
    @contextmanager
    def _wav_for_provider(
        file_path: Path,
    ) -> Generator[Path, None, None]:
        """Yield a WAV path suitable for the provider API.

        If the file's suffix is already handled by libsndfile the original
        path is yielded unchanged.  Otherwise the audio is re-encoded to
        16 kHz mono WAV via *ffmpeg* so that the vLLM ``librosa.load()``
        call can decode it.

        Args:
            file_path: Source audio file.

        Yields:
            Path to a WAV file (either the original or a temporary copy).
        """
        if file_path.suffix.lower() in _LIBSNDFILE_EXTENSIONS:
            yield file_path
            return

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
            tmp_path = Path(tmp.name)

        try:
            cmd = [
                "ffmpeg",
                "-y",
                "-nostdin",
                "-threads",
                "0",
                "-i",
                str(file_path),
                "-ac",
                "1",
                "-ar",
                "16000",
                "-acodec",
                "pcm_s16le",
                str(tmp_path),
            ]
            subprocess.run(cmd, capture_output=True, check=True)  # noqa: S603
            logger.debug(
                "Converted {} → {} for provider API",
                file_path.name,
                tmp_path.name,
            )
            yield tmp_path
        finally:
            tmp_path.unlink(missing_ok=True)

    def __init__(
        self,
        *,
        api_base: str,
        api_key: str,
        max_retries: int,
        model_id: str,
        timeout: float,
    ) -> None:
        """Initialize the provider-backed audio client.

        Args:
            api_base: OpenAI-compatible base URL.
            api_key: API key used for authorization.
            max_retries: Maximum request retries.
            model_id: Served ASR model identifier.
            timeout: Request timeout in seconds.
        """

        self.model_id = model_id
        self._client = _OpenAIClient(
            api_key=api_key,
            base_url=api_base,
            max_retries=max_retries,
            timeout=timeout,
        )

    @staticmethod
    def _response_to_dict(response: Any) -> dict[str, Any]:
        """Normalize OpenAI client responses into plain dictionaries.

        Args:
            response: The raw response object from the OpenAI client.

        Returns:
            dict[str, Any]: A normalized dictionary representation of the response.
        """

        if isinstance(response, dict):
            return cast(dict[str, Any], response)
        model_dump = getattr(response, "model_dump", None)
        if callable(model_dump):
            dumped = model_dump()
            if isinstance(dumped, dict):
                return cast(dict[str, Any], dumped)
        return cast(dict[str, Any], vars(response))

    @staticmethod
    def _translation_supported(model_id: str) -> bool:
        """Return whether the configured ASR model should support translation.

        Args:
            model_id: The ASR model identifier to evaluate.

        Returns:
            bool: True when the model is expected to support translation, otherwise False.
        """

        normalized = model_id.strip().lower()
        return normalized != "turbo" and "whisper-large-v3-turbo" not in normalized

    def _transcribe_via_provider(
        self, file_path: Path, *, language: str | None = None
    ) -> dict[str, Any]:
        """Call the provider transcriptions endpoint with verbose segments.

        Args:
            file_path: The path to the audio file to transcribe.
            language: Optional source language override.

        Returns:
            dict[str, Any]: The normalized transcription result from the provider.
        """

        kwargs: dict[str, Any] = {
            "file": None,
            "model": self.model_id,
            "response_format": "verbose_json",
            "timestamp_granularities": ["segment"],
        }
        if language:
            kwargs["language"] = language

        with (
            self._wav_for_provider(file_path) as send_path,
            send_path.open("rb") as audio_file,
        ):
            kwargs["file"] = audio_file
            response = self._client.audio.transcriptions.create(**kwargs)
        return self._response_to_dict(response)

    def _translate_via_provider(self, file_path: Path) -> dict[str, Any]:
        """Call the provider translations endpoint with verbose output.

        Args:
            file_path: The path to the audio file to translate.

        Returns:
            dict[str, Any]: The normalized translation result from the provider.
        """

        with (
            self._wav_for_provider(file_path) as send_path,
            send_path.open("rb") as audio_file,
        ):
            response = self._client.audio.translations.create(
                file=audio_file,
                model=self.model_id,
                response_format="verbose_json",
            )
        return self._response_to_dict(response)

    def transcribe_file(
        self,
        file_path: Path,
        *,
        configured_task: WhisperTask,
        src_language: str | None = None,
    ) -> dict[str, Any]:
        """Transcribe or translate a file through the provider backend.

        Args:
            file_path: Source audio file.
            configured_task: Requested task from configuration.
            src_language: Optional source language override.

        Returns:
            A normalized transcription payload.
        """

        transcription = self._transcribe_via_provider(file_path, language=src_language)
        detected_language = _normalize_language_code(
            src_language or cast(str | None, transcription.get("language"))
        )
        if configured_task == "translate" and detected_language is None:
            logger.warning(
                "Provider-backed Whisper translation requested but language detection was unavailable; falling back to transcription."
            )
        selected_task = _select_whisper_task(configured_task, detected_language)
        if selected_task != "translate":
            return {
                "detected_language": detected_language,
                "result": transcription,
                "selected_task": "transcribe",
            }

        if not self._translation_supported(self.model_id):
            logger.warning(
                "Provider-backed Whisper model '{}' does not support translation; falling back to transcription.",
                self.model_id,
            )
            return {
                "detected_language": detected_language,
                "result": transcription,
                "selected_task": "transcribe",
            }

        try:
            translated = self._translate_via_provider(file_path)
        except Exception as exc:
            logger.warning(
                "Provider-backed Whisper translation failed for '{}': {}. Falling back to transcription.",
                file_path,
                exc,
            )
            return {
                "detected_language": detected_language,
                "result": transcription,
                "selected_task": "transcribe",
            }

        return {
            "detected_language": detected_language,
            "result": translated,
            "selected_task": "translate",
        }


class AudioReader(BaseReader):
    """A reader for audio files using the Whisper model.

    Args:
        BaseReader: Base class for all LlamaIndex readers.
    """

    def __init__(self, device: str | None = "cpu") -> None:
        """Initialize the AudioReader.

        Args:
            device (str | None, optional): The device to use for inference. Defaults to "cpu".
        """
        self.device: str | None = device
        self.model_id: str = load_model_env().whisper_model
        openai_cfg = load_openai_env()
        self.inference_provider: str = openai_cfg.inference_provider.lower()
        whisper_cfg = load_whisper_env()
        self.max_workers: int = whisper_cfg.max_workers
        self.src_language: str | None = _normalize_language_code(
            whisper_cfg.src_language
        )
        self.task: WhisperTask = whisper_cfg.task
        self._model: whisper.Whisper | None = None
        self._provider_backend: OpenAICompatibleAudioBackend | None = None
        if self._uses_provider_audio_backend():
            self._provider_backend = OpenAICompatibleAudioBackend(
                api_base=openai_cfg.api_base,
                api_key=openai_cfg.api_key,
                max_retries=openai_cfg.max_retries,
                model_id=self.model_id,
                timeout=openai_cfg.timeout,
            )
        self.result: dict[str, str | list[Any]] | None = None

    def _uses_provider_audio_backend(self) -> bool:
        """Return whether audio should be routed to a provider API.

        Returns:
            bool: True when the configured inference provider exposes
            OpenAI-compatible audio endpoints that should be used instead of
            local Whisper inference.
        """

        provider = self.inference_provider.strip().lower()
        return provider in {"openai", "vllm"}

    def _load_model(self) -> whisper.Whisper:
        """Load the Whisper model.

        Returns:
            whisper.Whisper: The loaded Whisper model.

        Raises:
            ValueError: If the model ID is not set.
        """
        if self.model_id is None:
            logger.error("ValueError: Model ID is not set.")
            raise ValueError("Model ID is not set.")
        if self._model is None:
            self._model = whisper.load_model(self.model_id, device=self.device)
        return self._model

    def _load_audio(self, file_path: str | Path) -> NDArray[floating[Any]]:
        """Load audio from a file path.

        Args:
            file_path (str | Path): The path to the audio file.

        Returns:
            NDArray[floating[Any]]: The loaded audio as a NumPy array.
        """
        file = file_path if isinstance(file_path, str) else str(file_path)
        return whisper.load_audio(file=file)

    def _transcribe_audio(
        self,
        audio: NDArray[floating[Any]],
        model: whisper.Whisper,
        *,
        task: WhisperTask,
        language: str | None = None,
    ) -> dict[str, Any]:
        """Transcribe the given audio using the specified model and optional language.

        Args:
            audio (NDArray[floating[Any]]): The audio data to transcribe.
            model (whisper.Whisper): The Whisper model to use for transcription.
            task (WhisperTask): The Whisper task to run for this file.
            language (str | None): The detected language code, when available.

        Returns:
            dict[str, Any]: The transcription result.
        """
        return _run_whisper_transcribe(
            audio=audio,
            model=model,
            task=task,
            language=language,
        )

    def _detect_language(
        self, audio: NDArray[floating[Any]], model: whisper.Whisper
    ) -> str | None:
        """Detect the language for an audio sample when translation is enabled.

        Args:
            audio (NDArray[floating[Any]]): The audio sample to analyze.
            model (whisper.Whisper): The loaded Whisper model.

        Returns:
            str | None: The detected language code, or ``None`` when detection
            fails.
        """

        try:
            return _detect_language_with_model(
                audio=audio,
                model=model,
                device=self.device,
            )
        except Exception as exc:
            logger.warning("Whisper language detection failed: {}", exc)
            return None

    def _enrich_document(
        self,
        file_path: Path,
        text: str,
        source: str = "transcript",
        file_hash: str | None = None,
        extra_metadata: dict[str, Any] | None = None,
    ) -> Document:
        """Enrich a document with metadata from the image file.

        Args:
            file_path (Path): The path to the image file.
            text (str): The text content extracted from the image.
            source (str, optional): The source type. Defaults to "transcript".
            file_hash (str | None, optional): The hash of the audio file. Defaults to None.

        Returns:
            Document: The enriched document.

        Raises:
            ValueError: If file_path is not set.
        """
        if file_path is None:
            logger.error("ValueError: file_path is not set.")
            raise ValueError("file_path is not set.")
        filename = file_path.name
        mimetype = get_mimetype(file_path)

        try:
            source = mimetype.split("/")[0]
        except Exception:
            logger.warning(
                "ValueError: Could not determine source from mimetype: {}", mimetype
            )
            pass

        metadata: dict[str, Any] = {
            "file_path": str(file_path),
            "file_name": filename,
            "filename": filename,
            "file_type": mimetype,
            "mimetype": mimetype,
            "source": source,
            "origin": {
                "filename": filename,
                "mimetype": mimetype,
            },
        }
        ensure_file_hash(
            metadata,
            file_hash=file_hash if file_hash is not None else None,
            path=file_path if file_hash is None else None,
        )

        if extra_metadata:
            metadata.update(extra_metadata)

        return Document(
            text_resource=MediaResource(text=text, mimetype=mimetype),
            metadata=metadata,
        )

    @staticmethod
    def _format_timestamp(seconds: float | int) -> str:
        """Format seconds into HH:MM:SS format.

        Args:
            seconds (float | int): The number of seconds to format.

        Returns:
            str: The formatted timestamp in HH:MM:SS format.
        """
        total_seconds = int(seconds)
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        secs = total_seconds % 60
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    @staticmethod
    def _ends_with_punctuation(text: str) -> bool:
        """Return True if text ends with sentence-ending punctuation.

        Args:
            text (str): The text to check.

        Returns:
            bool: True if text ends with '.', '!', or '?', False otherwise.
        """
        return text.strip().endswith((".", "!", "?"))

    def _merge_segments_into_sentences(
        self, segments: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Merge whisper segments into full sentences based on trailing punctuation.
        Each merged item keeps earliest start and latest end.

        Args:
            segments (list[dict[str, Any]]): The list of segments to merge.

        Returns:
            list[dict[str, Any]]: The merged list of segments.
        """

        merged: list[dict[str, Any]] = []
        current: dict[str, Any] | None = None

        for seg in segments:
            if not isinstance(seg, dict):
                continue
            seg_text = seg.get("text", "")
            if not isinstance(seg_text, str) or not seg_text.strip():
                continue

            if current is None:
                current = {
                    "start": seg.get("start"),
                    "end": seg.get("end"),
                    "text": seg_text.strip(),
                }
            else:
                current["end"] = seg.get("end", current.get("end"))
                current_text = current.get("text", "")
                current["text"] = f"{current_text} {seg_text.strip()}".strip()

            if self._ends_with_punctuation(seg_text):
                merged.append(current)
                current = None

        if current:
            current_text = current.get("text")
            if isinstance(current_text, str) and current_text.strip():
                merged.append(current)

        return merged

    def _build_segment_documents(
        self,
        result: dict[str, Any],
        file_path: Path,
        file_hash: str | None,
        *,
        detected_language: str | None = None,
        selected_task: WhisperTask = "transcribe",
    ) -> list[Document]:
        """Create one Document per full sentence by merging segments until punctuation.
        Timing metadata reflects the merged start/end.

        Args:
            result (dict[str, Any]): The transcription result containing segments.
            file_path (Path): The path to the audio file.
            file_hash (str | None): The hash of the audio file.
            detected_language (str | None): The detected source language code.
            selected_task (WhisperTask): The applied Whisper task.

        Returns:
            list[Document]: A list of enriched Document objects for each sentence.
        """
        base_metadata: dict[str, Any] = {"whisper_task": selected_task}
        if detected_language is not None:
            base_metadata["whisper_language"] = detected_language

        segments = result.get("segments") if isinstance(result, dict) else None
        if not isinstance(segments, list):
            fallback = result.get("text") if isinstance(result, dict) else ""
            if isinstance(fallback, str) and fallback.strip():
                return [
                    self._enrich_document(
                        file_path,
                        fallback,
                        file_hash=file_hash,
                        extra_metadata=base_metadata,
                    )
                ]
            return []

        sentence_segments = self._merge_segments_into_sentences(segments)
        docs: list[Document] = []
        for idx, seg in enumerate(sentence_segments):
            seg_text = seg.get("text", "")
            if not isinstance(seg_text, str) or not seg_text.strip():
                continue
            start = seg.get("start")
            end = seg.get("end")
            extra: dict[str, Any] = {"sentence_index": idx}
            extra.update(base_metadata)
            if isinstance(start, (int, float)):
                extra["start_ts"] = self._format_timestamp(start)
                extra["start_seconds"] = float(start)
            if isinstance(end, (int, float)):
                extra["end_ts"] = self._format_timestamp(end)
                extra["end_seconds"] = float(end)
            docs.append(
                self._enrich_document(
                    file_path,
                    seg_text,
                    file_hash=file_hash,
                    extra_metadata=extra,
                )
            )

        if not docs:
            fallback = result.get("text") if isinstance(result, dict) else ""
            if isinstance(fallback, str) and fallback.strip():
                return [
                    self._enrich_document(
                        file_path,
                        fallback,
                        file_hash=file_hash,
                        extra_metadata=base_metadata,
                    )
                ]
        return docs

    def _effective_worker_count(self, file_count: int) -> int:
        """Resolve the safe worker count for a batch of files.

        Args:
            file_count (int): The number of files in the batch.

        Returns:
            int: The safe number of workers to use for the batch.
        """

        if self._provider_backend is not None:
            return 1

        if file_count <= 1:
            return 1

        max_workers = max(1, self.max_workers)
        device_name = (self.device or "").strip().lower()
        if max_workers > 1 and device_name not in _CPU_DEVICE_NAMES:
            logger.warning(
                "Whisper multiprocessing is disabled for device '{}'; falling back to a single worker.",
                self.device,
            )
            return 1
        return min(max_workers, file_count)

    def _resolve_task_for_audio(
        self, audio: NDArray[floating[Any]], model: whisper.Whisper
    ) -> tuple[WhisperTask, str | None]:
        """Resolve the effective Whisper task and detected language for a file.

        Args:
            audio (NDArray[floating[Any]]): The audio sample to process.
            model (whisper.Whisper): The loaded Whisper model.

        Returns:
            tuple[WhisperTask, str | None]: The selected task and detected
            language code for the file.
        """

        detected_language: str | None = self.src_language
        if detected_language is None and self.task == "translate":
            detected_language = self._detect_language(audio, model)
            if detected_language is None:
                logger.warning(
                    "Whisper translation requested but language detection was unavailable; falling back to transcription."
                )

        selected_task = _select_whisper_task(self.task, detected_language)
        return selected_task, detected_language or self.src_language

    def _transcribe_file_payload(
        self, file_path: Path, file_hash: str | None
    ) -> dict[str, Any]:
        """Transcribe a single file and return a normalized payload.

        Args:
            file_path (Path): The audio file to transcribe.
            file_hash (str | None): The precomputed file hash, if available.

        Returns:
            dict[str, Any]: A normalized payload containing the transcription
            result and selected Whisper metadata.
        """
        if self._provider_backend is not None:
            payload = self._provider_backend.transcribe_file(
                file_path,
                configured_task=self.task,
                src_language=self.src_language,
            )
            payload["file_hash"] = file_hash
            payload["file_path"] = str(file_path)
            return payload

        model = self._load_model()
        audio = self._load_audio(file_path)
        selected_task, detected_language = self._resolve_task_for_audio(audio, model)
        result = self._transcribe_audio(
            audio,
            model,
            task=selected_task,
            language=detected_language,
        )
        return {
            "detected_language": detected_language,
            "file_hash": file_hash,
            "file_path": str(file_path),
            "result": result,
            "selected_task": selected_task,
        }

    @staticmethod
    def _log_file_start(
        file_path: Path,
        *,
        index: int,
        total: int,
        mode: str,
    ) -> None:
        """Log the start of a file transcription operation.

        Args:
            file_path (Path): The file being transcribed.
            index (int): The one-based position of the file in the batch.
            total (int): The total number of files in the operation.
            mode (str): A short mode label such as ``"single"`` or ``"parallel"``.
        """

        logger.info(
            "[AudioReader] Starting {}/{} ({}) for {}",
            index,
            total,
            mode,
            file_path,
        )

    @staticmethod
    def _log_file_result(
        file_path: Path,
        *,
        index: int,
        total: int,
        payload: dict[str, Any] | None,
        doc_count: int,
    ) -> bool:
        """Log the outcome of a file transcription operation.

        Args:
            file_path (Path): The file that was processed.
            index (int): The one-based completion count for the batch.
            total (int): The total number of files in the operation.
            payload (dict[str, Any] | None): The transcription payload, if one was produced.
            doc_count (int): The number of documents derived from the transcription.

        Returns:
            bool: True when the file completed successfully, otherwise False.
        """

        error = payload.get("error") if isinstance(payload, dict) else None
        if isinstance(error, str) and error:
            logger.warning(
                "[AudioReader] Failed {}/{} for {}: {}",
                index,
                total,
                file_path,
                error,
            )
            return False

        selected_task = (
            _coerce_whisper_task(payload.get("selected_task"))
            if isinstance(payload, dict)
            else "transcribe"
        )
        detected_language = (
            _normalize_language_code(payload.get("detected_language"))
            if isinstance(payload, dict)
            else None
        )
        language_display = detected_language or "unknown"
        logger.info(
            "[AudioReader] Finished {}/{} for {} -> {} document(s) [task={}, language={}]",
            index,
            total,
            file_path,
            doc_count,
            selected_task,
            language_display,
        )
        return True

    def _payload_to_documents(
        self, payload: dict[str, Any], file_path: Path
    ) -> list[Document]:
        """Convert a transcription payload into Documents.

        Args:
            payload (dict[str, Any]): The normalized transcription payload.
            file_path (Path): The source audio file path.

        Returns:
            list[Document]: The documents derived from the payload.
        """

        result = payload.get("result")
        if not isinstance(result, dict):
            return []
        return self._build_segment_documents(
            result=result,
            file_path=file_path,
            file_hash=payload.get("file_hash"),
            detected_language=_normalize_language_code(
                payload.get("detected_language")
            ),
            selected_task=_coerce_whisper_task(payload.get("selected_task")),
        )

    @staticmethod
    def _resolve_file_hash(
        file_path: Path,
        metadata: dict[str, Any] | None,
    ) -> str:
        """Resolve the file hash for a batch input.

        Args:
            file_path (Path): The source file path.
            metadata (dict[str, Any] | None): Optional metadata for the file.

        Returns:
            str: The existing or computed file hash.
        """

        file_hash = metadata.get("file_hash") if isinstance(metadata, dict) else None
        return (
            file_hash
            if isinstance(file_hash, str) and file_hash
            else compute_file_hash(file_path)
        )

    def load_batch_data(
        self,
        files: Sequence[str | Path],
        *,
        extra_info: Sequence[dict[str, Any] | None] | None = None,
    ) -> list[list[Document]]:
        """Transcribe multiple files, using multiprocessing when configured.

        Args:
            files (Sequence[str | Path]): The audio files to transcribe.
            extra_info (Sequence[dict[str, Any] | None] | None): Optional
                metadata entries aligned one-to-one with ``files``.

        Returns:
            list[list[Document]]: A list of per-file document lists that
            preserves the input file order.

        Raises:
            ValueError: If ``extra_info`` does not align with the number of
                input files.
        """

        file_paths = [
            Path(file) if not isinstance(file, Path) else file for file in files
        ]
        if not file_paths:
            return []

        if extra_info is not None and len(extra_info) != len(file_paths):
            raise ValueError("extra_info must match the number of input files.")

        normalized_extra_info = (
            list(extra_info) if extra_info is not None else [None] * len(file_paths)
        )

        worker_count = self._effective_worker_count(len(file_paths))
        logger.info(
            "[AudioReader] Starting batch transcription for {} file(s) with {} worker(s).",
            len(file_paths),
            worker_count,
        )
        if worker_count <= 1:
            serial_results: list[list[Document]] = []
            success_count = 0
            for idx, (file_path, file_hash) in enumerate(
                (
                    (
                        file_path,
                        self._resolve_file_hash(file_path, metadata),
                    )
                    for file_path, metadata in zip(
                        file_paths, normalized_extra_info, strict=False
                    )
                ),
                start=1,
            ):
                self._log_file_start(
                    file_path,
                    index=idx,
                    total=len(file_paths),
                    mode="single-worker",
                )
                try:
                    payload = self._transcribe_file_payload(file_path, file_hash)
                    docs = self._payload_to_documents(payload, file_path)
                    serial_results.append(docs)
                    success_count += int(
                        self._log_file_result(
                            file_path,
                            index=idx,
                            total=len(file_paths),
                            payload=payload,
                            doc_count=len(docs),
                        )
                    )
                except Exception as exc:
                    logger.warning("Failed to transcribe {}: {}", file_path, exc)
                    serial_results.append([])
                    self._log_file_result(
                        file_path,
                        index=idx,
                        total=len(file_paths),
                        payload={"error": str(exc)},
                        doc_count=0,
                    )
            logger.info(
                "[AudioReader] Finished batch transcription: {}/{} file(s) succeeded.",
                success_count,
                len(file_paths),
            )
            return serial_results

        results: list[list[Document]] = [[] for _ in file_paths]
        failed_indices: set[int] = set()
        completed_count = 0
        success_count = 0

        try:
            with ProcessPoolExecutor(
                max_workers=worker_count,
                mp_context=get_context("spawn"),
                initializer=_init_whisper_worker,
                initargs=(self.model_id, self.device, self.task, self.src_language),
            ) as executor:
                future_to_index: dict[Any, int] = {}
                next_submit_index = 0

                def _submit_next_job() -> None:
                    nonlocal next_submit_index
                    if next_submit_index >= len(file_paths):
                        return
                    file_path = file_paths[next_submit_index]
                    metadata = normalized_extra_info[next_submit_index]
                    file_hash = self._resolve_file_hash(file_path, metadata)
                    future = executor.submit(
                        _transcribe_audio_job,
                        (str(file_path), file_hash),
                    )
                    future_to_index[future] = next_submit_index
                    self._log_file_start(
                        file_path,
                        index=next_submit_index + 1,
                        total=len(file_paths),
                        mode="parallel",
                    )
                    next_submit_index += 1

                for _ in range(min(worker_count, len(file_paths))):
                    _submit_next_job()

                while future_to_index:
                    future = next(iter(as_completed(list(future_to_index))))
                    file_index = future_to_index.pop(future)
                    try:
                        payload = future.result()
                    except Exception as exc:
                        failed_indices.add(file_index)
                        logger.warning(
                            "Whisper worker crashed for {}: {}. Retrying in the main process.",
                            file_paths[file_index],
                            exc,
                        )
                        _submit_next_job()
                        continue

                    error = payload.get("error") if isinstance(payload, dict) else None
                    if isinstance(error, str) and error:
                        logger.warning(
                            "Failed to transcribe {}: {}",
                            file_paths[file_index],
                            error,
                        )
                        completed_count += 1
                        self._log_file_result(
                            file_paths[file_index],
                            index=completed_count,
                            total=len(file_paths),
                            payload=payload,
                            doc_count=0,
                        )
                        _submit_next_job()
                        continue

                    docs = self._payload_to_documents(payload, file_paths[file_index])
                    results[file_index] = docs
                    completed_count += 1
                    success_count += int(
                        self._log_file_result(
                            file_paths[file_index],
                            index=completed_count,
                            total=len(file_paths),
                            payload=payload,
                            doc_count=len(docs),
                        )
                    )
                    _submit_next_job()
        except Exception as exc:
            logger.warning(
                "Whisper multiprocessing setup failed: {}. Falling back to single-process transcription.",
                exc,
            )
            failed_indices = set(range(len(file_paths)))

        for idx in sorted(failed_indices):
            self._log_file_start(
                file_paths[idx],
                index=idx + 1,
                total=len(file_paths),
                mode="main-process-retry",
            )
            try:
                payload = self._transcribe_file_payload(
                    file_paths[idx],
                    self._resolve_file_hash(
                        file_paths[idx],
                        normalized_extra_info[idx],
                    ),
                )
            except Exception as exc:
                logger.warning("Failed to transcribe {}: {}", file_paths[idx], exc)
                completed_count += 1
                self._log_file_result(
                    file_paths[idx],
                    index=completed_count,
                    total=len(file_paths),
                    payload={"error": str(exc)},
                    doc_count=0,
                )
                continue

            error = payload.get("error") if isinstance(payload, dict) else None
            if isinstance(error, str) and error:
                logger.warning("Failed to transcribe {}: {}", file_paths[idx], error)
                completed_count += 1
                self._log_file_result(
                    file_paths[idx],
                    index=completed_count,
                    total=len(file_paths),
                    payload=payload,
                    doc_count=0,
                )
                continue

            docs = self._payload_to_documents(payload, file_paths[idx])
            results[idx] = docs
            completed_count += 1
            success_count += int(
                self._log_file_result(
                    file_paths[idx],
                    index=completed_count,
                    total=len(file_paths),
                    payload=payload,
                    doc_count=len(docs),
                )
            )

        logger.info(
            "[AudioReader] Finished batch transcription: {}/{} file(s) succeeded.",
            success_count,
            len(file_paths),
        )
        return results

    def load_data(self, file: str | Path, **kwargs) -> list[Document]:
        """Transcribe audio from a file using the Whisper model.

        Args:
            file (str | Path): The path to the audio file.
            **kwargs: Additional keyword arguments.

        Returns:
            list[Document]: The segmented transcription documents for the file.
        """
        file_path = Path(file) if not isinstance(file, Path) else file
        extra_info = kwargs.get("extra_info", {})

        file_hash = (
            extra_info.get("file_hash") if isinstance(extra_info, dict) else None
        )
        if file_hash is None:
            file_hash = compute_file_hash(file_path)

        self._log_file_start(file_path, index=1, total=1, mode="single")
        payload = self._transcribe_file_payload(file_path, file_hash)
        result = payload.get("result")
        self.result = result if isinstance(result, dict) else None
        if self.result is None:
            self._log_file_result(
                file_path,
                index=1,
                total=1,
                payload={"error": "No transcription result returned."},
                doc_count=0,
            )
            return []
        docs = self._build_segment_documents(
            self.result,
            file_path,
            file_hash,
            detected_language=_normalize_language_code(
                payload.get("detected_language")
            ),
            selected_task=_coerce_whisper_task(payload.get("selected_task")),
        )
        self._log_file_result(
            file_path,
            index=1,
            total=1,
            payload=payload,
            doc_count=len(docs),
        )
        return docs
