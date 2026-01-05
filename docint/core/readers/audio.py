from pathlib import Path
from typing import Any

import whisper
from llama_index.core import Document
from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import MediaResource
from loguru import logger
from numpy import floating
from numpy.typing import NDArray

from docint.utils.env_cfg import load_model_env
from docint.utils.hashing import compute_file_hash, ensure_file_hash
from docint.utils.mimetype import get_mimetype


class AudioReader(BaseReader):
    """
    A reader for audio files using the Whisper model.

    Args:
        BaseReader: Base class for all LlamaIndex readers.
    """

    def __init__(self, device: str | None = "cpu") -> None:
        """
        Initialize the AudioReader.

        Args:
            device (str | None, optional): The device to use for inference. Defaults to "cpu".
        """
        self.device: str | None = device
        self.model_id: str = load_model_env().whisper_model
        self.result: dict[str, str | list[Any]] | None = None

    def _load_model(self) -> whisper.Whisper:
        """
        Load the Whisper model.

        Returns:
            whisper.Whisper: The loaded Whisper model.

        Raises:
            ValueError: If the model ID is not set.
        """
        if self.model_id is None:
            logger.error("ValueError: Model ID is not set.")
            raise ValueError("Model ID is not set.")
        return whisper.load_model(self.model_id, device=self.device)

    def _load_audio(self, file_path: str | Path) -> NDArray[floating[Any]]:
        """
        Load audio from a file path.

        Args:
            file_path (str | Path): The path to the audio file.

        Returns:
            NDArray[floating[Any]]: The loaded audio as a NumPy array.
        """
        file = file_path if isinstance(file_path, str) else str(file_path)
        return whisper.load_audio(file=file)

    def _transcribe_audio(
        self, audio: NDArray[floating[Any]], model: whisper.Whisper
    ) -> dict[str, str | list[Any]]:
        """
        Transcribe the given audio using the specified model and optional language.

        Args:
            audio (NDArray[floating[Any]]): The audio data to transcribe.
            model (whisper.Whisper): The Whisper model to use for transcription.

        Returns:
            dict[str, str | list[Any]]: The transcription result.

        Raises:
            TypeError: If the transcription result is not a dictionary.
        """
        return whisper.transcribe(model=model, audio=audio)

    def _enrich_document(
        self,
        file_path: Path,
        text: str,
        source: str = "transcript",
        file_hash: str | None = None,
        extra_metadata: dict[str, Any] | None = None,
    ) -> Document:
        """
        Enrich a document with metadata from the image file.

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
        """
        Format seconds into HH:MM:SS format.

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
        """
        Return True if text ends with sentence-ending punctuation.

        Args:
            text (str): The text to check.

        Returns:
            bool: True if text ends with '.', '!', or '?', False otherwise.
        """
        return text.strip().endswith((".", "!", "?"))

    def _merge_segments_into_sentences(
        self, segments: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """
        Merge whisper segments into full sentences based on trailing punctuation.
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
        self, result: dict[str, Any], file_path: Path, file_hash: str | None
    ) -> list[Document]:
        """
        Create one Document per full sentence by merging segments until punctuation.
        Timing metadata reflects the merged start/end.
        """

        segments = result.get("segments") if isinstance(result, dict) else None
        if not isinstance(segments, list):
            fallback = result.get("text") if isinstance(result, dict) else ""
            if isinstance(fallback, str) and fallback.strip():
                return [self._enrich_document(file_path, fallback, file_hash=file_hash)]
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
                return [self._enrich_document(file_path, fallback, file_hash=file_hash)]
        return docs

    def load_data(self, file: str | Path, **kwargs) -> list[Document]:
        """
        Transcribe audio from a file using the Whisper model.

        Args:
            audio_file (Path): The path to the audio file.
            **kwargs: Additional keyword arguments.

        Returns:
            str: The transcribed text.
        """
        logger.info("[AudioReader] Loading audio from {}", file)
        file_path = Path(file) if not isinstance(file, Path) else file
        extra_info = kwargs.get("extra_info", {})

        file_hash = (
            extra_info.get("file_hash") if isinstance(extra_info, dict) else None
        )
        if file_hash is None:
            file_hash = compute_file_hash(file_path)

        model = self._load_model()
        audio = self._load_audio(file_path)
        self.result = self._transcribe_audio(audio, model)
        if self.result is None:
            return []
        return self._build_segment_documents(self.result, file_path, file_hash)
