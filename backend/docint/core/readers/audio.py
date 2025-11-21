import os
from pathlib import Path
from typing import Any

import whisper
from dotenv import load_dotenv
from llama_index.core import Document
from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import MediaResource
from loguru import logger
from numpy import floating
from numpy.typing import NDArray

from docint.utils.hashing import compute_file_hash, ensure_file_hash
from docint.utils.mimetype import get_mimetype


# --- Environment variables ---
load_dotenv()
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "turbo")


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
        self.model_id: str = WHISPER_MODEL
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
        return whisper.load_model(self.model_id)

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

        metadata = {
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

        return Document(
            text_resource=MediaResource(text=text, mimetype=mimetype),
            metadata=metadata,
        )

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
        text = self.result.get("text", "")
        if self.result is None or not isinstance(text, str):
            return []
        return [self._enrich_document(file_path, text, file_hash=file_hash)]
