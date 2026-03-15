"""Shared pytest configuration and fixtures for the docint test suite."""

import sys
import types


class _MagicModule(types.ModuleType):
    """Magic module for handling file types."""

    class Magic:
        """Magic class for handling file types."""

        def __init__(self, mime: bool = True) -> None:
            """Initialize the Magic class.

            Args:
                mime (bool, optional): Whether to use MIME types. Defaults to True.
            """
            self.mime = mime

        def from_file(self, path: str) -> str:
            """Get the MIME type of a file.

            Args:
                path (str): The path to the file.

            Returns:
                str: The MIME type of the file.
            """
            return "application/octet-stream"


def _install_magic_stub() -> None:
    """Install a stub for the magic module."""
    sys.modules.setdefault("magic", _MagicModule("magic"))


def _install_whisper_stub() -> None:
    """Install a stub for the whisper module."""
    module = types.ModuleType("whisper")

    class Whisper:
        def detect_language(self, mel):
            return None, {"en": 1.0}

    def load_model(model_id: str, device: str | None = None):
        return Whisper()

    def load_audio(file: str):
        return f"audio:{file}"

    def transcribe(model, audio, **kwargs):
        return {"text": "transcribed"}

    def pad_or_trim(audio):
        return audio

    class _Mel:
        def to(self, device: str | None):
            return self

    def log_mel_spectrogram(audio):
        return _Mel()

    module.Whisper = Whisper  # type: ignore[attr-defined]
    module.load_model = load_model  # type: ignore[attr-defined]
    module.load_audio = load_audio  # type: ignore[attr-defined]
    module.transcribe = transcribe  # type: ignore[attr-defined]
    module.pad_or_trim = pad_or_trim  # type: ignore[attr-defined]
    module.log_mel_spectrogram = log_mel_spectrogram  # type: ignore[attr-defined]
    sys.modules.setdefault("whisper", module)


def pytest_configure() -> None:
    """Configure pytest by installing necessary stubs."""
    _install_magic_stub()
    _install_whisper_stub()
