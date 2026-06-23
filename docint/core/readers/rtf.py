r"""Rich Text Format reader that strips RTF markup to plain text.

The ingestion pipeline previously had no extractor registered for ``.rtf``, so
SimpleDirectoryReader fell back to its generic UTF-8 text path — the raw RTF
control words (``\\rtf1``, ``\\fonttbl`` …) were embedded verbatim. This
reader runs ``striprtf`` over the file body so only the human-readable text is
chunked and embedded.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llama_index.core import Document
from llama_index.core.readers.base import BaseReader
from loguru import logger
from striprtf.striprtf import rtf_to_text
from typing_extensions import override

from docint.utils.clean_text import basic_clean
from docint.utils.hashing import ensure_file_hash
from docint.utils.mimetype import get_mimetype


@dataclass
class RTFReader(BaseReader):
    r"""Read an RTF file and emit a single plain-text :class:`Document`.

    The reader decodes the file as Latin-1 (a 1:1 byte→codepoint mapping that
    never raises on a binary stream) and lets ``striprtf`` handle the
    in-document codepage signalled by ``\\ansicpgNNNN``. The resulting text is
    normalized via :func:`basic_clean`.
    """

    def _build_document(
        self,
        file_path: Path,
        text: str,
        extra_info: dict[str, Any] | None,
    ) -> Document:
        filename = file_path.name
        mimetype = get_mimetype(file_path)
        metadata: dict[str, Any] = {
            "file_path": str(file_path),
            "file_name": filename,
            "filename": filename,
            "file_type": mimetype,
            "mimetype": mimetype,
            "source": "text",
        }
        if extra_info:
            metadata.update(extra_info)
        file_hash = metadata.get("file_hash") if isinstance(metadata.get("file_hash"), str) else None
        ensure_file_hash(
            metadata,
            path=file_path if not file_hash else None,
            file_hash=file_hash,
        )
        return Document(text=text, metadata=metadata)

    def iter_documents(self, file: str | Path, **kwargs: Any) -> Iterator[Document]:
        """Yield exactly one ``Document`` containing the stripped RTF body.

        Args:
            file: Path to the RTF file on disk.
            **kwargs: Forwarded by the pipeline; ``extra_info`` (when present)
                carries the pre-computed file hash and canonical file metadata.

        Yields:
            Document: The plain-text body wrapped with ingestion metadata.
        """
        file_path = file if isinstance(file, Path) else Path(file)
        extra_info = kwargs.get("extra_info")
        extra_info = extra_info if isinstance(extra_info, dict) else None

        raw = file_path.read_text(encoding="latin-1")
        try:
            stripped = rtf_to_text(raw, errors="ignore")  # type: ignore[no-untyped-call]
        except Exception as exc:
            logger.warning("[RTFReader] striprtf failed on {}: {}; falling back to raw body", file_path, exc)
            stripped = raw

        text = basic_clean(stripped)
        yield self._build_document(file_path, text, extra_info)

    @override
    def load_data(self, file: str | Path, **kwargs: Any) -> list[Document]:  # pyrefly: ignore[bad-override]  # llama-index BaseReader.load_data; runtime-compatible
        """Eager-list shim over :meth:`iter_documents` for legacy callers.

        Args:
            file: Path to the RTF file on disk.
            **kwargs: Forwarded to :meth:`iter_documents`.

        Returns:
            list[Document]: Single-element list with the stripped RTF body.
        """
        return list(self.iter_documents(file, **kwargs))
