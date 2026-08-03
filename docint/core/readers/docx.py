"""Word ``.docx`` reader that extracts document text via docling.

The ingestion pipeline previously had no extractor registered for ``.docx``, so
``SimpleDirectoryReader`` fell back to its generic UTF-8 text path and embedded
the raw ZIP container verbatim (``PK`` headers, ``[Content_Types].xml``,
``word/document.xml`` …) — a ``.docx`` *is* a ZIP archive. This reader converts
the file with docling's declarative :class:`MsWordDocumentBackend` and emits a
single ``Document`` whose text is a compact, serialized Docling document, so it
flows through the already-wired :class:`DoclingNodeParser` exactly like a PDF.

The backend is driven directly (via :class:`InputDocument`) rather than through
``docling.document_converter.DocumentConverter``: the converter module eagerly
imports the model-backed PDF pipeline (``docling_ibm_models`` → torch), which
docint deliberately does not ship — all ML inference is remote. The DOCX
backend is pure XML parsing (python-docx/lxml): conversion loads or downloads
nothing, keeping the airgapped, CPU-only deployment intact.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from docling.backend.abstract_backend import DeclarativeDocumentBackend
from docling.backend.msword_backend import MsWordDocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import InputDocument
from docling_core.types.doc import DoclingDocument
from llama_index.core import Document
from llama_index.core.readers.base import BaseReader
from loguru import logger
from typing_extensions import override

from docint.utils.hashing import ensure_file_hash
from docint.utils.mimetype import get_mimetype


@dataclass
class DocxReader(BaseReader):
    """Read a ``.docx`` file and emit a single Docling-JSON :class:`Document`.

    Each file gets its own :class:`MsWordDocumentBackend` (declarative, no
    models) built through :class:`InputDocument`, which owns validation and
    hashing. The emitted text is
    ``json.dumps(DoclingDocument.export_to_dict())`` with compact separators so
    the pipeline's ``basic_clean`` pass cannot perturb structural whitespace and
    break JSON validity. If JSON serialization fails the reader degrades to
    Markdown; if conversion itself fails it logs and yields nothing — it never
    falls back to the raw bytes that caused the original bug.
    """

    def _convert(self, file_path: Path) -> DoclingDocument:
        """Convert one ``.docx`` file to a :class:`DoclingDocument`.

        Mirrors what ``SimplePipeline`` does for declarative backends, without
        importing ``DocumentConverter`` (whose module pulls in the model-backed
        PDF pipeline).

        Args:
            file_path: Path to the ``.docx`` file on disk.

        Returns:
            DoclingDocument: The parsed document.

        Raises:
            ValueError: If docling rejects the file as an invalid docx input.
        """
        in_doc = InputDocument(
            path_or_stream=file_path,
            format=InputFormat.DOCX,
            backend=MsWordDocumentBackend,
        )
        if not in_doc.valid:
            raise ValueError(f"docling rejected {file_path.name} as an invalid docx input")
        return cast(DeclarativeDocumentBackend, in_doc._backend).convert()

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
            "source": "document",
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
        """Yield exactly one ``Document`` containing the converted docx body.

        Args:
            file: Path to the ``.docx`` file on disk.
            **kwargs: Forwarded by the pipeline; ``extra_info`` (when present)
                carries the pre-computed file hash and canonical file metadata.

        Yields:
            Document: The Docling-JSON body (or Markdown fallback) wrapped with
            ingestion metadata. Nothing is yielded if conversion fails.
        """
        file_path = file if isinstance(file, Path) else Path(file)
        extra_info = kwargs.get("extra_info")
        extra_info = extra_info if isinstance(extra_info, dict) else None

        try:
            document = self._convert(file_path)
        except Exception as exc:
            logger.warning("[DocxReader] docling conversion failed on {}: {}; skipping", file_path, exc)
            return

        try:
            text = json.dumps(document.export_to_dict(), separators=(",", ":"), ensure_ascii=False)
        except Exception as exc:
            logger.warning(
                "[DocxReader] Docling-JSON export failed on {}: {}; falling back to markdown", file_path, exc
            )
            try:
                text = document.export_to_markdown()
            except Exception as exc2:
                logger.warning("[DocxReader] markdown export failed on {}: {}; skipping", file_path, exc2)
                return

        yield self._build_document(file_path, text, extra_info)

    @override
    def load_data(self, file: str | Path, **kwargs: Any) -> list[Document]:  # pyrefly: ignore[bad-override]  # llama-index BaseReader.load_data; runtime-compatible
        """Eager-list shim over :meth:`iter_documents` for legacy callers.

        Args:
            file: Path to the ``.docx`` file on disk.
            **kwargs: Forwarded to :meth:`iter_documents`.

        Returns:
            list[Document]: Single-element list, or empty if conversion failed.
        """
        return list(self.iter_documents(file, **kwargs))
