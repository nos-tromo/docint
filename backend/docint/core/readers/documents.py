from pathlib import Path

import pymupdf.layout  # noqa: F401
import pymupdf4llm
from llama_index.core import Document
from llama_index.core.readers.base import BaseReader
from llama_index.readers.docling import DoclingReader
from loguru import logger

from docint.utils.hashing import compute_file_hash, ensure_file_hash
from docint.utils.mimetype import get_mimetype


class HybridPDFReader(BaseReader):
    """
    Unified PDF reader that tries pymupdf4llm first (fast, markdown-based),
    and falls back to DoclingReader (robust, structured) if needed.

    Ensures consistent metadata for all outputs.
    """

    def __init__(self, export_type: str = "json") -> None:
        """
        Initializes the HybridPDFReader with the specified export type.

        Args:
            export_type (str): 'json' or 'markdown' for Docling fallback.
        """
        self.docling_reader = DoclingReader(
            export_type=(
                DoclingReader.ExportType.JSON
                if export_type.lower() == "json"
                else DoclingReader.ExportType.MARKDOWN
            )
        )

    def _standardize_metadata(
        self,
        file_path: Path,
        page_meta: dict | None = None,
        file_hash: str | None = None,
    ) -> dict:
        """
        Returns a unified metadata dict compatible with both PyMuPDF and Docling outputs.

        Args:
            path (Path): The file path of the document.
            page_meta (dict | None): Optional page-specific metadata.
            file_hash (str | None): Optional pre-computed file hash.

        Returns:
            dict: The standardized metadata dictionary.
        """
        filename = file_path.name

        # `file_hash` is expected to be provided by the caller (computed once per file).
        # Do not compute it here to avoid repeated expensive IO.

        mimetype = get_mimetype(file_path)
        base_meta = {
            "file_path": str(file_path),
            "file_name": filename,
            "filename": filename,
            "file_hash": file_hash,
            "file_type": mimetype,
            "mimetype": mimetype,
            "source": "document",
            "origin": {
                "filename": filename,
                "mimetype": mimetype,
            },
        }
        if page_meta and isinstance(page_meta, dict):
            if "page" in page_meta:
                base_meta["page_number"] = str(page_meta["page"])
            elif "page_number" in page_meta:
                base_meta["page_number"] = str(page_meta["page_number"])
        # Ensure the canonical file_hash is present on metadata (caller must supply it)
        ensure_file_hash(base_meta, file_hash=file_hash)
        return base_meta

    def _from_pymupdf(self, path: Path, file_hash: str | None) -> list[Document]:
        """
        Loads the document using pymupdf4llm and applies metadata normalization.

        Args:
            path (Path): The file path of the document.

        Returns:
            list[Document]: The list of loaded documents.

        Raises:
            ValueError: If the PDF is empty or unreadable.
        """
        reader = pymupdf4llm.LlamaMarkdownReader()
        docs = reader.load_data(str(path))
        if not docs:
            logger.error("ValueError: Empty or unreadable PDF")
            raise ValueError("Empty or unreadable PDF")

        nonempty_docs = [d for d in docs if getattr(d, "text", None) and d.text.strip()]
        if not nonempty_docs:
            logger.error("ValueError: PyMuPDF produced only empty pages")
            raise ValueError("PyMuPDF produced only empty pages")

        normalized_docs = []
        for d in nonempty_docs:
            page_meta = getattr(d, "metadata", {}) or {}
            meta = self._standardize_metadata(path, page_meta, file_hash=file_hash)
            meta["doc_format"] = "markdown"
            normalized_docs.append(
                Document(
                    text=d.text,
                    metadata=meta,
                    id_=getattr(d, "id_", None),
                )
            )

        logger.info(
            "[HybridPDFReader] Loaded {} pages via PyMuPDF: {}",
            len(normalized_docs),
            path.name,
        )
        return normalized_docs

    def _from_docling(self, file_path: Path, file_hash: str | None) -> list[Document]:
        """
        Loads the document using DoclingReader (fallback) and applies metadata normalization.

        Args:
            path (Path): The file path of the document.

        Returns:
            list[Document]: The list of loaded documents.
        """
        docs = self.docling_reader.load_data(file_path)
        normalized_docs = []
        for d in docs:
            page_meta = getattr(d, "metadata", {}) or {}
            meta = self._standardize_metadata(file_path, page_meta, file_hash=file_hash)
            meta["doc_format"] = "json"
            # merge existing Docling metadata where available (without overwriting)
            meta.update({k: v for k, v in page_meta.items() if k not in meta})
            normalized_docs.append(
                Document(
                    text=getattr(d, "text", "") or getattr(d, "text_resource", None),
                    metadata=meta,
                    id_=getattr(d, "id_", None),
                )
            )

        logger.info(
            "[HybridPDFReader] Loaded {} pages via Docling: {}",
            len(normalized_docs),
            file_path.name,
        )
        return normalized_docs

    def load_data(self, file: str | Path, **kwargs) -> list[Document]:
        """
        Try PyMuPDF first, then fall back to Docling on failure.

        Args:
            file (str | Path): The file path of the document.
            **kwargs: Optional arguments like extra_info passed by SimpleDirectoryReader.

        Returns:
            list[Document]: The list of loaded documents.

        Raises:
            RuntimeError: If both PyMuPDF and Docling fail to read the document.
        """
        file_path = Path(file) if not isinstance(file, Path) else file
        extra_info = kwargs.get("extra_info", {})
        file_hash = (
            extra_info.get("file_hash") if isinstance(extra_info, dict) else None
        )

        # Compute the file hash once here if not supplied by the caller.
        if file_hash is None:
            file_hash = compute_file_hash(file_path)

        try:
            docs = self._from_pymupdf(file_path, file_hash)
        except Exception as e:
            logger.warning(
                "[HybridPDFReader] PyMuPDF failed for {}: {} → falling back to Docling",
                file_path.name,
                e,
            )
            try:
                docs = self._from_docling(file_path, file_hash)
            except Exception as e2:
                logger.error(
                    "[HybridPDFReader] Docling failed for {}: {}", file_path.name, e2
                )
                logger.error(
                    "RuntimeError: Both PyMuPDF and Docling failed to read {}",
                    file_path,
                )
                raise RuntimeError(
                    f"Both PyMuPDF and Docling failed to read {file_path}"
                ) from e2

        # Optionally merge extra_info into metadata
        for d in docs:
            if extra_info:
                d.metadata.update(extra_info)
        return docs
