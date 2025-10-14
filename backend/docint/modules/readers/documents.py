import logging
from pathlib import Path

import pymupdf4llm
from llama_index.core import Document
from llama_index.core.readers.base import BaseReader
from llama_index.readers.docling import DoclingReader

logger = logging.getLogger(__name__)


class HybridPDFReader(BaseReader):
    """
    Unified PDF reader that tries pymupdf4llm first (fast, markdown-based),
    and falls back to DoclingReader (robust, structured) if needed.

    Ensures consistent metadata for all outputs.
    """

    def __init__(self, export_type: str = "json"):
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

    def _standardize_metadata(self, path: Path, page_meta: dict | None = None) -> dict:
        """
        Returns a unified metadata dict compatible with both PyMuPDF and Docling outputs.

        Args:
            path (Path): The file path of the document.
            page_meta (dict | None): Optional page-specific metadata.
        """
        filename = path.name
        base_meta = {
            "file_path": str(path),
            "file_name": filename,
            "filename": filename,
            "file_type": "application/pdf",
            "mimetype": "application/pdf",
            "source": "document",
            "origin": {
                "filename": filename,
                "mimetype": "application/pdf",
            },
        }
        if page_meta and isinstance(page_meta, dict):
            if "page" in page_meta:
                base_meta["page_number"] = int(page_meta["page"])
            elif "page_number" in page_meta:
                base_meta["page_number"] = int(page_meta["page_number"])
        return base_meta

    def _from_pymupdf(self, path: Path) -> list[Document]:
        """
        Loads the document using pymupdf4llm and applies metadata normalization.

        Args:
            path (Path): The file path of the document.

        Raises:
            ValueError: If the PDF is empty or unreadable.
        """
        reader = pymupdf4llm.LlamaMarkdownReader()
        docs = reader.load_data(str(path))
        if not docs:
            raise ValueError("Empty or unreadable PDF")

        nonempty_docs = [
            d for d in docs if getattr(d, "text", None) and d.text.strip()
        ]
        if not nonempty_docs:
            raise ValueError("PyMuPDF produced only empty pages")

        normalized_docs = []
        for d in nonempty_docs:
            page_meta = getattr(d, "metadata", {}) or {}
            meta = self._standardize_metadata(path, page_meta)
            meta["doc_format"] = "markdown"
            normalized_docs.append(
                Document(
                    text=d.text,
                    metadata=meta,
                    id_=getattr(d, "id_", None),
                )
            )

        logger.info(
            "[HybridPDFReader] Loaded %d pages via PyMuPDF: %s",
            len(normalized_docs),
            path.name,
        )
        return normalized_docs

    def _from_docling(self, path: Path) -> list[Document]:
        """
        Loads the document using DoclingReader (fallback) and applies metadata normalization.

        Args:
            path (Path): The file path of the document.
        """
        docs = self.docling_reader.load_data(file=path)
        normalized_docs = []
        for d in docs:
            page_meta = getattr(d, "metadata", {}) or {}
            meta = self._standardize_metadata(path, page_meta)
            meta["doc_format"] = "json"
            # merge existing Docling metadata where available
            meta.update({k: v for k, v in page_meta.items() if k not in meta})
            normalized_docs.append(
                Document(
                    text=getattr(d, "text", "") or getattr(d, "text_resource", None),
                    metadata=meta,
                    id_=getattr(d, "id_", None),
                )
            )

        logger.info(
            "[HybridPDFReader] Loaded %d pages via Docling: %s",
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
        """
        path = Path(file) if not isinstance(file, Path) else file
        extra_info = kwargs.get("extra_info", {})

        try:
            docs = self._from_pymupdf(path)
        except Exception as e:
            logger.warning(
                "[HybridPDFReader] PyMuPDF failed for %s: %s → falling back to Docling",
                path.name,
                e,
            )
            try:
                docs = self._from_docling(path)
            except Exception as e2:
                logger.error(
                    "[HybridPDFReader] Docling failed for %s: %s", path.name, e2
                )
                raise RuntimeError(f"Both PyMuPDF and Docling failed to read {path}")

        # Optionally merge extra_info into metadata
        for d in docs:
            if extra_info:
                d.metadata.update(extra_info)
        return docs
