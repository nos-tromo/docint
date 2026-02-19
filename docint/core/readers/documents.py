from pathlib import Path

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    AcceleratorDevice,
    AcceleratorOptions,
    PdfPipelineOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from llama_index.core import Document
from llama_index.core.readers.base import BaseReader
from llama_index.readers.docling import DoclingReader
from loguru import logger

from docint.utils.env_cfg import load_ingestion_env
from docint.utils.hashing import compute_file_hash, ensure_file_hash
from docint.utils.mimetype import get_mimetype


class CustomDoclingReader(BaseReader):
    """
    PDF reader that uses DoclingReader (robust, structured).

    Ensures consistent metadata for all outputs.
    """

    def __init__(self, export_type: str = "json", device: str = "cpu") -> None:
        """
        Initializes the CustomDoclingReader with the specified export type.

        Args:
            export_type (str): 'json' or 'markdown'.
            device (str): 'cpu', 'cuda', or 'mps'.
        """
        # Map device string to AcceleratorDevice
        accelerator = AcceleratorDevice.CPU
        if device == "cuda":
            accelerator = AcceleratorDevice.CUDA
        elif device == "mps":
            accelerator = AcceleratorDevice.MPS

        # Configure pipeline with device
        num_threads = load_ingestion_env().docling_accelerator_num_threads
        acc_opts = AcceleratorOptions(num_threads=num_threads, device=accelerator)
        pipeline_opts = PdfPipelineOptions(
            accelerator_options=acc_opts,
            do_ocr=True,
            do_table_structure=True,
        )

        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_opts)
            }
        )

        self.docling_reader = DoclingReader(
            export_type=(
                DoclingReader.ExportType.JSON
                if export_type.lower() == "json"
                else DoclingReader.ExportType.MARKDOWN
            ),
            doc_converter=converter,
        )

    def _standardize_metadata(
        self,
        file_path: Path,
        page_meta: dict | None = None,
        file_hash: str | None = None,
    ) -> dict:
        """
        Returns a unified metadata dict compatible with Docling outputs.

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

    def load_data(self, file: str | Path, **kwargs) -> list[Document]:
        """
        Load data using Docling.

        Args:
            file (str | Path): The file path of the document.
            **kwargs: Optional arguments like extra_info passed by SimpleDirectoryReader.

        Returns:
            list[Document]: The list of loaded documents.

        Raises:
            RuntimeError: If Docling fails to read the document.
        """
        file_path = Path(file) if not isinstance(file, Path) else file
        extra_info = kwargs.get("extra_info", {})
        file_hash = (
            extra_info.get("file_hash") if isinstance(extra_info, dict) else None
        )

        # Compute the file hash once here if not supplied by the caller.
        if file_hash is None:
            file_hash = compute_file_hash(file_path)

        docs = []
        try:
            docs = self.docling_reader.load_data(file_path)
        except Exception as e:
            # Attempt PDF repair if Docling fails
            if get_mimetype(file_path) == "application/pdf":
                logger.warning(
                    "[CustomDoclingReader] Docling failed for {}, attempting repair with pypdf...",
                    file_path.name,
                )
                try:
                    import tempfile

                    import pypdf

                    with tempfile.NamedTemporaryFile(
                        suffix=".pdf", delete=False
                    ) as tmp:
                        tmp_path = Path(tmp.name)

                    try:
                        # Repair: Read with pypdf and write to new file
                        reader = pypdf.PdfReader(file_path)
                        writer = pypdf.PdfWriter()
                        for page in reader.pages:
                            writer.add_page(page)
                        writer.write(tmp_path)

                        # Retry Docling with repaired file
                        docs = self.docling_reader.load_data(tmp_path)
                        logger.info(
                            "[CustomDoclingReader] Successfully loaded repaired PDF: {}",
                            file_path.name,
                        )
                    finally:
                        if tmp_path.exists():
                            tmp_path.unlink()
                except Exception as repair_error:
                    logger.error(
                        "[CustomDoclingReader] Repair failed for {}: {}",
                        file_path.name,
                        repair_error,
                    )
                    raise RuntimeError(f"Docling failed to read {file_path}") from e
            else:
                logger.error(
                    "[CustomDoclingReader] Docling failed for {}: {}", file_path.name, e
                )
                raise RuntimeError(f"Docling failed to read {file_path}") from e

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

        # Optionally merge extra_info into metadata
        for d in normalized_docs:
            if extra_info:
                d.metadata.update(extra_info)

        logger.info(
            "[CustomDoclingReader] Loaded {} pages via Docling: {}",
            len(normalized_docs),
            file_path.name,
        )
        return normalized_docs
