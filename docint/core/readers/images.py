from dataclasses import dataclass, field
from pathlib import Path

from llama_index.core import Document
from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import MediaResource
from loguru import logger

from docint.core.ingest.images_service import (
    ImageAsset,
    ImageIngestionService,
    IngestContext,
)
from docint.utils.hashing import compute_file_hash, ensure_file_hash
from docint.utils.mimetype import get_mimetype


@dataclass
class ImageReader(BaseReader):
    """Image reader that routes image ingestion through the shared image service."""

    image_ingestion_service: ImageIngestionService = field(
        default_factory=ImageIngestionService
    )
    source_collection: str | None = None

    def _enrich_document(
        self,
        file_path: Path,
        text: str,
        source: str = "image",
        file_hash: str | None = None,
        image_metadata: dict[str, object] | None = None,
    ) -> Document:
        """Enrich a document with metadata from the image file.

        Args:
            file_path (Path): The path to the image file.
            text (str): The text content extracted from the image.
            source (str, optional): The source type. Defaults to "image".
            file_hash (str | None, optional): Pre-computed file hash. Defaults to None.
            image_metadata (dict[str, object] | None, optional): Additional metadata from image ingestion.
                Defaults to None.

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
        metadata: dict[str, object] = {
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
        if image_metadata:
            metadata.update(image_metadata)
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
        """Load and process image data through the shared image ingestion service.

        Args:
            file (str | Path): The path to the image file.
            **kwargs: Additional keyword arguments.

        Returns:
            list[Document]: A list containing a single Document object with the processed image data.
        """
        logger.info("[ImageReader] Loading image from {}", file)
        file_path = Path(file) if not isinstance(file, Path) else file
        extra_info = kwargs.get("extra_info", {})

        file_hash = (
            extra_info.get("file_hash") if isinstance(extra_info, dict) else None
        )
        if file_hash is None:
            file_hash = compute_file_hash(file_path)

        image_bytes = file_path.read_bytes()
        mime_type = get_mimetype(file_path)

        record = self.image_ingestion_service.ingest_image(
            ImageAsset(
                source_type="standalone",
                image_path=file_path,
                image_bytes=image_bytes,
                source_path=str(file_path),
                mime_type=mime_type,
            ),
            context=IngestContext(source_collection=self.source_collection),
        )
        text = record.llm_description.strip()
        if record.llm_tags:
            text = (
                f"{text}\n\nTags: {', '.join(record.llm_tags)}"
                if text
                else f"Tags: {', '.join(record.llm_tags)}"
            )
        if not text:
            text = f"Image file: {file_path.name}"

        image_meta: dict[str, object] = {}
        if record.image_id:
            image_meta["image_id"] = record.image_id
        if record.point_id:
            image_meta["image_point_id"] = record.point_id
        if record.llm_description:
            image_meta["llm_description"] = record.llm_description
        if record.llm_tags:
            image_meta["llm_tags"] = record.llm_tags
        if record.error:
            image_meta["image_ingest_error"] = record.error
        if record.status:
            image_meta["image_ingest_status"] = record.status

        return [
            self._enrich_document(
                file_path,
                text,
                file_hash=file_hash,
                image_metadata=image_meta,
            )
        ]
