"""Shared image ingestion service for standalone and extracted images."""

from __future__ import annotations

import base64
import json
import os
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from typing import Any, Protocol, TypeVar, cast

import torch
from llama_index.core.schema import ImageNode
from llama_index.vector_stores.qdrant import QdrantVectorStore
from loguru import logger
from PIL import Image
from qdrant_client import QdrantClient, models
from transformers import AutoProcessor, CLIPModel

from docint.utils.env_cfg import (
    ImageIngestionConfig,
    ModelConfig,
    load_host_env,
    load_image_ingestion_config,
    load_model_env,
    load_path_env,
    resolve_hf_cache_path,
)
from docint.utils.mimetype import get_mimetype
from docint.utils.openai_cfg import OpenAIPipeline

T = TypeVar("T")


@dataclass(frozen=True)
class IngestContext:
    """Context attached to an image ingestion call."""

    source_collection: str | None = None


@dataclass(frozen=True)
class ImageAsset:
    """Image asset input accepted by the shared ingestion service."""

    source_type: str
    image_path: Path | None = None
    image_bytes: bytes | None = None
    source_doc_id: str | None = None
    source_path: str | None = None
    page_number: int | None = None
    bbox: dict[str, float] | None = None
    mime_type: str | None = None
    extra_metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_path(
        cls,
        *,
        path: Path,
        source_type: str,
        source_doc_id: str | None = None,
        source_path: str | None = None,
        page_number: int | None = None,
        bbox: dict[str, float] | None = None,
        extra_metadata: dict[str, Any] | None = None,
    ) -> ImageAsset:
        """Construct an image asset from a file path.

        Args:
            path (Path): The file path of the image.
            source_type (str): The source type of the image.
            source_doc_id (str | None): The ID of the source document.
            source_path (str | None): The path of the source document.
            page_number (int | None): The page number of the image in the source document.
            bbox (dict[str, float] | None): The bounding box of the image in the source document.
            extra_metadata (dict[str, Any] | None): Additional metadata for the image.

        Returns:
            ImageAsset: The constructed image asset.
        """
        return cls(
            source_type=source_type,
            image_path=path,
            source_doc_id=source_doc_id,
            source_path=source_path or str(path),
            page_number=page_number,
            bbox=bbox,
            mime_type=get_mimetype(path),
            extra_metadata=extra_metadata or {},
        )


@dataclass(frozen=True)
class StoredImageRecord:
    """Result payload returned by shared image ingestion."""

    point_id: str | None
    image_id: str | None
    status: str
    payload: dict[str, Any]
    llm_description: str
    llm_tags: list[str]
    error: str | None = None


class ImageEmbeddingBackend(Protocol):
    """Protocol for image embedding backends."""

    @property
    def dimension(self) -> int:
        """Return embedding vector length."""

    def embed(self, image_bytes: bytes) -> list[float]:
        """Return a dense embedding for image bytes."""

    def embed_text(self, text: str) -> list[float]:
        """Return a dense embedding for text queries."""


@dataclass
class CLIPImageEmbeddingBackend:
    """Image embedding backend based on CLIP from ``transformers``."""

    image_embed_model_id: str
    cache_dir: Path
    device: str = field(default="cpu")
    model: Any = field(init=False)
    processor: Any = field(init=False)
    _dimension: int = field(init=False)

    def __post_init__(self) -> None:
        resolved = resolve_hf_cache_path(
            cache_dir=self.cache_dir, repo_id=self.image_embed_model_id
        )
        resolved_model = str(resolved) if resolved else self.image_embed_model_id
        local_only = os.getenv("HF_HUB_OFFLINE", "0") == "1"
        self.processor = AutoProcessor.from_pretrained(
            pretrained_model_name_or_path=resolved_model,
            cache_dir=str(self.cache_dir),
            local_files_only=local_only,
        )
        self.model = CLIPModel.from_pretrained(
            pretrained_model_name_or_path=resolved_model,
            cache_dir=str(self.cache_dir),
            local_files_only=local_only,
        )
        self.model.eval()
        self.model.to(self.device)
        self._dimension = int(self.model.config.projection_dim)

    @property
    def dimension(self) -> int:
        """Return CLIP projection dimensionality."""
        return self._dimension

    def embed(self, image_bytes: bytes) -> list[float]:
        """Embed image bytes with CLIP image tower.

        Args:
            image_bytes: Raw bytes of the image to embed.

        Returns:
            A list of floats representing the normalized image embedding vector.
        """
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            features = self.model.get_image_features(**inputs)
            features = features / features.norm(dim=-1, keepdim=True)
        return features[0].detach().cpu().tolist()

    def embed_text(self, text: str) -> list[float]:
        """Embed query text with CLIP text tower.

        Args:
            text: The input text to embed.

        Returns:
            A list of floats representing the normalized text embedding vector.
        """
        inputs = self.processor(
            text=[text], return_tensors="pt", padding=True, truncation=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            features = self.model.get_text_features(**inputs)
            features = features / features.norm(dim=-1, keepdim=True)
        return features[0].detach().cpu().tolist()


class ImageTaggingBackend(Protocol):
    """Protocol for LLM-based image caption/tag generation."""

    def describe_and_tag(
        self, image_bytes: bytes, mime_type: str
    ) -> tuple[str, list[str]]:
        """Return ``(description, tags)`` for the image."""


@dataclass
class VisionJSONTagger:
    """OpenAI-compatible vision tagger that returns structured description/tags."""

    pipeline: OpenAIPipeline = field(default_factory=OpenAIPipeline)
    prompt_template: str = field(
        default=(
            "Return strict JSON with keys description and tags.\n"
            "description: concise factual caption in <= 8 sentences.\n"
            "tags: 5-20 concise tags, each 1-3 words.\n"
            "Do not include markdown or prose outside JSON."
        )
    )

    @staticmethod
    def parse_tag_payload(raw: str) -> tuple[str, list[str]]:
        """Parse a raw vision response into ``(description, tags)``.

        Args:
            raw: Raw model output string.

        Returns:
            A tuple of cleaned description and tags.
        """
        payload: dict[str, Any] = {}
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                payload = parsed
        except Exception:
            start = raw.find("{")
            end = raw.rfind("}")
            if start != -1 and end > start:
                try:
                    parsed = json.loads(raw[start : end + 1])
                    if isinstance(parsed, dict):
                        payload = parsed
                except Exception:
                    payload = {}

        description = str(payload.get("description") or "").strip()
        tags_raw = payload.get("tags")
        tags: list[str] = []
        if isinstance(tags_raw, list):
            for tag in tags_raw:
                tag_str = str(tag).strip()
                if not tag_str:
                    continue
                # Constrain tags to concise 1-3 word phrases.
                if len(tag_str.split()) > 3:
                    continue
                tags.append(tag_str)

        # Fallback to raw content when JSON parse fails.
        if not description and raw.strip():
            description = raw.strip()

        deduped: list[str] = []
        seen: set[str] = set()
        for tag in tags:
            key = tag.lower()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(tag)
        return description, deduped[:20]

    # MIME types commonly supported by vision APIs.
    _SUPPORTED_MIME_TYPES: frozenset[str] = frozenset(
        {"image/jpeg", "image/png", "image/gif", "image/webp"}
    )

    def describe_and_tag(
        self, image_bytes: bytes, mime_type: str
    ) -> tuple[str, list[str]]:
        """Generate image description/tags via OpenAI-compatible vision API.

        If the image is not in a format natively supported by most vision
        models (JPEG, PNG, GIF, WebP), it is transparently converted to PNG
        before being sent.

        Args:
            image_bytes: Raw bytes of the image to describe and tag.
            mime_type: The MIME type of the image (e.g., "image/png").

        Returns:
            A tuple of (description, tags) generated for the image.
        """
        image_bytes, mime_type = self._normalize_image(image_bytes, mime_type)
        encoded = base64.b64encode(image_bytes).decode("utf-8")
        raw = self.pipeline.call_vision(
            prompt=self.prompt_template,
            img_base64=encoded,
            mime_type=mime_type,
        )
        return self.parse_tag_payload(raw)

    def _normalize_image(self, image_bytes: bytes, mime_type: str) -> tuple[bytes, str]:
        """Ensure image bytes are in a vision-API-friendly format.

        If *mime_type* is not one of the commonly supported types (JPEG, PNG,
        GIF, WebP), the image is re-encoded as PNG so the vision model can
        process it.

        Args:
            image_bytes: Raw image data.
            mime_type: Original MIME type.

        Returns:
            A tuple of ``(possibly converted bytes, mime_type)``.
        """
        if mime_type in self._SUPPORTED_MIME_TYPES:
            return image_bytes, mime_type

        try:
            img = Image.open(BytesIO(image_bytes))
            buf = BytesIO()
            img.convert("RGB").save(buf, format="PNG")
            logger.debug(
                "Converted image from {} to image/png for vision API",
                mime_type,
            )
            return buf.getvalue(), "image/png"
        except Exception:
            logger.warning(
                "Cannot convert image (mime_type={}); sending as-is",
                mime_type,
            )
            return image_bytes, mime_type


@dataclass
class ImageIngestionService:
    """Store image embeddings and metadata into a dedicated Qdrant collection."""

    device: str = field(default="cpu")
    img_ingestion_config: ImageIngestionConfig = field(
        default_factory=load_image_ingestion_config
    )
    model_config: ModelConfig = field(default_factory=load_model_env)
    qdrant_client: QdrantClient | None = None
    vector_store: QdrantVectorStore | None = None
    embedding_backend: ImageEmbeddingBackend | None = None
    tagging_backend: ImageTaggingBackend | None = None
    _vector_stores: dict[str, QdrantVectorStore] = field(
        default_factory=dict, init=False, repr=False
    )
    _embedding_backend_error: str | None = field(default=None, init=False, repr=False)
    _tagging_backend_error: str | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.qdrant_client is None:
            self.qdrant_client = QdrantClient(url=load_host_env().qdrant_host)

    def _resolve_collection_name(self, source_collection: str | None = None) -> str:
        """Resolve target image collection name from configured template.

        Args:
            source_collection: The source collection name to interpolate into the template.

        Returns:
            The resolved collection name for storing the image.
        """
        template = (self.img_ingestion_config.collection_name or "").strip()
        if not template:
            template = "{collection}_images"

        if "{collection}" in template:
            if source_collection and source_collection.strip():
                return template.format(collection=source_collection.strip())
            raise ValueError(
                "IMAGE_QDRANT_COLLECTION uses '{collection}' but source collection is missing."
            )
        return template

    def _get_vector_store(self, collection_name: str) -> QdrantVectorStore:
        """Resolve the image vector store lazily for a collection.

        Args:
            collection_name: The name of the collection for which to get the vector store.

        Returns:
            The QdrantVectorStore instance for the specified collection.
        """
        if self.vector_store is not None:
            return self.vector_store
        cached = self._vector_stores.get(collection_name)
        if cached is not None:
            return cached
        if self.qdrant_client is None:
            raise RuntimeError("Qdrant client is not initialized.")
        vector_store = QdrantVectorStore(
            collection_name=collection_name,
            client=self.qdrant_client,
            enable_hybrid=False,
            dense_vector_name=self.img_ingestion_config.vector_name,
        )
        self._vector_stores[collection_name] = vector_store
        return vector_store

    def _get_embedding_backend(self) -> ImageEmbeddingBackend | None:
        """Resolve the configured embedding backend lazily.

        Returns:
            The ImageEmbeddingBackend instance if available and enabled, otherwise None.
        """
        if self.embedding_backend is not None:
            return self.embedding_backend
        if self._embedding_backend_error is not None:
            return None
        if not self.img_ingestion_config.embedding_enabled:
            return None
        cache_dir = load_path_env().hf_hub_cache
        try:
            self.embedding_backend = CLIPImageEmbeddingBackend(
                image_embed_model_id=self.model_config.image_embed_model,
                cache_dir=cache_dir,
                device=self.device,
            )
        except Exception as exc:
            self._embedding_backend_error = str(exc)
            if self.img_ingestion_config.fail_on_embedding_error:
                raise
            logger.warning(
                "Image embedding backend unavailable (continuing without image vectors): {}",
                exc,
            )
            return None
        return self.embedding_backend

    def _get_tagging_backend(self) -> ImageTaggingBackend | None:
        """Resolve the configured image-tagging backend lazily.

        Returns:
            The ImageTaggingBackend instance if available and enabled, otherwise None.
        """
        if self.tagging_backend is not None:
            return self.tagging_backend
        if self._tagging_backend_error is not None:
            return None
        if not self.img_ingestion_config.tagging_enabled:
            return None
        try:
            self.tagging_backend = VisionJSONTagger()
        except Exception as exc:
            self._tagging_backend_error = str(exc)
            if self.img_ingestion_config.fail_on_tagging_error:
                raise
            logger.warning(
                "Image tagging backend unavailable (continuing without tags): {}",
                exc,
            )
            return None
        return self.tagging_backend

    @staticmethod
    def _hash_image_bytes(image_bytes: bytes) -> str:
        """Return SHA-256 digest for image bytes.

        Args:
            image_bytes: The raw bytes of the image to hash.

        Returns:
            A hexadecimal string representing the SHA-256 hash of the image bytes.
        """
        return sha256(image_bytes).hexdigest()

    @staticmethod
    def _point_id_from_image_id(image_id: str) -> str:
        """Return deterministic UUID point id derived from ``image_id``.

        Args:
            image_id: The unique identifier for the image (e.g., a hash).

        Returns:
            A string representing the UUID derived from the image ID, suitable for use as a Qdrant point ID.
        """
        return str(uuid.uuid5(uuid.NAMESPACE_URL, f"docint:image:{image_id}"))

    def _resolve_image_bytes(self, asset: ImageAsset) -> tuple[bytes, str]:
        """Resolve image bytes and MIME type from an asset.

        Args:
            asset: The ImageAsset containing either raw bytes or a file path to the image.

        Returns:
            A tuple of (image_bytes, mime_type) extracted from the asset.
        """
        if asset.image_bytes is not None:
            if asset.mime_type:
                return asset.image_bytes, asset.mime_type
            return asset.image_bytes, "image/png"

        if asset.image_path and asset.image_path.is_file():
            image_bytes = asset.image_path.read_bytes()
            mime_type = asset.mime_type or get_mimetype(asset.image_path)
            return image_bytes, mime_type

        raise ValueError("Image asset does not include readable bytes or path.")

    @staticmethod
    def _image_size(image_bytes: bytes) -> tuple[int | None, int | None]:
        """Return ``(width, height)`` for image bytes, best effort.

        Args:
            image_bytes: The raw bytes of the image to analyze.

        Returns:
            A tuple of (width, height) in pixels if determinable, otherwise (None, None).
        """
        try:
            with Image.open(BytesIO(image_bytes)) as image:
                return image.width, image.height
        except Exception:
            return None, None

    def _existing_by_image_id(
        self,
        image_id: str,
        *,
        collection_name: str,
    ) -> tuple[str, dict[str, Any]] | None:
        """Fetch an existing point by ``image_id`` from the image collection.

        Args:
            image_id: The unique identifier for the image (e.g., a hash).
            collection_name: The name of the collection to search for the image.

        Returns:
            A tuple of (point_id, payload) if the image exists, otherwise None.
        """
        if self.qdrant_client is None:
            return None
        try:
            points, _ = self.qdrant_client.scroll(
                collection_name=collection_name,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="image_id",
                            match=models.MatchValue(value=image_id),
                        )
                    ]
                ),
                limit=1,
                with_payload=True,
                with_vectors=False,
            )
            if not points:
                return None
            point = points[0]
            point_id = str(getattr(point, "id", ""))
            payload = dict(getattr(point, "payload", {}) or {})
            return point_id, payload
        except Exception:
            return None

    @staticmethod
    def _occurrence(asset: ImageAsset, context: IngestContext) -> dict[str, Any]:
        """Build a source occurrence payload for deduped images.

        Args:
            asset: The ImageAsset for which to build the occurrence.
            context: The IngestContext providing additional context for the occurrence.

        Returns:
            A dictionary representing the occurrence metadata to append to existing points
            when an image is deduplicated.
        """
        return {
            "source_type": asset.source_type,
            "source_collection": context.source_collection,
            "source_doc_id": asset.source_doc_id,
            "source_path": asset.source_path
            or (str(asset.image_path) if asset.image_path else None),
            "page_number": asset.page_number,
            "bbox": asset.bbox,
        }

    def _append_occurrence(
        self,
        *,
        collection_name: str,
        point_id: str,
        payload: dict[str, Any],
        occurrence: dict[str, Any],
    ) -> None:
        """Append source occurrence metadata to an existing point payload.
        An "occurrences" list is maintained in the point payload to track all source
        references for a deduplicated image.

        Args:
            collection_name: The name of the collection containing the point.
            point_id: The ID of the point to update.
            payload: The existing payload of the point, which may contain an "occurrences" list.
            occurrence: The new occurrence metadata to append to the point's "occurrences" list.
        """
        if self.qdrant_client is None:
            return
        occurrences = payload.get("occurrences")
        if not isinstance(occurrences, list):
            occurrences = []
        if occurrence not in occurrences:
            occurrences.append(occurrence)
            self.qdrant_client.set_payload(
                collection_name=collection_name,
                payload={"occurrences": occurrences},
                points=[point_id],
            )

    def _ensure_collection(self, *, collection_name: str, vector_dim: int) -> None:
        """Create or validate the image collection and vector schema.

        Args:
            collection_name: The name of the collection to create or validate.
            vector_dim: The expected dimensionality of the image embedding vectors.
        """
        if self.qdrant_client is None:
            return
        try:
            info = self.qdrant_client.get_collection(collection_name)
        except Exception:
            self.qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config={
                    self.img_ingestion_config.vector_name: models.VectorParams(
                        size=vector_dim,
                        distance=models.Distance.COSINE,
                    )
                },
            )
            logger.info(
                "Created image collection '{}' with vector '{}'",
                collection_name,
                self.img_ingestion_config.vector_name,
            )
            return

        vectors = info.config.params.vectors
        named_config = None
        if isinstance(vectors, dict):
            named_config = vectors.get(self.img_ingestion_config.vector_name)
        elif hasattr(vectors, "get"):
            named_config = cast(Any, vectors).get(self.img_ingestion_config.vector_name)

        if named_config is None:
            raise ValueError(
                "Image collection '{}' exists but vector '{}' is missing.".format(
                    collection_name,
                    self.img_ingestion_config.vector_name,
                )
            )
        if int(named_config.size) != int(vector_dim):
            raise ValueError(
                "Image collection '{}' vector '{}' dimension mismatch: "
                "expected {}, found {}".format(
                    collection_name,
                    self.img_ingestion_config.vector_name,
                    vector_dim,
                    named_config.size,
                )
            )

    @staticmethod
    def _run_with_retries(
        fn: Callable[[], T],
        *,
        attempts: int = 2,
    ) -> T:
        """Run backend inference with bounded retries.

        Args:
            fn: A callable that performs the inference and returns a result.
            attempts: The maximum number of attempts to run the callable before giving up.

        Returns:
            The result returned by the callable if successful.
        """
        last_error: Exception | None = None
        for attempt in range(1, attempts + 1):
            try:
                return fn()
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "Image inference attempt {}/{} failed: {}", attempt, attempts, exc
                )
        if last_error is None:
            raise RuntimeError("Image inference failed without error details.")
        raise last_error

    def ingest_image(
        self,
        asset: ImageAsset,
        *,
        context: IngestContext,
    ) -> StoredImageRecord:
        """Ingest one image into the shared image vector collection.

        Args:
            asset: Image asset from standalone or extracted document source.
            context: Collection-specific ingestion context.

        Returns:
            A ``StoredImageRecord`` describing the result.
        """
        if not self.img_ingestion_config.enabled:
            return StoredImageRecord(
                point_id=None,
                image_id=None,
                status="disabled",
                payload={},
                llm_description="",
                llm_tags=[],
            )

        try:
            image_bytes, mime_type = self._resolve_image_bytes(asset)
        except Exception as exc:
            return StoredImageRecord(
                point_id=None,
                image_id=None,
                status="failed",
                payload={},
                llm_description="",
                llm_tags=[],
                error=str(exc),
            )

        image_id = self._hash_image_bytes(image_bytes)
        point_id = self._point_id_from_image_id(image_id)
        occurrence = self._occurrence(asset, context)
        try:
            target_collection = self._resolve_collection_name(context.source_collection)
        except Exception as exc:
            return StoredImageRecord(
                point_id=point_id,
                image_id=image_id,
                status="failed",
                payload={"image_id": image_id},
                llm_description="",
                llm_tags=[],
                error=str(exc),
            )

        if self.img_ingestion_config.cache_by_hash:
            existing = self._existing_by_image_id(
                image_id,
                collection_name=target_collection,
            )
            if existing is not None:
                existing_point_id, existing_payload = existing
                self._append_occurrence(
                    collection_name=target_collection,
                    point_id=existing_point_id,
                    payload=existing_payload,
                    occurrence=occurrence,
                )
                description = str(existing_payload.get("llm_description") or "")
                existing_tags = existing_payload.get("llm_tags")
                tags_list = (
                    [str(tag) for tag in existing_tags]
                    if isinstance(existing_tags, list)
                    else []
                )
                return StoredImageRecord(
                    point_id=existing_point_id,
                    image_id=image_id,
                    status="cached",
                    payload=existing_payload,
                    llm_description=description,
                    llm_tags=tags_list,
                )

        description = ""
        tags: list[str] = []
        tag_error: str | None = None
        tagger = self._get_tagging_backend()
        if self.img_ingestion_config.tagging_enabled and tagger is not None:
            try:
                description, tags = self._run_with_retries(
                    lambda: tagger.describe_and_tag(image_bytes, mime_type),
                )
            except Exception as exc:
                tag_error = str(exc)
                if self.img_ingestion_config.fail_on_tagging_error:
                    raise
                logger.warning("Image tagging failed for {}: {}", image_id[:12], exc)

        embedding: list[float] | None = None
        embed_error: str | None = None
        embedding_backend = self._get_embedding_backend()
        if embedding_backend is None and self._embedding_backend_error:
            embed_error = self._embedding_backend_error
        if (
            self.img_ingestion_config.embedding_enabled
            and embedding_backend is not None
        ):
            try:
                embedding = self._run_with_retries(
                    lambda: embedding_backend.embed(image_bytes),
                )
            except Exception as exc:
                embed_error = str(exc)
                if self.img_ingestion_config.fail_on_embedding_error:
                    raise
                logger.warning("Image embedding failed for {}: {}", image_id[:12], exc)

        if not embedding:
            return StoredImageRecord(
                point_id=point_id,
                image_id=image_id,
                status="failed",
                payload={"image_id": image_id},
                llm_description=description,
                llm_tags=tags,
                error=embed_error or tag_error or "Image embedding unavailable.",
            )

        width, height = self._image_size(image_bytes)
        image_payload: dict[str, Any] = {
            "image_id": image_id,
            "source_type": asset.source_type,
            "source_collection": context.source_collection,
            "source_doc_id": asset.source_doc_id,
            "source_path": asset.source_path
            or (str(asset.image_path) if asset.image_path else None),
            "page_number": asset.page_number,
            "bbox": asset.bbox,
            "mime_type": mime_type,
            "mimetype": mime_type,
            "file_type": mime_type,
            "width": width,
            "height": height,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "llm_description": description,
            "llm_tags": tags,
            "vector_name": self.img_ingestion_config.vector_name,
            "image_collection": target_collection,
            "occurrences": [occurrence],
        }
        if asset.image_path:
            image_payload["file_path"] = str(asset.image_path)
            image_payload["file_name"] = asset.image_path.name
            image_payload["filename"] = asset.image_path.name
        if asset.extra_metadata:
            image_payload.update(asset.extra_metadata)
        if tag_error:
            image_payload["tagging_error"] = tag_error

        text_parts = [description.strip()]
        if tags:
            text_parts.append("Tags: " + ", ".join(tags))
        node_text = "\n\n".join([part for part in text_parts if part]).strip()

        image_node = ImageNode(
            id_=point_id,
            text=node_text,
            metadata=image_payload,
            image_path=str(asset.image_path) if asset.image_path else None,
            image_mimetype=mime_type,
            embedding=embedding,
        )

        try:
            self._ensure_collection(
                collection_name=target_collection,
                vector_dim=len(embedding),
            )
            vector_store = self._get_vector_store(target_collection)
            vector_store.add([image_node])
        except Exception as exc:
            return StoredImageRecord(
                point_id=point_id,
                image_id=image_id,
                status="failed",
                payload=image_payload,
                llm_description=description,
                llm_tags=tags,
                error=str(exc),
            )

        return StoredImageRecord(
            point_id=point_id,
            image_id=image_id,
            status="stored",
            payload=image_payload,
            llm_description=description,
            llm_tags=tags,
            error=tag_error,
        )

    def query_similar_images(
        self,
        image: Path | bytes,
        top_k: int = 5,
        *,
        source_collection: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return image records similar to the provided image.

        Args:
            image: Query image path or bytes.
            top_k: Number of nearest neighbors to return.
            source_collection: Optional source collection to resolve the target collection.

        Returns:
            A list of payload dictionaries with similarity scores.
        """
        embedding_backend = self._get_embedding_backend()
        if embedding_backend is None:
            return []
        try:
            target_collection = self._resolve_collection_name(source_collection)
        except Exception as exc:
            logger.warning("Image query skipped: {}", exc)
            return []
        vector_store = self._get_vector_store(target_collection)

        if isinstance(image, Path):
            image_bytes = image.read_bytes()
        else:
            image_bytes = image
        query_embedding = embedding_backend.embed(image_bytes)

        from llama_index.core.vector_stores.types import VectorStoreQuery

        result = vector_store.query(
            VectorStoreQuery(query_embedding=query_embedding, similarity_top_k=top_k)
        )
        output: list[dict[str, Any]] = []
        nodes = result.nodes or []
        similarities = result.similarities or []
        for i, node in enumerate(nodes):
            payload = dict(getattr(node, "metadata", {}) or {})
            payload["score"] = similarities[i] if i < len(similarities) else None
            payload.setdefault("image_collection", target_collection)
            output.append(payload)
        return output

    def query_similar_images_by_text(
        self,
        query_text: str,
        top_k: int = 5,
        *,
        source_collection: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return image records similar to a text query via CLIP text embeddings.

        A CLIP-based embedding backend is required for this functionality. The text
        query is embedded using the CLIP text tower, and then a similarity search is
        performed against the image embeddings in the target collection. This allows
        for retrieving images that are semantically similar to the text query, even
        if they do not contain the exact query terms in their metadata or descriptions.
        The results include similarity scores to indicate how closely each image matches
        the text query.

        Args:
            query_text: The input text query to find similar images for.
            top_k: The number of similar images to return.
            source_collection: Optional source collection to resolve the target collection.

        Returns:
            A list of payload dictionaries for the similar images, each including a similarity score.
        """
        if not query_text.strip():
            return []
        embedding_backend = self._get_embedding_backend()
        if embedding_backend is None:
            return []
        try:
            target_collection = self._resolve_collection_name(source_collection)
        except Exception as exc:
            logger.warning("Image text-query skipped: {}", exc)
            return []
        try:
            query_embedding = embedding_backend.embed_text(query_text)
        except Exception as exc:
            logger.warning("Image text embedding failed: {}", exc)
            return []

        from llama_index.core.vector_stores.types import VectorStoreQuery

        vector_store = self._get_vector_store(target_collection)
        result = vector_store.query(
            VectorStoreQuery(query_embedding=query_embedding, similarity_top_k=top_k)
        )
        output: list[dict[str, Any]] = []
        nodes = result.nodes or []
        similarities = result.similarities or []
        for i, node in enumerate(nodes):
            payload = dict(getattr(node, "metadata", {}) or {})
            payload["score"] = similarities[i] if i < len(similarities) else None
            payload.setdefault("image_collection", target_collection)
            output.append(payload)
        return output
