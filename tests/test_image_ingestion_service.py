from __future__ import annotations

import hashlib
import json
import uuid
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import patch

from PIL import Image

from docint.core.ingest.images_service import (
    ImageAsset,
    ImageIngestionConfig,
    ImageIngestionService,
    VisionJSONTagger,
    IngestContext,
)

"""Unit tests for the image ingestion service.

Covers hashing, tagging, embedding, caching, format normalisation,
image resizing, and graceful degradation when backends fail.
All tests run without GPU or network access by using fake backends
and an in-memory Qdrant client double.
"""


class FakeEmbeddingBackend:
    """Deterministic embedding backend for CPU-only tests."""

    @property
    def dimension(self) -> int:
        """Returns the dimensionality of the embedding vectors.

        Returns:
            int: The number of dimensions in the embedding vectors.
        """        
        return 3

    def embed(self, image_bytes: bytes) -> list[float]:
        """Generates a deterministic embedding vector based on the input image bytes.

        Args:
            image_bytes (bytes): The raw bytes of the image to be embedded.

        Returns:
            list[float]: A list of floats representing the embedding vector for the 
            input image. The values are derived from the first byte of the image to 
            ensure consistency across test runs. The vector has a fixed length of 3, 
            with the first value influenced by the image content and the remaining 
            values set to constants for simplicity.
        """        
        seed = image_bytes[0] if image_bytes else 1
        return [float(seed), 0.5, 0.25]

    def embed_text(self, text: str) -> list[float]:
        """Generates a deterministic embedding vector based on the input text.

        Args:
            text: The input text to be embedded.

        Returns:
            A 3-element float list derived from the stripped text length.
        """
        seed = float(len(text.strip()) or 1)
        return [seed, 0.5, 0.25]


class FakeTaggingBackend:
    """Deterministic image tagger for tests."""

    def describe_and_tag(
        self, image_bytes: bytes, mime_type: str
    ) -> tuple[str, list[str]]:
        """Return a fixed description and tag list for any image.

        Args:
            image_bytes: Raw bytes of the image (unused).
            mime_type: MIME type string included in the description.

        Returns:
            A tuple of (description, tags).
        """
        return f"Test image ({mime_type})", ["diagram", "paper", "figure"]


class FakeQdrantClient:
    """In-memory qdrant client double for image ingestion tests."""

    def __init__(self) -> None:
        """Initialise empty collection and record stores."""
        self.collections: dict[str, Any] = {}
        self.records: dict[str, dict[str, Any]] = {}

    def get_collection(self, collection_name: str) -> Any:
        """Return collection metadata or raise if it does not exist.

        Args:
            collection_name: Name of the collection to look up.

        Returns:
            A ``SimpleNamespace`` that mimics Qdrant collection info.

        Raises:
            RuntimeError: If the collection has not been created.
        """
        if collection_name not in self.collections:
            raise RuntimeError("missing")
        return self.collections[collection_name]

    def create_collection(
        self, collection_name: str, vectors_config: dict[str, Any]
    ) -> None:
        """Register a new collection with the given vector configuration.

        Args:
            collection_name: Name of the collection to create.
            vectors_config: Mapping of vector names to their parameters.
        """
        self.collections[collection_name] = SimpleNamespace(
            config=SimpleNamespace(params=SimpleNamespace(vectors=vectors_config))
        )

    def scroll(
        self,
        collection_name: str,
        scroll_filter: Any,
        limit: int,
        with_payload: bool,
        with_vectors: bool,
    ) -> tuple[list[Any], None]:
        """Search records by ``image_id`` filter, returning at most one match.

        Args:
            collection_name: Ignored (single-collection fake).
            scroll_filter: Filter object with a ``must`` list of conditions.
            limit: Ignored.
            with_payload: Ignored.
            with_vectors: Ignored.

        Returns:
            A tuple of (matching records list, ``None`` cursor).
        """
        del collection_name, limit, with_payload, with_vectors
        image_id: str | None = None
        must = getattr(scroll_filter, "must", []) or []
        for cond in must:
            if getattr(cond, "key", "") == "image_id":
                match = getattr(cond, "match", None)
                image_id = getattr(match, "value", None)
                break
        if not image_id:
            return [], None
        for point_id, payload in self.records.items():
            if payload.get("image_id") == image_id:
                return [SimpleNamespace(id=point_id, payload=dict(payload))], None
        return [], None

    def set_payload(
        self, collection_name: str, payload: dict[str, Any], points: list[str]
    ) -> None:
        """Merge *payload* into the stored record for each point.

        Args:
            collection_name: Ignored.
            payload: Key-value pairs to merge.
            points: Point IDs whose payloads are updated.
        """
        del collection_name
        for point_id in points:
            existing = self.records.get(point_id, {})
            existing.update(payload)
            self.records[point_id] = existing


class FakeVectorStore:
    """Captures upserted image nodes."""

    def __init__(self, client: FakeQdrantClient) -> None:
        """Initialise with a reference to the fake Qdrant client.

        Args:
            client: The ``FakeQdrantClient`` that stores point records.
        """
        self.client = client
        self.add_calls: list[list[Any]] = []

    def add(self, nodes: list[Any]) -> list[str]:
        """Persist nodes as records in the fake client and track calls.

        Args:
            nodes: Image nodes to upsert.

        Returns:
            List of point IDs that were stored.
        """
        self.add_calls.append(nodes)
        ids: list[str] = []
        for node in nodes:
            point_id = str(node.node_id)
            self.client.records[point_id] = dict(node.metadata)
            ids.append(point_id)
        return ids


def _make_png_bytes(color: tuple[int, int, int] = (120, 10, 10)) -> bytes:
    """Create a minimal 6x4 PNG image in memory.

    Args:
        color: RGB tuple used to fill the image.

    Returns:
        Raw PNG bytes.
    """
    img = Image.new("RGB", (6, 4), color=color)
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


def _build_service() -> tuple[ImageIngestionService, FakeQdrantClient, FakeVectorStore]:
    """Construct an ``ImageIngestionService`` wired to fake backends.

    Returns:
        A tuple of (service, fake Qdrant client, fake vector store).
    """
    cfg = ImageIngestionConfig(
        enabled=True,
        embedding_enabled=True,
        tagging_enabled=True,
        collection_name="test-images",
        vector_name="image-dense",
        cache_by_hash=True,
        fail_on_embedding_error=False,
        fail_on_tagging_error=False,
        retrieve_top_k=5,
    )
    model_cfg = SimpleNamespace(image_embed_model="openai/clip-vit-base-patch32")
    client = FakeQdrantClient()
    vector_store = FakeVectorStore(client)
    service = ImageIngestionService(
        img_ingestion_config=cfg,
        model_config=cast(Any, model_cfg),
        qdrant_client=cast(Any, client),
        vector_store=cast(Any, vector_store),
        embedding_backend=FakeEmbeddingBackend(),
        tagging_backend=FakeTaggingBackend(),
    )
    return service, client, vector_store


def test_image_id_and_point_id_are_deterministic() -> None:
    """Image ID (SHA-256) and derived point ID (UUID-5) must be stable across calls."""
    image_bytes = b"docint-image-bytes"
    image_id = ImageIngestionService._hash_image_bytes(image_bytes)
    expected = hashlib.sha256(image_bytes).hexdigest()
    assert image_id == expected

    point_id = ImageIngestionService._point_id_from_image_id(image_id)
    assert point_id == ImageIngestionService._point_id_from_image_id(image_id)
    assert str(uuid.UUID(point_id)) == point_id


def test_parse_tag_payload_extracts_structured_json() -> None:
    """``parse_tag_payload`` should extract description and deduplicated, length-filtered tags."""
    raw = json.dumps(
        {
            "description": "A technical architecture diagram.",
            "tags": [
                "diagram",
                "system design",
                "very long tag should be ignored here",
                "diagram",
            ],
        }
    )

    description, tags = VisionJSONTagger.parse_tag_payload(raw)

    assert description == "A technical architecture diagram."
    assert tags == ["diagram", "system design"]


def test_ingest_image_stores_expected_payload_and_vector() -> None:
    """A standalone PNG should be embedded, tagged, and stored with all required payload keys."""
    service, client, vector_store = _build_service()
    img_bytes = _make_png_bytes()

    record = service.ingest_image(
        ImageAsset(
            source_type="standalone",
            image_bytes=img_bytes,
            source_path="/tmp/a.png",
            mime_type="image/png",
        ),
        context=IngestContext(source_collection="att-2"),
    )

    assert record.status == "stored"
    assert str(uuid.UUID(record.point_id or "")) == record.point_id
    assert record.payload["vector_name"] == "image-dense"
    assert record.payload["source_type"] == "standalone"
    assert record.payload["mime_type"] == "image/png"
    assert record.payload["width"] == 6
    assert record.payload["height"] == 4
    assert record.payload["llm_description"]
    assert record.payload["llm_tags"] == ["diagram", "paper", "figure"]

    required_keys = {
        "image_id",
        "source_type",
        "source_doc_id",
        "source_path",
        "page_number",
        "bbox",
        "mime_type",
        "width",
        "height",
        "created_at",
        "llm_description",
        "llm_tags",
    }
    assert required_keys.issubset(record.payload.keys())

    assert len(vector_store.add_calls) == 1
    node = vector_store.add_calls[0][0]
    assert node.metadata["image_id"] == record.image_id
    assert isinstance(node.embedding, list)
    assert len(node.embedding) == 3

    info = client.get_collection("test-images")
    vectors = info.config.params.vectors
    assert "image-dense" in vectors
    assert int(vectors["image-dense"].size) == 3


def test_document_and_standalone_follow_same_shared_schema(tmp_path: Path) -> None:
    """Identical images ingested as standalone then document should share a point and track occurrences.
    
    Args:
    tmp_path: pytest fixture providing a temporary directory for test files.
    """
    service, client, vector_store = _build_service()
    image_bytes = _make_png_bytes(color=(1, 2, 3))

    standalone_path = tmp_path / "standalone.png"
    standalone_path.write_bytes(image_bytes)

    extracted_path = tmp_path / "artifact-image.png"
    extracted_path.write_bytes(image_bytes)

    standalone_record = service.ingest_image(
        ImageAsset.from_path(
            path=standalone_path,
            source_type="standalone",
            source_path=str(standalone_path),
        ),
        context=IngestContext(source_collection="att-2"),
    )
    document_record = service.ingest_image(
        ImageAsset.from_path(
            path=extracted_path,
            source_type="document",
            source_doc_id="doc-abc",
            source_path="/tmp/source.pdf",
            page_number=3,
            bbox={"x0": 1.0, "y0": 2.0, "x1": 3.0, "y1": 4.0},
        ),
        context=IngestContext(source_collection="att-2"),
    )

    assert standalone_record.status == "stored"
    assert document_record.status == "cached"
    assert standalone_record.image_id == document_record.image_id
    assert standalone_record.point_id == document_record.point_id
    assert len(vector_store.add_calls) == 1

    point_id = standalone_record.point_id or ""
    stored_payload = client.records[point_id]
    required_keys = {
        "image_id",
        "source_type",
        "source_doc_id",
        "source_path",
        "page_number",
        "bbox",
        "mime_type",
        "width",
        "height",
        "created_at",
        "llm_description",
        "llm_tags",
        "vector_name",
        "occurrences",
    }
    assert required_keys.issubset(stored_payload.keys())
    assert stored_payload["vector_name"] == "image-dense"
    assert isinstance(stored_payload["occurrences"], list)
    assert len(stored_payload["occurrences"]) == 2


def test_ingest_image_degrades_when_embedding_backend_init_fails() -> None:
    """When the CLIP backend raises on init, ingestion should fail gracefully with an error message."""
    cfg = ImageIngestionConfig(
        enabled=True,
        embedding_enabled=True,
        tagging_enabled=False,
        collection_name="test-images",
        vector_name="image-dense",
        cache_by_hash=True,
        fail_on_embedding_error=False,
        fail_on_tagging_error=False,
        retrieve_top_k=5,
    )
    model_cfg = SimpleNamespace(image_embed_model="missing/model")
    client = FakeQdrantClient()
    vector_store = FakeVectorStore(client)
    service = ImageIngestionService(
        img_ingestion_config=cfg,
        model_config=cast(Any, model_cfg),
        qdrant_client=cast(Any, client),
        vector_store=cast(Any, vector_store),
        tagging_backend=FakeTaggingBackend(),
    )

    with patch(
        "docint.core.ingest.images_service.CLIPImageEmbeddingBackend",
        side_effect=RuntimeError("clip init failed"),
    ):
        record = service.ingest_image(
            ImageAsset(
                source_type="standalone",
                image_bytes=_make_png_bytes(),
                source_path="/tmp/a.png",
                mime_type="image/png",
            ),
            context=IngestContext(source_collection="att-2"),
        )

    assert record.status == "failed"
    assert "clip init failed" in (record.error or "")


def _make_bmp_bytes(color: tuple[int, int, int] = (50, 50, 50)) -> bytes:
    """Create a small BMP image so we can test unsupported-format normalisation."""
    img = Image.new("RGB", (4, 4), color=color)
    buffer = BytesIO()
    img.save(buffer, format="BMP")
    return buffer.getvalue()


def test_normalize_image_passes_supported_formats_through() -> None:
    """JPEG, PNG, GIF, and WebP must not be re-encoded."""
    tagger = VisionJSONTagger.__new__(VisionJSONTagger)
    png_bytes = _make_png_bytes()

    for mime in ("image/jpeg", "image/png", "image/gif", "image/webp"):
        out_bytes, out_mime = tagger._normalize_image(png_bytes, mime)
        assert out_bytes is png_bytes, f"{mime} should pass through unchanged"
        assert out_mime == mime


def test_normalize_image_converts_unsupported_format_to_png() -> None:
    """Non-standard MIME types (e.g. BMP) should be re-encoded as PNG."""
    tagger = VisionJSONTagger.__new__(VisionJSONTagger)
    bmp_bytes = _make_bmp_bytes()

    out_bytes, out_mime = tagger._normalize_image(bmp_bytes, "image/bmp")

    assert out_mime == "image/png"
    assert out_bytes != bmp_bytes
    # Verify the output is valid PNG
    img = Image.open(BytesIO(out_bytes))
    assert img.format == "PNG"


def test_normalize_image_returns_original_on_corrupt_bytes() -> None:
    """If Pillow cannot open the bytes, return them unchanged."""
    tagger = VisionJSONTagger.__new__(VisionJSONTagger)
    bad_bytes = b"not-an-image"

    out_bytes, out_mime = tagger._normalize_image(bad_bytes, "application/octet-stream")
    assert out_bytes is bad_bytes
    assert out_mime == "application/octet-stream"


def test_collection_template_resolves_with_source_collection() -> None:
    """A ``{collection}_images`` template should resolve using the source collection name."""
    cfg = ImageIngestionConfig(
        enabled=True,
        embedding_enabled=True,
        tagging_enabled=True,
        collection_name="{collection}_images",
        vector_name="image-dense",
        cache_by_hash=True,
        fail_on_embedding_error=False,
        fail_on_tagging_error=False,
        retrieve_top_k=5,
        tagging_max_image_dimension=1024,
    )
    model_cfg = SimpleNamespace(image_embed_model="openai/clip-vit-base-patch32")
    client = FakeQdrantClient()
    vector_store = FakeVectorStore(client)
    service = ImageIngestionService(
        img_ingestion_config=cfg,
        model_config=cast(Any, model_cfg),
        qdrant_client=cast(Any, client),
        vector_store=cast(Any, vector_store),
        embedding_backend=FakeEmbeddingBackend(),
        tagging_backend=FakeTaggingBackend(),
    )

    record = service.ingest_image(
        ImageAsset(
            source_type="standalone",
            image_bytes=_make_png_bytes(),
            source_path="/tmp/a.png",
            mime_type="image/png",
        ),
        context=IngestContext(source_collection="att-2"),
    )

    assert record.status == "stored"
    assert record.payload["image_collection"] == "att-2_images"


def test_cap_image_size_shrinks_large_image() -> None:
    """Images exceeding max_image_dimension should be resized and re-encoded as JPEG."""
    tagger = VisionJSONTagger.__new__(VisionJSONTagger)
    tagger.max_image_dimension = 512

    big_img = Image.new("RGB", (2048, 1024), color=(100, 100, 100))
    buf = BytesIO()
    big_img.save(buf, format="PNG")
    big_bytes = buf.getvalue()

    out_bytes, out_mime = tagger._cap_image_size(big_bytes, "image/png")

    assert out_mime == "image/jpeg"
    result = Image.open(BytesIO(out_bytes))
    assert max(result.width, result.height) == 512
    assert result.width == 512
    assert result.height == 256


def test_cap_image_size_passes_small_image_through() -> None:
    """Small images should be returned unchanged."""
    tagger = VisionJSONTagger.__new__(VisionJSONTagger)
    tagger.max_image_dimension = 1024

    small_bytes = _make_png_bytes()  # 6×4

    out_bytes, out_mime = tagger._cap_image_size(small_bytes, "image/png")

    assert out_bytes is small_bytes
    assert out_mime == "image/png"
