"""Tests for the shared image-ingestion service used by document readers."""

from __future__ import annotations

import base64
import hashlib
import json
import uuid
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from docint.core.ingest.images_service import (
    ImageAsset,
    ImageIngestionConfig,
    ImageIngestionService,
    IngestContext,
    VisionJSONTagger,
    _caption_prompt,
)
from docint.utils.prompt_loader import load_localized_prompt

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

    def describe_and_tag(self, image_bytes: bytes, mime_type: str) -> tuple[str, list[str]]:
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

    def create_collection(self, collection_name: str, vectors_config: dict[str, Any], **kwargs: Any) -> None:
        """Register a new collection with the given vector configuration.

        Args:
            collection_name: Name of the collection to create.
            vectors_config: Mapping of vector names to their parameters.
            **kwargs: Additional creation options (e.g. ``quantization_config``),
                accepted and ignored like the real client.
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

    def set_payload(self, collection_name: str, payload: dict[str, Any], points: list[str]) -> None:
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


def _build_service(
    *, ocr_enabled: bool = False, keyframe_ocr_enabled: bool = False
) -> tuple[ImageIngestionService, FakeQdrantClient, FakeVectorStore]:
    """Construct an ``ImageIngestionService`` wired to fake backends.

    Args:
        ocr_enabled (bool): Whether the text inside an image is read.
        keyframe_ocr_enabled (bool): Whether video keyframes are read too.

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
        ocr_enabled=ocr_enabled,
        keyframe_ocr_enabled=keyframe_ocr_enabled,
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


def test_parse_tag_payload_strips_think_before_json() -> None:
    """Reasoning scratchpads must be removed before JSON parsing."""
    raw = "<think>the user wants strict JSON</think>" + json.dumps(
        {"description": "A flowchart.", "tags": ["flowchart", "ops"]}
    )

    description, tags = VisionJSONTagger.parse_tag_payload(raw)

    assert description == "A flowchart."
    assert tags == ["flowchart", "ops"]


def test_parse_tag_payload_returns_empty_when_reasoning_only() -> None:
    """A response that is only a reasoning block must not poison the store."""
    raw = "The user wants a strict JSON with keys 'description' and 'tags'. I should produce JSON only.\n</think>"

    description, tags = VisionJSONTagger.parse_tag_payload(raw)

    assert description == ""
    assert tags == []


def test_parse_tag_payload_returns_empty_when_json_missing() -> None:
    """Non-JSON responses must not fall back to the raw content as description."""
    raw = "Sorry, I don't have enough context to describe this image."

    description, tags = VisionJSONTagger.parse_tag_payload(raw)

    assert description == ""
    assert tags == []


def test_parse_tag_payload_brace_extractor_handles_prose_wrapped_json() -> None:
    """A JSON object embedded in prose should still be parsed."""
    raw = "Here you go: " + json.dumps({"description": "A diagram.", "tags": ["one", "two"]}) + " — hope this helps!"

    description, tags = VisionJSONTagger.parse_tag_payload(raw)

    assert description == "A diagram."
    assert tags == ["one", "two"]


def test_parse_tag_payload_brace_extractor_handles_nested_strings() -> None:
    """Braces inside string values must not confuse the balanced extractor."""
    raw = 'prefix {"description": "a {not json} caption", "tags": ["x"]} suffix'

    description, tags = VisionJSONTagger.parse_tag_payload(raw)

    assert description == "a {not json} caption"
    assert tags == ["x"]


def test_parse_tag_payload_accepts_tags_only_json_without_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Valid JSON lacking a description key must not trigger a spurious warning."""
    raw = json.dumps({"tags": ["cat", "photo"]})

    with caplog.at_level("WARNING"):
        description, tags = VisionJSONTagger.parse_tag_payload(raw)

    assert description == ""
    assert tags == ["cat", "photo"]
    assert "non-JSON output" not in caplog.text


class _StubPipeline:
    """Minimal ``OpenAIPipeline`` stand-in that returns a canned vision response."""

    def __init__(self, response: str) -> None:
        self._response = response
        self.calls: list[tuple[str, str, str]] = []

    def call_vision(self, *, prompt: str, img_base64: str, mime_type: str) -> str:
        self.calls.append((prompt, img_base64, mime_type))
        return self._response


def test_describe_and_tag_returns_empty_on_no_image_refusal() -> None:
    """A no-image refusal from the pipeline should map to empty description/tags."""
    tagger = VisionJSONTagger.__new__(VisionJSONTagger)
    tagger.pipeline = cast(
        Any,
        _StubPipeline(response="I don't see any image attached to your message."),
    )
    tagger.max_image_dimension = 1024
    tagger.prompt_template = "Return strict JSON."

    description, tags = tagger.describe_and_tag(_make_png_bytes(), "image/png")

    assert description == ""
    assert tags == []


def test_describe_and_tag_strips_reasoning_wrapped_json() -> None:
    """Reasoning-wrapped JSON should still produce clean description and tags."""
    tagger = VisionJSONTagger.__new__(VisionJSONTagger)
    payload = json.dumps({"description": "A cat.", "tags": ["animal", "cat"]})
    tagger.pipeline = cast(Any, _StubPipeline(response=f"<think>user wants JSON</think>{payload}"))
    tagger.max_image_dimension = 1024
    tagger.prompt_template = "Return strict JSON."

    description, tags = tagger.describe_and_tag(_make_png_bytes(), "image/png")

    assert description == "A cat."
    assert tags == ["animal", "cat"]


def test_describe_and_tag_discards_pure_reasoning_response() -> None:
    """Reasoning-only responses must not leak into description."""
    tagger = VisionJSONTagger.__new__(VisionJSONTagger)
    tagger.pipeline = cast(
        Any,
        _StubPipeline(response="The user wants a strict JSON with keys description and tags.</think>"),
    )
    tagger.max_image_dimension = 1024
    tagger.prompt_template = "Return strict JSON."

    description, tags = tagger.describe_and_tag(_make_png_bytes(), "image/png")

    assert description == ""
    assert tags == []


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


def test_deduped_social_image_occurrence_records_posting_link(tmp_path: Path) -> None:
    """A second posting referencing the same image bytes stays traceable via occurrences.

    The point's top-level payload (incl. ``reference_metadata``) is first-wins,
    so the dedup occurrence must carry the second posting's link ids.

    Args:
        tmp_path: pytest fixture providing a temporary directory for test files.
    """
    service, client, _vector_store = _build_service()
    image_bytes = _make_png_bytes(color=(4, 5, 6))

    first_path = tmp_path / "first.png"
    first_path.write_bytes(image_bytes)
    second_path = tmp_path / "second.png"
    second_path.write_bytes(image_bytes)

    def _social_asset(path: Path, posting_uuid: str, posting_id: str, media_id: str) -> ImageAsset:
        return ImageAsset.from_path(
            path=path,
            source_type="social_media",
            source_doc_id=posting_uuid,
            extra_metadata={
                "posting_uuid": posting_uuid,
                "posting_id": posting_id,
                "media_id": media_id,
                "source_type": "social_media",
            },
        )

    first = service.ingest_image(
        _social_asset(first_path, "u1", "P_1", "P_1_0"),
        context=IngestContext(source_collection="att-3"),
    )
    second = service.ingest_image(
        _social_asset(second_path, "u2", "P_2", "P_2_0"),
        context=IngestContext(source_collection="att-3"),
    )

    assert first.status == "stored"
    assert second.status == "cached"
    stored_payload = client.records[first.point_id or ""]
    # First-wins on the top-level link ids ...
    assert stored_payload["posting_uuid"] == "u1"
    # ... but the deduped occurrence records the second posting's link.
    occurrences = stored_payload["occurrences"]
    assert any(occ.get("posting_uuid") == "u2" and occ.get("media_id") == "P_2_0" for occ in occurrences)


def test_ingest_image_degrades_when_embedding_backend_init_fails() -> None:
    """When the remote CLIP backend raises on init, ingestion should fail gracefully with an error message."""
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
        "docint.core.ingest.images_service.RemoteCLIPBackend",
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

    small_bytes = _make_png_bytes()  # 6x4

    out_bytes, out_mime = tagger._cap_image_size(small_bytes, "image/png")

    assert out_bytes is small_bytes
    assert out_mime == "image/png"


def test_query_similar_images_by_text_returns_empty_when_collection_missing() -> None:
    """Text-image retrieval should return empty when the image collection does not exist."""

    class NoEmbedBackend:
        def embed_text(self, text: str) -> list[float]:
            raise AssertionError("embed_text should not be called")

    service, _, _ = _build_service()
    service.qdrant_client = cast(Any, SimpleNamespace(collection_exists=lambda collection_name: False))
    service.embedding_backend = cast(Any, NoEmbedBackend())

    matches = service.query_similar_images_by_text(
        query_text="diagram",
        top_k=3,
        source_collection="spiegel-data",
    )

    assert matches == []


def test_query_similar_images_returns_empty_when_collection_missing() -> None:
    """Image-image retrieval should return empty when the image collection does not exist."""

    class NoImageEmbedBackend:
        def embed(self, image_bytes: bytes) -> list[float]:
            raise AssertionError("embed should not be called")

    service, _, _ = _build_service()
    service.qdrant_client = cast(Any, SimpleNamespace(collection_exists=lambda collection_name: False))
    service.embedding_backend = cast(Any, NoImageEmbedBackend())

    matches = service.query_similar_images(
        image=b"binary-image-data",
        source_collection="spiegel-data",
    )

    assert matches == []


def test_query_similar_images_by_text_propagates_the_node_id() -> None:
    """The retrieved node's id must reach the caller so an image citation is traceable.

    ``node.metadata`` alone carries no id, so without this the source built from a
    match renders as "Chunk-ID: n/a" and two images cannot be told apart.
    """

    class OneHitVectorStore:
        """Vector store stub returning a single identified image node."""

        def query(self, query: Any) -> Any:
            """Return one node with a distinct node id.

            Args:
                query: The vector store query (ignored by the stub).
            """
            node = SimpleNamespace(
                node_id="0298c8c6-aaab-559b-bd58-2bb428b853b2",
                metadata={"image_id": "105cc611", "file_name": "cert.jpg"},
            )
            return SimpleNamespace(nodes=[node], similarities=[0.42])

    service, _, _ = _build_service()
    service.qdrant_client = cast(Any, SimpleNamespace(collection_exists=lambda collection_name: True))
    service._get_vector_store = cast(Any, lambda collection: OneHitVectorStore())

    matches = service.query_similar_images_by_text("a query", top_k=1, source_collection="coll")

    assert matches[0]["node_id"] == "0298c8c6-aaab-559b-bd58-2bb428b853b2"


# ---------------------------------------------------------------------------
# Reading the text inside an image
# ---------------------------------------------------------------------------


def _ocr_engine(text: str) -> Any:
    """An OCR engine stand-in that reads one block of *text* from any image."""
    engine = MagicMock()
    blocks: list[SimpleNamespace] = [SimpleNamespace(text=text, category="text", bbox=None, cells=None)] if text else []
    engine.read_image.return_value = blocks
    return engine


def test_the_words_inside_an_image_are_read_and_stored() -> None:
    """A screenshot's own words are what a reader searched for, so they are kept."""
    service, _, _ = _build_service(ocr_enabled=True)
    engine = _ocr_engine("INVOICE 2031-0042\nTotal due: 1.240,00")

    with patch("docint.core.ingest.images_service.build_engine", return_value=engine):
        record = service.ingest_image(
            ImageAsset(source_type="standalone", image_bytes=_make_png_bytes(), mime_type="image/png"),
            context=IngestContext(source_collection="docs"),
        )

    assert record.status == "stored"
    assert record.ocr_text.startswith("INVOICE 2031-0042")
    assert record.payload["ocr_text"].startswith("INVOICE 2031-0042")
    engine.read_image.assert_called_once()


def test_the_read_words_come_before_the_caption_in_the_node_text() -> None:
    """The caption paraphrases; the words are the thing itself."""
    service, _, vector_store = _build_service(ocr_enabled=True)

    with patch("docint.core.ingest.images_service.build_engine", return_value=_ocr_engine("INVOICE 2031-0042")):
        service.ingest_image(
            ImageAsset(source_type="standalone", image_bytes=_make_png_bytes(), mime_type="image/png"),
            context=IngestContext(source_collection="docs"),
        )

    text = vector_store.add_calls[0][0].text
    assert text.index("INVOICE 2031-0042") < text.index("Tags:")
    assert text.startswith("INVOICE 2031-0042")


def test_an_image_is_not_read_when_the_lane_is_off() -> None:
    """An unchanged stack pays no OCR call per image."""
    service, _, _ = _build_service(ocr_enabled=False)

    with patch("docint.core.ingest.images_service.build_engine") as build:
        record = service.ingest_image(
            ImageAsset(source_type="standalone", image_bytes=_make_png_bytes(), mime_type="image/png"),
            context=IngestContext(source_collection="docs"),
        )

    build.assert_not_called()
    assert record.ocr_text == ""
    assert record.payload["ocr_text"] == ""


def test_an_unreadable_image_still_stores_its_caption() -> None:
    """Most images carry no text at all; that is not a failed ingestion."""
    service, _, _ = _build_service(ocr_enabled=True)
    engine = MagicMock()
    engine.read_image.side_effect = RuntimeError("endpoint down")

    with patch("docint.core.ingest.images_service.build_engine", return_value=engine):
        record = service.ingest_image(
            ImageAsset(source_type="standalone", image_bytes=_make_png_bytes(), mime_type="image/png"),
            context=IngestContext(source_collection="docs"),
        )

    assert record.status == "stored"
    assert record.ocr_text == ""
    assert record.llm_description


def test_one_engine_serves_every_image_of_a_batch() -> None:
    """Building a client per image would cost more than the reads do."""
    service, _, _ = _build_service(ocr_enabled=True)

    with patch("docint.core.ingest.images_service.build_engine", return_value=_ocr_engine("A")) as build:
        for colour in ((10, 20, 30), (40, 50, 60), (70, 80, 90)):
            service.ingest_image(
                ImageAsset(source_type="standalone", image_bytes=_make_png_bytes(colour), mime_type="image/png"),
                context=IngestContext(source_collection="docs"),
            )

    build.assert_called_once()


def test_keyframes_are_not_read_unless_asked() -> None:
    """A clip contributes many frames and most carry no text."""
    service, _, _ = _build_service(ocr_enabled=True, keyframe_ocr_enabled=False)
    engine = _ocr_engine("SLIDE 3")

    with patch("docint.core.ingest.images_service.build_engine", return_value=engine):
        records = service.ingest_keyframe_set(
            [_make_png_bytes((1, 2, 3))],
            context=IngestContext(source_collection="docs"),
            source_doc_id="clip-1",
        )

    engine.read_image.assert_not_called()
    assert records and records[0].ocr_text == ""


def test_keyframes_are_read_when_asked() -> None:
    """Slides are the case that carries text, and they are worth having."""
    service, _, _ = _build_service(ocr_enabled=True, keyframe_ocr_enabled=True)
    engine = _ocr_engine("SLIDE 3: Results")

    with patch("docint.core.ingest.images_service.build_engine", return_value=engine):
        records = service.ingest_keyframe_set(
            [_make_png_bytes((1, 2, 3))],
            context=IngestContext(source_collection="docs"),
            source_doc_id="clip-1",
        )

    assert records[0].ocr_text == "SLIDE 3: Results"
    assert records[0].payload["ocr_text"] == "SLIDE 3: Results"


# ---------------------------------------------------------------------------
# Thumbnails — the pixels a report can show
# ---------------------------------------------------------------------------


def _decode_thumbnail(payload: dict[str, Any]) -> Image.Image:
    """Decode a payload's stored thumbnail into a PIL image.

    Args:
        payload: The stored image point payload.

    Returns:
        The decoded thumbnail image.
    """
    raw = base64.b64decode(payload["thumbnail_b64"])
    return Image.open(BytesIO(raw))


def test_ingest_image_stores_a_capped_thumbnail() -> None:
    """A stored image carries a JPEG thumbnail capped at the thumbnail bound."""
    service, _, _ = _build_service()
    img = Image.new("RGB", (800, 600), color=(30, 60, 90))
    buffer = BytesIO()
    img.save(buffer, format="PNG")

    record = service.ingest_image(
        ImageAsset(source_type="standalone", image_bytes=buffer.getvalue(), mime_type="image/png"),
        context=IngestContext(source_collection="att-2"),
    )

    assert record.status == "stored"
    assert record.payload["thumbnail_mime"] == "image/jpeg"
    thumb = _decode_thumbnail(record.payload)
    assert thumb.format == "JPEG"
    assert max(thumb.width, thumb.height) == 768
    assert thumb.width / thumb.height == pytest.approx(800 / 600, abs=0.02)


def test_thumbnail_never_enters_node_text() -> None:
    """The base64 blob is payload-only: node text (and thus search_text) stays clean."""
    service, _, vector_store = _build_service()

    record = service.ingest_image(
        ImageAsset(source_type="standalone", image_bytes=_make_png_bytes(), mime_type="image/png"),
        context=IngestContext(source_collection="att-2"),
    )

    node = vector_store.add_calls[0][0]
    assert record.payload["thumbnail_b64"] not in node.text
    assert "thumbnail" not in node.text


def test_keyframe_set_stores_thumbnails_and_dimensions_per_survivor() -> None:
    """Keyframe points gain a thumbnail plus width/height like document images."""
    service, _, _ = _build_service()

    records = service.ingest_keyframe_set(
        [_make_png_bytes((1, 2, 3))],
        context=IngestContext(source_collection="docs"),
        source_doc_id="clip-1",
    )

    assert records
    payload = records[0].payload
    assert payload["thumbnail_mime"] == "image/jpeg"
    thumb = _decode_thumbnail(payload)
    assert max(thumb.width, thumb.height) <= 768
    assert payload["width"] == 6
    assert payload["height"] == 4


def test_cache_hit_backfills_missing_thumbnail_via_set_payload() -> None:
    """Re-ingesting bytes an old collection already holds upgrades the point in place."""
    service, client, _ = _build_service()
    img_bytes = _make_png_bytes((7, 7, 7))

    first = service.ingest_image(
        ImageAsset(source_type="standalone", image_bytes=img_bytes, mime_type="image/png"),
        context=IngestContext(source_collection="att-2"),
    )
    point_id = first.point_id or ""
    # Simulate a pre-thumbnail point written before this feature shipped.
    client.records[point_id].pop("thumbnail_b64", None)
    client.records[point_id].pop("thumbnail_mime", None)

    second = service.ingest_image(
        ImageAsset(source_type="standalone", image_bytes=img_bytes, mime_type="image/png"),
        context=IngestContext(source_collection="att-2"),
    )

    assert second.status == "cached"
    stored = client.records[point_id]
    assert stored["thumbnail_mime"] == "image/jpeg"
    assert max(_decode_thumbnail(stored).size) <= 768


def test_cache_hit_leaves_a_current_size_thumbnail_alone() -> None:
    """A cached point whose thumbnail was made at today's cap is not regenerated."""
    service, _, _ = _build_service()
    img_bytes = _make_png_bytes((9, 9, 9))
    service.ingest_image(
        ImageAsset(source_type="standalone", image_bytes=img_bytes, mime_type="image/png"),
        context=IngestContext(source_collection="att-2"),
    )

    with patch.object(ImageIngestionService, "_thumbnail_fields", MagicMock()) as make_thumb:
        second = service.ingest_image(
            ImageAsset(source_type="standalone", image_bytes=img_bytes, mime_type="image/png"),
            context=IngestContext(source_collection="att-2"),
        )

    assert second.status == "cached"
    make_thumb.assert_not_called()


def test_an_image_without_a_thumbnail_still_ingests() -> None:
    """Thumbnail generation is fail-soft: a refusal costs the field, not the point."""
    service, _, _ = _build_service()

    with patch.object(ImageIngestionService, "_thumbnail_fields", MagicMock(return_value=None)):
        record = service.ingest_image(
            ImageAsset(source_type="standalone", image_bytes=_make_png_bytes(), mime_type="image/png"),
            context=IngestContext(source_collection="att-2"),
        )

    assert record.status == "stored"
    assert "thumbnail_b64" not in record.payload
    assert "thumbnail_mime" not in record.payload


def test_make_thumbnail_flattens_rgba_and_respects_exif() -> None:
    """Transparency is flattened for JPEG and EXIF orientation is applied."""
    rgba = Image.new("RGBA", (400, 200), color=(10, 20, 30, 128))
    buffer = BytesIO()
    rgba.save(buffer, format="PNG")
    result = ImageIngestionService._thumbnail_fields(buffer.getvalue())
    assert result is not None
    assert result["thumbnail_mime"] == "image/jpeg"
    flat = Image.open(BytesIO(base64.b64decode(result["thumbnail_b64"])))
    assert flat.mode == "RGB"
    assert (flat.width, flat.height) == (400, 200)

    exif = Image.Exif()
    exif[0x0112] = 6  # rotate 90 CW on load
    rotated_src = Image.new("RGB", (400, 200), color=(1, 2, 3))
    buffer = BytesIO()
    rotated_src.save(buffer, format="JPEG", exif=exif)
    result = ImageIngestionService._thumbnail_fields(buffer.getvalue())
    assert result is not None
    upright = Image.open(BytesIO(base64.b64decode(result["thumbnail_b64"])))
    assert (upright.width, upright.height) == (200, 400)


def test_make_thumbnail_returns_none_on_corrupt_bytes() -> None:
    """Bytes PIL cannot open yield no thumbnail rather than an exception."""
    assert ImageIngestionService._thumbnail_fields(b"definitely not an image") is None


def test_thumbnail_is_written_payload_only_never_as_node_metadata() -> None:
    """The blob reaches the point through set_payload, not through the node.

    Node metadata is serialized into ``_node_content`` as well as written flat,
    so a thumbnail carried there is stored twice — measured at 24KB per point
    for a 320px thumbnail before this was fixed.
    """
    service, client, vector_store = _build_service()

    record = service.ingest_image(
        ImageAsset(source_type="standalone", image_bytes=_make_png_bytes(), mime_type="image/png"),
        context=IngestContext(source_collection="att-2"),
    )

    node = vector_store.add_calls[0][0]
    assert "thumbnail_b64" not in node.metadata
    assert "thumbnail_mime" not in node.metadata
    stored = client.records[record.point_id or ""]
    assert stored["thumbnail_b64"] == record.payload["thumbnail_b64"]


def test_keyframe_thumbnail_is_written_payload_only_never_as_node_metadata() -> None:
    """Keyframes take the same payload-only route as document images."""
    service, client, vector_store = _build_service()

    records = service.ingest_keyframe_set(
        [_make_png_bytes((4, 5, 6))],
        context=IngestContext(source_collection="docs"),
        source_doc_id="clip-9",
    )

    node = vector_store.add_calls[0][0]
    assert "thumbnail_b64" not in node.metadata
    assert client.records[records[0].point_id or ""]["thumbnail_b64"] == records[0].payload["thumbnail_b64"]


def test_thumbnail_records_the_cap_it_was_made_with() -> None:
    """The stored cap is what lets a later ingest tell an undersized thumbnail apart."""
    service, _, _ = _build_service()

    record = service.ingest_image(
        ImageAsset(source_type="standalone", image_bytes=_make_png_bytes(), mime_type="image/png"),
        context=IngestContext(source_collection="att-2"),
    )

    assert record.payload["thumbnail_max_dim"] == 768


def test_cache_hit_upgrades_a_thumbnail_made_at_a_smaller_cap() -> None:
    """A collection ingested when the cap was 320px gains today's resolution."""
    service, client, _ = _build_service()
    img = Image.new("RGB", (1600, 1200), color=(30, 60, 90))
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    img_bytes = buffer.getvalue()

    first = service.ingest_image(
        ImageAsset(source_type="standalone", image_bytes=img_bytes, mime_type="image/png"),
        context=IngestContext(source_collection="att-2"),
    )
    point_id = first.point_id or ""
    # Simulate the point as an older, smaller-capped ingest wrote it.
    small = Image.open(BytesIO(img_bytes))
    small.thumbnail((320, 320))
    old = BytesIO()
    small.convert("RGB").save(old, format="JPEG", quality=70)
    client.records[point_id]["thumbnail_b64"] = base64.b64encode(old.getvalue()).decode("ascii")
    client.records[point_id]["thumbnail_max_dim"] = 320

    second = service.ingest_image(
        ImageAsset(source_type="standalone", image_bytes=img_bytes, mime_type="image/png"),
        context=IngestContext(source_collection="att-2"),
    )

    assert second.status == "cached"
    stored = client.records[point_id]
    assert stored["thumbnail_max_dim"] == 768
    assert max(_decode_thumbnail(stored).size) == 768


def test_cache_hit_upgrades_a_thumbnail_that_predates_the_recorded_cap() -> None:
    """A point written before the cap was recorded counts as undersized."""
    service, client, _ = _build_service()
    img = Image.new("RGB", (1600, 1200), color=(90, 60, 30))
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    img_bytes = buffer.getvalue()

    first = service.ingest_image(
        ImageAsset(source_type="standalone", image_bytes=img_bytes, mime_type="image/png"),
        context=IngestContext(source_collection="att-2"),
    )
    point_id = first.point_id or ""
    client.records[point_id]["thumbnail_b64"] = "QUJD"
    client.records[point_id].pop("thumbnail_max_dim", None)

    service.ingest_image(
        ImageAsset(source_type="standalone", image_bytes=img_bytes, mime_type="image/png"),
        context=IngestContext(source_collection="att-2"),
    )

    assert client.records[point_id]["thumbnail_max_dim"] == 768
    assert max(_decode_thumbnail(client.records[point_id]).size) == 768


def test_upgrading_a_point_drops_the_duplicate_blob_from_node_content() -> None:
    """The stale copy an earlier write left inside the serialized node goes too.

    Otherwise upgrading a collection is a net loss: today's larger thumbnail
    lands beside the old one instead of replacing it.
    """
    service, client, _ = _build_service()
    img_bytes = _make_png_bytes((2, 4, 6))

    first = service.ingest_image(
        ImageAsset(source_type="standalone", image_bytes=img_bytes, mime_type="image/png"),
        context=IngestContext(source_collection="att-2"),
    )
    point_id = first.point_id or ""
    client.records[point_id]["thumbnail_max_dim"] = 320
    client.records[point_id]["_node_content"] = json.dumps(
        {"id_": point_id, "text": "a caption", "metadata": {"image_id": first.image_id, "thumbnail_b64": "QUJD"}}
    )

    service.ingest_image(
        ImageAsset(source_type="standalone", image_bytes=img_bytes, mime_type="image/png"),
        context=IngestContext(source_collection="att-2"),
    )

    node_content = json.loads(client.records[point_id]["_node_content"])
    assert "thumbnail_b64" not in node_content["metadata"]
    assert node_content["text"] == "a caption"
    assert node_content["metadata"]["image_id"] == first.image_id


def test_a_small_image_is_never_upscaled_and_never_re_stamped() -> None:
    """A 6x4 source stays 6x4, and its recorded cap keeps re-ingest from looping."""
    service, client, _ = _build_service()
    img_bytes = _make_png_bytes((5, 5, 5))

    first = service.ingest_image(
        ImageAsset(source_type="standalone", image_bytes=img_bytes, mime_type="image/png"),
        context=IngestContext(source_collection="att-2"),
    )

    assert _decode_thumbnail(first.payload).size == (6, 4)

    with patch.object(ImageIngestionService, "_thumbnail_fields", MagicMock()) as make_thumb:
        service.ingest_image(
            ImageAsset(source_type="standalone", image_bytes=img_bytes, mime_type="image/png"),
            context=IngestContext(source_collection="att-2"),
        )

    make_thumb.assert_not_called()
    assert client.records[first.point_id or ""]["thumbnail_max_dim"] == 768


def test_write_image_search_text_also_ensures_the_companion_field_indexes() -> None:
    """A freshly ingested collection with images must need no operator step.

    ``ensure_search_index`` alone only covers the default text lane; the
    "Search in" field picker also needs the companion's metadata TEXT
    indexes, or a freshly ingested collection would report ``not_indexed``
    for Author/Network/etc. against its images until a manual
    `make search-index` backfill ran.
    """
    service, client, _ = _build_service()

    with (
        patch("docint.core.ingest.images_service.ensure_search_index", return_value=True) as ensure_text,
        patch("docint.core.ingest.images_service.ensure_field_indexes", return_value=True) as ensure_fields,
        patch("docint.core.ingest.images_service.write_search_text", return_value=1) as write_text,
    ):
        service._write_image_search_text("docs_images", "point-1", "a caption")

    ensure_text.assert_called_once_with(client, "docs_images")
    ensure_fields.assert_called_once_with(client, "docs_images")
    write_text.assert_called_once_with(client, "docs_images", {"point-1": "a caption"})


def test_write_image_search_text_is_fail_soft_when_field_indexing_breaks() -> None:
    """A field-index failure must degrade the image, not the ingest.

    Mirrors the existing fail-soft posture around ``ensure_search_index`` and
    ``write_search_text`` in the same method: an outage here costs one image
    a backfill, never the ingest itself.
    """
    service, _, _ = _build_service()

    with (
        patch("docint.core.ingest.images_service.ensure_search_index", return_value=True),
        patch(
            "docint.core.ingest.images_service.ensure_field_indexes",
            side_effect=RuntimeError("qdrant unreachable"),
        ),
    ):
        service._write_image_search_text("docs_images", "point-1", "a caption")


# --------------------------------------------------------------------------- #
# Caption locale
# --------------------------------------------------------------------------- #
def test_the_caption_prompt_follows_the_response_language(monkeypatch: pytest.MonkeyPatch) -> None:
    """A caption is prose an investigator reads, so it follows the locale.

    German operators were getting English captions in a German report because
    this was the one model prompt that never consulted ``RESPONSE_LANGUAGE``.
    """
    monkeypatch.setenv("RESPONSE_LANGUAGE", "de")
    assert "Deutsch" in _caption_prompt()

    monkeypatch.setenv("RESPONSE_LANGUAGE", "en")
    assert "English" in _caption_prompt()


def test_the_caption_prompt_keeps_its_json_keys_english_in_every_locale() -> None:
    """The keys are protocol; only the values they carry are prose."""
    for locale in ("en", "de"):
        text = load_localized_prompt("image_caption", default="", lang=locale)
        assert "description" in text
        assert "tags" in text


def test_an_unknown_locale_still_yields_a_usable_caption_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    """A missing language pack must not leave the tagger promptless."""
    monkeypatch.setenv("RESPONSE_LANGUAGE", "xx")
    prompt = _caption_prompt()
    assert "description" in prompt
    assert "tags" in prompt


def test_the_tagger_reads_its_prompt_at_construction(monkeypatch: pytest.MonkeyPatch) -> None:
    """Read per instance, so a locale set after import still takes effect."""
    monkeypatch.setenv("RESPONSE_LANGUAGE", "de")
    assert "Deutsch" in VisionJSONTagger().prompt_template
