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


class FakeEmbeddingBackend:
    """Deterministic embedding backend for CPU-only tests."""

    @property
    def dimension(self) -> int:
        return 3

    def embed(self, image_bytes: bytes) -> list[float]:
        seed = image_bytes[0] if image_bytes else 1
        return [float(seed), 0.5, 0.25]

    def embed_text(self, text: str) -> list[float]:
        seed = float(len(text.strip()) or 1)
        return [seed, 0.5, 0.25]


class FakeTaggingBackend:
    """Deterministic image tagger for tests."""

    def describe_and_tag(
        self, image_bytes: bytes, mime_type: str
    ) -> tuple[str, list[str]]:
        return f"Test image ({mime_type})", ["diagram", "paper", "figure"]


class FakeQdrantClient:
    """In-memory qdrant client double for image ingestion tests."""

    def __init__(self) -> None:
        self.collections: dict[str, Any] = {}
        self.records: dict[str, dict[str, Any]] = {}

    def get_collection(self, collection_name: str) -> Any:
        if collection_name not in self.collections:
            raise RuntimeError("missing")
        return self.collections[collection_name]

    def create_collection(
        self, collection_name: str, vectors_config: dict[str, Any]
    ) -> None:
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
        del collection_name
        for point_id in points:
            existing = self.records.get(point_id, {})
            existing.update(payload)
            self.records[point_id] = existing


class FakeVectorStore:
    """Captures upserted image nodes."""

    def __init__(self, client: FakeQdrantClient) -> None:
        self.client = client
        self.add_calls: list[list[Any]] = []

    def add(self, nodes: list[Any]) -> list[str]:
        self.add_calls.append(nodes)
        ids: list[str] = []
        for node in nodes:
            point_id = str(node.node_id)
            self.client.records[point_id] = dict(node.metadata)
            ids.append(point_id)
        return ids


def _make_png_bytes(color: tuple[int, int, int] = (120, 10, 10)) -> bytes:
    img = Image.new("RGB", (6, 4), color=color)
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


def _build_service() -> tuple[ImageIngestionService, FakeQdrantClient, FakeVectorStore]:
    cfg = ImageIngestionConfig(
        enabled=True,
        embedding_enabled=True,
        tagging_enabled=True,
        collection_name="test-images",
        vector_name="image-dense",
        cache_by_hash=True,
        fail_on_embedding_error=False,
        fail_on_tagging_error=False,
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
    image_bytes = b"docint-image-bytes"
    image_id = ImageIngestionService._hash_image_bytes(image_bytes)
    expected = hashlib.sha256(image_bytes).hexdigest()
    assert image_id == expected

    point_id = ImageIngestionService._point_id_from_image_id(image_id)
    assert point_id == ImageIngestionService._point_id_from_image_id(image_id)
    assert str(uuid.UUID(point_id)) == point_id


def test_parse_tag_payload_extracts_structured_json() -> None:
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
    cfg = ImageIngestionConfig(
        enabled=True,
        embedding_enabled=True,
        tagging_enabled=False,
        collection_name="test-images",
        vector_name="image-dense",
        cache_by_hash=True,
        fail_on_embedding_error=False,
        fail_on_tagging_error=False,
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


def test_collection_template_resolves_with_source_collection() -> None:
    cfg = ImageIngestionConfig(
        enabled=True,
        embedding_enabled=True,
        tagging_enabled=True,
        collection_name="{collection}_images",
        vector_name="image-dense",
        cache_by_hash=True,
        fail_on_embedding_error=False,
        fail_on_tagging_error=False,
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
