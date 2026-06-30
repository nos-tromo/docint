"""Test keyframe near-duplicate pruning with CLIP embeddings."""

from typing import Any

import pytest

from docint.core.ingest.images_service import ImageIngestionService, IngestContext


class _FakeEmbed:
    """Returns a preset unit vector per frame; records calls."""

    def __init__(self, vectors: list[list[float]]) -> None:
        self.vectors = vectors
        self.calls = 0

    @property
    def dimension(self) -> int:
        return len(self.vectors[0])

    def embed(self, image_bytes: bytes) -> list[float]:
        v = self.vectors[self.calls]
        self.calls += 1
        return v

    def embed_text(self, text: str) -> list[float]:  # pragma: no cover - unused
        return self.vectors[0]


class _FakeTagger:
    def __init__(self) -> None:
        self.calls = 0

    def describe_and_tag(self, image_bytes: bytes, mime_type: str) -> tuple[str, list[str]]:
        self.calls += 1
        return (f"caption {self.calls}", ["tag"])


def _service(monkeypatch: pytest.MonkeyPatch, embed: _FakeEmbed, tagger: _FakeTagger) -> ImageIngestionService:
    svc = ImageIngestionService(qdrant_client=None)
    monkeypatch.setattr(svc, "_get_embedding_backend", lambda: embed)
    monkeypatch.setattr(svc, "_get_tagging_backend", lambda: tagger)
    stored: list[Any] = []
    monkeypatch.setattr(svc, "_ensure_collection", lambda **kw: None)

    class _Store:
        def add(self, nodes: list[Any]) -> None:
            stored.extend(nodes)

    monkeypatch.setattr(svc, "_get_vector_store", lambda name: _Store())
    svc._stored_nodes = stored  # type: ignore[attr-defined]
    return svc


def test_prunes_near_duplicate_frames(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that near-duplicate frames are pruned and only survivors are tagged.

    Frame 0 and frame 1 are identical (cosine 1.0 -> drop frame 1);
    frame 2 is orthogonal (keep).
    """
    embed = _FakeEmbed([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    tagger = _FakeTagger()
    svc = _service(monkeypatch, embed, tagger)
    records = svc.ingest_keyframe_set(
        [b"f0", b"f1", b"f2"],
        context=IngestContext(source_collection="c"),
        source_doc_id="uuid-1",
        dedup_cosine=0.95,
    )
    stored = [r for r in records if r.status == "stored"]
    assert len(stored) == 2  # frame1 pruned
    assert tagger.calls == 2  # caption only the 2 survivors, not the pruned dup
    assert all(r.payload.get("posting_uuid") == "uuid-1" for r in stored)
