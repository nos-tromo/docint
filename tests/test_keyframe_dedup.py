"""Test keyframe near-duplicate pruning with CLIP embeddings."""

import dataclasses
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


def _service(monkeypatch: pytest.MonkeyPatch, embed: Any, tagger: _FakeTagger) -> ImageIngestionService:
    svc = ImageIngestionService(qdrant_client=None)
    monkeypatch.setattr(svc, "_get_embedding_backend", lambda: embed)
    monkeypatch.setattr(svc, "_get_tagging_backend", lambda: tagger)
    stored: list[Any] = []
    monkeypatch.setattr(svc, "_ensure_collection", lambda **kw: None)

    class _Store:
        def add(self, nodes: list[Any]) -> None:
            stored.extend(nodes)

    monkeypatch.setattr(svc, "_get_vector_store", lambda name: _Store())
    # Pin config so test is environment-independent
    svc.img_ingestion_config = dataclasses.replace(svc.img_ingestion_config, enabled=True, tagging_enabled=True)
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
        extra_metadata={"posting_id": "p1"},
        dedup_cosine=0.95,
    )
    stored = [r for r in records if r.status == "stored"]
    assert len(stored) == 2  # frame1 pruned
    assert tagger.calls == 2  # caption only the 2 survivors, not the pruned dup
    assert embed.calls == 3  # embed once per input frame
    assert all(r.payload.get("posting_uuid") == "uuid-1" for r in stored)
    assert all(r.payload.get("posting_id") == "p1" for r in stored)


def test_embed_failure_is_skipped_failsoft(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that embed failure on one frame is fail-soft and skipped.

    Frame 0 embeds successfully, frame 1 raises (skipped), frame 2 embeds
    successfully. Only frames 0 and 2 are stored.
    """

    class _FailingEmbed:
        """Raises on the second embed call, succeeds for others."""

        def __init__(self, vectors: list[list[float]]) -> None:
            self.vectors = vectors
            self.calls = 0

        @property
        def dimension(self) -> int:
            return len(self.vectors[0])

        def embed(self, image_bytes: bytes) -> list[float]:
            if self.calls == 1:  # Second frame
                self.calls += 1
                raise RuntimeError("Simulated embed failure")
            v = self.vectors[self.calls]
            self.calls += 1
            return v

        def embed_text(self, text: str) -> list[float]:  # pragma: no cover - unused
            return self.vectors[0]

    embed = _FailingEmbed([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]])
    tagger = _FakeTagger()
    svc = _service(monkeypatch, embed, tagger)
    records = svc.ingest_keyframe_set(
        [b"f0", b"f1", b"f2"],
        context=IngestContext(source_collection="c"),
        source_doc_id="uuid-2",
    )
    stored = [r for r in records if r.status == "stored"]
    assert len(stored) == 2  # frame 1 embed failed, so skipped
    assert embed.calls == 3  # all three frames were attempted
