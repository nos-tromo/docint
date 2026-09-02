"""Test that ingest_keyframe_set's source_type and link field are parameterized."""

from typing import Any

import pytest

from docint.core.ingest.images_service import ImageIngestionService, IngestContext


class _FakeEmbed:
    def __init__(self) -> None:
        self.calls = 0

    @property
    def dimension(self) -> int:
        return 2

    def embed(self, image_bytes: bytes) -> list[float]:
        self.calls += 1
        return [1.0, 0.0]

    def embed_text(self, text: str) -> list[float]:  # pragma: no cover - unused
        return [1.0, 0.0]


def _service(monkeypatch: pytest.MonkeyPatch) -> tuple[ImageIngestionService, list[Any]]:
    svc = ImageIngestionService(qdrant_client=None)
    monkeypatch.setattr(svc, "_get_embedding_backend", lambda: _FakeEmbed())
    monkeypatch.setattr(svc, "_get_tagging_backend", lambda: None)
    monkeypatch.setattr(svc, "_resolve_collection_name", lambda name: f"{name}_images")
    monkeypatch.setattr(svc, "_ensure_collection", lambda **kw: None)
    stored: list[Any] = []

    class _Store:
        def add(self, nodes: list[Any]) -> None:
            stored.extend(nodes)

    monkeypatch.setattr(svc, "_get_vector_store", lambda name: _Store())
    return svc, stored


def test_defaults_preserve_social_keyframe_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default args preserve the social keyframe payload byte-for-byte."""
    svc, stored = _service(monkeypatch)
    svc.ingest_keyframe_set(
        [b"f0"],
        context=IngestContext(source_collection="c"),
        source_doc_id="uuid-1",
        extra_metadata={"posting_id": "P", "media_id": "P_0", "source_type": "social_media", "posting_uuid": "uuid-1"},
    )
    payload = stored[0].metadata
    assert payload["source_type"] == "social_media"  # extra_metadata override, exactly as today
    assert payload["posting_uuid"] == "uuid-1"
    assert payload["source_doc_id"] == "uuid-1"


def test_standalone_video_keyframe_has_no_posting_uuid(monkeypatch: pytest.MonkeyPatch) -> None:
    """A standalone video keyframe carries source_type video_keyframe and no posting_uuid."""
    svc, stored = _service(monkeypatch)
    svc.ingest_keyframe_set(
        [b"f0"],
        context=IngestContext(source_collection="c"),
        source_doc_id="hash-1",
        keyframe_source_type="video_keyframe",
        link_field=None,
        extra_metadata={"media_file_hash": "hash-1", "source_file": "clip.mp4"},
    )
    payload = stored[0].metadata
    assert payload["source_type"] == "video_keyframe"
    assert "posting_uuid" not in payload
    assert payload["source_doc_id"] == "hash-1"
    assert payload["source_file"] == "clip.mp4"


def test_keyframe_payload_carries_index_and_time(monkeypatch: pytest.MonkeyPatch) -> None:
    """A timed frame records where in the clip it came from."""
    svc, stored = _service(monkeypatch)
    svc.ingest_keyframe_set(
        [b"f0", b"f1"],
        context=IngestContext(source_collection="c"),
        source_doc_id="hash-1",
        keyframe_source_type="video_keyframe",
        link_field=None,
        frame_times=[0.0, 12.5],
        dedup_cosine=1.01,
    )
    assert [(n.metadata["keyframe_index"], n.metadata["keyframe_time_sec"]) for n in stored] == [(0, 0.0), (1, 12.5)]


def test_keyframe_index_survives_the_dedup_prune(monkeypatch: pytest.MonkeyPatch) -> None:
    """A pruned frame does not renumber its neighbours; the index stays Nextext's."""
    svc, stored = _service(monkeypatch)
    svc.ingest_keyframe_set(
        [b"f0", b"f1"],
        context=IngestContext(source_collection="c"),
        source_doc_id="hash-1",
        link_field=None,
        frame_times=[0.0, 12.5],
    )
    assert [(n.metadata["keyframe_index"], n.metadata["keyframe_time_sec"]) for n in stored] == [(0, 0.0)]


def test_keyframe_payload_without_times_keeps_its_index(monkeypatch: pytest.MonkeyPatch) -> None:
    """An untimed frame still records its position, so ordering survives."""
    svc, stored = _service(monkeypatch)
    svc.ingest_keyframe_set(
        [b"f0"],
        context=IngestContext(source_collection="c"),
        source_doc_id="hash-1",
        link_field=None,
    )
    assert stored[0].metadata["keyframe_index"] == 0
    assert stored[0].metadata["keyframe_time_sec"] is None


def test_keyframe_times_that_do_not_pair_are_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mislabelling a frame is worse than leaving it untimed."""
    svc, stored = _service(monkeypatch)
    svc.ingest_keyframe_set(
        [b"f0", b"f1"],
        context=IngestContext(source_collection="c"),
        source_doc_id="hash-1",
        link_field=None,
        frame_times=[3.0],
        dedup_cosine=1.01,
    )
    assert [n.metadata["keyframe_time_sec"] for n in stored] == [None, None]
