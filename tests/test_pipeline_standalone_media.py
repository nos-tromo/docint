"""Integration tests for wiring the standalone media pass into the pipeline."""

from pathlib import Path
from typing import Any

import pytest
from llama_index.core import Document

from docint.core.ingest import ingestion_pipeline as pipe_mod
from docint.core.ingest.ingestion_pipeline import DocumentIngestionPipeline
from docint.core.ingest.media_transcribe import MediaTranscribeResult


class _StubManifest:
    """No-op manifest stub so ``_run_standalone_media`` touches no SQLite."""

    def close(self) -> None:
        """No-op close (satisfies the manifest interface)."""


def _pipeline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> DocumentIngestionPipeline:
    """Build a pipeline with real services/SQLite isolated from ``_run_standalone_media``.

    Args:
        tmp_path: The batch tree root (pytest tmp dir).
        monkeypatch: Pytest's monkeypatch fixture.

    Returns:
        A ``DocumentIngestionPipeline`` whose ``ImageIngestionService`` construction
        and manifest opening are stubbed out.
    """
    pipeline = DocumentIngestionPipeline(
        data_dir=tmp_path, ner_model=None, progress_callback=None, target_collection="c"
    )
    # Isolate _run_standalone_media from real services/disk: it constructs an
    # ImageIngestionService and opens a SQLite manifest before delegating.
    monkeypatch.setattr(pipe_mod, "ImageIngestionService", lambda *a, **k: object())
    # DocumentIngestionPipeline is @dataclass(slots=True), so instance-level
    # attribute patching isn't possible (mirrors the RAG convention documented in
    # tests/test_build_query_engine_linkfollow.py) — patch at the class level.
    monkeypatch.setattr(DocumentIngestionPipeline, "_open_ingest_manifest", lambda self: _StubManifest())
    return pipeline


def test_standalone_pass_merges_into_prepass_accumulators(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The standalone pass's consumed paths + transcript Documents merge into the pre-pass accumulators."""
    (tmp_path / "a.mp4").write_bytes(b"v")
    pipeline = _pipeline(tmp_path, monkeypatch)

    doc = Document(text="hello", metadata={"docint_doc_kind": "transcript_segment"})
    captured: dict[str, Any] = {}

    class _FakeIngestor:
        def run(self, data_dir: Path, already_consumed: set[Path]) -> MediaTranscribeResult:
            captured["already_consumed"] = set(already_consumed)
            return MediaTranscribeResult(consumed_paths={tmp_path / "a.mp4"}, transcript_documents=[doc])

    monkeypatch.setattr(pipe_mod, "StandaloneMediaIngestor", lambda *a, **k: _FakeIngestor())

    pipeline._run_standalone_media()

    assert (tmp_path / "a.mp4") in pipeline.social_link_consumed
    assert doc in pipeline.social_link_documents
    assert captured["already_consumed"] == set()  # social consumed nothing


def test_standalone_pass_excludes_socially_consumed_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Paths the social linker already claimed are passed through as ``already_consumed``."""
    claimed = tmp_path / "linked.mp4"
    claimed.write_bytes(b"v")
    pipeline = _pipeline(tmp_path, monkeypatch)
    pipeline.social_link_consumed = {claimed}

    seen: dict[str, Any] = {}

    class _FakeIngestor:
        def run(self, data_dir: Path, already_consumed: set[Path]) -> MediaTranscribeResult:
            seen["already_consumed"] = set(already_consumed)
            return MediaTranscribeResult()

    monkeypatch.setattr(pipe_mod, "StandaloneMediaIngestor", lambda *a, **k: _FakeIngestor())
    pipeline._run_standalone_media()
    assert claimed in seen["already_consumed"]  # the linker's claim is passed through so it is skipped


def test_pipeline_coexistence_merges_social_and_standalone(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A mixed batch merges the social linker's and the standalone pass's contributions.

    Simulates the social linker having already claimed one clip and produced a
    transcript, then runs the standalone pass: the social-claimed path is excluded
    from the standalone walk (no double-ingest), and both passes' consumed paths and
    transcript Documents end up merged (social first, standalone appended).
    """
    social_clip = tmp_path / "linked.mp4"
    social_clip.write_bytes(b"s")
    loose_clip = tmp_path / "loose.mp4"
    loose_clip.write_bytes(b"l")
    pipeline = _pipeline(tmp_path, monkeypatch)

    # Simulate the social linker's output (it runs first in _load_doc_readers).
    social_doc = Document(text="social transcript", metadata={"docint_doc_kind": "transcript_segment"})
    pipeline.social_link_consumed = {social_clip}
    pipeline.social_link_documents = [social_doc]

    standalone_doc = Document(text="loose transcript", metadata={"docint_doc_kind": "transcript_segment"})
    seen: dict[str, Any] = {}

    class _FakeIngestor:
        def run(self, data_dir: Path, already_consumed: set[Path]) -> MediaTranscribeResult:
            seen["already_consumed"] = set(already_consumed)
            return MediaTranscribeResult(consumed_paths={loose_clip}, transcript_documents=[standalone_doc])

    monkeypatch.setattr(pipe_mod, "StandaloneMediaIngestor", lambda *a, **k: _FakeIngestor())

    pipeline._run_standalone_media()

    assert social_clip in seen["already_consumed"]  # social's claim excluded from the standalone walk
    assert pipeline.social_link_consumed == {social_clip, loose_clip}  # both merged
    assert pipeline.social_link_documents == [social_doc, standalone_doc]  # social first, standalone appended
