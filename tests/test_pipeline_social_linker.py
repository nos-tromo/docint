"""Tests for social-linker integration in DocumentIngestionPipeline."""

import json
from pathlib import Path
from typing import Any

import pytest

from docint.core.ingest.ingestion_pipeline import DocumentIngestionPipeline
from docint.core.ingest.social_linker import SocialLinkResult
from docint.utils.hashing import compute_file_hash


def test_pipeline_skips_consumed_and_yields_linker_documents(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify the pipeline skips consumed paths and injects the linker's Documents.

    Checks that:
    - consumed files (the dossier, a linked a.jpg) are excluded from the sweep;
    - Documents produced by the social linker are yielded;
    - non-consumed files (notes.txt) still flow through.
    """
    (tmp_path / "dossier.json").write_text('{"schema": "me-dossier/1"}', encoding="utf-8")
    (tmp_path / "a.jpg").write_bytes(b"\xff\xd8\xff")
    (tmp_path / "notes.txt").write_text("hello", encoding="utf-8")

    from llama_index.core import Document

    fake_doc = Document(text="spoken", metadata={"posting_uuid": "u1", "docint_doc_kind": "transcript_segment"})

    def fake_run(self: Any, data_dir: Path) -> SocialLinkResult:
        return SocialLinkResult(
            consumed_paths={tmp_path / "dossier.json", tmp_path / "a.jpg"},
            documents=[fake_doc],
        )

    monkeypatch.setattr("docint.core.ingest.social_linker.SocialLinker.run", fake_run)

    pipeline = DocumentIngestionPipeline(
        data_dir=tmp_path, ner_model=None, progress_callback=None, target_collection="c"
    )
    pipeline._load_doc_readers()
    batches = list(pipeline._iter_loaded_documents())
    loaded = [doc for batch in batches for doc in batch]

    texts = {doc.text for doc in loaded}
    assert "spoken" in texts  # linker doc injected
    # The consumed dossier + a.jpg are not re-ingested by the generic sweep.
    filenames = {doc.metadata.get("filename") for doc in loaded}
    assert "a.jpg" not in filenames
    assert "dossier.json" not in filenames
    assert "notes.txt" in filenames


class _StubManifest:
    """No-op manifest stub so the linker touches no SQLite."""

    def close(self) -> None:
        """No-op close (satisfies the manifest interface)."""


class _RecordingImageService:
    """Image-service stub recording the assets the real linker routes to it."""

    def __init__(self) -> None:
        """Initialise with an empty asset list."""
        self.images: list[Any] = []

    def ingest_image(self, asset: Any, *, context: Any) -> None:
        """Record the asset.

        Args:
            asset: The image asset the linker resolved.
            context: Ingestion context (ignored).
        """
        self.images.append(asset)


def test_pipeline_skips_nested_media_the_real_linker_consumed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A nested file claimed by the real linker is not swept up a second time.

    The pipeline subtracts consumed paths by exact ``Path`` equality against
    ``SimpleDirectoryReader.input_files``, so the linker's paths must keep the
    same form as ``data_dir``. The batch is reached through a symlink here so
    that form is observable: the reader keeps the symlinked path, and a linker
    that normalised its own paths (``.resolve()``) would emit the real ones, the
    subtraction would silently stop matching, and every linked image would be
    ingested twice — once linked to its posting, once as a standalone.
    """
    from docint.core.ingest import ingestion_pipeline as pipe_mod

    real_root = tmp_path / "real"
    real_root.mkdir()
    batch = tmp_path / "batch"
    batch.symlink_to(real_root, target_is_directory=True)
    profile = _write_dossier(batch)

    images: Any = _RecordingImageService()
    monkeypatch.setattr(pipe_mod, "ImageIngestionService", lambda *a, **k: object())
    monkeypatch.setattr(DocumentIngestionPipeline, "_open_ingest_manifest", lambda self: _StubManifest())
    pipeline = DocumentIngestionPipeline(
        data_dir=batch,
        ner_model=None,
        progress_callback=None,
        target_collection="c",
        image_ingestion_service=images,
    )
    pipeline._load_doc_readers()

    # The real linker resolved the nested file and claimed it.
    assert [asset.source_doc_id for asset in images.images] == ["u1"]
    assert profile / "media" / "photos" / "shot.jpg" in pipeline.social_link_consumed

    loaded = [doc for batch in pipeline._iter_loaded_documents() for doc in batch]
    filenames = {doc.metadata.get("filename") for doc in loaded}
    assert "shot.jpg" not in filenames
    # The posting arrives once, from the linker — the generic JSON reader never saw the dossier.
    assert "dossier.json" not in filenames
    assert [doc.text for doc in loaded if (doc.metadata.get("table") or {}).get("style") == "postings"] == ["hello"]


def _write_dossier(root: Path) -> Path:
    """Write a one-posting, one-photo ``me-dossier/1`` profile folder under *root*.

    Args:
        root: The export root.

    Returns:
        Path: The profile folder.
    """
    profile = root / "jane.poster - facebook"
    (profile / "media" / "photos").mkdir(parents=True)
    (profile / "media" / "photos" / "shot.jpg").write_bytes(b"\xff\xd8\xff")
    posting = {
        "id": "u1",
        "platformId": "P_1",
        "publishedAt": "2023-01-01T10:00:00+00:00",
        "text": "hello",
        "network": "facebook",
        "author": {"name": "Jane Poster", "vanity": "jane.poster", "platformId": "42"},
        "mediaIds": ["m1"],
    }
    media = {"id": "m1", "platformId": "M_1", "status": "complete", "file": f"{profile.name}/media/photos/shot.jpg"}
    dossier = {"schema": "me-dossier/1", "postings": [posting], "media": [media]}
    (profile / "dossier.json").write_text(json.dumps(dossier), encoding="utf-8")
    return profile


def test_prefilter_leaves_claimed_files_to_the_linker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A re-ingested dossier is counted skipped once, by the linker's rows — not a second time by the prefilter.

    The dossier is a supported ``.json``, so the generic reader lists it; its
    hash is already in the collection because its rows carry it. Counting it
    in the prefilter too would report every profile twice in ``files_skipped``.
    """
    from docint.core.ingest import ingestion_pipeline as pipe_mod

    profile = _write_dossier(tmp_path)
    images: Any = _RecordingImageService()
    monkeypatch.setattr(pipe_mod, "ImageIngestionService", lambda *a, **k: object())
    monkeypatch.setattr(DocumentIngestionPipeline, "_open_ingest_manifest", lambda self: _StubManifest())
    pipeline = DocumentIngestionPipeline(
        data_dir=tmp_path,
        ner_model=None,
        progress_callback=None,
        target_collection="c",
        image_ingestion_service=images,
    )
    pipeline._load_doc_readers()

    pipeline._filter_input_files({compute_file_hash(profile / "dossier.json")})

    assert pipeline.prefilter_skipped == 0
