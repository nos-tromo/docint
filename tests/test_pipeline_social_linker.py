"""Tests for social-linker integration in DocumentIngestionPipeline (Task 11)."""

from pathlib import Path
from typing import Any

import pytest

from docint.core.ingest.ingestion_pipeline import DocumentIngestionPipeline
from docint.core.ingest.social_linker import SocialLinkResult


def test_pipeline_skips_consumed_and_yields_transcripts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify the pipeline skips consumed paths and injects transcript Documents.

    Checks that:
    - consumed media files (media.csv, a.jpg) are excluded from the sweep;
    - transcript Documents produced by the social linker are yielded;
    - non-consumed files (postings.csv) still flow through.
    """
    (tmp_path / "media.csv").write_text("Media ID,Exported media filename\nP_1_0,a.jpg\n", encoding="utf-8")
    (tmp_path / "a.jpg").write_bytes(b"\xff\xd8\xff")
    (tmp_path / "postings.csv").write_text("Posting ID,UUID,Text Content\nP_1,u1,hello\n", encoding="utf-8")

    from llama_index.core import Document

    fake_doc = Document(text="spoken", metadata={"posting_uuid": "u1", "docint_doc_kind": "transcript_segment"})

    def fake_run(self: Any, data_dir: Path) -> SocialLinkResult:
        return SocialLinkResult(
            consumed_paths={tmp_path / "media.csv", tmp_path / "a.jpg"},
            transcript_documents=[fake_doc],
        )

    monkeypatch.setattr("docint.core.ingest.social_linker.SocialLinker.run", fake_run)

    pipeline = DocumentIngestionPipeline(
        data_dir=tmp_path, ner_model=None, progress_callback=None, target_collection="c"
    )
    pipeline._load_doc_readers()
    batches = list(pipeline._iter_loaded_documents())
    loaded = [doc for batch in batches for doc in batch]

    texts = {doc.text for doc in loaded}
    assert "spoken" in texts  # transcript doc injected
    # The consumed media.csv + a.jpg are not re-ingested by the generic sweep.
    filenames = {doc.metadata.get("filename") for doc in loaded}
    assert "a.jpg" not in filenames
    assert "media.csv" not in filenames
    # postings.csv still flows through the sweep (not consumed).
    # Note: the 3-column CSV doesn't match the full postings schema profile (25 cols
    # required for exact-match detection), so _guess_text_cols falls back to the
    # first column ("Posting ID").  We verify presence via the filename metadata key
    # instead of by text content.
    assert "postings.csv" in filenames


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
    import pandas as pd

    from docint.core.ingest import ingestion_pipeline as pipe_mod
    from docint.core.readers.tables import TableReader

    real_root = tmp_path / "real"
    real_root.mkdir()
    batch = tmp_path / "batch"
    batch.symlink_to(real_root, target_is_directory=True)
    photos = batch / "dir" / "photos"
    photos.mkdir(parents=True)
    (photos / "shot.jpg").write_bytes(b"\xff\xd8\xff")
    (batch / "media.csv").write_text("Media ID,Exported media filename\nP_1_0,shot.jpg\n", encoding="utf-8")
    # The postings table is detected by exact header-set equality, so build the
    # full profile from its single source of truth rather than restating it.
    columns = next(profile.headers for profile in TableReader.schema_profiles if profile.style == "postings")
    postings_data: dict[str, list[str]] = {column: [""] for column in columns}
    postings_data["UUID"] = ["u1"]
    postings_data["Posting ID"] = ["P_1"]
    postings_data["Text Content"] = ["hello"]
    pd.DataFrame(postings_data).to_csv(batch / "postings.csv", index=False)

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
    assert photos / "shot.jpg" in pipeline.social_link_consumed

    loaded = [doc for batch in pipeline._iter_loaded_documents() for doc in batch]
    assert "shot.jpg" not in {doc.metadata.get("filename") for doc in loaded}
