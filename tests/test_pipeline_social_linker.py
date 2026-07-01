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
