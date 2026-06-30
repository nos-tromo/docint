"""Test that transcript segments carry posting link metadata."""

from pathlib import Path

from docint.core.readers.json import CustomJSONReader


def test_transcript_segments_carry_posting_uuid(tmp_path: Path) -> None:
    """Verify that posting_uuid is copied into segment reference_metadata."""
    jsonl = tmp_path / "clip.nextext.jsonl"
    jsonl.write_text(
        '{"text":"hello","start_seconds":0,"end_seconds":1}\n{"text":"world","start_seconds":1,"end_seconds":2}\n',
        encoding="utf-8",
    )
    docs = list(
        CustomJSONReader(is_jsonl=True).iter_documents(
            jsonl,
            extra_info={"posting_uuid": "uuid-9", "posting_id": "P_1", "media_id": "P_1_0"},
        )
    )
    assert len(docs) == 2
    for doc in docs:
        assert doc.metadata["posting_uuid"] == "uuid-9"
        assert doc.metadata["reference_metadata"]["posting_uuid"] == "uuid-9"
