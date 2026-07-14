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


def test_transcript_segments_merge_posting_reference_fields_additively(tmp_path: Path) -> None:
    """Posting reference fields join the segment's reference_metadata without replacing the Nextext identity."""
    jsonl = tmp_path / "clip.nextext.jsonl"
    jsonl.write_text('{"text":"hello","start_seconds":0,"end_seconds":1}\n', encoding="utf-8")
    docs = list(
        CustomJSONReader(is_jsonl=True).iter_documents(
            jsonl,
            extra_info={
                "posting_uuid": "uuid-9",
                "posting_id": "P_1",
                "media_id": "P_1_0",
                "posting_network": "Facebook",
                "posting_author": "Jane Poster",
                "posting_author_id": "42",
                "posting_vanity": "jane.poster",
                "posting_timestamp": "2023-01-01 10:00",
                "posting_url": "https://fb.example/p1",
                "posting_text": "Original post body",
            },
        )
    )
    assert len(docs) == 1
    ref = docs[0].metadata["reference_metadata"]
    # Additive: the segment's own identity fields are untouched.
    assert ref["network"] == "nextext"
    assert ref["type"] == "transcript_segment"
    assert ref["text_id"].endswith(":0")
    # The posting's fields ride along, prefixed.
    assert ref["posting_network"] == "Facebook"
    assert ref["posting_author"] == "Jane Poster"
    assert ref["posting_author_id"] == "42"
    assert ref["posting_vanity"] == "jane.poster"
    assert ref["posting_timestamp"] == "2023-01-01 10:00"
    assert ref["posting_url"] == "https://fb.example/p1"
    assert ref["posting_text"] == "Original post body"


def test_transcript_segments_exclude_reference_metadata_from_embedding(tmp_path: Path) -> None:
    """reference_metadata (segment text + posting text) must not enter the embed input."""
    jsonl = tmp_path / "clip.nextext.jsonl"
    jsonl.write_text('{"text":"hello","start_seconds":0,"end_seconds":1}\n', encoding="utf-8")
    docs = list(CustomJSONReader(is_jsonl=True).iter_documents(jsonl, extra_info={}))
    assert len(docs) == 1
    assert "reference_metadata" in docs[0].excluded_embed_metadata_keys
