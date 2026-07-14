"""Tests for posting reference metadata on retrieved sources.

Covers the query-side half of the posting-enrichment feature: the
``_extract_reference_metadata`` projection must include the ``posting_*``
registry fields, and ``_retrieve_image_sources`` must attach the nested
``reference_metadata`` block stamped on social-linked image/keyframe payloads.
"""

from __future__ import annotations

from typing import Any, cast

import pytest

import docint.core.rag as rag_module
from docint.core.rag import RAG


def test_extract_reference_metadata_projects_posting_fields() -> None:
    """The projection keeps posting_* keys alongside the artifact's own fields."""
    payload = {
        "reference_metadata": {
            "network": "nextext",
            "type": "transcript_segment",
            "posting_uuid": "u1",
            "posting_network": "Facebook",
            "posting_author": "Jane Poster",
            "posting_url": "https://fb.example/p1",
            "posting_text": "Original post body",
            "not_registered": "dropped",
        }
    }
    extracted = RAG._extract_reference_metadata(payload)
    assert extracted is not None
    assert extracted["network"] == "nextext"
    assert extracted["posting_uuid"] == "u1"
    assert extracted["posting_network"] == "Facebook"
    assert extracted["posting_author"] == "Jane Poster"
    assert extracted["posting_url"] == "https://fb.example/p1"
    assert extracted["posting_text"] == "Original post body"
    assert "not_registered" not in extracted


class _StubImageService:
    """Image-service stub returning canned payload matches."""

    def __init__(self, matches: list[dict[str, Any]]) -> None:
        """Store the canned matches.

        Args:
            matches: Payload dicts returned by the text query.
        """
        self._matches = matches

    def _resolve_collection_name(self, collection: str) -> str:
        """Mirror the real ``{collection}_images`` naming."""
        return f"{collection}_images"

    def query_similar_images_by_text(self, **_kwargs: Any) -> list[dict[str, Any]]:
        """Return the canned matches regardless of the query."""
        return self._matches


def test_retrieve_image_sources_attaches_reference_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    """Image sources carry the posting reference block stamped at ingest."""
    payload = {
        "image_id": "img-1",
        "source_type": "social_media_keyframe",
        "posting_uuid": "u1",
        "llm_description": "a red banner",
        "reference_metadata": {
            "type": "keyframe",
            "posting_uuid": "u1",
            "posting_network": "Facebook",
            "posting_author": "Jane Poster",
            "posting_url": "https://fb.example/p1",
        },
    }
    plain_payload = {"image_id": "img-2", "llm_description": "an unlinked image"}

    rag = object.__new__(RAG)
    rag._image_ingestion_service = cast(Any, _StubImageService([payload, plain_payload]))
    rag._qdrant_client = cast(Any, object())
    monkeypatch.setattr(rag_module, "qdrant_collection_exists", lambda *_a, **_k: True)

    with rag.collection_scope("col"):
        sources = rag._retrieve_image_sources("banner")

    assert len(sources) == 2
    ref = sources[0]["reference_metadata"]
    assert ref["type"] == "keyframe"
    assert ref["posting_uuid"] == "u1"
    assert ref["posting_network"] == "Facebook"
    assert ref["posting_author"] == "Jane Poster"
    assert ref["posting_url"] == "https://fb.example/p1"
    assert sources[0]["posting_uuid"] == "u1"
    # An image without a stamped block gets no reference_metadata key at all.
    assert "reference_metadata" not in sources[1]
