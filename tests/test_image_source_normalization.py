"""Tests that image payloads normalize through the shared source normalizer.

Images used to build their source dicts in a hand-written block inside
``_retrieve_image_sources`` while every other source type went through
``_source_from_payload``. The two drifted -- separate preview-url assembly,
separate file-hash key, separate text assembly, a second copy of the
citation-identity rule -- and only one of them could answer "what is a
source?". These tests pin the image payload shape onto the shared normalizer
so there is one answer.
"""

from typing import Any

from docint.core.rag import RAG

IMAGE_PAYLOAD: dict[str, Any] = {
    "image_id": "img-9f2c",
    "node_id": "point-77",
    "source_doc_id": "hash-abc123",
    "source_path": "/ingest/batch/site-plan.png",
    "mime_type": "image/png",
    "source_type": "image",
    "page_number": 4,
    "llm_description": "A hand-drawn site plan with a numbered legend.",
    "llm_tags": ["plan", "legend"],
    "image_collection": "uabc__docs_images",
    "bbox": {"x": 1, "y": 2, "w": 3, "h": 4},
}


def test_image_payload_yields_the_caption_as_source_text() -> None:
    """The stored caption and tags become the source body."""
    src = RAG._source_from_payload(collection="uabc__docs", payload=IMAGE_PAYLOAD)

    assert "hand-drawn site plan" in src["text"]
    assert "plan, legend" in src["text"]
    assert src["preview_text"]


def test_image_payload_carries_its_citation_identity() -> None:
    """An image is traceable to one point and one durable content hash."""
    src = RAG._source_from_payload(collection="uabc__docs", payload=IMAGE_PAYLOAD)

    assert src["id"] == "point-77"
    assert src["chunk_id"] == "img-9f2c"


def test_image_payload_resolves_its_locators() -> None:
    """Filename, type, page and hash come off the image-specific keys."""
    src = RAG._source_from_payload(collection="uabc__docs", payload=IMAGE_PAYLOAD)

    assert src["filename"] == "site-plan.png"
    assert src["filetype"] == "image/png"
    assert src["source"] == "image"
    assert src["page"] == 4
    assert src["file_hash"] == "hash-abc123"


def test_image_payload_builds_the_preview_link_like_any_source() -> None:
    """The preview link is assembled once, by the shared normalizer."""
    src = RAG._source_from_payload(collection="uabc__docs", payload=IMAGE_PAYLOAD)

    assert src["preview_url"] == "/sources/preview?collection=uabc__docs&file_hash=hash-abc123"
    assert src["document_url"] == src["preview_url"]


def test_image_payload_keeps_the_image_only_extras() -> None:
    """Image id, companion collection and bbox survive normalization."""
    src = RAG._source_from_payload(collection="uabc__docs", payload=IMAGE_PAYLOAD)

    assert src["image_id"] == "img-9f2c"
    assert src["image_collection"] == "uabc__docs_images"
    assert src["bbox"] == {"x": 1, "y": 2, "w": 3, "h": 4}


def test_image_id_backs_the_identity_when_no_point_id_is_present() -> None:
    """A payload scrolled without its point id still cites something."""
    payload = {k: v for k, v in IMAGE_PAYLOAD.items() if k != "node_id"}

    src = RAG._source_from_payload(collection="uabc__docs", payload=payload)

    assert src["id"] == "img-9f2c"


def test_the_words_inside_the_image_lead_its_evidence_body() -> None:
    """Stored but unread would mean the ranker still judges the caption alone.

    An image's ``ocr_text`` is what a reader searched for; it has to reach the
    reranker and the generator, not only the keyword index.
    """
    payload = dict(IMAGE_PAYLOAD, ocr_text="Bauantrag 2031/44 — Erweiterung Halle B")

    src = RAG._source_from_payload(collection="uabc__docs", payload=payload)

    assert src["text"].startswith("Bauantrag 2031/44")
    assert "hand-drawn site plan" in src["text"]
    assert src["text"].index("Bauantrag") < src["text"].index("hand-drawn")


def test_an_image_with_only_printed_words_still_has_a_body() -> None:
    """A screenshot the captioner could not describe is still evidence."""
    payload = {k: v for k, v in IMAGE_PAYLOAD.items() if k not in {"llm_description", "llm_tags"}}
    payload["ocr_text"] = "Bauantrag 2031/44"

    src = RAG._source_from_payload(collection="uabc__docs", payload=payload)

    assert src["text"] == "Bauantrag 2031/44"


def test_an_image_without_printed_words_reads_as_before() -> None:
    """The commonest image carries no text, and nothing about it changes."""
    src = RAG._source_from_payload(collection="uabc__docs", payload=IMAGE_PAYLOAD)

    assert src["text"] == "A hand-drawn site plan with a numbered legend.\n\nTags: plan, legend"


def test_tags_alone_still_produce_a_body() -> None:
    """An image with tags but no caption is not left with empty evidence."""
    payload = {k: v for k, v in IMAGE_PAYLOAD.items() if k != "llm_description"}

    src = RAG._source_from_payload(collection="uabc__docs", payload=payload)

    assert src["text"] == "Tags: plan, legend"


def test_text_sources_are_unaffected_by_the_image_keys() -> None:
    """A document payload normalizes exactly as before."""
    src = RAG._source_from_payload(
        collection="uabc__docs",
        payload={"filename": "handbook.pdf", "page": 26, "text": "Station 3."},
        node_id="node-1",
    )

    assert src["filename"] == "handbook.pdf"
    assert src["text"] == "Station 3."
    assert "image_id" not in src
    assert "bbox" not in src
