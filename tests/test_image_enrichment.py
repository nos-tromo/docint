"""Tests for the ingest job's NER + hate-speech pass over the image companion."""

from __future__ import annotations

import asyncio
import json
import uuid
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from qdrant_client import QdrantClient, models

import docint.core.ingest.ingestion_pipeline as pipeline_module
import docint.core.rag as rag_module
from docint.core.ingest.image_enrichment import enrich_pending_images, hate_speech_text, image_text_fields
from docint.core.ingest.images_service import ImageIngestionService
from docint.core.ingest.ingestion_pipeline import DocumentIngestionPipeline
from docint.core.rag import RAG

_COMPANION = "docs_images"
_FIGURE_OCR = "HATEFUL slogan printed on the poster"
_FIGURE_CAPTION = "Poster held up in a town square"
_KEYFRAME_CAPTION = "Speaker at a lectern"
_SYMBOL_CAPTION = "A flag bearing a HATEFUL symbol hangs on a wall"


def _point_id(name: str) -> str:
    """Return a stable UUID point id for *name*.

    Args:
        name: A label for the point.

    Returns:
        The point id.
    """
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"test:{name}"))


FIGURE = _point_id("figure")
KEYFRAME = _point_id("keyframe")
FINISHED = _point_id("finished")
LEGACY = _point_id("legacy")
TWIN = _point_id("twin")
SYMBOL = _point_id("symbol")


def _figure_payload(**extra: Any) -> dict[str, Any]:
    """Payload of a PDF figure whose printed words are hateful.

    Args:
        **extra: Keys to add or override.

    Returns:
        The payload.
    """
    return {
        "image_id": "img-figure",
        "source_type": "document",
        "source_doc_id": "pdf-hash",
        "source_path": "batch/report.pdf",
        "page_number": 3,
        "ocr_text": _FIGURE_OCR,
        "llm_description": _FIGURE_CAPTION,
        "llm_tags": ["poster"],
        **extra,
    }


def _store(client: QdrantClient, collection: str, points: dict[str, dict[str, Any]]) -> None:
    """Create *collection* (if needed) and upsert *points* into it.

    Args:
        client: The in-memory client.
        collection: Collection name.
        points: ``{point_id: payload}``.
    """
    if not client.collection_exists(collection):
        client.create_collection(
            collection,
            vectors_config={"dense": models.VectorParams(size=3, distance=models.Distance.COSINE)},
        )
    client.upsert(
        collection,
        points=[
            models.PointStruct(id=point_id, vector={"dense": [0.1, 0.2, 0.3]}, payload=payload)
            for point_id, payload in points.items()
        ],
    )


def _payload(client: QdrantClient, collection: str, point_id: str) -> dict[str, Any]:
    """Return one stored payload.

    Args:
        client: The in-memory client.
        collection: Collection name.
        point_id: The point.

    Returns:
        Its payload.
    """
    (record,) = client.retrieve(collection, ids=[point_id], with_payload=True)
    return dict(record.payload or {})


class _Verdicts:
    """Chat-model stand-in that endorses only the hateful marker, recording every prompt."""

    def __init__(self) -> None:
        """Start with no prompts."""
        self.prompts: list[str] = []

    def complete(self, prompt: str, **_kwargs: Any) -> SimpleNamespace:
        """Return a verdict for the text rendered into *prompt*.

        Args:
            prompt: The rendered hate-speech prompt.
            **_kwargs: Request options such as ``response_format`` (ignored).

        Returns:
            A response carrying the verdict as JSON ``text``.
        """
        self.prompts.append(prompt)
        endorsed = "HATEFUL" in prompt
        verdict = {
            "hate_speech": endorsed,
            "stance": "endorses" if endorsed else "none",
            "category": "religion" if endorsed else "none",
            "confidence": "high" if endorsed else "low",
            "reason": "Endorses excluding a group." if endorsed else "",
        }
        return SimpleNamespace(text=json.dumps(verdict))


def _pipeline(
    monkeypatch: pytest.MonkeyPatch, data_dir: Path, model: _Verdicts, read: list[str]
) -> DocumentIngestionPipeline:
    """Build a real pipeline with hate speech on and a recording NER extractor.

    Only the env, the prompt file and the remote models are stubbed; the
    enrichment code is the pipeline's own.

    Args:
        monkeypatch: The monkeypatch fixture.
        data_dir: Any directory.
        model: The hate-speech model stand-in.
        read: Receives every text handed to NER.

    Returns:
        The pipeline.
    """
    ner_cfg = replace(pipeline_module.load_ner_env(), enabled=False, max_workers=1)
    hate_cfg = replace(pipeline_module.load_hate_speech_env(), enabled=True, max_workers=1)
    monkeypatch.setattr(pipeline_module, "load_ner_env", lambda: ner_cfg)
    monkeypatch.setattr(pipeline_module, "load_hate_speech_env", lambda: hate_cfg)
    monkeypatch.setattr(
        pipeline_module, "OpenAIPipeline", lambda: SimpleNamespace(load_prompt=lambda kw: "Classify:\n{text}")
    )
    pipeline = DocumentIngestionPipeline(
        data_dir=data_dir,
        ner_model=None,
        progress_callback=None,
        hate_speech_model=cast(Any, model),
    )

    def _extract(text: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        read.append(text)
        return [{"text": text.split()[0], "type": "org"}], []

    pipeline.entity_extractor = _extract
    return pipeline


def test_pending_images_get_ner_on_words_and_caption_and_hate_speech_on_every_labelled_part(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """NER reads the words and the caption; hate-speech detection reads words, description and tags, labelled.

    A picture whose hate is purely visual has no printed words, so its
    description is the only text that shows it. The labels tell the
    classifier the text describes an image.

    Args:
        monkeypatch: The monkeypatch fixture.
        tmp_path: Temporary directory path for the test.
    """
    monkeypatch.setenv("RESPONSE_LANGUAGE", "en")
    client = QdrantClient(location=":memory:")
    _store(
        client,
        _COMPANION,
        {
            FIGURE: _figure_payload(enrichment="pending"),
            KEYFRAME: {
                "image_id": "img-keyframe",
                "source_type": "video_keyframe",
                "ocr_text": "",
                "llm_description": _KEYFRAME_CAPTION,
                "enrichment": "pending",
            },
            SYMBOL: {
                "image_id": "img-symbol",
                "source_type": "social_media",
                "llm_description": _SYMBOL_CAPTION,
                "llm_tags": ["flag", "wall"],
                "enrichment": "pending",
            },
            FINISHED: _figure_payload(image_id="img-finished", enrichment="done"),
            LEGACY: _figure_payload(image_id="img-legacy"),
        },
    )
    model = _Verdicts()
    read: list[str] = []
    pipeline = _pipeline(monkeypatch, tmp_path, model, read)

    enriched = enrich_pending_images(client, _COMPANION, pipeline.enrich_nodes)

    figure_input = f"Image description: {_FIGURE_CAPTION}\n\nTags: poster\n\nText in the image: {_FIGURE_OCR}"
    keyframe_input = f"Image description: {_KEYFRAME_CAPTION}"
    symbol_input = f"Image description: {_SYMBOL_CAPTION}\n\nTags: flag, wall"
    assert enriched == 3
    assert sorted(read) == sorted([f"{_FIGURE_OCR}\n\n{_FIGURE_CAPTION}", _KEYFRAME_CAPTION, _SYMBOL_CAPTION])
    assert sorted(model.prompts) == sorted(
        f"Classify:\n{text}" for text in (figure_input, keyframe_input, symbol_input)
    )

    figure = _payload(client, _COMPANION, FIGURE)
    assert figure["enrichment"] == "done"
    assert figure["entities"] == [{"text": "HATEFUL", "type": "org"}]
    assert figure["hate_speech"]["hate_speech"] is True
    assert figure["hate_speech"]["chunk_text"] == figure_input
    assert figure["hate_speech"]["chunk_id"] == FIGURE

    # No printed words at all: the description alone carries the finding.
    symbol = _payload(client, _COMPANION, SYMBOL)
    assert symbol["hate_speech"]["hate_speech"] is True
    assert symbol["hate_speech"]["chunk_text"] == symbol_input

    keyframe = _payload(client, _COMPANION, KEYFRAME)
    assert keyframe["enrichment"] == "done"
    assert keyframe["entities"] == [{"text": "Speaker", "type": "org"}]
    assert "hate_speech" not in keyframe

    # Only points waiting for the pass are touched: a finished point is not
    # paid for twice, and a point stored before the marker existed is left as
    # its collection was ingested.
    assert "entities" not in _payload(client, _COMPANION, FINISHED)
    legacy = _payload(client, _COMPANION, LEGACY)
    assert "enrichment" not in legacy
    assert "entities" not in legacy


def test_hate_speech_labels_follow_the_response_language(monkeypatch: pytest.MonkeyPatch) -> None:
    """The labels are in the prompt's language, and an image with no text gives no input.

    Args:
        monkeypatch: The monkeypatch fixture.
    """
    monkeypatch.setenv("RESPONSE_LANGUAGE", "de")
    payload = {"ocr_text": "Gedruckt", "llm_description": "Beschreibung", "llm_tags": ["eins", " ", "zwei"]}

    assert (
        hate_speech_text(payload)
        == "Bildbeschreibung: Beschreibung\n\nSchlagworte: eins, zwei\n\nText im Bild: Gedruckt"
    )
    assert hate_speech_text({"ocr_text": " ", "llm_tags": []}) == ""


def test_a_long_printed_text_cannot_push_the_description_out_of_the_judged_text(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Only the first ``HATE_SPEECH_MAX_CHARS`` are judged, so the short description goes first.

    Args:
        monkeypatch: The monkeypatch fixture.
        tmp_path: Temporary directory path for the test.
    """
    monkeypatch.setenv("RESPONSE_LANGUAGE", "en")
    client = QdrantClient(location=":memory:")
    _store(
        client,
        _COMPANION,
        {
            SYMBOL: {
                "image_id": "img-dense",
                "ocr_text": "word " * 2000,
                "llm_description": _SYMBOL_CAPTION,
                "enrichment": "pending",
            }
        },
    )
    pipeline = _pipeline(monkeypatch, tmp_path, _Verdicts(), [])

    enrich_pending_images(client, _COMPANION, pipeline.enrich_nodes)

    assert _payload(client, _COMPANION, SYMBOL)["hate_speech"]["hate_speech"] is True


def test_every_pending_image_is_reached_across_pages(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Settling a page removes it from the filter being paged; no point may be skipped for it.

    Args:
        monkeypatch: The monkeypatch fixture.
        tmp_path: Temporary directory path for the test.
    """
    client = QdrantClient(location=":memory:")
    ids = [_point_id(f"page-{i}") for i in range(5)]
    _store(
        client,
        _COMPANION,
        {
            point_id: {"image_id": f"img-{i}", "llm_description": f"Caption{i} text", "enrichment": "pending"}
            for i, point_id in enumerate(ids)
        },
    )
    pipeline = _pipeline(monkeypatch, tmp_path, _Verdicts(), [])

    enriched = enrich_pending_images(client, _COMPANION, pipeline.enrich_nodes, page_size=2)

    assert enriched == 5
    assert [_payload(client, _COMPANION, point_id)["enrichment"] for point_id in ids] == ["done"] * 5


def _rag_over(client: QdrantClient, monkeypatch: pytest.MonkeyPatch) -> RAG:
    """Return a RAG for collection ``docs`` reading *client*.

    Args:
        client: The in-memory client.
        monkeypatch: The monkeypatch fixture.

    Returns:
        The RAG.
    """
    monkeypatch.setattr(RAG, "qdrant_client", property(lambda self: client))
    rag = RAG(qdrant_collection="docs")
    # Resolves the companion's name without building a client of its own.
    rag._image_ingestion_service = ImageIngestionService(qdrant_client=client)
    return rag


@pytest.mark.parametrize("mode", ["sync", "async"])
def test_ingest_docs_enriches_the_images_waiting_for_it(
    mode: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The job runs the pass with its own pipeline, after every lane has stored its images.

    Args:
        mode: ``sync`` drives ``ingest_docs``, ``async`` drives ``asingest_docs``.
        monkeypatch: The monkeypatch fixture.
        tmp_path: Temporary directory path for the test.
    """
    client = QdrantClient(location=":memory:")
    _store(client, "docs", {_point_id("chunk"): {"text": "Already ingested chunk."}})
    _store(client, _COMPANION, {FIGURE: _figure_payload(enrichment="pending")})
    pipeline = _pipeline(monkeypatch, tmp_path, _Verdicts(), [])

    monkeypatch.setattr(DocumentIngestionPipeline, "build_streaming", lambda self, existing_hashes=None: iter(()))
    monkeypatch.setattr(RAG, "_build_ingestion_pipeline", lambda self, progress_callback=None, **_kw: pipeline)
    monkeypatch.setattr(RAG, "create_collection_if_missing", lambda self: None)
    monkeypatch.setattr(RAG, "probe_sparse_endpoint", lambda self: None)
    monkeypatch.setattr(RAG, "probe_embed_endpoint", lambda self: None)
    monkeypatch.setattr(RAG, "_prepare_sources_dir", lambda self, path: path)
    monkeypatch.setattr(RAG, "_vector_store", lambda self: object())
    monkeypatch.setattr(RAG, "_storage_context", lambda self, vector_store: object())
    monkeypatch.setattr(RAG, "embed_model", property(lambda self: object()))
    monkeypatch.setattr(RAG, "_get_existing_file_hashes", lambda self: set())
    monkeypatch.setattr(RAG, "_build_ingest_manifest", lambda self, *a, **k: rag_module.NullIngestManifest())
    monkeypatch.setattr(RAG, "reset_session_state", lambda self: None)
    monkeypatch.setattr(RAG, "_bump_summary_revision", lambda self, collection: None)
    monkeypatch.setattr(rag_module, "VectorStoreIndex", lambda **_kw: SimpleNamespace())
    monkeypatch.setattr(rag_module, "ensure_search_index", lambda client, collection: None)
    monkeypatch.setattr(rag_module, "prefetch_batch", lambda *a, **k: 0)
    rag = _rag_over(client, monkeypatch)

    if mode == "sync":
        rag.ingest_docs(tmp_path, build_query_engine=False)
    else:
        asyncio.run(rag.asingest_docs(tmp_path, build_query_engine=False))

    figure = _payload(client, _COMPANION, FIGURE)
    assert figure["enrichment"] == "done"
    assert figure["hate_speech"]["hate_speech"] is True


def _analysed_collection() -> QdrantClient:
    """A collection with one analysed chunk, one analysed figure, and one image that has a text twin.

    Returns:
        The in-memory client.
    """
    client = QdrantClient(location=":memory:")
    finding = {
        "hate_speech": True,
        "category": "religion",
        "confidence": "high",
        "reason": "Endorses excluding a group.",
    }
    _store(
        client,
        "docs",
        {
            _point_id("chunk"): {
                "text": "Chunk text naming Acme.",
                "filename": "notes.txt",
                "entities": [{"text": "Acme", "type": "org"}],
                "hate_speech": {**finding, "chunk_text": "Chunk text naming Acme."},
            }
        },
    )
    _store(
        client,
        _COMPANION,
        {
            FIGURE: _figure_payload(
                enrichment="done",
                entities=[{"text": "Berlin", "type": "location"}],
                hate_speech={**finding, "chunk_id": FIGURE, "chunk_text": _FIGURE_OCR, "source_ref": ""},
            ),
            # A standalone file is also a main-collection document; that
            # document carries its findings.
            TWIN: _figure_payload(
                image_id="img-twin",
                source_type="social_media",
                occurrences=[{"source_type": "social_media"}, {"source_type": "standalone"}],
                enrichment="done",
                entities=[{"text": "Berlin", "type": "location"}],
                hate_speech={**finding, "chunk_id": TWIN, "chunk_text": _FIGURE_OCR, "source_ref": ""},
            ),
        },
    )
    return client


def test_hate_speech_findings_include_the_words_inside_images(monkeypatch: pytest.MonkeyPatch) -> None:
    """A finding read off an image is listed beside the text findings, named after its document.

    Args:
        monkeypatch: The monkeypatch fixture.
    """
    rag = _rag_over(_analysed_collection(), monkeypatch)

    rows = {row["chunk_id"]: row for row in rag.get_collection_hate_speech()}

    assert set(rows) == {_point_id("chunk"), FIGURE}
    figure = rows[FIGURE]
    assert figure["chunk_text"] == _FIGURE_OCR
    assert figure["source_ref"] == "report.pdf"
    assert figure["page"] == 3
    assert figure["image_id"] == "img-figure"
    assert figure["category"] == "religion"


def test_hate_speech_rows_say_whether_text_or_an_image_was_judged(monkeypatch: pytest.MonkeyPatch) -> None:
    """A row judged from an image says so, whether it lives on the companion or is a standalone file's document.

    Args:
        monkeypatch: The monkeypatch fixture.
    """
    client = _analysed_collection()
    standalone = _point_id("standalone-document")
    _store(
        client,
        "docs",
        {
            standalone: {
                "text": f"{_SYMBOL_CAPTION}\n\nTags: flag",
                "filename": "flag.png",
                "image_id": "img-standalone",
                "llm_description": _SYMBOL_CAPTION,
                "hate_speech": {
                    "hate_speech": True,
                    "category": "extremism",
                    "confidence": "high",
                    "reason": "Shows a hate symbol without distance.",
                    "chunk_text": f"{_SYMBOL_CAPTION}\n\nTags: flag",
                },
            }
        },
    )
    rag = _rag_over(client, monkeypatch)

    rows = {row["chunk_id"]: row for row in rag.get_collection_hate_speech()}

    assert rows[_point_id("chunk")]["basis"] == "text"
    assert rows[FIGURE]["basis"] == "image"
    assert rows[standalone]["basis"] == "image"


def test_entity_sources_include_the_words_and_caption_of_images(monkeypatch: pytest.MonkeyPatch) -> None:
    """An image's entities feed every NER view, quoting the text they were read from.

    Args:
        monkeypatch: The monkeypatch fixture.
    """
    rag = _rag_over(_analysed_collection(), monkeypatch)

    rows = {row["chunk_id"]: row for row in rag._load_collection_ner_sources()}

    assert set(rows) == {_point_id("chunk"), FIGURE}
    figure = rows[FIGURE]
    assert figure["entities"] == [{"text": "Berlin", "type": "location"}]
    assert figure["chunk_text"] == f"{_FIGURE_OCR}\n\n{_FIGURE_CAPTION}"
    assert figure["filename"] == "report.pdf"
    assert figure["image_id"] == "img-figure"


def test_image_text_fields_keep_an_images_words_description_and_tags_apart() -> None:
    """Each part is its own field, trimmed, and only when the image carries it."""
    padded = {"ocr_text": f"  {_FIGURE_OCR}\n", "llm_description": f"{_FIGURE_CAPTION} ", "llm_tags": [" ", "crowd "]}
    assert image_text_fields(padded) == {
        "ocr_text": _FIGURE_OCR,
        "image_description": _FIGURE_CAPTION,
        "image_tags": ["crowd"],
    }
    untagged = {"ocr_text": "", "llm_description": _KEYFRAME_CAPTION, "llm_tags": "not-a-list"}
    assert image_text_fields(untagged) == {"image_description": _KEYFRAME_CAPTION}
    assert image_text_fields({"text": "Chunk text naming Acme.", "filename": "notes.txt"}) == {}


def test_finding_rows_carry_an_images_parts_apart_from_its_judged_text(monkeypatch: pytest.MonkeyPatch) -> None:
    """Image rows name their printed words, description and tags separately; text rows gain nothing.

    Args:
        monkeypatch: The monkeypatch fixture.
    """
    client = _analysed_collection()
    standalone = _point_id("standalone-document")
    _store(
        client,
        "docs",
        {
            standalone: {
                "text": f"{_FIGURE_OCR}\n\n{_SYMBOL_CAPTION}\n\nTags: flag",
                "filename": "flag.png",
                "image_id": "img-standalone",
                "ocr_text": _FIGURE_OCR,
                "llm_description": _SYMBOL_CAPTION,
                "llm_tags": ["flag"],
                "entities": [{"text": "Berlin", "type": "location"}],
                "hate_speech": {
                    "hate_speech": True,
                    "category": "extremism",
                    "confidence": "high",
                    "reason": "Shows a hate symbol without distance.",
                    "chunk_text": f"{_FIGURE_OCR}\n\n{_SYMBOL_CAPTION}\n\nTags: flag",
                },
            }
        },
    )
    rag = _rag_over(client, monkeypatch)
    figure_parts = {"ocr_text": _FIGURE_OCR, "image_description": _FIGURE_CAPTION, "image_tags": ["poster"]}
    standalone_parts = {"ocr_text": _FIGURE_OCR, "image_description": _SYMBOL_CAPTION, "image_tags": ["flag"]}

    for rows in (
        {row["chunk_id"]: row for row in rag.get_collection_hate_speech()},
        {row["chunk_id"]: row for row in rag._load_collection_ner_sources()},
    ):
        assert {key: rows[FIGURE][key] for key in figure_parts} == figure_parts
        assert {key: rows[standalone][key] for key in standalone_parts} == standalone_parts
        assert not set(figure_parts) & set(rows[_point_id("chunk")])
