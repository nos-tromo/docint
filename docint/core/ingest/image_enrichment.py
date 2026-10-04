"""Run the ingest job's NER + hate-speech pass over the image companion's points.

An image's words live only in the ``{collection}_images`` companion: the text
printed in it (``ocr_text``) and the vision model's caption
(``llm_description``). No lane the pipeline enriches reads them — except a
standalone image file's, which ``ImageReader`` also writes into the main
collection as a document — so the words on a poster in a posting, a slide in
a video or a figure in a PDF reached neither the Entities nor the Hate speech
view.

The image service marks every point it writes fresh ``pending``, and the job
runs this pass once its lanes are done. Points written at upload time, before
any job existed to say which stages are on, are therefore enriched with the
stages of the job that ingests them. Points stored before the marker existed
carry none and are left as they were ingested, like the text chunks of a
collection ingested before a stage was switched on.
"""

from __future__ import annotations

from typing import Any, Protocol

from llama_index.core.schema import BaseNode, TextNode
from qdrant_client import models

from docint.core.storage.scroll import iter_scroll
from docint.utils.ui_strings import ui_string

ENRICHMENT_FIELD: str = "enrichment"
"""Payload key holding an image point's enrichment state."""

ENRICHMENT_PENDING: str = "pending"
"""Written fresh, waiting for the job's pass."""

ENRICHMENT_DONE: str = "done"
"""Settled by the pass, whether or not a stage found anything."""

ENRICHMENT_RESULT_KEYS: tuple[str, ...] = ("entities", "relations", "hate_speech")
"""Payload keys the pass writes."""

FINDING_BASIS_IMAGE: str = "image"
"""A finding judged from an image's printed words, description and tags."""

FINDING_BASIS_TEXT: str = "text"
"""A finding judged from a text chunk or transcript segment."""

_SOURCE_PAYLOAD_KEYS: list[str] = ["ocr_text", "llm_description", "llm_tags"]


class NodeEnricher(Protocol):
    """The run's enrichment entry point, ``DocumentIngestionPipeline.enrich_nodes``."""

    def __call__(
        self,
        nodes: list[BaseNode],
        *,
        ner: bool,
        hate_speech: bool,
        progress_offset: int,
        progress_total: int | None,
    ) -> None:
        """Enrich *nodes* in place with the selected stages.

        Args:
            nodes (list[BaseNode]): Nodes to enrich.
            ner (bool): Run entity extraction.
            hate_speech (bool): Run hate-speech detection.
            progress_offset (int): Processed count offset for cumulative progress.
            progress_total (int | None): Total count for progress display.
        """


def has_text_twin(payload: dict[str, Any]) -> bool:
    """Report whether an image point is also a main-collection document.

    A standalone image file is read twice: ``ImageReader`` writes its words
    and caption into the main collection as a document, which the generic
    lane enriches, and the image service writes this point. Identity is
    first-wins, so a file seen standalone after a posting claimed it keeps the
    posting's ``source_type``; its occurrences still record the standalone
    copy.

    Args:
        payload (dict[str, Any]): The image point's payload.

    Returns:
        bool: ``True`` when the point's findings belong to its document.
    """
    if payload.get("source_type") == "standalone":
        return True
    occurrences = payload.get("occurrences")
    return isinstance(occurrences, list) and any(
        isinstance(occurrence, dict) and occurrence.get("source_type") == "standalone" for occurrence in occurrences
    )


def _tag_list(payload: dict[str, Any]) -> list[str]:
    """Return an image's non-blank tags, in stored order.

    Args:
        payload (dict[str, Any]): The image point's payload.

    Returns:
        list[str]: The tags, empty when it has none.
    """
    tags = payload.get("llm_tags")
    if not isinstance(tags, list):
        return []
    return [text for text in (str(tag).strip() for tag in tags) if text]


def image_text_fields(payload: dict[str, Any]) -> dict[str, Any]:
    """Return an image finding's printed words, description and tags as separate row fields.

    A finding's ``chunk_text`` is the text a stage judged, its parts run
    together. Kept apart, they let a view say which words were printed in the
    picture and which are docint's own description of it. A text chunk carries
    none of these payload keys, so its row gains nothing.

    Args:
        payload (dict[str, Any]): A main-collection or image-companion payload.

    Returns:
        dict[str, Any]: ``ocr_text``, ``image_description`` and ``image_tags``,
        each only when the image carries it.
    """
    fields: dict[str, Any] = {}
    ocr_text = str(payload.get("ocr_text") or "").strip()
    if ocr_text:
        fields["ocr_text"] = ocr_text
    description = str(payload.get("llm_description") or "").strip()
    if description:
        fields["image_description"] = description
    tags = _tag_list(payload)
    if tags:
        fields["image_tags"] = tags
    return fields


def entity_text(payload: dict[str, Any]) -> str:
    """Return the text NER reads off an image: its printed words, then its caption.

    Args:
        payload (dict[str, Any]): The image point's payload.

    Returns:
        str: The text, empty when the image has neither.
    """
    parts = (str(payload.get("ocr_text") or "").strip(), str(payload.get("llm_description") or "").strip())
    return "\n\n".join(part for part in parts if part)


def hate_speech_text(payload: dict[str, Any]) -> str:
    """Return the text hate-speech detection judges: the image's description, tags and printed words.

    A picture whose hate is purely visual has no printed words, so the
    description is the only text that shows it. The short description and
    tags come first, so a long printed text cannot push them past
    ``HATE_SPEECH_MAX_CHARS``. Each part is labelled, which tells the
    classifier it is reading an image and lets it judge the message the
    picture conveys rather than the neutral voice describing it. The labels
    follow ``RESPONSE_LANGUAGE`` and stay in the finding's quoted text.

    Args:
        payload (dict[str, Any]): The image point's payload.

    Returns:
        str: The labelled parts, empty when the image has none.
    """
    parts = (
        ("image_label_description", str(payload.get("llm_description") or "").strip()),
        ("image_label_tags", ", ".join(_tag_list(payload))),
        ("image_label_text", str(payload.get("ocr_text") or "").strip()),
    )
    return "\n\n".join(f"{ui_string(key)}: {value}" for key, value in parts if value)


def finding_basis(payload: dict[str, Any]) -> str:
    """Return what a main-collection point's finding was judged from.

    The only main-collection points carrying an image's caption or printed
    words are the documents ``ImageReader`` writes for standalone image files.

    Args:
        payload (dict[str, Any]): The main-collection point's payload.

    Returns:
        str: :data:`FINDING_BASIS_IMAGE` for an image's document, else :data:`FINDING_BASIS_TEXT`.
    """
    if payload.get("llm_description") or payload.get("ocr_text"):
        return FINDING_BASIS_IMAGE
    return FINDING_BASIS_TEXT


def enrichment_carryover(cached_payload: dict[str, Any] | None) -> dict[str, Any]:
    """Return the enrichment fields a point rewritten from *cached_payload* starts with.

    A rewrite that reuses a cached point's words and caption keeps the results
    computed from them; anything else waits for the pass.

    Args:
        cached_payload (dict[str, Any] | None): The stored point being rewritten, if any.

    Returns:
        dict[str, Any]: The enrichment state and any results to keep.
    """
    if cached_payload and cached_payload.get(ENRICHMENT_FIELD) == ENRICHMENT_DONE:
        return {
            key: cached_payload[key] for key in (ENRICHMENT_FIELD, *ENRICHMENT_RESULT_KEYS) if key in cached_payload
        }
    return {ENRICHMENT_FIELD: ENRICHMENT_PENDING}


def _pending_filter() -> models.Filter:
    """Return the filter matching the points waiting for the pass.

    Returns:
        models.Filter: ``enrichment == "pending"``.
    """
    return models.Filter(
        must=[models.FieldCondition(key=ENRICHMENT_FIELD, match=models.MatchValue(value=ENRICHMENT_PENDING))]
    )


def _settled_payload(ner_node: BaseNode, hate_node: BaseNode) -> dict[str, Any]:
    """Return the payload update recording one point's results.

    Args:
        ner_node (BaseNode): The point's node after entity extraction.
        hate_node (BaseNode): The point's node after hate-speech detection.

    Returns:
        dict[str, Any]: The results found, plus the ``done`` marker.
    """
    update: dict[str, Any] = {ENRICHMENT_FIELD: ENRICHMENT_DONE}
    for key in ("entities", "relations"):
        if ner_node.metadata.get(key):
            update[key] = ner_node.metadata[key]
    if hate_node.metadata.get("hate_speech"):
        update["hate_speech"] = hate_node.metadata["hate_speech"]
    return update


def enrich_pending_images(
    client: Any,
    collection_name: str,
    enrich: NodeEnricher,
    *,
    page_size: int = 64,
) -> int:
    """Enrich every image point waiting for the pass, then mark it done.

    Entities are read from :func:`entity_text` and the hate-speech verdict
    from :func:`hate_speech_text`, both through the run's own pipeline, so the
    stages, worker limits and per-request overrides are the text lanes'. The
    results are written payload-only. Only the points pending when the pass
    begins are settled; one written meanwhile waits for the next run.

    Args:
        client (Any): Qdrant client.
        collection_name (str): The image companion collection.
        enrich (NodeEnricher): The run's enrichment entry point.
        page_size (int): Points enriched and written per round.

    Returns:
        int: How many points were settled.
    """
    total = int(client.count(collection_name=collection_name, count_filter=_pending_filter(), exact=True).count)
    settled = 0
    if not total:
        return settled
    # Paging continues from the next point id, so settling a page (which
    # drops it out of the filter) cannot make the scroll skip a point.
    for page in iter_scroll(
        client,
        collection_name=collection_name,
        scroll_filter=_pending_filter(),
        page_size=page_size,
        with_payload=_SOURCE_PAYLOAD_KEYS,
        on_error="raise",
        error_context="pending images",
    ):
        points = page[: total - settled]
        ner_nodes: list[BaseNode] = [
            TextNode(id_=str(point.id), text=entity_text(point.payload or {})) for point in points
        ]
        hate_nodes: list[BaseNode] = [
            TextNode(id_=str(point.id), text=hate_speech_text(point.payload or {})) for point in points
        ]
        enrich(ner_nodes, ner=True, hate_speech=False, progress_offset=settled, progress_total=total)
        enrich(hate_nodes, ner=False, hate_speech=True, progress_offset=settled, progress_total=total)
        client.batch_update_points(
            collection_name=collection_name,
            update_operations=[
                models.SetPayloadOperation(
                    set_payload=models.SetPayload(payload=_settled_payload(ner_node, hate_node), points=[point.id])
                )
                for point, ner_node, hate_node in zip(points, ner_nodes, hate_nodes, strict=True)
            ],
            wait=True,
        )
        settled += len(points)
        if settled >= total:
            break
    return settled
