"""Typed ingestion events for richer SSE progress and observability.

Phase 5 of the streaming ingestion generalisation. Today the ingestion
pipeline only fires a single string-shaped progress callback. The
Streamlit UI cannot render per-stage progress (e.g. "embed batch 3/12
of file foo.pdf") because the existing callback is opaque
``Callable[[str], None]``.

This module introduces:

* :class:`IngestionEvent` — a :class:`TypedDict` discriminator-stage
  payload that ingestion subsystems can fire alongside the existing
  string callback. Strict superset of the current contract; old
  callers that only consume the string callback keep working.
* :func:`format_progress_message` — produces the existing
  human-readable string from a typed event so the legacy string
  callback can be derived from one source of truth.

Wiring sites (incremental rollout):

* :meth:`docint.core.rag.RAG._persist_node_batches` —
  ``stage="persist"`` events per micro-batch.
* :meth:`docint.core.rag.RAG._prepare_vector_nodes_for_insert` —
  ``stage="embed"`` events per HTTP-batch chunk.
* :meth:`docint.core.ingest.ingestion_pipeline.DocumentIngestionPipeline._enrich_nodes_in_place`
  — ``stage="enrich"`` events per chunk.

The SSE endpoint (``docint/core/api.py:1277-1505``) maps these typed
events into the ``ingestion_progress_typed`` SSE event name. Old
clients that only subscribe to the legacy ``ingestion_progress``
text events continue to work unchanged.
"""

from __future__ import annotations

from typing import Callable, Literal, NotRequired, TypedDict

IngestionStage = Literal[
    "read",
    "enrich",
    "embed",
    "persist",
    "manifest",
    "summary",
]


class IngestionEvent(TypedDict, total=False):
    """Discriminator-stage progress event.

    The ``stage`` field is the discriminator; downstream consumers
    can branch on it. All other fields are optional payload. Using
    ``total=False`` so callers can add per-stage detail without
    breaking the type contract for unrelated stages.

    Fields:
        stage: One of :data:`IngestionStage`.
        file_hash: SHA-256 of the source file the event pertains to.
        batch: Current batch index (1-based) within a stage.
        of: Total number of batches in this stage for this file.
        nodes: Count of nodes affected by this event (e.g. nodes in
            this persist batch).
        attempts: Retry attempt number for this stage's operation
            (1 for first attempt, >1 for retries).
        message: Human-readable message — kept for back-compat with
            the legacy string callback.
    """

    stage: IngestionStage
    file_hash: NotRequired[str]
    batch: NotRequired[int]
    of: NotRequired[int]
    nodes: NotRequired[int]
    attempts: NotRequired[int]
    message: NotRequired[str]


EventCallback = Callable[[IngestionEvent], None]


def format_progress_message(event: IngestionEvent) -> str:
    """Render a typed event as a human-readable progress string.

    Mirrors the format the legacy string callback emits today so the
    SSE stream and the ``progress_callback`` argument can both be
    fed from a single source of truth.

    Args:
        event: Typed ingestion event.

    Returns:
        A short single-line message suitable for log output and the
        legacy SSE ``ingestion_progress`` event.
    """
    if "message" in event and event["message"]:
        return str(event["message"])

    stage = event.get("stage", "ingest")
    parts: list[str] = [f"[{stage}]"]
    if "batch" in event and "of" in event:
        parts.append(f"batch {event['batch']}/{event['of']}")
    elif "batch" in event:
        parts.append(f"batch {event['batch']}")
    if "nodes" in event:
        parts.append(f"{event['nodes']} node(s)")
    if "file_hash" in event:
        fh = str(event["file_hash"])
        parts.append(f"file_hash={fh[:8]}")
    if "attempts" in event and event["attempts"] > 1:
        parts.append(f"attempt {event['attempts']}")
    return " ".join(parts)


__all__ = [
    "EventCallback",
    "IngestionEvent",
    "IngestionStage",
    "format_progress_message",
]
