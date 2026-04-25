"""Generic batching helpers shared across the ingestion pipeline.

Hoisted from :class:`docint.core.rag.RAG` and
:class:`docint.core.ingest.ingestion_pipeline.DocumentIngestionPipeline`
where the same ``_chunk_nodes`` static method was duplicated. Centralising
it under :mod:`docint.utils` matches the project convention that
cross-cutting helpers live alongside :mod:`docint.utils.retry`.
"""

from __future__ import annotations

from typing import Sequence, TypeVar

T = TypeVar("T")


def chunk_nodes(items: Sequence[T], batch_size: int) -> list[list[T]]:
    """Split *items* into non-empty batches of at most *batch_size* elements.

    Args:
        items: Sequence of items to batch (typically ingestion nodes).
        batch_size: Preferred maximum batch size. Values below 1 are
            clamped to 1; non-integer inputs are coerced via ``int()``.

    Returns:
        List of batches in input order. Empty input yields an empty list.
    """
    if not items:
        return []
    effective = max(1, int(batch_size))
    materialised = list(items)
    return [
        materialised[i : i + effective]
        for i in range(0, len(materialised), effective)
    ]
