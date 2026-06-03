"""Shared Qdrant scroll iterators.

The same scroll loop is reproduced in roughly eight places in
:mod:`docint.core.rag`. These helpers concentrate it so paginated HTTP
endpoints, full-collection scans, and the existing in-memory aggregators all
share one well-tested implementation.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, Literal

from loguru import logger

from docint.utils.cursor import decode_cursor, encode_cursor

ScrollErrorMode = Literal["raise", "warn", "debug"]


def _handle_scroll_error(
    exc: Exception,
    *,
    collection_name: str,
    mode: ScrollErrorMode,
    context: str,
) -> None:
    """Apply the configured error mode to a scroll exception."""
    if mode == "raise":
        raise exc
    message = "Qdrant scroll failed for '{}'{}: {}".format(
        collection_name,
        f" ({context})" if context else "",
        exc,
    )
    if mode == "debug":
        logger.debug(message)
    else:
        logger.warning(message)


def iter_scroll(
    qdrant_client: Any,
    *,
    collection_name: str,
    scroll_filter: Any | None = None,
    page_size: int = 256,
    with_payload: bool = True,
    with_vectors: bool = False,
    max_pages: int | None = None,
    on_error: ScrollErrorMode = "warn",
    error_context: str = "",
) -> Iterator[list[Any]]:
    """Yield successive Qdrant scroll pages until the collection is exhausted.

    Args:
        qdrant_client (Any): Qdrant client exposing a ``.scroll`` method.
        collection_name (str): Collection to scroll.
        scroll_filter (Any | None): Optional Qdrant ``Filter`` applied during scroll.
        page_size (int): Points per page (the Qdrant ``limit`` parameter).
        with_payload (bool): Whether to fetch payloads.
        with_vectors (bool): Whether to fetch vectors.
        max_pages (int | None): If set, stop after this many pages.
        on_error (ScrollErrorMode): How to react to scroll exceptions:
            ``"raise"`` propagates the exception; ``"warn"`` logs at WARNING
            and stops cleanly; ``"debug"`` logs at DEBUG and stops cleanly.
        error_context (str): Free-form context string included in error logs
            (e.g. ``"NER sources"``).

    Yields:
        list[Any]: Successive non-empty pages of Qdrant point records.
    """
    offset: Any = None
    pages_yielded = 0
    while True:
        if max_pages is not None and pages_yielded >= max_pages:
            return
        try:
            points, offset = qdrant_client.scroll(
                collection_name=collection_name,
                limit=page_size,
                offset=offset,
                scroll_filter=scroll_filter,
                with_payload=with_payload,
                with_vectors=with_vectors,
            )
        except Exception as exc:
            _handle_scroll_error(
                exc,
                collection_name=collection_name,
                mode=on_error,
                context=error_context,
            )
            return

        if not points:
            return

        yield list(points)
        pages_yielded += 1
        if offset is None:
            return


def paginate_scroll(
    qdrant_client: Any,
    *,
    collection_name: str,
    cursor: str | None = None,
    limit: int = 50,
    scroll_filter: Any | None = None,
    with_payload: bool = True,
    with_vectors: bool = False,
) -> tuple[list[Any], str | None]:
    """Fetch a single page of points for cursor-paginated endpoints.

    Wraps :func:`docint.utils.cursor.decode_cursor` /
    :func:`docint.utils.cursor.encode_cursor` so callers can pass opaque
    cursor tokens straight through from query parameters.

    Args:
        qdrant_client (Any): Qdrant client exposing a ``.scroll`` method.
        collection_name (str): Collection to scroll.
        cursor (str | None): Opaque cursor token from a previous call, or
            ``None`` for the first page.
        limit (int): Number of points to return for this page.
        scroll_filter (Any | None): Optional Qdrant filter.
        with_payload (bool): Whether to fetch payloads.
        with_vectors (bool): Whether to fetch vectors.

    Returns:
        tuple[list[Any], str | None]: The fetched points and the cursor token
        for the next page (or ``None`` if exhausted).
    """
    decoded = decode_cursor(cursor)
    offset = decoded.get("o")
    points, next_offset = qdrant_client.scroll(
        collection_name=collection_name,
        limit=limit,
        offset=offset,
        scroll_filter=scroll_filter,
        with_payload=with_payload,
        with_vectors=with_vectors,
    )
    next_token = encode_cursor(next_offset)
    return list(points or []), next_token


__all__ = [
    "ScrollErrorMode",
    "iter_scroll",
    "paginate_scroll",
]
