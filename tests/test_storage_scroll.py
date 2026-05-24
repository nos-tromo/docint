"""Tests for the shared Qdrant scroll iterators."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from docint.core.storage.scroll import iter_scroll, paginate_scroll
from docint.utils.cursor import decode_cursor, encode_cursor


@dataclass
class FakePoint:
    """Lightweight stand-in for ``qdrant_client.models.Record``."""

    id: int
    payload: dict[str, Any] = field(default_factory=dict)


class FakeQdrant:
    """Paged scroll mock returning slices of a fixed payload list.

    Tracks calls so tests can assert on offset wiring.
    """

    def __init__(self, points: list[FakePoint]) -> None:
        self.points = points
        self.calls: list[dict[str, Any]] = []

    def scroll(
        self,
        *,
        collection_name: str,
        limit: int,
        offset: Any,
        scroll_filter: Any,
        with_payload: bool,
        with_vectors: bool,
    ) -> tuple[list[FakePoint], Any]:
        self.calls.append(
            {
                "collection_name": collection_name,
                "limit": limit,
                "offset": offset,
                "scroll_filter": scroll_filter,
                "with_payload": with_payload,
                "with_vectors": with_vectors,
            }
        )
        start = 0 if offset is None else int(offset)
        end = min(len(self.points), start + limit)
        page = self.points[start:end]
        next_offset = end if end < len(self.points) else None
        return page, next_offset


def _points(n: int) -> list[FakePoint]:
    return [FakePoint(id=i, payload={"n": i}) for i in range(n)]


def test_iter_scroll_yields_all_pages_in_order() -> None:
    client = FakeQdrant(_points(550))
    pages = list(iter_scroll(client, collection_name="c", page_size=200))
    assert [len(p) for p in pages] == [200, 200, 150]
    flat = [pt.id for page in pages for pt in page]
    assert flat == list(range(550))


def test_iter_scroll_stops_on_empty_page() -> None:
    client = FakeQdrant([])
    pages = list(iter_scroll(client, collection_name="c", page_size=100))
    assert pages == []
    assert len(client.calls) == 1


def test_iter_scroll_respects_max_pages() -> None:
    client = FakeQdrant(_points(1000))
    pages = list(iter_scroll(client, collection_name="c", page_size=100, max_pages=3))
    assert len(pages) == 3
    assert sum(len(p) for p in pages) == 300


def test_iter_scroll_warns_and_stops_by_default() -> None:
    class BoomClient:
        def scroll(self, **_: Any) -> Any:
            raise RuntimeError("boom")

    pages = list(iter_scroll(BoomClient(), collection_name="c"))
    assert pages == []


def test_iter_scroll_raises_when_configured() -> None:
    class BoomClient:
        def scroll(self, **_: Any) -> Any:
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        list(iter_scroll(BoomClient(), collection_name="c", on_error="raise"))


def test_iter_scroll_passes_scroll_filter_through() -> None:
    client = FakeQdrant(_points(10))
    sentinel = object()
    list(iter_scroll(client, collection_name="c", scroll_filter=sentinel, page_size=5))
    for call in client.calls:
        assert call["scroll_filter"] is sentinel


def test_iter_scroll_defaults_with_vectors_false() -> None:
    client = FakeQdrant(_points(3))
    list(iter_scroll(client, collection_name="c"))
    assert client.calls[0]["with_vectors"] is False


def test_paginate_scroll_round_trips_cursor() -> None:
    client = FakeQdrant(_points(75))
    points, cursor = paginate_scroll(client, collection_name="c", cursor=None, limit=30)
    assert [p.id for p in points] == list(range(30))
    assert cursor is not None
    assert decode_cursor(cursor)["o"] == 30

    points2, cursor2 = paginate_scroll(client, collection_name="c", cursor=cursor, limit=30)
    assert [p.id for p in points2] == list(range(30, 60))
    assert cursor2 is not None

    points3, cursor3 = paginate_scroll(client, collection_name="c", cursor=cursor2, limit=30)
    assert [p.id for p in points3] == list(range(60, 75))
    assert cursor3 is None


def test_paginate_scroll_accepts_none_cursor() -> None:
    client = FakeQdrant(_points(5))
    points, _ = paginate_scroll(client, collection_name="c", cursor=None, limit=10)
    assert [p.id for p in points] == [0, 1, 2, 3, 4]


def test_paginate_scroll_accepts_extra_cursor_payload() -> None:
    client = FakeQdrant(_points(10))
    token = encode_cursor(3, extra={"flushed": ["a"]})
    points, _ = paginate_scroll(client, collection_name="c", cursor=token, limit=2)
    assert [p.id for p in points] == [3, 4]
