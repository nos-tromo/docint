"""Tests for the Qdrant scan behind an extract.

Payloads are synthetic; the fake client records the filters it was handed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from qdrant_client.http import models

from docint.core.extract.gather import scroll_collection, source_filters


@dataclass
class FakePoint:
    """Stand-in for a Qdrant record."""

    id: str
    payload: dict[str, Any] = field(default_factory=dict)


class FakeQdrant:
    """Scroll mock returning one page per collection and recording its calls."""

    def __init__(self, pages: dict[str, list[FakePoint]], *, missing: set[str] | None = None) -> None:
        """Seed per-collection points and the collections that do not exist."""
        self.pages = pages
        self.missing = missing or set()
        self.calls: list[dict[str, Any]] = []

    def collection_exists(self, collection_name: str) -> bool:
        """Report whether a collection exists."""
        return collection_name not in self.missing

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
        """Record the call and return the collection's single page."""
        self.calls.append({"collection_name": collection_name, "scroll_filter": scroll_filter})
        if offset is not None:
            return [], None
        return self.pages.get(collection_name, []), None


def _keys(scroll_filter: models.Filter | None, attr: str) -> list[str]:
    """Return the payload keys a filter's clause matches on."""
    conditions = getattr(scroll_filter, attr, None) or []
    return [getattr(condition, "key", "") for condition in conditions]


def test_scroll_returns_both_lanes() -> None:
    """The scan covers the collection and its images companion."""
    client = FakeQdrant(
        {"col": [FakePoint("a", {"file_hash": "h"})], "col_images": [FakePoint("i", {"image_id": "img"})]}
    )
    main, images = scroll_collection(client, "col", "col_images")
    assert [point[0] for point in main] == ["a"]
    assert [point[1]["image_id"] for point in images] == ["img"]


def test_main_scan_excludes_coarse_parents() -> None:
    """Coarse chunks duplicate their children, so they never reach a bundle."""
    client = FakeQdrant({"col": [], "col_images": []})
    scroll_collection(client, "col", "col_images")
    main_call = next(call for call in client.calls if call["collection_name"] == "col")
    assert _keys(main_call["scroll_filter"], "must_not") == ["docint_hier_type"]


def test_a_missing_companion_is_not_an_error() -> None:
    """A text-only collection has no images companion and still extracts."""
    client = FakeQdrant({"col": [FakePoint("a", {})]}, missing={"col_images"})
    main, images = scroll_collection(client, "col", "col_images")
    assert len(main) == 1
    assert images == []
    assert all(call["collection_name"] == "col" for call in client.calls)


def test_source_filters_cover_every_identity_a_source_can_have() -> None:
    """A source id may be a file hash, a posting uuid or an image id."""
    main_filter, image_filter = source_filters("abc")
    assert _keys(main_filter, "should") == ["file_hash", "posting_uuid", "reference_metadata.uuid"]
    assert _keys(image_filter, "should") == ["source_doc_id", "posting_uuid", "image_id", "file_hash"]


def test_source_scan_keeps_the_coarse_exclusion() -> None:
    """A per-source extract must not double a hierarchical document's text."""
    client = FakeQdrant({"col": [], "col_images": []})
    scroll_collection(client, "col", "col_images", source_id="abc")
    main_call = next(call for call in client.calls if call["collection_name"] == "col")
    assert _keys(main_call["scroll_filter"], "must_not") == ["docint_hier_type"]
    assert _keys(main_call["scroll_filter"], "must")
