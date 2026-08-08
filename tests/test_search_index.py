"""Unit tests for the full-text search payload field and its index."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from qdrant_client import models
from typing_extensions import override

from docint.core.search.index import (
    SEARCH_TEXT_FIELD,
    BackfillSummary,
    backfill_search_text,
    ensure_search_index,
    search_index_params,
    search_index_status,
    write_search_text,
)


class _FakeClient:
    """Records payload-index calls without talking to a server."""

    def __init__(self, *, fail: bool = False) -> None:
        """Initialize the fake.

        Args:
            fail (bool): When True, every call raises, to exercise fail-soft paths.
        """
        self.fail = fail
        self.index_calls: list[dict[str, Any]] = []

    def create_payload_index(self, **kwargs: Any) -> None:
        """Record an index creation, or raise when configured to fail."""
        if self.fail:
            raise RuntimeError("qdrant unreachable")
        self.index_calls.append(kwargs)


def test_search_index_uses_a_lowercase_prefix_tokenizer() -> None:
    """Case-insensitivity on non-ASCII text depends on this exact index.

    Un-indexed MatchText only case-folds ASCII, so a German title-case token
    would not match its lowercase form without a lowercase text index.
    """
    params = search_index_params()

    assert params.tokenizer == models.TokenizerType.PREFIX
    assert params.lowercase is True
    assert params.min_token_len == 2


def test_ensure_search_index_creates_the_index_on_the_field() -> None:
    """The index must be created on the dedicated search field."""
    client = _FakeClient()

    assert ensure_search_index(client, "col") is True
    assert len(client.index_calls) == 1
    assert client.index_calls[0]["field_name"] == SEARCH_TEXT_FIELD
    assert client.index_calls[0]["collection_name"] == "col"


def test_ensure_search_index_is_fail_soft() -> None:
    """A Qdrant outage must not break the caller — search degrades, not crashes."""
    client = _FakeClient(fail=True)

    assert ensure_search_index(client, "col") is False


class _RecordingClient(_FakeClient):
    """Fake client that records batched payload updates."""

    def __init__(self, *, fail: bool = False) -> None:
        """Initialize the recorder."""
        super().__init__(fail=fail)
        self.batches: list[list[Any]] = []

    def batch_update_points(self, **kwargs: Any) -> list[Any]:
        """Record one batched update call."""
        if self.fail:
            raise RuntimeError("qdrant unreachable")
        self.batches.append(list(kwargs["update_operations"]))
        return []


def _written(client: _RecordingClient) -> dict[Any, str]:
    """Flatten recorded operations into ``{point_id: text}``.

    Args:
        client (_RecordingClient): The fake that recorded the calls.

    Returns:
        dict[Any, str]: Every point id written, mapped to its text.
    """
    out: dict[Any, str] = {}
    for batch in client.batches:
        for op in batch:
            payload = op.set_payload
            for point_id in payload.points or []:
                out[point_id] = payload.payload[SEARCH_TEXT_FIELD]
    return out


def test_write_search_text_writes_one_operation_per_point() -> None:
    """Distinct texts per point must survive into distinct operations."""
    client = _RecordingClient()

    assert write_search_text(client, "col", {"a": "first chunk", "b": "second chunk"}) == 2
    assert _written(client) == {"a": "first chunk", "b": "second chunk"}


def test_write_search_text_preserves_the_point_id_type() -> None:
    """Qdrant ids are unsigned ints or UUIDs; coercing an int id writes nothing.

    A str("1") targets a point that does not exist, so the write silently
    lands nowhere and the collection stays unsearchable.
    """
    client = _RecordingClient()

    assert write_search_text(client, "col", {7: "seventh chunk"}) == 1
    assert _written(client) == {7: "seventh chunk"}


def test_write_search_text_batches_by_batch_size() -> None:
    """A large collection must not become one enormous request."""
    client = _RecordingClient()
    texts = {str(i): f"chunk {i}" for i in range(7)}

    assert write_search_text(client, "col", texts, batch_size=3) == 7
    assert [len(batch) for batch in client.batches] == [3, 3, 1]


def test_write_search_text_ignores_empty_input() -> None:
    """No points means no request at all."""
    client = _RecordingClient()

    assert write_search_text(client, "col", {}) == 0
    assert client.batches == []


class _ScrollingClient(_RecordingClient):
    """Fake client that serves canned scroll pages."""

    def __init__(self, points: list[Any]) -> None:
        """Initialize with the points one scroll should yield.

        Args:
            points (list[Any]): Point stand-ins exposing ``id`` and ``payload``.
        """
        super().__init__()
        self.points = points

    def scroll(self, **kwargs: Any) -> tuple[list[Any], Any]:
        """Return every point in one page, then stop."""
        if kwargs.get("offset") is not None:
            return [], None
        return list(self.points), None


class _Point:
    """Minimal Qdrant point stand-in."""

    def __init__(self, point_id: Any, payload: dict[str, Any]) -> None:
        """Store the id and payload.

        Args:
            point_id (Any): Point identifier.
            payload (dict[str, Any]): Point payload.
        """
        self.id = point_id
        self.payload = payload


def _extract(payload: Mapping[str, Any]) -> str:
    """Test extractor standing in for ``RAG._extract_payload_text``."""
    return str(payload.get("body") or "")


def test_backfill_writes_text_for_points_that_lack_it() -> None:
    """The migration must fill in every point that has text but no search_text."""
    client = _ScrollingClient([_Point("a", {"body": "alpha"}), _Point("b", {"body": "beta"})])

    summary = backfill_search_text(client, "col", extract_text=_extract)

    assert summary == BackfillSummary(scanned=2, written=2, skipped=0)
    assert _written(client) == {"a": "alpha", "b": "beta"}


def test_backfill_preserves_integer_point_ids() -> None:
    """Coercing an int id to a string would write to a nonexistent point."""
    client = _ScrollingClient([_Point(7, {"body": "seventh"})])

    backfill_search_text(client, "col", extract_text=_extract)

    assert _written(client) == {7: "seventh"}


def test_backfill_skips_points_that_already_have_search_text() -> None:
    """Re-running the migration must be cheap and idempotent."""
    client = _ScrollingClient(
        [
            _Point("a", {"body": "alpha", SEARCH_TEXT_FIELD: "alpha"}),
            _Point("b", {"body": "beta"}),
        ]
    )

    summary = backfill_search_text(client, "col", extract_text=_extract)

    assert summary == BackfillSummary(scanned=2, written=1, skipped=1)
    assert _written(client) == {"b": "beta"}


def test_backfill_force_rewrites_existing_search_text() -> None:
    """``force`` exists for when the extractor itself changed."""
    client = _ScrollingClient([_Point("a", {"body": "alpha", SEARCH_TEXT_FIELD: "stale"})])

    summary = backfill_search_text(client, "col", extract_text=_extract, force=True)

    assert summary == BackfillSummary(scanned=1, written=1, skipped=0)
    assert _written(client) == {"a": "alpha"}


def test_backfill_counts_textless_points_separately_from_written_ones() -> None:
    """A point carrying no text is not an error — it is simply not searchable.

    It is still recorded as processed (see the coverage test below), but must
    not inflate the count of points that actually became searchable.
    """
    client = _ScrollingClient([_Point("a", {"body": ""}), _Point("b", {"body": "beta"})])

    summary = backfill_search_text(client, "col", extract_text=_extract)

    assert summary == BackfillSummary(scanned=2, written=1, skipped=0, empty=1)


def test_backfill_reports_progress_when_a_sink_is_supplied() -> None:
    """A long migration must report progress, not sit silently.

    The ``search-index`` CLI is the only consumer, and a multi-minute scroll
    with no output is indistinguishable from a hang.
    """
    client = _ScrollingClient([_Point(str(i), {"body": f"chunk {i}"}) for i in range(3)])
    messages: list[str] = []

    backfill_search_text(client, "col", extract_text=_extract, progress=messages.append)

    assert messages
    assert "3 scanned" in messages[-1]


class _CountingClient(_ScrollingClient):
    """Fake client answering coverage counts."""

    def __init__(self, points: list[Any], *, total: int, with_text: int) -> None:
        """Initialize with the counts ``count`` should report.

        Args:
            points (list[Any]): Points a scroll would yield.
            total (int): Total points in the collection.
            with_text (int): Points already carrying ``search_text``.
        """
        super().__init__(points)
        self.total = total
        self.with_text = with_text

    def get_collection(self, **kwargs: Any) -> Any:
        """Report a collection whose search field is indexed."""
        return type("_Info", (), {"payload_schema": {SEARCH_TEXT_FIELD: object()}})()

    def count(self, collection_name: str, **kwargs: Any) -> Any:
        """Return the filtered or unfiltered count."""
        value = self.with_text if kwargs.get("count_filter") is not None else self.total
        return type("_Count", (), {"count": value})()


def test_status_counts_coverage_instead_of_sampling_the_head() -> None:
    """A head sample cannot distinguish a finished backfill from a started one.

    The backfill walks the collection from the same offset a sample would, so
    once its first page lands, a first-hit sample reports the whole collection
    ready — and a search issued mid-migration silently omits everything not yet
    written, while still reporting success.
    """
    client = _CountingClient([], total=1000, with_text=64)

    status = search_index_status(client, "col")

    assert status["total"] == 1000
    assert status["with_search_text"] == 64
    assert status["missing"] == 936
    assert status["complete"] is False


def test_status_is_complete_when_every_point_carries_the_field() -> None:
    """A finished backfill must be distinguishable from a partial one."""
    client = _CountingClient([], total=500, with_text=500)

    status = search_index_status(client, "col")

    assert status["missing"] == 0
    assert status["complete"] is True


def test_status_reports_no_coverage_for_an_unmigrated_collection() -> None:
    """Zero coverage is the "run the migration" signal."""
    client = _CountingClient([], total=500, with_text=0)

    status = search_index_status(client, "col")

    assert status["with_search_text"] == 0
    assert status["complete"] is False


class _FlakyClient(_ScrollingClient):
    """Fails the first N batched writes, then succeeds."""

    def __init__(self, points: list[Any], *, failures: int) -> None:
        """Initialize with a number of transient failures to emit.

        Args:
            points (list[Any]): Points a scroll would yield.
            failures (int): Consecutive write failures before succeeding.
        """
        super().__init__(points)
        self.remaining_failures = failures

    @override
    def batch_update_points(self, **kwargs: Any) -> list[Any]:
        """Fail transiently, then record the write."""
        if self.remaining_failures > 0:
            self.remaining_failures -= 1
            raise ConnectionError("connection reset by peer")
        return super().batch_update_points(**kwargs)


def test_backfill_retries_a_transient_write_failure() -> None:
    """A blip mid-migration must not leave the collection half-indexed.

    Without a retry, one transient error kills the run partway and the
    collection sits in a "some points indexed" state indefinitely — which a
    search would then serve from, silently omitting the rest.
    """
    client = _FlakyClient([_Point("a", {"body": "alpha"})], failures=2)

    summary = backfill_search_text(client, "col", extract_text=_extract, retry_sleep=lambda _s: None)

    assert summary == BackfillSummary(scanned=1, written=1, skipped=0)
    assert _written(client) == {"a": "alpha"}


def test_backfill_marks_textless_points_so_coverage_can_reach_zero() -> None:
    """A point with no text must still be recorded as processed.

    Otherwise it counts as "missing" forever, the collection reports `partial`
    on every search no matter how many times the migration is re-run, and a
    permanently-on warning is one nobody reads. An empty string is a value, so
    Qdrant's is-empty check treats such a point as covered.
    """
    client = _ScrollingClient([_Point("a", {"body": ""}), _Point("b", {"body": "beta"})])

    summary = backfill_search_text(client, "col", extract_text=_extract)

    assert summary == BackfillSummary(scanned=2, written=1, skipped=0, empty=1)
    assert _written(client) == {"a": "", "b": "beta"}


class _CompanionClient(_CountingClient):
    """Counting client that also answers for an image companion."""

    def __init__(self, *, main: tuple[int, int], images: tuple[int, int]) -> None:
        """Initialize with per-collection ``(total, with_text)`` counts.

        Args:
            main (tuple[int, int]): Counts for the main collection.
            images (tuple[int, int]): Counts for the ``_images`` companion.
        """
        super().__init__([], total=main[0], with_text=main[1])
        self.images = images

    @override
    def count(self, collection_name: str, **kwargs: Any) -> Any:
        """Return counts for whichever collection was asked about."""
        total, with_text = self.images if collection_name.endswith("_images") else (self.total, self.with_text)
        value = with_text if kwargs.get("count_filter") is not None else total
        return type("_Count", (), {"count": value})()

    def collection_exists(self, collection_name: str) -> bool:
        """Report that every collection exists."""
        return True


def test_status_counts_the_image_companion_too() -> None:
    """An unindexed companion means image hits silently missing.

    Coverage that ignored the companion would read "complete" while every
    figure and keyframe stayed unfindable.
    """
    client = _CompanionClient(main=(100, 100), images=(22, 0))

    status = search_index_status(client, "col")

    assert status["total"] == 122
    assert status["with_search_text"] == 100
    assert status["missing"] == 22
    assert status["complete"] is False
