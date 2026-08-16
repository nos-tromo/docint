"""Tests for the progress-log throttle.

Every filename here is invented; none of it corresponds to real data.
"""

from __future__ import annotations

import threading

import pytest

from docint.utils.logfmt import ProgressLogThrottle, parse_counter, progress_key


class _FakeClock:
    """A monotonic clock a test can advance by hand."""

    def __init__(self) -> None:
        """Start at zero."""
        self.now = 0.0

    def __call__(self) -> float:
        """Return the current fake time.

        Returns:
            float: Seconds since the clock was created.
        """
        return self.now

    def advance(self, seconds: float) -> None:
        """Move the clock forward.

        Args:
            seconds (float): Seconds to advance by.
        """
        self.now += seconds


# ---------------------------------------------------------------------------
# progress_key — the projection the whole throttle rests on
# ---------------------------------------------------------------------------


def test_per_file_messages_get_distinct_keys() -> None:
    """Each file must announce itself; masking digits keeps the filename."""
    a = progress_key("Core pipeline processing PDF (1/3): annual-report.pdf")
    b = progress_key("Core pipeline processing PDF (2/3): meeting-notes.pdf")

    assert a != b


def test_per_chunk_ticks_share_one_key() -> None:
    """Ticks differing only by their counter must collapse to one stage."""
    a = progress_key("Extracting entities: 1/2000 chunks processed")
    b = progress_key("Extracting entities: 1999/2000 chunks processed")

    assert a == b


def test_distinct_stages_keep_distinct_keys() -> None:
    """NER and hate-speech run interleaved and must heartbeat independently."""
    ner = progress_key("Extracting entities: 5/2000 chunks processed")
    hate = progress_key("Detecting hate speech: 5/2000 chunks processed")

    assert ner != hate


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("Summarizing 12/412", (12, 412)),
        ("Extracting entities: 840/2000 chunks processed", (840, 2000)),
        ("Core pipeline processing PDF (1/3): a.pdf", (1, 3)),
        ("Core pipeline indexed 240 chunks: a.pdf", None),
        ("Building collection summary...", None),
    ],
)
def test_parse_counter(message: str, expected: tuple[int, int] | None) -> None:
    """The n/m counter drives the always-log-the-final-value rule.

    Args:
        message (str): A progress message.
        expected (tuple[int, int] | None): The counter it should yield.
    """
    assert parse_counter(message) == expected


# ---------------------------------------------------------------------------
# throttling
# ---------------------------------------------------------------------------


def test_stage_transition_always_logs() -> None:
    """A new stage is news regardless of how recently anything else logged."""
    clock = _FakeClock()
    throttle = ProgressLogThrottle(30.0, time_fn=clock)

    assert throttle.observe("Core pipeline processing PDF (1/2): alpha.pdf") == [
        "Core pipeline processing PDF (1/2): alpha.pdf"
    ]
    assert throttle.observe("Core pipeline processing PDF (2/2): beta.pdf") == [
        "Core pipeline processing PDF (2/2): beta.pdf"
    ]


def test_ticks_inside_the_interval_are_dropped() -> None:
    """The complaint is silence, not detail — but not 2000 lines of detail."""
    clock = _FakeClock()
    throttle = ProgressLogThrottle(30.0, time_fn=clock)

    throttle.observe("Extracting entities: 1/2000 chunks processed")
    dropped = [throttle.observe(f"Extracting entities: {n}/2000 chunks processed") for n in range(2, 50)]

    assert all(out == [] for out in dropped)


def test_a_heartbeat_escapes_once_the_interval_passes() -> None:
    """No gap may exceed the interval while a stage is still working."""
    clock = _FakeClock()
    throttle = ProgressLogThrottle(30.0, time_fn=clock)

    throttle.observe("Extracting entities: 1/2000 chunks processed")
    clock.advance(29.0)
    assert throttle.observe("Extracting entities: 2/2000 chunks processed") == []

    clock.advance(1.0)
    assert throttle.observe("Extracting entities: 3/2000 chunks processed") == [
        "Extracting entities: 3/2000 chunks processed"
    ]


def test_the_final_value_always_logs() -> None:
    """A stage must report its own completion even if it finishes fast."""
    clock = _FakeClock()
    throttle = ProgressLogThrottle(30.0, time_fn=clock)

    throttle.observe("Extracting entities: 1/3 chunks processed")
    assert throttle.observe("Extracting entities: 2/3 chunks processed") == []
    assert throttle.observe("Extracting entities: 3/3 chunks processed") == [
        "Extracting entities: 3/3 chunks processed"
    ]


def test_a_stage_that_stops_short_is_released_by_the_next_stage() -> None:
    """A stage that never reaches its total must not trail off mid-count."""
    clock = _FakeClock()
    throttle = ProgressLogThrottle(30.0, time_fn=clock)

    throttle.observe("Extracting entities: 1/2000 chunks processed")
    throttle.observe("Extracting entities: 7/2000 chunks processed")  # held

    out = throttle.observe("Detecting hate speech: 1/2000 chunks processed")

    assert out == [
        "Extracting entities: 7/2000 chunks processed",
        "Detecting hate speech: 1/2000 chunks processed",
    ]


def test_flush_releases_a_held_tick() -> None:
    """The job's terminal path must not swallow the last thing a stage said."""
    clock = _FakeClock()
    throttle = ProgressLogThrottle(30.0, time_fn=clock)

    throttle.observe("Extracting entities: 1/2000 chunks processed")
    throttle.observe("Extracting entities: 7/2000 chunks processed")

    assert throttle.flush() == ["Extracting entities: 7/2000 chunks processed"]
    assert throttle.flush() == []


def test_flush_is_empty_when_nothing_is_held() -> None:
    """A stage that logged its final value leaves nothing behind."""
    clock = _FakeClock()
    throttle = ProgressLogThrottle(30.0, time_fn=clock)

    throttle.observe("Extracting entities: 3/3 chunks processed")

    assert throttle.flush() == []


def test_zero_interval_disables_throttling() -> None:
    """The debug escape hatch: LOG_PROGRESS_INTERVAL_S=0 logs everything."""
    clock = _FakeClock()
    throttle = ProgressLogThrottle(0.0, time_fn=clock)

    out = [throttle.observe(f"Extracting entities: {n}/2000 chunks processed") for n in range(1, 6)]

    assert out == [[f"Extracting entities: {n}/2000 chunks processed"] for n in range(1, 6)]


def test_messages_are_never_rewritten() -> None:
    """The SPA anchors regexes on these strings; the log layer is additive."""
    clock = _FakeClock()
    throttle = ProgressLogThrottle(30.0, time_fn=clock)
    message = "Core pipeline indexed 240 chunks: annual-report.pdf"

    assert throttle.observe(message) == [message]


def test_concurrent_ticks_do_not_corrupt_state() -> None:
    """NER reports from a ThreadPoolExecutor, so ticks genuinely race."""
    clock = _FakeClock()
    throttle = ProgressLogThrottle(30.0, time_fn=clock)
    emitted: list[str] = []
    lock = threading.Lock()

    def worker(index: int) -> None:
        """Emit one tick.

        Args:
            index (int): This worker's tick number.
        """
        out = throttle.observe(f"Extracting entities: {index}/500 chunks processed")
        with lock:
            emitted.extend(out)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(1, 201)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    # Exactly one tick escapes: the first to arrive opens the stage, and the
    # clock never advances, so every later tick is inside the interval.
    assert len(emitted) == 1
    assert emitted[0].startswith("Extracting entities: ")
