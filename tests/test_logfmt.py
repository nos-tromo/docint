"""Tests for the progress-log throttle.

Every filename here is invented; none of it corresponds to real data.
"""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from docint.utils.logfmt import (
    ProgressLogThrottle,
    describe_inputs,
    format_by_type,
    format_bytes,
    parse_counter,
    progress_key,
)


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
# format_bytes — must agree with the SPA, which describes the same upload
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        # These are the cases frontend/src/lib/ingestStatus.test.ts pins for
        # formatBytes. The log and the ingest card must not disagree about
        # the size of the same file.
        (0, "0 B"),
        (-5, "0 B"),
        (1023, "1023 B"),
        (1024, "1.0 KB"),
        (1024 * 1024, "1.0 MB"),
        (1_500_000, "1.4 MB"),
    ],
)
def test_format_bytes_matches_the_spa(value: int, expected: str) -> None:
    """Port parity with ``formatBytes``, truncation included.

    Args:
        value (int): Byte count.
        expected (str): What the SPA renders for it.
    """
    assert format_bytes(value) == expected


def test_format_bytes_caps_at_terabytes() -> None:
    """The unit ladder ends at TB rather than overflowing its list."""
    assert format_bytes(1024**5).endswith(" TB")


def test_format_bytes_survives_a_nonfinite_count() -> None:
    """A size we cannot render must not raise inside a log call."""
    assert format_bytes(float("inf")) == "0 B"
    assert format_bytes(float("nan")) == "0 B"


# ---------------------------------------------------------------------------
# describe_inputs — the banner's inventory
# ---------------------------------------------------------------------------


def test_describe_inputs_reports_names_types_and_sizes(tmp_path: Path) -> None:
    """The whole point: what was staged, how big, and of what type.

    Args:
        tmp_path (Path): Temporary batch directory.
    """
    (tmp_path / "annual-report.pdf").write_bytes(b"x" * 2048)
    (tmp_path / "meeting-notes.docx").write_bytes(b"y" * 512)

    inventory = describe_inputs(tmp_path)

    assert inventory.total_files == 2
    assert inventory.total_bytes == 2560
    assert [(f.name, f.kind, f.size_bytes) for f in inventory.files] == [
        ("annual-report.pdf", "pdf", 2048),
        ("meeting-notes.docx", "docx", 512),
    ]


def test_describe_inputs_walks_nested_directories(tmp_path: Path) -> None:
    """Uploads preserve subdirectories, so a flat listing would miss files.

    Args:
        tmp_path (Path): Temporary batch directory.
    """
    nested = tmp_path / "archive" / "2024"
    nested.mkdir(parents=True)
    (nested / "survey.csv").write_bytes(b"z" * 10)

    inventory = describe_inputs(tmp_path)

    assert inventory.total_files == 1
    assert inventory.files[0].name == str(Path("archive") / "2024" / "survey.csv")


def test_describe_inputs_counts_by_type(tmp_path: Path) -> None:
    """The header rollup is what makes a long inventory scannable.

    Args:
        tmp_path (Path): Temporary batch directory.
    """
    for name in ("a.pdf", "b.pdf", "c.docx"):
        (tmp_path / name).write_bytes(b"x")

    inventory = describe_inputs(tmp_path)

    assert dict(inventory.by_type) == {"pdf": 2, "docx": 1}
    assert format_by_type(inventory.by_type).startswith("pdf:2")


def test_describe_inputs_reports_extensionless_files(tmp_path: Path) -> None:
    """A file with no extension still has to be accounted for.

    Args:
        tmp_path (Path): Temporary batch directory.
    """
    (tmp_path / "README").write_bytes(b"x")

    assert describe_inputs(tmp_path).files[0].kind == "none"


def test_describe_inputs_caps_the_listing_but_not_the_totals(tmp_path: Path) -> None:
    """A 500-file batch must not print 500 lines, nor under-report itself.

    Args:
        tmp_path (Path): Temporary batch directory.
    """
    for i in range(10):
        (tmp_path / f"doc-{i:02d}.pdf").write_bytes(b"x" * 100)

    inventory = describe_inputs(tmp_path, limit=3)

    assert len(inventory.files) == 3
    assert inventory.omitted == 7
    assert inventory.total_files == 10
    assert inventory.total_bytes == 1000


def test_describe_inputs_on_a_missing_directory_is_empty_not_an_error(tmp_path: Path) -> None:
    """A banner is a log line; it must never be able to fail a run.

    Args:
        tmp_path (Path): Temporary directory whose child does not exist.
    """
    inventory = describe_inputs(tmp_path / "nope")

    assert inventory.total_files == 0
    assert inventory.total_bytes == 0
    assert inventory.files == ()


def test_format_by_type_handles_an_empty_rollup() -> None:
    """An empty batch renders a word, not an empty field."""
    assert format_by_type(()) == "none"


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
