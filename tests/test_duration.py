"""Tests for the shared elapsed-duration formatter."""

import pytest

from docint.utils.duration import format_elapsed


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [
        (0.0, "00:00"),
        (-5.0, "00:00"),
        (0.9, "00:00"),
        (59.0, "00:59"),
        (60.0, "01:00"),
        (3_599.0, "59:59"),
        (3_600.0, "1:00:00"),
        (86_399.0, "23:59:59"),
        (86_400.0, "1d 00:00:00"),
        (151_237.0, "1d 18:00:37"),
    ],
)
def test_format_elapsed_scales_past_mm_ss(seconds: float, expected: str) -> None:
    """The log's duration must scale rather than overflow one column.

    Mirrors the SPA's ``formatDuration``
    (``frontend/src/lib/ingestStatus.ts``) exactly so an operator can compare
    a log line against the ingest card without converting units: MM:SS under
    an hour, H:MM:SS under a day, and ``Nd HH:MM:SS`` beyond. Rolling hours
    into the minutes column is the bug this pins — a ~42 h run must not read
    as ``2500:37``.

    Args:
        seconds (float): Elapsed wall-clock seconds.
        expected (str): The formatted duration.
    """
    assert format_elapsed(seconds) == expected


@pytest.mark.parametrize("seconds", [float("nan"), float("inf"), float("-inf")])
def test_format_elapsed_never_raises_on_a_non_finite_duration(seconds: float) -> None:
    """A completion line must survive a nonsense duration.

    ``int(nan)`` raises, so an unguarded formatter would turn a run that
    finished into a traceback at the moment it reported success.

    Args:
        seconds (float): A non-finite elapsed duration.
    """
    assert format_elapsed(seconds) == "00:00"
