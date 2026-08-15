"""Elapsed-duration formatting for operator-facing log lines.

Kept here rather than beside any one caller because every boundary that owns
a complete run renders the same number: the ingest CLI, the legacy
synchronous ingest endpoint, and the job registry's terminal paths. The SPA's
ingest card renders its own copy in the same shape
(``formatDuration`` in ``frontend/src/lib/ingestStatus.ts``), so an operator
comparing a log line against the card never has to convert units.
"""

import math

_SECONDS_PER_DAY = 86_400
_SECONDS_PER_HOUR = 3_600


def format_elapsed(seconds: float) -> str:
    """Format an elapsed duration for a completion log line.

    The duration *scales* rather than overflowing one column: rolling hours
    into the minutes place would render a ~42 h run as ``2500:37``.

    Args:
        seconds (float): Elapsed wall-clock seconds. Non-positive and
            non-finite inputs yield ``"00:00"`` rather than a negative or
            nonsense duration — a log line must never raise (``int(nan)``
            does) on the way out of a run that otherwise succeeded.

    Returns:
        str: ``MM:SS`` under an hour, ``H:MM:SS`` under a day, and
        ``Nd HH:MM:SS`` beyond (DIN 1301 day symbol, shared across locales).
    """
    if not math.isfinite(seconds):
        return "00:00"
    total = int(seconds)
    if total <= 0:
        return "00:00"
    days, remainder = divmod(total, _SECONDS_PER_DAY)
    hours, remainder = divmod(remainder, _SECONDS_PER_HOUR)
    minutes, secs = divmod(remainder, 60)
    if days > 0:
        return f"{days}d {hours:02d}:{minutes:02d}:{secs:02d}"
    if hours > 0:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"
