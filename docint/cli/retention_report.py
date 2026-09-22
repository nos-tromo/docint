"""Read-only retention report: what ``COLLECTION_RETENTION`` deletes, and when.

Lists every collection's last activity and deadline under the configured
window, or under ``--window`` as if it were switched on now, so an operator
can see what retention will delete before enabling it. It deletes nothing and
records no window: only the API's startup does that (``docs/retention.md``).
"""

import argparse
import sys
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path

from loguru import logger

from docint.core.rag import RAG
from docint.core.retention import NOTICE_DAYS, expires_at, is_expired, is_warning
from docint.core.state.collection_owner_manager import CollectionActivity, CollectionOwnerManager
from docint.utils.env_cfg import RETENTION_WINDOWS, load_retention_env, set_offline_env
from docint.utils.logger_cfg import init_logger


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the retention report's arguments.

    Args:
        argv (list[str] | None): Argument vector; defaults to ``sys.argv``.

    Returns:
        argparse.Namespace: ``window``, or ``None`` for the configured one.
    """
    parser = argparse.ArgumentParser(description="Show what collection retention deletes, and when. Read-only.")
    parser.add_argument(
        "--window",
        choices=list(RETENTION_WINDOWS),
        help="Evaluate this window as if it were switched on now (default: COLLECTION_RETENTION).",
    )
    return parser.parse_args(argv)


def _day(moment: datetime) -> str:
    """Format a timestamp as its UTC date."""
    return moment.astimezone(UTC).date().isoformat()


def _columns(table: Sequence[Sequence[str]]) -> list[str]:
    """Left-align a table's cells into padded columns."""
    widths = [max(len(row[i]) for row in table) for i in range(len(table[0]))]
    return ["  ".join(cell.ljust(width) for cell, width in zip(row, widths, strict=True)).rstrip() for row in table]


def render_report(
    rows: Sequence[CollectionActivity],
    *,
    window: str,
    months: int,
    window_set_at: datetime,
    hypothetical: bool,
    now: datetime,
    unowned: Sequence[str],
) -> list[str]:
    """Render the report's lines.

    Args:
        rows (Sequence[CollectionActivity]): Every owned collection's clock.
        window (str): The window evaluated, ``off`` included.
        months (int): The window in months; ``0`` when off.
        window_set_at (datetime): When the window was set; anchors the grace.
        hypothetical (bool): Whether the window is evaluated as if switched
            on now rather than as recorded by the API's last startup.
        now (datetime): The evaluation time, timezone-aware.
        unowned (Sequence[str]): Qdrant collections with no owner row.

    Returns:
        list[str]: The report, one line per entry.
    """
    lines: list[str] = []
    if months <= 0:
        lines.append("Collection retention is off (COLLECTION_RETENTION): nothing is deleted.")
        lines.append("Pass --window 6m|12m|18m|24m to see what a window would delete.")
    else:
        basis = f"as if switched on now ({_day(now)})" if hypothetical else f"set {_day(window_set_at)}"
        lines.append(f"Collection retention: {window}, {basis}.")
        grace_end = window_set_at + timedelta(days=NOTICE_DAYS)
        if grace_end > now:
            lines.append(f"Grace period: nothing is deleted before {_day(grace_end)}.")
    lines.append("")

    deadlines = {row.physical: expires_at(row.last_activity_at, months, window_set_at=window_set_at) for row in rows}
    ordered = sorted(
        rows,
        key=lambda row: (deadlines[row.physical] is None, deadlines[row.physical] or now, row.owner or "", row.logical),
    )
    table = [("OWNER", "COLLECTION", "LAST ACTIVITY", "DUE", "NOTE")]
    due_now = soon = never = 0
    for row in ordered:
        deadline = deadlines[row.physical]
        last = _day(row.last_activity_at) if row.last_activity_at is not None else "never recorded"
        due, note = "-", ""
        if months > 0 and deadline is None:
            due, note = "never", "no activity recorded"
            never += 1
        elif deadline is not None and is_expired(deadline, now):
            due, note = _day(deadline), "due now"
            due_now += 1
        elif deadline is not None:
            due = _day(deadline)
            if is_warning(deadline, now):
                note = f"within {NOTICE_DAYS} days"
                soon += 1
        table.append((row.owner or "-", row.logical, last, due, note))
    lines.extend(_columns(table))
    lines.append("")

    if months > 0:
        lines.append(
            f"{len(rows)} collection(s): {due_now} due now, {soon} within {NOTICE_DAYS} days, {never} never expire."
        )
    else:
        lines.append(f"{len(rows)} collection(s).")
    if unowned:
        lines.append(f"Not covered, never deleted (no owner, so no activity clock): {', '.join(unowned)}")
    return lines


def _unowned(rag: RAG, rows: Sequence[CollectionActivity]) -> list[str]:
    """Name the Qdrant collections no ownership row covers.

    Args:
        rag (RAG): The engine, for its Qdrant listing.
        rows (Sequence[CollectionActivity]): Every owned collection.

    Returns:
        list[str]: Sorted physical names; empty when Qdrant cannot be listed.
    """
    try:
        existing = rag.list_collections()
    except Exception as exc:
        logger.warning("Could not list Qdrant collections, so unowned ones are not shown: {}", exc)
        return []
    owned = {row.physical for row in rows}
    return sorted(name for name in existing if name not in owned)


def main(argv: list[str] | None = None) -> None:
    """Run the retention report.

    Args:
        argv (list[str] | None): Argument vector; defaults to ``sys.argv``.
    """
    init_logger()
    set_offline_env()
    args = parse_args(argv)
    window = args.window or load_retention_env().window
    rag = RAG(qdrant_collection="")
    try:
        owners = CollectionOwnerManager(rag)
        rows = owners.list_all_activity()
        state = owners.retention_window()
        unowned = _unowned(rag, rows)
    finally:
        rag.unload_models()
    now = datetime.now(UTC)
    recorded = state is not None and state.window == window
    lines = render_report(
        rows,
        window=window,
        months=RETENTION_WINDOWS[window],
        window_set_at=state.set_at if state is not None and recorded else now,
        hypothetical=not recorded,
        now=now,
        unowned=unowned,
    )
    print("\n".join(lines))


if __name__ == "__main__":
    sys.path.append(str(Path(__file__).parents[2].resolve()))
    main()
