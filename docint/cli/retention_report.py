"""Read-only retention report: what document retention deletes, and when.

Lists every document, chat and report with its deadline under the period in
force (``DOCUMENT_RETENTION_DAYS``), or every document under ``--days`` as if
that period were switched on now, so an operator can see what retention will
delete before changing it. It deletes nothing and records no period: only the
API's startup does that (``docs/retention.md``). It is not audited, so it names
items by reference only (file hash, chat id, report id), never by file name,
question or title.
"""

import argparse
import sys
from collections.abc import Sequence
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

from docint.core.rag import RAG
from docint.core.retention import ExpiringItem, RetentionPeriodState, document_expires_at, is_due, is_warning_within
from docint.core.state.collection_owner_manager import CollectionOwnerManager
from docint.core.state.document_retention import DocumentRetention
from docint.core.state.document_retention_manager import DocumentRetentionManager
from docint.core.state.report_manager import ReportManager
from docint.core.state.session_manager import SessionManager
from docint.utils.env_cfg import load_document_retention_env, set_offline_env
from docint.utils.logger_cfg import init_logger

#: The periods ``--days`` accepts, as ``DOCUMENT_RETENTION_DAYS`` does.
_MIN_DAYS, _MAX_DAYS = 1, 3650


def _period(raw: str) -> int:
    """Parse ``--days``: a whole number of days in the accepted range.

    Args:
        raw (str): The argument.

    Returns:
        int: The period in days.

    Raises:
        argparse.ArgumentTypeError: Outside the range or not a number.
    """
    try:
        days = int(raw)
    except ValueError:
        raise argparse.ArgumentTypeError(f"'{raw}' is not a number of days") from None
    if not _MIN_DAYS <= days <= _MAX_DAYS:
        raise argparse.ArgumentTypeError(f"the period must be {_MIN_DAYS} to {_MAX_DAYS} days")
    return days


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the retention report's arguments.

    Args:
        argv (list[str] | None): Argument vector; defaults to ``sys.argv``.

    Returns:
        argparse.Namespace: ``days``, or ``None`` for the configured period.
    """
    parser = argparse.ArgumentParser(description="Show what document retention deletes, and when. Read-only.")
    parser.add_argument(
        "--days",
        type=_period,
        help="Evaluate this period as if it were switched on now (default: DOCUMENT_RETENTION_DAYS).",
    )
    return parser.parse_args(argv)


def _day(moment: datetime) -> str:
    """Format a timestamp as its UTC date."""
    return moment.astimezone(UTC).date().isoformat()


def _columns(table: Sequence[Sequence[str]]) -> list[str]:
    """Left-align a table's cells into padded columns."""
    widths = [max(len(row[i]) for row in table) for i in range(len(table[0]))]
    return ["  ".join(cell.ljust(width) for cell, width in zip(row, widths, strict=True)).rstrip() for row in table]


def hypothetical_items(
    rows: Sequence[DocumentRetention], period: RetentionPeriodState, *, notice_days: int
) -> list[ExpiringItem]:
    """Give every indexed document the deadline ``period`` would give it.

    Args:
        rows (Sequence[DocumentRetention]): Every indexed document.
        period (RetentionPeriodState): The period evaluated and when it would take effect.
        notice_days (int): The grace after it takes effect.

    Returns:
        list[ExpiringItem]: One per document.
    """
    if period.days is None:
        return []
    return [
        ExpiringItem(
            "document",
            row.physical,
            row.logical,
            row.owner,
            row.file_hash,
            row.filename or row.file_hash,
            row.ingested_at,
            document_expires_at(row.ingested_at, period.days, set_at=period.set_at, notice_days=notice_days),
        )
        for row in rows
    ]


def render_report(
    items: Sequence[ExpiringItem],
    *,
    period_days: int | None,
    notice_days: int,
    set_at: datetime,
    hypothetical: bool,
    now: datetime,
    unowned: Sequence[str],
) -> list[str]:
    """Render the report's lines.

    Args:
        items (Sequence[ExpiringItem]): What is listed, with its deadline.
        period_days (int | None): The period evaluated; ``None`` is off.
        notice_days (int): The notice and grace in days.
        set_at (datetime): When the period took effect; anchors the grace.
        hypothetical (bool): Whether the period is evaluated as if switched
            on now rather than as recorded by the API's last startup.
        now (datetime): The evaluation time, timezone-aware.
        unowned (Sequence[str]): Collections whose documents have no owner.

    Returns:
        list[str]: The report, one line per entry.
    """
    lines: list[str] = []
    if period_days is None:
        lines.append("Document retention is off (DOCUMENT_RETENTION_DAYS=off): nothing is deleted.")
        lines.append("Pass --days N to see what a period would delete.")
        return lines
    basis = f"as if switched on now ({_day(now)})" if hypothetical else f"in force since {_day(set_at)}"
    lines.append(f"Document retention: {period_days} days after ingest, {basis}.")
    grace_end = set_at + timedelta(days=notice_days)
    if grace_end > now:
        lines.append(f"Grace period: nothing is deleted before {_day(grace_end)}.")
    if hypothetical:
        lines.append("Chats, reports and extracts follow the earliest of their sources; documents are listed.")
    lines.append("")

    ordered = sorted(items, key=lambda i: (i.expires_at or now, i.owner or "", i.logical or "", i.kind, i.id))
    table = [("OWNER", "COLLECTION", "KIND", "ID", "INGESTED/CREATED", "DUE", "NOTE")]
    due_now = soon = 0
    for item in ordered:
        note = ""
        if item.expires_at is not None and is_due(item.expires_at, now):
            note = "due now"
            due_now += 1
        elif is_warning_within(item.expires_at, now, notice_days):
            note = f"within {notice_days} days"
            soon += 1
        table.append(
            (
                item.owner or "-",
                item.logical or item.physical or "-",
                item.kind,
                item.id,
                _day(item.since) if item.since is not None else "-",
                _day(item.expires_at) if item.expires_at is not None else "-",
                note,
            )
        )
    lines.extend(_columns(table))
    lines.append("")
    lines.append(f"{len(items)} item(s): {due_now} due now, {soon} within {notice_days} days.")
    if unowned:
        lines.append(f"Collections without an owner (deleted like any other; no one is notified): {', '.join(unowned)}")
    return lines


def main(argv: list[str] | None = None) -> None:
    """Run the retention report.

    Args:
        argv (list[str] | None): Argument vector; defaults to ``sys.argv``.
    """
    init_logger()
    set_offline_env()
    args = parse_args(argv)
    configured = load_document_retention_env()
    days = args.days if args.days is not None else configured.days
    rag = RAG(qdrant_collection="")
    try:
        owners = CollectionOwnerManager(rag)
        state = owners.retention_period()
        logical_of = {row.physical: row.logical for row in owners.list_all_activity()}
        documents = DocumentRetentionManager(rag)
        rows = documents.all_rows()
        now = datetime.now(UTC)
        recorded = days is not None and state is not None and state.days == days
        if recorded:
            assert state is not None
            set_at = state.set_at
            items = documents.expiring_items(None)
            items += [
                replace(i, logical=logical_of.get(i.physical or "", i.physical))
                for i in SessionManager(rag).expiring_sessions(None)
            ]
            items += ReportManager(rag).expiring_reports(None)
        else:
            set_at = now
            items = hypothetical_items(rows, RetentionPeriodState(days, now), notice_days=configured.notice_days)
    finally:
        rag.unload_models()
    unowned = sorted({row.physical for row in rows if row.owner is None})
    lines = render_report(
        items,
        period_days=days,
        notice_days=configured.notice_days,
        set_at=set_at,
        hypothetical=not recorded,
        now=now,
        unowned=unowned,
    )
    print("\n".join(lines))


if __name__ == "__main__":
    sys.path.append(str(Path(__file__).parents[2].resolve()))
    main()
