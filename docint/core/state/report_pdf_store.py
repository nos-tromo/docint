"""On-disk store of rendered report PDFs: the newest render of each report.

A report PDF is rendered by a background job and fetched once the job is done,
so it has to outlive the request that asked for it — but not much more, since
it can always be rendered again. The store keeps one PDF per report beside a
JSON sidecar, under scratch space by default (``REPORT_PDF_DIR``), and is only
ever read through the report it belongs to; nothing lists it.

SQLite hands a deleted report's id to the next report created, so a read and a
guarded delete also match the report's creation time: a render of a deleted
report is never served for the report that inherited its id.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypedDict, cast


class ReportPdfRecord(TypedDict):
    """What the store records beside a rendered PDF.

    Attributes:
        report_id: The report the PDF was rendered from.
        filename: The download name, fixed at render time.
        size: The PDF's size in bytes.
        created_at: When the PDF was rendered (ISO, UTC).
        report_created_at: The report's creation time, which tells a reused id apart.
        report_updated_at: The report's ``updated_at`` at render time; the
            render is current while the report's still matches it.
        pages: How many pages the PDF has, when known.
    """

    report_id: int
    filename: str
    size: int
    created_at: str
    report_created_at: str
    report_updated_at: str
    pages: int | None


def _write_private(path: Path, data: bytes) -> None:
    """Write ``data`` to ``path`` atomically, readable by the process user only."""
    staging = path.with_name(f".{path.name}.tmp")
    staging.unlink(missing_ok=True)
    descriptor = os.open(staging, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(data)
    os.replace(staging, path)


class ReportPdfStore:
    """The newest rendered PDF of each report, keyed by report id."""

    def __init__(self, root: Path) -> None:
        """Bind the store to its directory, created on first write.

        Args:
            root (Path): Directory holding the PDFs and their sidecars.
        """
        self._root = root

    def path(self, report_id: int) -> Path:
        """Return where a report's PDF is stored.

        Args:
            report_id (int): The report id.

        Returns:
            Path: The PDF's path; the file may not exist.
        """
        return self._root / f"{int(report_id)}.pdf"

    def _sidecar(self, report_id: int) -> Path:
        """Return where a report's PDF metadata is stored."""
        return self._root / f"{int(report_id)}.json"

    def _read_sidecar(self, report_id: int) -> dict[str, Any] | None:
        """Return a report's stored metadata, or ``None`` when absent or unreadable."""
        try:
            data = json.loads(self._sidecar(report_id).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return data if isinstance(data, dict) else None

    def write(
        self,
        report_id: int,
        pdf: bytes,
        *,
        report_created_at: str,
        report_updated_at: str,
        filename: str,
        pages: int | None,
        now: datetime,
    ) -> ReportPdfRecord:
        """Store a report's PDF, replacing any earlier render.

        The PDF is replaced before its sidecar, so a reader can never pair new
        metadata with an old file and call a stale render current.

        Args:
            report_id (int): The report the PDF was rendered from.
            pdf (bytes): The rendered document.
            report_created_at (str): The report's creation time.
            report_updated_at (str): The report's ``updated_at`` at render time.
            filename (str): The download name.
            pages (int | None): How many pages the PDF has, when known.
            now (datetime): The render's completion time.

        Returns:
            ReportPdfRecord: The stored metadata.
        """
        self._root.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.chmod(self._root, 0o700)
        record: ReportPdfRecord = {
            "report_id": int(report_id),
            "filename": filename,
            "size": len(pdf),
            "created_at": now.astimezone(UTC).isoformat(),
            "report_created_at": report_created_at,
            "report_updated_at": report_updated_at,
            "pages": pages,
        }
        _write_private(self.path(report_id), pdf)
        _write_private(self._sidecar(report_id), json.dumps(record).encode("utf-8"))
        return record

    def get(self, report_id: int, report_created_at: str) -> ReportPdfRecord | None:
        """Return a report's stored render, when it belongs to that report.

        Args:
            report_id (int): The report id.
            report_created_at (str): The live report's creation time.

        Returns:
            ReportPdfRecord | None: The metadata, or ``None`` when nothing is
                stored, the render belongs to an earlier report with this id,
                or its PDF is gone.
        """
        stored = self._read_sidecar(report_id)
        if stored is None or stored.get("report_created_at") != report_created_at:
            return None
        if not self.path(report_id).is_file():
            return None
        return cast(ReportPdfRecord, stored)

    def delete(self, report_id: int, *, report_created_at: str | None = None) -> bool:
        """Remove a report's stored render.

        Args:
            report_id (int): The report id.
            report_created_at (str | None): When given, remove the render only
                if it belongs to the report created then — the guard a late
                cleanup needs once a new report may have reused the id.

        Returns:
            bool: ``True`` when anything was removed.
        """
        if report_created_at is not None:
            stored = self._read_sidecar(report_id)
            if stored is None or stored.get("report_created_at") != report_created_at:
                return False
        removed = False
        for path in (self.path(report_id), self._sidecar(report_id)):
            try:
                path.unlink()
                removed = True
            except FileNotFoundError:
                continue
        return removed
