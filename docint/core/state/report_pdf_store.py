"""On-disk store of rendered report PDFs: the newest render of each report.

A report PDF is rendered by a background job and fetched once the job is done,
so it has to outlive the request that asked for it — but not much more, since
it can always be rendered again. The store keeps one PDF per report beside a
JSON sidecar, under scratch space by default (``REPORT_PDF_DIR``), and is only
ever read through the report it belongs to; nothing lists it.

SQLite hands a deleted report's id to the next report created, so the files
are named for the report *incarnation* — its id and a digest of its creation
time — never for the id alone. Keyed by id, a render for the report that
inherited the id could replace a PDF between a download's check and its read,
and serve one owner another owner's evidence; keyed by incarnation, a path
checked for a report only ever holds that report's renders.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
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


def _incarnation(report_id: int, report_created_at: str) -> str:
    """Name one report's files: its id plus a digest of its creation time."""
    digest = hashlib.sha256(report_created_at.encode("utf-8")).hexdigest()[:16]
    return f"{int(report_id)}-{digest}"


def _write_private(path: Path, data: bytes) -> None:
    """Write ``data`` to ``path`` atomically, readable by the process user only."""
    descriptor, staging = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
        os.replace(staging, path)
    except BaseException:
        Path(staging).unlink(missing_ok=True)
        raise


class ReportPdfStore:
    """The newest rendered PDF of each report, keyed by report id."""

    def __init__(self, root: Path) -> None:
        """Bind the store to its directory, created on first write.

        Args:
            root (Path): Directory holding the PDFs and their sidecars.
        """
        self._root = root

    def path(self, report_id: int, report_created_at: str) -> Path:
        """Return where a report's PDF is stored.

        Args:
            report_id (int): The report id.
            report_created_at (str): The report's creation time.

        Returns:
            Path: The PDF's path; the file may not exist.
        """
        return self._root / f"{_incarnation(report_id, report_created_at)}.pdf"

    def _sidecar(self, report_id: int, report_created_at: str) -> Path:
        """Return where a report's PDF metadata is stored."""
        return self._root / f"{_incarnation(report_id, report_created_at)}.json"

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
        _write_private(self.path(report_id, report_created_at), pdf)
        _write_private(self._sidecar(report_id, report_created_at), json.dumps(record).encode("utf-8"))
        return record

    def get(self, report_id: int, report_created_at: str) -> ReportPdfRecord | None:
        """Return a report's stored render, when it belongs to that report.

        Args:
            report_id (int): The report id.
            report_created_at (str): The live report's creation time.

        Returns:
            ReportPdfRecord | None: The metadata, or ``None`` when nothing is
                stored for this report or its PDF is gone.
        """
        try:
            stored: Any = json.loads(self._sidecar(report_id, report_created_at).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if not isinstance(stored, dict) or stored.get("report_created_at") != report_created_at:
            return None
        if not self.path(report_id, report_created_at).is_file():
            return None
        return cast(ReportPdfRecord, stored)

    def delete(self, report_id: int, *, report_created_at: str | None = None) -> bool:
        """Remove a report's stored render.

        Args:
            report_id (int): The report id.
            report_created_at (str | None): When given, remove only the render
                of the report created then — what a late cleanup needs once a
                new report may have reused the id. Without it, everything
                rendered under the id goes, orphans of earlier reports included.

        Returns:
            bool: ``True`` when anything was removed.
        """
        if report_created_at is not None:
            targets = [self.path(report_id, report_created_at), self._sidecar(report_id, report_created_at)]
        else:
            targets = [*self._root.glob(f"{int(report_id)}-*.pdf"), *self._root.glob(f"{int(report_id)}-*.json")]
        removed = False
        for path in targets:
            try:
                path.unlink()
                removed = True
            except FileNotFoundError:
                continue
        return removed
