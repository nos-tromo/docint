"""SQLite-backed report manager: owner-scoped CRUD over curated reports.

Mirrors :class:`~docint.core.state.session_manager.SessionManager`'s store
plumbing (its ``_make_session_maker`` / ``_session_scope`` / owner-scoping
pattern) without any of its chat logic. Reports live in the same SQLite DB as
conversations, so both managers point at the same engine and tables.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, cast

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from docint.core.state.base import _make_session_maker
from docint.core.state.report import Report
from docint.core.state.report_item import ReportItem

if TYPE_CHECKING:
    from docint.core.rag import RAG


@dataclass(slots=True)
class ReportManager:
    """Owns curated-report persistence, scoped to the calling principal.

    Every method is owner-scoped: a report owned by a different principal is
    treated as not found (``None``/``[]``/``False``), so the API layer can
    return 404 without leaking whether the id exists — the same posture as
    :class:`SessionManager`.
    """

    rag: RAG
    _SessionMaker: Any | None = field(default=None, init=False, repr=False)
    session_store: str = field(default="", init=False)

    def __post_init__(self) -> None:
        """Adopt the RAG's configured session-store URL."""
        self.session_store = self.rag.session_store

    def init_session_store(self, db_url: str | None = None) -> None:
        """Initialize the report store (shares the conversations DB).

        Args:
            db_url (str | None): Optional database URL to override the default.
        """
        if db_url:
            self.session_store = db_url
        self._SessionMaker = _make_session_maker(self.session_store)

    def init_session_store_if_needed(self) -> None:
        """Initialize the store lazily on first use."""
        if self._SessionMaker is None:
            self.init_session_store()

    @contextmanager
    def _session_scope(self) -> Iterator[Session]:
        """Provide a transactional scope around a series of operations.

        Yields:
            Iterator[Session]: A new database session.

        Raises:
            RuntimeError: If the SessionMaker is not initialized.
        """
        self.init_session_store_if_needed()
        if self._SessionMaker is None:
            raise RuntimeError("SessionMaker is not initialized.")
        session = self._SessionMaker()
        try:
            yield session
        finally:
            session.close()

    # --- serialization helpers ---
    @staticmethod
    def _item_to_dict(item: ReportItem) -> dict[str, Any]:
        """Serialize a :class:`ReportItem`, decoding its JSON snapshot."""
        raw = cast(str | None, item.snapshot)
        try:
            snapshot = json.loads(raw) if raw else {}
        except (TypeError, ValueError):
            snapshot = {}
        created = cast(datetime | None, item.created_at)
        return {
            "id": item.id,
            "artifact_type": item.artifact_type,
            "dedupe_key": item.dedupe_key,
            "position": item.position,
            "note": item.note,
            "snapshot": snapshot,
            "created_at": created.isoformat() if created else None,
        }

    def _serialize_report(self, s: Session, report: Report, *, include_items: bool = True) -> dict[str, Any]:
        """Serialize a :class:`Report`, querying items fresh and ordered.

        Items are re-queried (rather than read off ``report.items``) so the
        returned order always reflects the latest ``position`` even after an
        in-session reorder, regardless of relationship-cache staleness.
        """
        created = cast(datetime | None, report.created_at)
        updated = cast(datetime | None, report.updated_at)
        data: dict[str, Any] = {
            "id": report.id,
            "title": report.title,
            "collection_name": report.collection_name,
            "operator": report.operator,
            "reference_number": report.reference_number,
            "show_toc": True if report.show_toc is None else bool(report.show_toc),
            "session_id": report.session_id,
            "created_at": created.isoformat() if created else None,
            "updated_at": updated.isoformat() if updated else None,
        }
        if include_items:
            items = (
                s.query(ReportItem).filter_by(report_id=report.id).order_by(ReportItem.position, ReportItem.id).all()
            )
            data["items"] = [self._item_to_dict(i) for i in items]
            data["item_count"] = len(items)
        else:
            data["item_count"] = s.query(ReportItem).filter_by(report_id=report.id).count()
        return data

    # --- report CRUD ---
    def create_report(
        self,
        *,
        title: str,
        owner: str | None,
        collection_name: str | None = None,
        operator: str | None = None,
        reference_number: str | None = None,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """Create a new, empty report owned by ``owner``.

        Args:
            title (str): Human-readable report title.
            owner (str | None): The principal that owns this report.
            collection_name (str | None): Collection the report is scoped to.
            operator (str | None): Case worker ("Bearbeiter/-in").
            reference_number (str | None): File reference ("Aktenzeichen").
            session_id (str | None): Optional originating chat session.

        Returns:
            dict[str, Any]: The created report (with an empty item list).
        """
        with self._session_scope() as s:
            report = Report(
                title=title or "Untitled report",
                owner=cast(Any, owner),
                collection_name=cast(Any, collection_name),
                operator=cast(Any, operator),
                reference_number=cast(Any, reference_number),
                session_id=cast(Any, session_id),
            )
            s.add(report)
            s.commit()
            s.refresh(report)
            return self._serialize_report(s, report)

    def list_reports(self, owner: str | None, collection: str | None = None) -> list[dict[str, Any]]:
        """List the caller's reports, most-recently-updated first.

        Args:
            owner (str | None): The principal whose reports to list.
            collection (str | None): Optional collection filter.

        Returns:
            list[dict[str, Any]]: Report summaries (no item bodies).
        """
        with self._session_scope() as s:
            query = s.query(Report).filter(Report.owner == owner)
            if collection is not None:
                query = query.filter(Report.collection_name == collection)
            reports = query.order_by(Report.updated_at.desc(), Report.id.desc()).all()
            return [self._serialize_report(s, r, include_items=False) for r in reports]

    def get_report(self, report_id: int, owner: str | None) -> dict[str, Any] | None:
        """Return a report the caller owns, including its ordered items.

        Args:
            report_id (int): The report id.
            owner (str | None): The principal requesting the report.

        Returns:
            dict[str, Any] | None: The report, or ``None`` when missing or
                not owned by ``owner``.
        """
        with self._session_scope() as s:
            report = s.get(Report, report_id)
            if report is None or report.owner != owner:
                return None
            return self._serialize_report(s, report)

    def update_report(
        self,
        report_id: int,
        owner: str | None,
        *,
        title: str | None = None,
        operator: str | None = None,
        reference_number: str | None = None,
        show_toc: bool | None = None,
    ) -> dict[str, Any] | None:
        """Update a report the caller owns.

        Only fields passed as non-``None`` are written, so a caller can patch
        one field at a time; pass an empty string to clear a field.

        Args:
            report_id (int): The report id.
            owner (str | None): The principal requesting the update.
            title (str | None): New title, when provided.
            operator (str | None): New case worker, when provided.
            reference_number (str | None): New file reference, when provided.
            show_toc (bool | None): Whether the exports render a contents
                section, when provided.

        Returns:
            dict[str, Any] | None: The updated report, or ``None`` when missing
                or not owned by ``owner``.
        """
        with self._session_scope() as s:
            report = s.get(Report, report_id)
            if report is None or report.owner != owner:
                return None
            if title is not None:
                report.title = cast(Any, title)
            if operator is not None:
                report.operator = cast(Any, operator)
            if reference_number is not None:
                report.reference_number = cast(Any, reference_number)
            if show_toc is not None:
                report.show_toc = cast(Any, show_toc)
            s.commit()
            s.refresh(report)
            return self._serialize_report(s, report)

    def delete_report(self, report_id: int, owner: str | None) -> bool:
        """Delete a report (and its items) the caller owns.

        Args:
            report_id (int): The report id.
            owner (str | None): The principal requesting the deletion.

        Returns:
            bool: ``True`` if deleted, ``False`` when missing or not owned.
        """
        with self._session_scope() as s:
            report = s.get(Report, report_id)
            if report is None or report.owner != owner:
                return False
            s.delete(report)
            s.commit()
            return True

    # --- item operations ---
    def add_item(
        self,
        report_id: int,
        owner: str | None,
        *,
        artifact_type: str,
        dedupe_key: str,
        snapshot: dict[str, Any],
        note: str | None = None,
    ) -> dict[str, Any] | None:
        """Append an artifact snapshot to a report, idempotently.

        If ``(report_id, dedupe_key)`` already exists the existing item is
        returned unchanged (re-adding the same view is a no-op).

        Args:
            report_id (int): The report id.
            owner (str | None): The principal requesting the add.
            artifact_type (str): ``chat_answer`` | ``entity_finding`` |
                ``hate_speech_finding`` | ``summary``.
            dedupe_key (str): Type-prefixed stable key (see :class:`ReportItem`).
            snapshot (dict[str, Any]): Frozen artifact content (JSON-serializable).
            note (str | None): Optional investigator note.

        Returns:
            dict[str, Any] | None: The added (or pre-existing) item, or
                ``None`` when the report is missing or not owned by ``owner``.
        """
        with self._session_scope() as s:
            report = s.get(Report, report_id)
            if report is None or report.owner != owner:
                return None
            existing = s.query(ReportItem).filter_by(report_id=report_id, dedupe_key=dedupe_key).one_or_none()
            if existing is not None:
                return self._item_to_dict(existing)
            max_pos = s.query(ReportItem).filter_by(report_id=report_id).count()
            item = ReportItem(
                report_id=report.id,
                artifact_type=artifact_type,
                dedupe_key=dedupe_key,
                position=max_pos,
                note=note,
                snapshot=json.dumps(snapshot, ensure_ascii=False),
            )
            s.add(item)
            report.updated_at = cast(Any, datetime.now(UTC))
            try:
                s.commit()
            except IntegrityError:
                # Concurrent add of the same dedupe_key: fall back to the winner.
                s.rollback()
                existing = s.query(ReportItem).filter_by(report_id=report_id, dedupe_key=dedupe_key).one_or_none()
                return self._item_to_dict(existing) if existing is not None else None
            s.refresh(item)
            return self._item_to_dict(item)

    def remove_item(self, report_id: int, owner: str | None, item_id: int) -> bool:
        """Remove a single item from a report the caller owns.

        Args:
            report_id (int): The report id.
            owner (str | None): The principal requesting the removal.
            item_id (int): The item id.

        Returns:
            bool: ``True`` if removed, ``False`` when the report/item is
                missing or not owned by ``owner``.
        """
        with self._session_scope() as s:
            report = s.get(Report, report_id)
            if report is None or report.owner != owner:
                return False
            item = s.get(ReportItem, item_id)
            if item is None or item.report_id != report_id:
                return False
            s.delete(item)
            report.updated_at = cast(Any, datetime.now(UTC))
            s.commit()
            return True

    def annotate_item(
        self, report_id: int, owner: str | None, item_id: int, *, note: str | None
    ) -> dict[str, Any] | None:
        """Set/clear the investigator note on an item.

        Args:
            report_id (int): The report id.
            owner (str | None): The principal requesting the change.
            item_id (int): The item id.
            note (str | None): The new note (``None`` clears it).

        Returns:
            dict[str, Any] | None: The updated item, or ``None`` when missing
                or not owned by ``owner``.
        """
        with self._session_scope() as s:
            report = s.get(Report, report_id)
            if report is None or report.owner != owner:
                return None
            item = s.get(ReportItem, item_id)
            if item is None or item.report_id != report_id:
                return None
            item.note = cast(Any, note)
            report.updated_at = cast(Any, datetime.now(UTC))
            s.commit()
            s.refresh(item)
            return self._item_to_dict(item)

    def reorder_items(self, report_id: int, owner: str | None, item_ids: Sequence[int]) -> dict[str, Any] | None:
        """Reassign item positions to match ``item_ids`` order.

        Ids not belonging to the report are ignored; items omitted from
        ``item_ids`` keep their relative order after the listed ones.

        Args:
            report_id (int): The report id.
            owner (str | None): The principal requesting the reorder.
            item_ids (Sequence[int]): Desired item id order.

        Returns:
            dict[str, Any] | None: The reordered report, or ``None`` when
                missing or not owned by ``owner``.
        """
        with self._session_scope() as s:
            report = s.get(Report, report_id)
            if report is None or report.owner != owner:
                return None
            items = (
                s.query(ReportItem).filter_by(report_id=report_id).order_by(ReportItem.position, ReportItem.id).all()
            )
            by_id = {cast(int, i.id): i for i in items}
            pos = 0
            seen: set[int] = set()
            for raw_id in item_ids:
                item = by_id.get(raw_id)
                if item is None:
                    continue
                item.position = cast(Any, pos)
                seen.add(raw_id)
                pos += 1
            for item in items:
                if cast(int, item.id) not in seen:
                    item.position = cast(Any, pos)
                    pos += 1
            report.updated_at = cast(Any, datetime.now(UTC))
            s.commit()
            s.refresh(report)
            return self._serialize_report(s, report)

    def list_dedupe_keys(self, report_id: int, owner: str | None) -> list[str]:
        """Return the dedupe keys present in a report (for "already added" UI).

        Args:
            report_id (int): The report id.
            owner (str | None): The principal requesting the keys.

        Returns:
            list[str]: Dedupe keys, or ``[]`` when missing or not owned.
        """
        with self._session_scope() as s:
            report = s.get(Report, report_id)
            if report is None or report.owner != owner:
                return []
            rows = s.query(ReportItem.dedupe_key).filter_by(report_id=report_id).all()
            return [str(r[0]) for r in rows]
