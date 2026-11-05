"""The document retention index: queries and writes over ``document_retention``.

The rule itself is :mod:`docint.core.retention`; this module keeps, per
document, when its ingest completed and when it is due, and answers the
questions the sweep, the notice and the inheritance of derived items ask.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, cast

from sqlalchemy import select
from sqlalchemy.orm import Session

from docint.core.retention import ExpiringItem, RetentionPeriodState, document_expires_at
from docint.core.state.base import _make_session_maker
from docint.core.state.document_retention import DocumentRetention

if TYPE_CHECKING:
    from docint.core.rag import RAG


@dataclass(frozen=True, slots=True)
class PayloadDocument:
    """One document as Qdrant's payloads describe it (for repairing the index).

    Attributes:
        file_hash (str): The document's key.
        point_hashes (frozenset[str]): Every hash its points carry.
        filename (str): Its file name.
        ingested_at (datetime | None): The ``docint_ingested_at`` its points carry, if any.
    """

    file_hash: str
    point_hashes: frozenset[str]
    filename: str
    ingested_at: datetime | None


def _utc(value: datetime | None) -> datetime | None:
    """Read back a stored time as aware UTC (SQLite returns naive datetimes).

    Args:
        value (datetime | None): The stored value.

    Returns:
        datetime | None: The same moment, aware.
    """
    if value is None:
        return None
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


def _deadline(ingested_at: datetime, period: RetentionPeriodState, notice_days: int) -> datetime | None:
    """Return a document's deadline under the period in force.

    Args:
        ingested_at (datetime): When its ingest completed.
        period (RetentionPeriodState): The period and when it took effect.
        notice_days (int): The grace after it took effect.

    Returns:
        datetime | None: The deadline; ``None`` while retention is off.
    """
    if period.days is None:
        return None
    return document_expires_at(ingested_at, period.days, set_at=period.set_at, notice_days=notice_days)


@dataclass(slots=True)
class DocumentRetentionManager:
    """Owns the ``document_retention`` table in the sessions store."""

    rag: RAG
    _SessionMaker: Any | None = field(default=None, init=False, repr=False)
    session_store: str = field(default="", init=False)

    def __post_init__(self) -> None:
        """Adopt the RAG's configured session-store URL."""
        self.session_store = self.rag.session_store

    def init_session_store_if_needed(self) -> None:
        """Initialize the store lazily on first use."""
        if self._SessionMaker is None:
            self._SessionMaker = _make_session_maker(self.session_store)

    @contextmanager
    def _session_scope(self) -> Iterator[Session]:
        """Provide a session that is closed afterwards.

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

    @staticmethod
    def _detached(row: DocumentRetention) -> DocumentRetention:
        """Copy a row out of its session with aware times.

        Args:
            row (DocumentRetention): The ORM row.

        Returns:
            DocumentRetention: A transient copy.
        """
        return DocumentRetention(
            physical=row.physical,
            file_hash=row.file_hash,
            owner=row.owner,
            logical=row.logical,
            filename=row.filename,
            point_hashes=list(row.point_hashes or []),
            ingested_at=_utc(row.ingested_at),
            expires_at=_utc(row.expires_at),
        )

    def record_ingest(
        self,
        physical: str,
        logical: str,
        owner: str | None,
        file_hash: str,
        filename: str,
        point_hashes: Sequence[str],
        *,
        now: datetime,
        period: RetentionPeriodState,
        notice_days: int,
    ) -> None:
        """Index a document whose ingest completed; an existing row is left as it is.

        Re-ingesting identical content while the document exists never moves
        its clock.

        Args:
            physical (str): The Qdrant collection.
            logical (str): The user-visible collection name.
            owner (str | None): The owning principal; ``None`` for CLI ingests.
            file_hash (str): The document's key.
            filename (str): Its file name.
            point_hashes (Sequence[str]): Every hash its points carry.
            now (datetime): When the ingest completed.
            period (RetentionPeriodState): The period in force.
            notice_days (int): The grace after the period took effect.
        """
        with self._session_scope() as s:
            row = s.get(DocumentRetention, (physical, file_hash))
            if row is not None:
                merged = sorted({*(row.point_hashes or []), *point_hashes})
                if merged != sorted(row.point_hashes or []):
                    row.point_hashes = merged
                    s.commit()
                return
            s.add(
                DocumentRetention(
                    physical=physical,
                    file_hash=file_hash,
                    owner=owner,
                    logical=logical,
                    filename=filename,
                    point_hashes=sorted({file_hash, *point_hashes}),
                    ingested_at=now,
                    expires_at=_deadline(now, period, notice_days),
                )
            )
            s.commit()

    def recompute(self, period: RetentionPeriodState, *, notice_days: int) -> int:
        """Set every document's deadline under the period in force (at startup).

        Args:
            period (RetentionPeriodState): The period and when it took effect.
            notice_days (int): The grace after it took effect.

        Returns:
            int: How many deadlines changed.
        """
        changed = 0
        with self._session_scope() as s:
            for row in s.scalars(select(DocumentRetention)):
                deadline = _deadline(cast(datetime, _utc(row.ingested_at)), period, notice_days)
                if _utc(row.expires_at) != deadline:
                    row.expires_at = deadline
                    changed += 1
            s.commit()
        return changed

    def all_rows(self) -> list[DocumentRetention]:
        """Return every indexed document of every collection.

        Returns:
            list[DocumentRetention]: Sorted by collection, then ingest time.
        """
        with self._session_scope() as s:
            rows = s.scalars(
                select(DocumentRetention).order_by(
                    DocumentRetention.physical, DocumentRetention.ingested_at, DocumentRetention.file_hash
                )
            )
            return [self._detached(row) for row in rows]

    def expiring_items(self, until: datetime | None) -> list[ExpiringItem]:
        """Return every owner's documents with a deadline by ``until``.

        Args:
            until (datetime | None): The horizon; ``None`` for every deadline.

        Returns:
            list[ExpiringItem]: Sorted by deadline.
        """
        query = select(DocumentRetention).where(DocumentRetention.expires_at.is_not(None))
        if until is not None:
            query = query.where(DocumentRetention.expires_at <= until)
        with self._session_scope() as s:
            rows = [self._detached(row) for row in s.scalars(query.order_by(DocumentRetention.expires_at))]
        return [
            ExpiringItem(
                "document",
                row.physical,
                row.logical,
                row.owner,
                row.file_hash,
                row.filename or row.file_hash,
                row.ingested_at,
                row.expires_at,
            )
            for row in rows
        ]

    def rows_for(self, physical: str) -> list[DocumentRetention]:
        """Return every indexed document of a collection.

        Args:
            physical (str): The Qdrant collection.

        Returns:
            list[DocumentRetention]: Sorted by ingest time.
        """
        with self._session_scope() as s:
            rows = s.scalars(
                select(DocumentRetention)
                .where(DocumentRetention.physical == physical)
                .order_by(DocumentRetention.ingested_at, DocumentRetention.file_hash)
            )
            return [self._detached(row) for row in rows]

    def due(self, now: datetime, *, limit: int) -> list[DocumentRetention]:
        """Return documents whose deadline has been reached, oldest deadline first.

        Args:
            now (datetime): The current time.
            limit (int): At most this many.

        Returns:
            list[DocumentRetention]: The due documents.
        """
        with self._session_scope() as s:
            rows = s.scalars(
                select(DocumentRetention)
                .where(DocumentRetention.expires_at.is_not(None), DocumentRetention.expires_at <= now)
                .order_by(DocumentRetention.expires_at, DocumentRetention.physical, DocumentRetention.file_hash)
                .limit(limit)
            )
            return [self._detached(row) for row in rows]

    def due_collections(self, now: datetime) -> list[str]:
        """Return every collection holding at least one due document.

        Args:
            now (datetime): The current time.

        Returns:
            list[str]: Physical names, sorted.
        """
        with self._session_scope() as s:
            found = s.scalars(
                select(DocumentRetention.physical)
                .where(DocumentRetention.expires_at.is_not(None), DocumentRetention.expires_at <= now)
                .distinct()
                .order_by(DocumentRetention.physical)
            )
            return [str(value) for value in found]

    def warning_for(self, owner: str | None, now: datetime, notice_days: int) -> list[DocumentRetention]:
        """Return an owner's documents due within the notice period (or overdue).

        Args:
            owner (str | None): The owner.
            now (datetime): The current time.
            notice_days (int): The notice lead time.

        Returns:
            list[DocumentRetention]: Sorted by deadline.
        """
        from datetime import timedelta

        horizon = now + timedelta(days=notice_days)
        with self._session_scope() as s:
            rows = s.scalars(
                select(DocumentRetention)
                .where(
                    DocumentRetention.owner.is_(owner) if owner is None else DocumentRetention.owner == owner,
                    DocumentRetention.expires_at.is_not(None),
                    DocumentRetention.expires_at <= horizon,
                )
                .order_by(DocumentRetention.expires_at, DocumentRetention.file_hash)
            )
            return [self._detached(row) for row in rows]

    def oldest_live(self, physical: str) -> DocumentRetention | None:
        """Return the collection's earliest-expiring document (by deadline, else by ingest).

        Args:
            physical (str): The Qdrant collection.

        Returns:
            DocumentRetention | None: ``None`` when the collection holds no indexed document.
        """
        rows = self.rows_for(physical)
        if not rows:
            return None
        return min(rows, key=lambda row: (row.expires_at is None, row.expires_at or row.ingested_at, row.ingested_at))

    def expiry_of(self, physical: str, file_hashes: Iterable[str]) -> dict[str, datetime]:
        """Return the deadlines of the given documents that are indexed and have one.

        Args:
            physical (str): The Qdrant collection.
            file_hashes (Iterable[str]): Document keys (or any hash their points carry).

        Returns:
            dict[str, datetime]: Deadline per hash that names an indexed document.
        """
        wanted = {value for value in file_hashes if value}
        found: dict[str, datetime] = {}
        for row in self.rows_for(physical):
            if row.expires_at is None:
                continue
            for value in wanted & {row.file_hash, *row.point_hashes}:
                found[value] = row.expires_at
        return found

    def delete(self, physical: str, file_hash: str) -> None:
        """Remove a document's row (after its data is gone).

        Args:
            physical (str): The Qdrant collection.
            file_hash (str): The document's key.
        """
        with self._session_scope() as s:
            row = s.get(DocumentRetention, (physical, file_hash))
            if row is not None:
                s.delete(row)
                s.commit()

    def delete_collection(self, physical: str) -> int:
        """Remove every row of a collection (after the collection is gone).

        Args:
            physical (str): The Qdrant collection.

        Returns:
            int: Rows removed.
        """
        with self._session_scope() as s:
            rows = list(s.scalars(select(DocumentRetention).where(DocumentRetention.physical == physical)))
            for row in rows:
                s.delete(row)
            s.commit()
            return len(rows)

    def repair_from_payload(
        self,
        physical: str,
        logical: str,
        owner: str | None,
        docs: Iterable[PayloadDocument],
        *,
        manifest_times: Mapping[str, datetime],
        period: RetentionPeriodState,
        notice_days: int,
    ) -> int:
        """Index documents Qdrant holds that the index lacks (an upgrade, a restore).

        The clock is the points' own ``docint_ingested_at``, else the ingest
        manifest's time for a completed row, else when the period took effect:
        never the time of the repair, so restored data never gets a fresh
        deadline.

        Args:
            physical (str): The Qdrant collection.
            logical (str): The user-visible collection name.
            owner (str | None): The owning principal.
            docs (Iterable[PayloadDocument]): What Qdrant holds.
            manifest_times (Mapping[str, datetime]): Completion times the manifest knows.
            period (RetentionPeriodState): The period in force.
            notice_days (int): The grace after the period took effect.

        Returns:
            int: Rows added.
        """
        known = {row.file_hash for row in self.rows_for(physical)}
        added = 0
        for doc in docs:
            if doc.file_hash in known:
                continue
            ingested = doc.ingested_at or manifest_times.get(doc.file_hash) or period.set_at
            self.record_ingest(
                physical,
                logical,
                owner,
                doc.file_hash,
                doc.filename,
                sorted(doc.point_hashes),
                now=ingested,
                period=period,
                notice_days=notice_days,
            )
            added += 1
        return added
