"""SQLite-backed collection-ownership manager: per-user collection namespacing.

Mirrors :class:`~docint.core.state.report_manager.ReportManager`'s store
plumbing (its ``_make_session_maker`` / ``_session_scope`` / owner-scoping
pattern). The manager owns the ``(owner, logical) <-> physical`` mapping that
isolates each user's Qdrant collections. Cross-owner access is "not found"
(``None`` / ``[]``), never an error — the same posture as reports and sessions.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from docint.core.state.base import _make_session_maker
from docint.core.state.collection_ownership import CollectionOwnership

if TYPE_CHECKING:
    from docint.core.rag import RAG


#: Characters a logical name may not carry. qdrant-client formats the
#: collection name straight into the request path with no percent-encoding,
#: so ``#`` (URL fragment), ``?`` (query), ``/`` (segment), ``%`` (escape) and
#: ``\`` each change which collection a request addresses — a name with ``#``
#: silently created and queried its own prefix. Spaces, brackets and unicode
#: are fine: Qdrant accepts them and existing collections use them.
_FORBIDDEN_NAME_CHARS: frozenset[str] = frozenset("#?/\\%")


class InvalidCollectionNameError(ValueError):
    """A logical collection name that cannot be addressed as a Qdrant collection.

    Attributes:
        offending (list[str]): The characters that were rejected, sorted;
            empty when the name itself was blank.
    """

    def __init__(self, offending: list[str]) -> None:
        """Build the client-facing message from the rejected characters."""
        self.offending = offending
        if offending:
            shown = ", ".join(repr(char) for char in offending)
            message = f"Collection name contains characters that cannot be used: {shown}."
        else:
            message = "Collection name must not be empty."
        super().__init__(message)


def validate_collection_name(logical: str) -> str:
    """Refuse a logical name that would not survive the trip to Qdrant.

    Args:
        logical (str): The user-visible collection name.

    Returns:
        str: The same name, when it is usable.

    Raises:
        InvalidCollectionNameError: For a blank name or one carrying a
            character from ``_FORBIDDEN_NAME_CHARS`` or a control character.
    """
    if not logical or not logical.strip():
        raise InvalidCollectionNameError([])
    offending = sorted({char for char in logical if char in _FORBIDDEN_NAME_CHARS or ord(char) < 32 or char == "\x7f"})
    if offending:
        raise InvalidCollectionNameError(offending)
    return logical


def physical_collection_name(owner: str | None, logical: str) -> str:
    """Compute the per-owner physical Qdrant name for a logical collection.

    Deterministic, so the same ``(owner, logical)`` always maps to the same
    physical name. The owner is hashed (not embedded verbatim) so arbitrary
    principal strings (emails, header values) can never produce an invalid
    Qdrant collection name; the logical name is embedded verbatim, so it is
    validated here — the one place a physical name is minted.

    Args:
        owner (str | None): The owning principal.
        logical (str): The user-visible collection name.

    Returns:
        str: The namespaced physical collection name, ``u{owner_hash}__{logical}``.

    Raises:
        InvalidCollectionNameError: When ``logical`` cannot be addressed in a
            Qdrant request URL (see :func:`validate_collection_name`).
    """
    validate_collection_name(logical)
    slug = hashlib.sha256((owner or "").encode("utf-8")).hexdigest()[:12]
    return f"u{slug}__{logical}"


@dataclass(slots=True)
class CollectionOwnerManager:
    """Owns the per-user collection namespace, scoped to the calling principal."""

    rag: RAG
    _SessionMaker: Any | None = field(default=None, init=False, repr=False)
    session_store: str = field(default="", init=False)

    def __post_init__(self) -> None:
        """Adopt the RAG's configured session-store URL (shared with conversations/reports)."""
        self.session_store = self.rag.session_store

    def init_session_store(self, db_url: str | None = None) -> None:
        """Initialize the ownership store (shares the conversations DB).

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

    def register(self, owner: str | None, logical: str) -> str:
        """Register (idempotently) that ``owner`` owns logical collection ``logical``.

        The first registrant owns the name; the physical Qdrant name never
        moves afterwards. A pre-existing mapping (including a legacy backfilled
        row whose physical name is the bare logical name) is returned unchanged.

        Args:
            owner (str | None): The owning principal.
            logical (str): The user-visible collection name.

        Returns:
            str: The physical Qdrant collection name to ingest into / query.
        """
        with self._session_scope() as s:
            existing = self._lookup(s, owner, logical)
            if existing is not None:
                return cast(str, existing.physical_name)
            physical = physical_collection_name(owner, logical)
            s.add(CollectionOwnership(physical_name=physical, owner=cast(Any, owner), logical_name=logical))
            try:
                s.commit()
            except IntegrityError:
                # Concurrent ingest of the same (owner, logical): fall back to the winner.
                s.rollback()
                existing = self._lookup(s, owner, logical)
                if existing is not None:
                    return cast(str, existing.physical_name)
                raise
            return physical

    def resolve(self, owner: str | None, logical: str) -> str | None:
        """Return the physical name for ``owner``'s ``logical`` collection, or ``None``.

        This is the access gate: a collection owned by another principal (or one
        that does not exist) resolves to ``None`` so the API can 404 without
        leaking whether the name exists.

        Args:
            owner (str | None): The requesting principal.
            logical (str): The user-visible collection name.

        Returns:
            str | None: The physical collection name, or ``None`` when not owned.
        """
        with self._session_scope() as s:
            row = self._lookup(s, owner, logical)
            return cast(str, row.physical_name) if row is not None else None

    def list_for(self, owner: str | None) -> list[str]:
        """Return the owner's logical collection names, sorted.

        Args:
            owner (str | None): The principal whose collections to list.

        Returns:
            list[str]: Logical (user-visible) names owned by ``owner``.
        """
        with self._session_scope() as s:
            rows = (
                s.query(CollectionOwnership.logical_name)
                .filter(CollectionOwnership.owner == owner)
                .order_by(CollectionOwnership.logical_name)
                .all()
            )
            return [str(r[0]) for r in rows]

    def list_all(self) -> list[tuple[str | None, str]]:
        """Return every ``(owner, logical_name)`` mapping, sorted by owner then name.

        Admin-only listing surface: callers are responsible for gating on the
        requesting principal's admin flag before exposing the result.

        Returns:
            list[tuple[str | None, str]]: All ownership rows, including legacy
                backfilled ones (whose logical name equals the physical name).
        """
        with self._session_scope() as s:
            rows = (
                s.query(CollectionOwnership.owner, CollectionOwnership.logical_name)
                .order_by(CollectionOwnership.owner, CollectionOwnership.logical_name)
                .all()
            )
            return [(r[0], str(r[1])) for r in rows]

    def delete(self, owner: str | None, logical: str) -> str | None:
        """Delete the mapping the caller owns; return its physical name, or ``None``.

        Returns ``None`` (not an error) when the mapping is missing or owned by
        another principal, so the API can 404 without leaking existence.

        Args:
            owner (str | None): The requesting principal.
            logical (str): The user-visible collection name.

        Returns:
            str | None: The physical collection name that was unmapped, or
                ``None`` when nothing was owned/found.
        """
        with self._session_scope() as s:
            row = self._lookup(s, owner, logical)
            if row is None:
                return None
            physical = cast(str, row.physical_name)
            s.delete(row)
            s.commit()
            return physical

    def backfill_legacy(self, physical_names: list[str], default_owner: str | None) -> None:
        """Assign pre-existing (ownerless) collections to ``default_owner``.

        Mirrors the conversation owner backfill in ``base.py``: every physical
        collection name not already present in the table is inserted as a legacy
        row (``logical_name == physical_name``, owner = default), so the current
        operator keeps access to data created before ownership shipped.
        Idempotent — names already mapped (by anyone) are left untouched.

        Args:
            physical_names (list[str]): Physical Qdrant collection names that
                currently exist (already filtered of hidden companions).
            default_owner (str | None): Identity to assign legacy collections to.
        """
        with self._session_scope() as s:
            known = {str(r[0]) for r in s.query(CollectionOwnership.physical_name).all()}
            added = False
            for name in physical_names:
                if name in known:
                    continue
                s.add(CollectionOwnership(physical_name=name, owner=cast(Any, default_owner), logical_name=name))
                known.add(name)
                added = True
            if added:
                s.commit()

    @staticmethod
    def _lookup(s: Session, owner: str | None, logical: str) -> CollectionOwnership | None:
        """Fetch the row for ``(owner, logical)`` within an open session."""
        return (
            s.query(CollectionOwnership)
            .filter(CollectionOwnership.owner == owner, CollectionOwnership.logical_name == logical)
            .one_or_none()
        )
