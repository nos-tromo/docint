"""CollectionOwnership ORM model: maps an owner's logical collection to its physical Qdrant name."""

from datetime import UTC, datetime

from sqlalchemy import Column, DateTime, Index, String

from docint.core.state.base import Base


class CollectionOwnership(Base):  # type: ignore[misc]
    """Ownership record for one user-visible collection.

    docint is multi-tenant: each user names collections logically (``mydocs``)
    while the physical Qdrant collection is namespaced per owner, so Alice's
    ``mydocs`` and Bob's ``mydocs`` are independent. This table is the source of
    truth for the ``(owner, logical_name) -> physical_name`` mapping; the access
    gate (``resolve``) and the per-user listing (``list_for``) both read it.

    Legacy collections created before ownership shipped keep their bare name
    (``physical_name == logical_name``) and are backfilled to the configured
    default identity, so no Qdrant rename is ever required.

    Args:
        Base (declarative_base): The declarative base class for SQLAlchemy models.
    """

    __tablename__ = "collection_owners"
    physical_name = Column(String, primary_key=True)
    owner = Column(String, nullable=True, index=True)
    logical_name = Column(String, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(UTC), nullable=False)

    # One logical name per owner. (NULL owners are distinct under SQLite, which
    # is fine: real principals are non-NULL — NULL only appears in tests.)
    __table_args__ = (Index("ix_collection_owners_owner_logical", "owner", "logical_name", unique=True),)
