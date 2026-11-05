"""DocumentRetention ORM model: every document's ingest time and deadline, per collection."""

from datetime import datetime

from sqlalchemy import JSON, DateTime, String
from sqlalchemy.orm import Mapped, mapped_column

from docint.core.state.base import Base


class DocumentRetention(Base):
    """The retention clock of one document in one collection.

    A document is keyed by its physical collection and the file hash its
    ingest recorded (``IngestManifest.mark_completed``); ``point_hashes``
    lists every hash its Qdrant points carry (``source_file_hash``), so the
    per-document delete finds them all. The same clock is also stamped on the
    points themselves (``docint_ingested_at``), so a restored collection keeps
    its deadlines and this index can be repaired from Qdrant
    (``docs/retention.md``).

    Args:
        Base (DeclarativeBase): The declarative base class for SQLAlchemy models.
    """

    __tablename__ = "document_retention"
    physical: Mapped[str] = mapped_column(String, primary_key=True)
    file_hash: Mapped[str] = mapped_column(String, primary_key=True)
    owner: Mapped[str | None] = mapped_column(String, nullable=True, index=True)
    logical: Mapped[str] = mapped_column(String, nullable=False)
    filename: Mapped[str] = mapped_column(String, nullable=False, default="")
    point_hashes: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    ingested_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True, index=True)
