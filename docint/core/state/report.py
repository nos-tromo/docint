"""Report ORM model: a curated, owner-scoped collection of hand-picked artifacts."""

from datetime import UTC, datetime

from sqlalchemy import Boolean, Column, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import relationship

from docint.core.state.base import Base


class Report(Base):  # type: ignore[misc]
    """A curated report grouping hand-picked artifacts for an investigation.

    A report is the unit an investigator assembles by cherry-picking individual
    chat answers, entity findings, and hate-speech findings out of the noisy
    "export everything" views. It is owner-scoped exactly like
    :class:`~docint.core.state.conversation.Conversation`. The optional
    ``session_id`` link uses ``ON DELETE SET NULL`` so deleting a chat session
    never removes a self-contained report.

    Args:
        Base (declarative_base): The declarative base class for SQLAlchemy models.
    """

    __tablename__ = "reports"
    id = Column(Integer, primary_key=True, autoincrement=True)
    title = Column(String, nullable=False)
    owner = Column(String, nullable=True, index=True)
    collection_name = Column(String, nullable=True)
    operator = Column(String, nullable=True)  # case worker — "Bearbeiter/-in"
    reference_number = Column(String, nullable=True)  # file reference — "Aktenzeichen"
    # Render a contents section (Inhaltsverzeichnis) in the document exports; on by
    # default, toggled off per report for short reports that don't warrant one.
    show_toc = Column(Boolean, nullable=False, default=True)
    # Render the collection's document overview (Dokumentenübersicht) as the
    # trailing report section; on by default, toggled off per report — the twin
    # of ``show_toc``. The manifest is frozen point-in-time in
    # ``collection_overview_snapshot`` (JSON), immune to re-ingestion until
    # explicitly refreshed.
    show_collection_overview = Column(Boolean, nullable=False, default=True)
    collection_overview_snapshot = Column(Text, nullable=True)
    session_id = Column(String, ForeignKey("conversations.id", ondelete="SET NULL"), nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(UTC), nullable=False)
    updated_at = Column(
        DateTime,
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
        nullable=False,
    )
    items = relationship(
        argument="ReportItem",
        back_populates="report",
        cascade="all, delete-orphan",
        order_by="ReportItem.position",
    )
