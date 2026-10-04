"""Report ORM model: a curated, owner-scoped collection of hand-picked artifacts."""

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from docint.core.state.base import Base

if TYPE_CHECKING:
    from docint.core.state.report_item import ReportItem


class Report(Base):
    """A curated report grouping hand-picked artifacts for an investigation.

    A report is the unit an investigator assembles by cherry-picking individual
    chat answers, entity findings, and hate-speech findings out of the noisy
    "export everything" views. It is owner-scoped exactly like
    :class:`~docint.core.state.conversation.Conversation`. The optional
    ``session_id`` link uses ``ON DELETE SET NULL`` so deleting a chat session
    never removes a self-contained report.

    Args:
        Base (DeclarativeBase): The declarative base class for SQLAlchemy models.
    """

    __tablename__ = "reports"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    title: Mapped[str] = mapped_column(String, nullable=False)
    owner: Mapped[str | None] = mapped_column(String, nullable=True, index=True)
    collection_name: Mapped[str | None] = mapped_column(String, nullable=True)
    operator: Mapped[str | None] = mapped_column(String, nullable=True)  # case worker — "Bearbeiter/-in"
    reference_number: Mapped[str | None] = mapped_column(String, nullable=True)  # file reference — "Aktenzeichen"
    # Render a contents section (Inhaltsverzeichnis) in the document exports; on by
    # default, toggled off per report for short reports that don't warrant one.
    show_toc: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    # Render the collection's document overview (Dokumentenübersicht) as the
    # trailing report section; on by default, toggled off per report — the twin
    # of ``show_toc``. The manifest is frozen point-in-time in
    # ``collection_overview_snapshot`` (JSON), immune to re-ingestion until
    # explicitly refreshed.
    show_collection_overview: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    collection_overview_snapshot: Mapped[str | None] = mapped_column(Text, nullable=True)
    session_id: Mapped[str | None] = mapped_column(
        String, ForeignKey("conversations.id", ondelete="SET NULL"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(UTC), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
        nullable=False,
    )
    items: Mapped[list["ReportItem"]] = relationship(
        argument="ReportItem",
        back_populates="report",
        cascade="all, delete-orphan",
        order_by="ReportItem.position",
    )
