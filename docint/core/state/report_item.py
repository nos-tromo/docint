"""ReportItem ORM model: one snapshotted artifact within a report."""

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from docint.core.state.base import Base

if TYPE_CHECKING:
    from docint.core.state.report import Report


class ReportItem(Base):
    """A single hand-picked artifact frozen into a report.

    The artifact's content is snapshotted as JSON in ``snapshot`` at add-time,
    so the rendered report is immune to later re-ingestion of the underlying
    collection (intentional point-in-time semantics). ``dedupe_key`` is
    type-prefixed (e.g. ``entity:<chunk_id>`` vs ``hate:<chunk_id>``) so the
    same chunk can appear as distinct evidence under different artifact types
    while re-adding the *same* view is a no-op, enforced by the
    ``(report_id, dedupe_key)`` unique constraint.

    Args:
        Base (DeclarativeBase): The declarative base class for SQLAlchemy models.
    """

    __tablename__ = "report_items"
    __table_args__ = (UniqueConstraint("report_id", "dedupe_key", name="uq_report_item_dedupe"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    report_id: Mapped[int] = mapped_column(Integer, ForeignKey("reports.id"), index=True, nullable=False)
    # chat_answer | entity_finding | hate_speech_finding | summary
    artifact_type: Mapped[str] = mapped_column(String, nullable=False)
    dedupe_key: Mapped[str] = mapped_column(String, nullable=False)
    position: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    note: Mapped[str | None] = mapped_column(Text, nullable=True)
    snapshot: Mapped[str] = mapped_column(Text, nullable=False)  # JSON-encoded artifact content
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(UTC), nullable=False)
    report: Mapped["Report"] = relationship("Report", back_populates="items")
