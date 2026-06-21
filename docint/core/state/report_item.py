"""ReportItem ORM model: one snapshotted artifact within a report."""

from datetime import UTC, datetime

from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import relationship

from docint.core.state.base import Base


class ReportItem(Base):  # type: ignore[misc]
    """A single hand-picked artifact frozen into a report.

    The artifact's content is snapshotted as JSON in ``snapshot`` at add-time,
    so the rendered report is immune to later re-ingestion of the underlying
    collection (intentional point-in-time semantics). ``dedupe_key`` is
    type-prefixed (e.g. ``entity:<chunk_id>`` vs ``hate:<chunk_id>``) so the
    same chunk can appear as distinct evidence under different artifact types
    while re-adding the *same* view is a no-op, enforced by the
    ``(report_id, dedupe_key)`` unique constraint.

    Args:
        Base (declarative_base): The declarative base class for SQLAlchemy models.
    """

    __tablename__ = "report_items"
    __table_args__ = (UniqueConstraint("report_id", "dedupe_key", name="uq_report_item_dedupe"),)

    id = Column(Integer, primary_key=True, autoincrement=True)
    report_id = Column(Integer, ForeignKey("reports.id"), index=True, nullable=False)
    artifact_type = Column(String, nullable=False)  # chat_answer | entity_finding | hate_speech_finding | summary
    dedupe_key = Column(String, nullable=False)
    position = Column(Integer, nullable=False, default=0)
    note = Column(Text, nullable=True)
    snapshot = Column(Text, nullable=False)  # JSON-encoded artifact content
    created_at = Column(DateTime, default=lambda: datetime.now(UTC), nullable=False)
    report = relationship("Report", back_populates="items")
