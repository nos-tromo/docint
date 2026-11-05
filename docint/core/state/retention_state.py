"""RetentionState ORM model: the retention window in force and when it was set."""

from datetime import datetime

from sqlalchemy import DateTime, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from docint.core.state.base import Base


class RetentionState(Base):
    """The document retention period last applied, and since when.

    A single row, written at startup. ``window`` holds ``<days>d`` or ``off``
    (a ``6m``-style value from the retired idle-window rule reads as no
    period); ``window_set_at`` anchors the grace period: whenever the period
    is switched on, off or changed, nothing is deleted sooner than the notice
    period after it (``docs/retention.md``).

    Args:
        Base (DeclarativeBase): The declarative base class for SQLAlchemy models.
    """

    __tablename__ = "retention_state"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    window: Mapped[str] = mapped_column(String, nullable=False)
    window_set_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
