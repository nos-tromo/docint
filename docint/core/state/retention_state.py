"""RetentionState ORM model: the retention window in force and when it was set."""

from datetime import datetime

from sqlalchemy import DateTime, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from docint.core.state.base import Base


class RetentionState(Base):
    """The collection retention window last applied, and since when.

    A single row, written at startup. ``window_set_at`` anchors the grace
    period: whenever the window is switched on, off or changed, no collection
    expires sooner than the notice period after it (``docs/retention.md``).

    Args:
        Base (DeclarativeBase): The declarative base class for SQLAlchemy models.
    """

    __tablename__ = "retention_state"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    window: Mapped[str] = mapped_column(String, nullable=False)
    window_set_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
