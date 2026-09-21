"""RetentionState ORM model: the retention window in force and when it was set."""

from sqlalchemy import Column, DateTime, Integer, String

from docint.core.state.base import Base


class RetentionState(Base):  # type: ignore[misc]
    """The collection retention window last applied, and since when.

    A single row, written at startup. ``window_set_at`` anchors the grace
    period: whenever the window is switched on, off or changed, no collection
    expires sooner than the notice period after it (``docs/retention.md``).

    Args:
        Base (declarative_base): The declarative base class for SQLAlchemy models.
    """

    __tablename__ = "retention_state"
    id = Column(Integer, primary_key=True)
    window = Column(String, nullable=False)
    window_set_at = Column(DateTime, nullable=False)
