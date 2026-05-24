"""Conversation ORM model grouping turns within a user session."""

from datetime import UTC, datetime

from sqlalchemy import Column, DateTime, String, Text
from sqlalchemy.orm import relationship

from docint.core.state.base import Base


class Conversation(Base):  # type: ignore[misc]
    """Represents a user conversation session.

    Args:
        Base (declarative_base): The declarative base class for SQLAlchemy models.
    """

    __tablename__ = "conversations"
    id = Column(String, primary_key=True)  # external session id
    created_at = Column(DateTime, default=lambda: datetime.now(UTC), nullable=False)
    collection_name = Column(String, nullable=True)
    rolling_summary = Column(Text, default="", nullable=False)
    turns = relationship(
        argument="Turn",
        back_populates="conversation",
        cascade="all, delete-orphan",
        order_by="Turn.idx",
    )
