from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, String, Text
from sqlalchemy.orm import relationship

from docint.core.chat.base import Base


class Conversation(Base):
    """
    Represents a user conversation session.

    Args:
        Base (declarative_base): The declarative base class for SQLAlchemy models.
    """

    __tablename__ = "conversations"
    id = Column(String, primary_key=True)  # external session id
    created_at = Column(
        DateTime, default=lambda: datetime.now(timezone.utc), nullable=False
    )
    rolling_summary = Column(Text, default="", nullable=False)
    turns = relationship(
        "Turn",
        back_populates="conversation",
        cascade="all, delete-orphan",
        order_by="Turn.idx",
    )
