"""Turn ORM model capturing a single user/assistant exchange within a conversation."""

from datetime import UTC, datetime

from sqlalchemy import Boolean, Column, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import relationship

from docint.core.state.base import Base


class Turn(Base):  # type: ignore[misc]
    """Represents a user turn within a conversation.

    Args:
        Base (declarative_base): The declarative base class for SQLAlchemy models.
    """

    __tablename__ = "turns"
    id = Column(Integer, primary_key=True, autoincrement=True)
    conversation_id = Column(String, ForeignKey("conversations.id"), index=True)
    idx = Column(Integer, nullable=False)  # 0..N
    user_text = Column(Text, nullable=False)
    rewritten_query = Column(Text, nullable=True)
    model_response = Column(Text, nullable=False)
    reasoning = Column(Text, nullable=True)
    validation_checked = Column(Boolean, nullable=True)
    validation_mismatch = Column(Boolean, nullable=True)
    validation_reason = Column(Text, nullable=True)
    # Corrective-retry provenance. The answer above may be a second attempt
    # after the first was rejected as ungrounded; a reloaded session has to be
    # able to say so, or the retry would read as the original answer.
    retried = Column(Boolean, nullable=True)
    retry_query = Column(Text, nullable=True)
    # Callable, like every other model here: a bare ``datetime.now(UTC)`` is
    # evaluated once at import, stamping every turn the process ever writes
    # with the moment it booted.
    created_at = Column(DateTime, default=lambda: datetime.now(UTC), nullable=False)
    conversation = relationship("Conversation", back_populates="turns")
    # Ordered by insertion id: citations are written in ``source_nodes``
    # order, which is the order the generator numbered them in. Replay reads
    # the number off that position, so an unordered relationship would hand
    # the chat window a different numbering than the stored answer used.
    citations = relationship(
        "Citation",
        back_populates="turn",
        cascade="all, delete-orphan",
        order_by="Citation.id",
    )
