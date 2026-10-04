"""Turn ORM model capturing a single user/assistant exchange within a conversation."""

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from docint.core.state.base import Base

if TYPE_CHECKING:
    from docint.core.state.citation import Citation
    from docint.core.state.conversation import Conversation


class Turn(Base):
    """Represents a user turn within a conversation.

    Args:
        Base (DeclarativeBase): The declarative base class for SQLAlchemy models.
    """

    __tablename__ = "turns"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    conversation_id: Mapped[str | None] = mapped_column(String, ForeignKey("conversations.id"), index=True)
    idx: Mapped[int] = mapped_column(Integer, nullable=False)  # 0..N
    user_text: Mapped[str] = mapped_column(Text, nullable=False)
    rewritten_query: Mapped[str | None] = mapped_column(Text, nullable=True)
    model_response: Mapped[str] = mapped_column(Text, nullable=False)
    reasoning: Mapped[str | None] = mapped_column(Text, nullable=True)
    validation_checked: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    validation_mismatch: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    validation_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Corrective-retry provenance. The answer above may be a second attempt
    # after the first was rejected as ungrounded; a reloaded session has to be
    # able to say so, or the retry would read as the original answer.
    retried: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    retry_query: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Callable, like every other model here: a bare ``datetime.now(UTC)`` is
    # evaluated once at import, stamping every turn the process ever writes
    # with the moment it booted.
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(UTC), nullable=False)
    conversation: Mapped["Conversation | None"] = relationship("Conversation", back_populates="turns")
    # Ordered by insertion id: citations are written in ``source_nodes``
    # order, which is the order the generator numbered them in. Replay reads
    # the number off that position, so an unordered relationship would hand
    # the chat window a different numbering than the stored answer used.
    citations: Mapped[list["Citation"]] = relationship(
        "Citation",
        back_populates="turn",
        cascade="all, delete-orphan",
        order_by="Citation.id",
    )
