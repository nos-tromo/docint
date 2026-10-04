"""Conversation ORM model grouping turns within a user session."""

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from sqlalchemy import DateTime, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from docint.core.state.base import Base

if TYPE_CHECKING:
    from docint.core.state.turn import Turn


class Conversation(Base):
    """Represents a user conversation session.

    Args:
        Base (DeclarativeBase): The declarative base class for SQLAlchemy models.
    """

    __tablename__ = "conversations"
    id: Mapped[str] = mapped_column(String, primary_key=True)  # external session id
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(UTC), nullable=False)
    collection_name: Mapped[str | None] = mapped_column(String, nullable=True)
    owner: Mapped[str | None] = mapped_column(String, nullable=True, index=True)
    rolling_summary: Mapped[str] = mapped_column(Text, default="", nullable=False)
    # JSON list of Qdrant point ids the session's answers are restricted to.
    # NULL means unscoped (normal retrieval). Stored on the conversation rather
    # than per turn so a scope survives a reload and reopening the session, the
    # way the pinned collection does.
    scope_chunk_ids: Mapped[str | None] = mapped_column(Text, nullable=True)
    scope_set_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    turns: Mapped[list["Turn"]] = relationship(
        argument="Turn",
        back_populates="conversation",
        cascade="all, delete-orphan",
        order_by="Turn.idx",
    )
