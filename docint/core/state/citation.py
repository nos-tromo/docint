"""Citation ORM model linking retrieved source chunks to conversation turns."""

from typing import TYPE_CHECKING

from sqlalchemy import (
    Float,
    ForeignKey,
    Integer,
    String,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from docint.core.state.base import Base

if TYPE_CHECKING:
    from docint.core.state.turn import Turn


class Citation(Base):
    """Represents a citation within a turn of a conversation.

    Args:
        Base (DeclarativeBase): The declarative base class for SQLAlchemy models.
    """

    __tablename__ = "citations"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    turn_id: Mapped[int | None] = mapped_column(Integer, ForeignKey("turns.id"), index=True)
    node_id: Mapped[str | None] = mapped_column(String, nullable=True)  # LlamaIndex node id or Qdrant point id
    score: Mapped[float | None] = mapped_column(Float, nullable=True)
    filename: Mapped[str | None] = mapped_column(String, nullable=True)
    file_hash: Mapped[str | None] = mapped_column(String, nullable=True)
    filetype: Mapped[str | None] = mapped_column(String, nullable=True)
    source: Mapped[str | None] = mapped_column(String, nullable=True)  # "table" or ""
    page: Mapped[int | None] = mapped_column(Integer, nullable=True)
    row: Mapped[int | None] = mapped_column(Integer, nullable=True)
    turn: Mapped["Turn | None"] = relationship("Turn", back_populates="citations")
