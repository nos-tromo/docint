from sqlalchemy import (
    Column,
    Float,
    ForeignKey,
    Integer,
    String,
)
from sqlalchemy.orm import relationship

from docint.core.state.base import Base


class Citation(Base):
    """
    Represents a citation within a turn of a conversation.

    Args:
        Base (declarative_base): The declarative base class for SQLAlchemy models.
    """

    __tablename__ = "citations"
    id = Column(Integer, primary_key=True, autoincrement=True)
    turn_id = Column(Integer, ForeignKey("turns.id"), index=True)
    node_id = Column(String, nullable=True)  # LlamaIndex node id or Qdrant point id
    score = Column(Float, nullable=True)
    filename = Column(String, nullable=True)
    filetype = Column(String, nullable=True)
    source = Column(String, nullable=True)  # "table" or ""
    page = Column(Integer, nullable=True)
    row = Column(Integer, nullable=True)
    turn = relationship("Turn", back_populates="citations")
