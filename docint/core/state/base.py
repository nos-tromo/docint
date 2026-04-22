"""SQLAlchemy declarative base and session factory for state persistence."""

from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker


# --- Session persistence (ORM) ---
Base = declarative_base()


def _ensure_sqlite_parent_dir(db_url: str) -> None:
    """Create the parent directory for file-backed SQLite URLs.

    Args:
        db_url (str): SQLAlchemy database URL.
    """
    sqlite_prefix = "sqlite:///"
    if not db_url.startswith(sqlite_prefix):
        return

    db_path_str = db_url[len(sqlite_prefix) :]
    if not db_path_str or db_path_str == ":memory:":
        return

    Path(db_path_str).expanduser().parent.mkdir(parents=True, exist_ok=True)


# --- Session maker ---
def _make_session_maker(db_url: str) -> sessionmaker:
    """Creates a new SQLAlchemy session maker.

    Args:
        db_url (str): The database URL.

    Returns:
        sessionmaker: The SQLAlchemy session maker.
    """
    _ensure_sqlite_parent_dir(db_url)
    engine = create_engine(db_url, future=True)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, expire_on_commit=False)
