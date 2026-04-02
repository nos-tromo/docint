from pathlib import Path
import shutil

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker


# --- Session persistence (ORM) ---
Base = declarative_base()


def _ensure_sqlite_parent_dir(db_url: str) -> None:
    """Create the parent directory for file-backed SQLite URLs.

    Args:
        db_url: SQLAlchemy database URL.

    Raises:
        IsADirectoryError: If the SQLite database path points to a non-empty directory.
    """
    sqlite_prefix = "sqlite:///"
    if not db_url.startswith(sqlite_prefix):
        return

    db_path_str = db_url[len(sqlite_prefix) :]
    if not db_path_str or db_path_str == ":memory:":
        return

    db_path = Path(db_path_str).expanduser()
    db_path.parent.mkdir(parents=True, exist_ok=True)

    if db_path.exists() and db_path.is_dir():
        if any(db_path.iterdir()):
            raise IsADirectoryError(
                f"SQLite database path points to a non-empty directory: {db_path}"
            )
        shutil.rmtree(db_path)


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
