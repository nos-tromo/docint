"""SQLAlchemy declarative base and session factory for state persistence."""

from pathlib import Path

from loguru import logger
from sqlalchemy import Engine, create_engine, inspect, text
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


def _ensure_turn_validation_columns(engine: Engine) -> None:
    """Backfill ``validation_*`` columns onto a pre-existing ``turns`` table.

    ``Base.metadata.create_all`` only creates missing tables, never adds
    columns to existing ones. Sessions DBs created before validation
    persistence shipped already have a ``turns`` table and would silently
    fail on inserts that touch the new columns.
    """
    try:
        inspector = inspect(engine)
        if "turns" not in inspector.get_table_names():
            return
        existing = {col["name"] for col in inspector.get_columns("turns")}
        pending = [
            ("validation_checked", "BOOLEAN"),
            ("validation_mismatch", "BOOLEAN"),
            ("validation_reason", "TEXT"),
        ]
        with engine.begin() as conn:
            for name, sql_type in pending:
                if name not in existing:
                    conn.execute(text(f"ALTER TABLE turns ADD COLUMN {name} {sql_type}"))
    except Exception as exc:
        logger.warning(
            "Skipping turns validation-column migration: {}: {}",
            type(exc).__name__,
            exc,
        )


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
    _ensure_turn_validation_columns(engine)
    return sessionmaker(bind=engine, expire_on_commit=False)
