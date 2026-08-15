"""SQLAlchemy declarative base and session factory for state persistence."""

from pathlib import Path

from sqlalchemy import Engine, create_engine, inspect, text
from sqlalchemy.orm import Session, declarative_base, sessionmaker

from docint.utils.env_cfg import load_principal_env

# --- Session persistence (ORM) ---
Base = declarative_base()


class SessionStoreMigrationError(RuntimeError):
    """A session-store schema migration could not be applied.

    The ORM models already reference the migrated columns, so continuing
    with a stale schema means every query touching them fails with
    ``no such column`` — this must abort startup instead.
    """


def _migration_failure(migration: str, engine: Engine, exc: Exception) -> SessionStoreMigrationError:
    """Build the loud, actionable error for a failed column migration.

    Args:
        migration (str): Human-readable name of the migration that failed.
        engine (Engine): The session-store engine (its URL names the DB).
        exc (Exception): The underlying failure.

    Returns:
        SessionStoreMigrationError: The error to raise ``from exc``.
    """
    hint = ""
    if "readonly database" in str(exc).lower():
        hint = (
            " The database is not writable by this process — on a hardened "
            "deployment (deploy ADR 0001) this usually means the sessions "
            "volume is still owned by another uid; the compose "
            "volume-permissions init service (or a one-time "
            "`chown -R 10001:10001` of the docint volumes) fixes it."
        )
    return SessionStoreMigrationError(
        f"Session-store migration ({migration}) failed on {engine.url}: "
        f"{type(exc).__name__}: {exc}.{hint} Refusing to start with a stale "
        "schema — the ORM would fail every query touching the missing columns."
    )


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
    """Backfill ``validation_*`` / ``retr*`` columns onto a pre-existing ``turns`` table.

    ``Base.metadata.create_all`` only creates missing tables, never adds
    columns to existing ones. Sessions DBs created before validation
    persistence shipped already have a ``turns`` table and would silently
    fail on inserts that touch the new columns.

    Raises:
        SessionStoreMigrationError: If the migration cannot be applied.
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
            # Corrective-retry provenance: a reloaded session must still show
            # that the answer on screen came from a second attempt.
            ("retried", "BOOLEAN"),
            ("retry_query", "TEXT"),
        ]
        with engine.begin() as conn:
            for name, sql_type in pending:
                if name not in existing:
                    conn.execute(text(f"ALTER TABLE turns ADD COLUMN {name} {sql_type}"))
    except Exception as exc:
        raise _migration_failure("turns validation-columns", engine, exc) from exc


def _ensure_conversation_owner_column(engine: Engine) -> None:
    """Backfill an ``owner`` column onto a pre-existing ``conversations`` table.

    ``Base.metadata.create_all`` only creates missing tables, never adds
    columns to existing ones. Sessions DBs created before ownership
    shipped already have a ``conversations`` table and would otherwise
    have no owner to scope list/history/delete by. This adds the column,
    its index, and idempotently backfills legacy rows to the configured
    default identity (the same value the principal resolver returns
    pre-auth) so existing sessions are owned, not orphaned.

    Raises:
        SessionStoreMigrationError: If the migration cannot be applied.
    """
    try:
        inspector = inspect(engine)
        if "conversations" not in inspector.get_table_names():
            return
        existing = {col["name"] for col in inspector.get_columns("conversations")}
        with engine.begin() as conn:
            if "owner" not in existing:
                conn.execute(text("ALTER TABLE conversations ADD COLUMN owner TEXT"))
            index_names = {ix["name"] for ix in inspector.get_indexes("conversations")}
            if "ix_conversations_owner" not in index_names:
                conn.execute(text("CREATE INDEX IF NOT EXISTS ix_conversations_owner ON conversations (owner)"))
            default_identity = load_principal_env().default_identity
            if default_identity:
                conn.execute(
                    text("UPDATE conversations SET owner = :default WHERE owner IS NULL"),
                    {"default": default_identity},
                )
    except Exception as exc:
        raise _migration_failure("conversations owner-column", engine, exc) from exc


def _ensure_conversation_scope_columns(engine: Engine) -> None:
    """Backfill the search-scope columns onto a pre-existing ``conversations`` table.

    ``Base.metadata.create_all`` only creates missing tables, never adds
    columns to existing ones, so a sessions DB created before search scopes
    shipped would fail on every insert touching them. Mirrors
    :func:`_ensure_conversation_owner_column`.

    Args:
        engine (Engine): The session-store engine.

    Raises:
        SessionStoreMigrationError: If the migration cannot be applied.
    """
    try:
        inspector = inspect(engine)
        if "conversations" not in inspector.get_table_names():
            return
        existing = {col["name"] for col in inspector.get_columns("conversations")}
        pending = [("scope_chunk_ids", "TEXT"), ("scope_set_at", "DATETIME")]
        with engine.begin() as conn:
            for name, sql_type in pending:
                if name not in existing:
                    conn.execute(text(f"ALTER TABLE conversations ADD COLUMN {name} {sql_type}"))
    except Exception as exc:
        raise _migration_failure("conversations scope-columns", engine, exc) from exc


def _ensure_report_columns(engine: Engine) -> None:
    """Backfill ``operator`` / ``reference_number`` onto a pre-existing reports table.

    ``Base.metadata.create_all`` never adds columns to an existing table, so a
    ``reports`` table created before these case-metadata fields shipped needs
    them added explicitly. Uses raw SQL + ``inspect`` (no model import) to avoid
    a base ↔ report import cycle.

    Raises:
        SessionStoreMigrationError: If the migration cannot be applied.
    """
    try:
        inspector = inspect(engine)
        if "reports" not in inspector.get_table_names():
            return
        existing = {col["name"] for col in inspector.get_columns("reports")}
        # ``BOOLEAN DEFAULT 1`` backfills pre-existing reports to TOC-on (the model
        # default), matching the "on by default" behavior for new reports.
        pending = [
            ("operator", "TEXT"),
            ("reference_number", "TEXT"),
            ("show_toc", "BOOLEAN DEFAULT 1"),
            ("show_collection_overview", "BOOLEAN DEFAULT 1"),
            ("collection_overview_snapshot", "TEXT"),
        ]
        with engine.begin() as conn:
            for name, sql_type in pending:
                if name not in existing:
                    conn.execute(text(f"ALTER TABLE reports ADD COLUMN {name} {sql_type}"))
    except Exception as exc:
        raise _migration_failure("reports columns", engine, exc) from exc


# --- Session maker ---
def _make_session_maker(db_url: str) -> sessionmaker[Session]:
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
    _ensure_conversation_owner_column(engine)
    _ensure_conversation_scope_columns(engine)
    # ``create_all`` above creates any missing table (incl. reports/report_items,
    # registered via ``docint.core.state.__init__``). Only added *columns* on a
    # pre-existing table need a manual backfill:
    _ensure_report_columns(engine)
    return sessionmaker(bind=engine, expire_on_commit=False)
