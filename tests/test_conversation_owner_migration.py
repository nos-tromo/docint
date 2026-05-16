"""Tests for the conversations.owner column and its idempotent migration."""

from sqlalchemy import (
    Column,
    DateTime,
    MetaData,
    String,
    Table,
    Text,
    create_engine,
    inspect,
    text,
)

from docint.core.state.base import (
    _ensure_conversation_owner_column,
    _make_session_maker,
)
from docint.core.state.conversation import Conversation


def test_conversation_model_declares_indexed_owner_column() -> None:
    """The ORM model must expose a nullable, indexed ``owner`` column."""
    owner_col = Conversation.__table__.columns["owner"]
    assert owner_col.nullable is True
    assert owner_col.index is True
    assert isinstance(owner_col.type, String)


def test_fresh_db_has_owner_column(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """A DB created via _make_session_maker exposes conversations.owner.

    Args:
        tmp_path: Temporary directory provided by pytest.
    """
    db_path = tmp_path / "fresh.db"
    _make_session_maker(f"sqlite:///{db_path}")

    engine = create_engine(f"sqlite:///{db_path}", future=True)
    columns = {c["name"] for c in inspect(engine).get_columns("conversations")}
    assert "owner" in columns
    engine.dispose()


def test_legacy_conversations_table_gets_column_and_backfill(
    monkeypatch,  # type: ignore[no-untyped-def]
) -> None:
    """A pre-existing conversations table without owner is upgraded + backfilled.

    Args:
        monkeypatch: The pytest monkeypatch fixture.
    """
    monkeypatch.setenv("DOCINT_DEFAULT_IDENTITY", "operator")

    engine = create_engine("sqlite:///:memory:", future=True)
    metadata = MetaData()
    # Mirror only the pre-owner conversations schema. The turns FK is
    # intentionally omitted so the legacy table stands alone for the
    # migration assertion (same approach as the turns migration test).
    Table(
        "conversations",
        metadata,
        Column("id", String, primary_key=True),
        Column("created_at", DateTime, nullable=False),
        Column("collection_name", String, nullable=True),
        Column("rolling_summary", Text, nullable=False),
    )
    metadata.create_all(engine)
    with engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO conversations "
                "(id, created_at, collection_name, rolling_summary) "
                "VALUES ('c1', '2026-01-01 00:00:00', 'alpha', '')"
            )
        )

    pre_columns = {c["name"] for c in inspect(engine).get_columns("conversations")}
    assert "owner" not in pre_columns

    _ensure_conversation_owner_column(engine)
    _ensure_conversation_owner_column(engine)  # second call must be a no-op

    post_columns = {c["name"] for c in inspect(engine).get_columns("conversations")}
    assert "owner" in post_columns
    indexes = inspect(engine).get_indexes("conversations")
    assert any("owner" in ix["column_names"] for ix in indexes)

    with engine.begin() as conn:
        backfilled = conn.execute(
            text("SELECT owner FROM conversations WHERE id = 'c1'")
        ).scalar()
    assert backfilled == "operator"
    engine.dispose()


def test_migration_helper_no_op_when_table_missing() -> None:
    """Helper must short-circuit cleanly if conversations doesn't exist yet."""
    engine = create_engine("sqlite:///:memory:", future=True)
    _ensure_conversation_owner_column(engine)  # no exception
    assert "conversations" not in inspect(engine).get_table_names()
    engine.dispose()


def test_backfill_skipped_when_no_default_identity(
    monkeypatch,  # type: ignore[no-untyped-def]
) -> None:
    """With no configured identity the column is added but rows stay NULL.

    Args:
        monkeypatch: The pytest monkeypatch fixture.
    """
    monkeypatch.delenv("DOCINT_DEFAULT_IDENTITY", raising=False)

    engine = create_engine("sqlite:///:memory:", future=True)
    metadata = MetaData()
    Table(
        "conversations",
        metadata,
        Column("id", String, primary_key=True),
        Column("created_at", DateTime, nullable=False),
        Column("collection_name", String, nullable=True),
        Column("rolling_summary", Text, nullable=False),
    )
    metadata.create_all(engine)
    with engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO conversations "
                "(id, created_at, collection_name, rolling_summary) "
                "VALUES ('c1', '2026-01-01 00:00:00', 'alpha', '')"
            )
        )

    _ensure_conversation_owner_column(engine)

    with engine.begin() as conn:
        owner_val = conn.execute(
            text("SELECT owner FROM conversations WHERE id = 'c1'")
        ).scalar()
    assert owner_val is None
    engine.dispose()
