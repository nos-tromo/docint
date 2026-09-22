"""Tests for ``collection_owners.last_activity_at`` and its migration.

The column is the retention clock (``docs/retention.md``), so the migration
decides when every pre-existing collection's clock starts.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from sqlalchemy import Column, DateTime, MetaData, String, Table, create_engine, inspect, text
from sqlalchemy.orm import sessionmaker

from docint.core.state.base import _ensure_collection_owner_activity_column, _make_session_maker
from docint.core.state.collection_ownership import CollectionOwnership

CREATED = "2023-02-01 09:00:00.000000"


def _legacy_engine(tmp_path: Path):  # noqa: ANN202 — local helper
    """A sessions DB whose ``collection_owners`` predates the activity column."""
    engine = create_engine(f"sqlite:///{tmp_path / 'sessions.sqlite3'}", future=True)
    Table(
        "collection_owners",
        MetaData(),
        Column("physical_name", String, primary_key=True),
        Column("owner", String, nullable=True),
        Column("logical_name", String, nullable=False),
        Column("created_at", DateTime, nullable=False),
    ).create(engine)
    with engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO collection_owners (physical_name, owner, logical_name, created_at) "
                "VALUES ('u1__old', 'alice', 'old', :created)"
            ),
            {"created": CREATED},
        )
    return engine


def test_the_model_declares_a_nullable_stamped_activity_column() -> None:
    """Nullable so a missing stamp can mean "never expires"; defaulted so new rows are never missing one."""
    column = CollectionOwnership.__table__.columns["last_activity_at"]
    assert column.nullable is True
    assert column.default is not None


def test_a_fresh_db_has_the_activity_column(tmp_path: Path) -> None:
    """The store factory creates the column for a brand-new DB."""
    db_url = f"sqlite:///{tmp_path / 'fresh.sqlite3'}"
    _make_session_maker(db_url)
    columns = {c["name"] for c in inspect(create_engine(db_url)).get_columns("collection_owners")}
    assert "last_activity_at" in columns


def test_legacy_rows_are_stamped_at_migration_time_not_at_creation(tmp_path: Path) -> None:
    """No history exists to reconstruct; guessing too old would delete data."""
    engine = _legacy_engine(tmp_path)
    before = datetime.now(UTC).replace(tzinfo=None)

    _ensure_collection_owner_activity_column(engine)

    row = sessionmaker(bind=engine)().get(CollectionOwnership, "u1__old")
    assert row is not None
    assert row.last_activity_at is not None
    assert row.last_activity_at >= before - timedelta(seconds=1)
    assert row.last_activity_at != row.created_at


def test_a_backfilled_stamp_compares_with_an_orm_written_one(tmp_path: Path) -> None:
    """Both read back naive: a bound datetime stored with an offset would read back aware and crash a comparison."""
    engine = _legacy_engine(tmp_path)
    _ensure_collection_owner_activity_column(engine)
    session = sessionmaker(bind=engine)()
    session.add(CollectionOwnership(physical_name="u2__new", owner="bob", logical_name="new"))
    session.commit()

    stamps = [row.last_activity_at for row in session.query(CollectionOwnership).all()]

    assert len(stamps) == 2
    assert all(stamp is not None and stamp.tzinfo is None for stamp in stamps)
    assert max(stamps) >= min(stamps)


def test_the_migration_never_moves_an_existing_stamp(tmp_path: Path) -> None:
    """Re-running on every startup must not reset anyone's clock."""
    engine = _legacy_engine(tmp_path)
    _ensure_collection_owner_activity_column(engine)
    with engine.begin() as conn:
        conn.execute(text("UPDATE collection_owners SET last_activity_at = '2024-05-05 05:05:05.000000'"))

    _ensure_collection_owner_activity_column(engine)

    row = sessionmaker(bind=engine)().get(CollectionOwnership, "u1__old")
    assert row is not None
    assert row.last_activity_at == datetime(2024, 5, 5, 5, 5, 5)


def test_a_row_left_unstamped_is_stamped_on_the_next_startup(tmp_path: Path) -> None:
    """A row an older release inserted without the column joins retention from now, never from the past."""
    engine = _legacy_engine(tmp_path)
    _ensure_collection_owner_activity_column(engine)
    with engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO collection_owners (physical_name, owner, logical_name, created_at) "
                "VALUES ('u3__rollback', 'carol', 'rollback', :created)"
            ),
            {"created": CREATED},
        )

    _ensure_collection_owner_activity_column(engine)

    row = sessionmaker(bind=engine)().get(CollectionOwnership, "u3__rollback")
    assert row is not None
    assert row.last_activity_at is not None
