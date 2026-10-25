"""Tests for the topic table."""

import numpy as np
import pytest

from docint.core.topics.model import (
    TopicInfo,
    TopicTable,
    central_members,
    centres_for,
    floors_for,
    merge_new,
    novel,
    unit_rows,
)


def _info(topic_id: int, floor: float = 0.0, **kw: str | None) -> TopicInfo:
    """Build a topic record.

    Args:
        topic_id: Topic id.
        floor: Admission floor.
        **kw: Optional first_seen / last_seen.

    Returns:
        The record.
    """
    return TopicInfo(
        id=topic_id,
        label=f"t{topic_id}",
        description="d",
        keywords=(("w", 0.5),),
        origin="build",
        added_at="2026-01-01T00:00:00+00:00",
        fit_size=10,
        floor=floor,
        first_seen=kw.get("first_seen"),
        last_seen=kw.get("last_seen"),
    )


def _table(centres: list[list[float]], floors: list[float], ids: list[int] | None = None) -> TopicTable:
    """Build a table from unit centres.

    Args:
        centres: Centre rows.
        floors: Floors per row.
        ids: Optional ids.

    Returns:
        The table.
    """
    ids = ids if ids is not None else list(range(len(centres)))
    arr = unit_rows(np.asarray(centres, dtype=np.float64)).astype(np.float32).astype(np.float64)
    return TopicTable(topics=tuple(_info(i, f) for i, f in zip(ids, floors, strict=True)), centres=arr)


def test_centres_are_unit_means_of_their_members_in_float32_precision() -> None:
    """Centres are normalised member means stored at float32 precision.

    Returns:
        None.
    """
    vecs = np.array([[1.0, 0.0], [0.0, 2.0], [3.0, 0.0]])
    labels = np.array([0, 0, 1])
    got = centres_for(vecs, labels, [0, 1])
    mean = unit_rows(vecs[:2]).mean(axis=0)
    want = (mean / np.linalg.norm(mean)).astype(np.float32).astype(np.float64)
    assert np.array_equal(got[0], want)
    assert np.array_equal(got[1], np.array([1.0, 0.0]))
    assert got.dtype == np.float64


def test_floors_are_the_member_cosine_percentile() -> None:
    """Three members at percentile 50 give the median cosine.

    Returns:
        None.
    """
    vecs = unit_rows(np.array([[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]))
    labels = np.array([0, 0, 0])
    centres = np.array([[1.0, 0.0]])
    floors = floors_for(vecs, labels, [0], centres, 50.0)
    assert floors[0] == pytest.approx(np.sqrt(0.5))


def test_align_admits_at_or_above_the_floor_and_leaves_the_rest_unassigned() -> None:
    """A score equal to the floor is admitted.

    Returns:
        None.
    """
    table = _table([[1.0, 0.0]], [0.5])
    exact = float(table.centres[0] @ unit_rows(np.array([[1.0, 1.0]]))[0])
    table = _table([[1.0, 0.0]], [exact])
    got = table.align(np.array([[1.0, 1.0], [0.0, 1.0]]))
    assert got.tolist() == [0, -1]
    assert got.dtype == np.int64


def test_align_breaks_ties_to_the_lowest_topic_id() -> None:
    """Equal scores go to the lowest id.

    Returns:
        None.
    """
    table = _table([[1.0, 0.0], [1.0, 0.0]], [0.0, 0.0], ids=[3, 7])
    assert table.align(np.array([[1.0, 0.0]])).tolist() == [3]


def test_align_of_a_zero_vector_is_unassigned() -> None:
    """A zero vector scores 0, below a positive floor.

    Returns:
        None.
    """
    table = _table([[1.0, 0.0]], [0.3])
    assert table.align(np.zeros((1, 2))).tolist() == [-1]


def test_align_without_topics_is_all_unassigned() -> None:
    """An empty table assigns nothing.

    Returns:
        None.
    """
    table = TopicTable(topics=(), centres=np.zeros((0, 2)))
    assert table.align(np.ones((3, 2))).tolist() == [-1, -1, -1]


def test_align_normalises_its_input() -> None:
    """A scaled copy lands where the original does.

    Returns:
        None.
    """
    table = _table([[1.0, 0.0], [0.0, 1.0]], [0.5, 0.5])
    v = np.array([[0.2, 1.0]])
    assert table.align(v).tolist() == table.align(v * 40).tolist() == [1]


def test_novel_keeps_candidates_below_the_similarity_bar() -> None:
    """0.69 is novel, 0.70 is not, at a bar of 0.7.

    Returns:
        None.
    """
    existing = np.array([[1.0, 0.0]])
    cands = np.array([[0.69, np.sqrt(1 - 0.69**2)], [0.70, np.sqrt(1 - 0.70**2)]])
    assert novel(existing, cands, min_similarity=0.7).tolist() == [True, False]


def test_merge_new_appends_novel_candidates_and_never_changes_existing() -> None:
    """Existing rows stay byte-equal and new ids follow the maximum.

    Returns:
        None.
    """
    table = _table([[1.0, 0.0], [0.0, 1.0]], [0.4, 0.5], ids=[0, 4])
    cands = [_info(5), _info(2), _info(9)]
    cc = unit_rows(np.array([[0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]]))
    merged, mapping = merge_new(table, cands, cc, min_similarity=0.7)
    assert mapping == {2: 5, 9: 6}
    assert merged.ids.tolist() == [0, 4, 5, 6]
    assert merged.ids[:2].tobytes() == table.ids.tobytes()
    assert merged.centres[:2].tobytes() == table.centres.tobytes()
    assert merged.floors[:2].tobytes() == table.floors.tobytes()
    assert np.array_equal(merged.centres[2], cc[1].astype(np.float32).astype(np.float64))


def test_merge_new_on_an_empty_table_numbers_from_zero() -> None:
    """An empty table numbers new topics from 0.

    Returns:
        None.
    """
    table = TopicTable(topics=(), centres=np.zeros((0, 2)))
    cc = unit_rows(np.array([[1.0, 0.0], [0.0, 1.0]]))
    merged, mapping = merge_new(table, [_info(7), _info(3)], cc, min_similarity=0.7)
    assert mapping == {3: 0, 7: 1}
    assert merged.ids.tolist() == [0, 1]


def test_a_table_rejects_unsorted_ids_or_mismatched_centres() -> None:
    """Invalid shapes and orderings raise.

    Returns:
        None.
    """
    with pytest.raises(ValueError):
        TopicTable(topics=(_info(2), _info(1)), centres=np.zeros((2, 2)))
    with pytest.raises(ValueError):
        TopicTable(topics=(_info(1), _info(1)), centres=np.zeros((2, 2)))
    with pytest.raises(ValueError):
        TopicTable(topics=(_info(1),), centres=np.zeros((2, 2)))
    with pytest.raises(ValueError):
        TopicTable(topics=(_info(1),), centres=np.zeros(2))


def test_topic_info_round_trips_through_json() -> None:
    """to_json then from_json restores the record.

    Returns:
        None.
    """
    info = _info(3, 0.25, first_seen="2026-01-01T00:00:00+00:00", last_seen="2026-02-01T00:00:00+00:00")
    assert TopicInfo.from_json(info.to_json()) == info


def test_with_seen_only_widens_the_range() -> None:
    """Narrower ranges change nothing; wider ones extend.

    Returns:
        None.
    """
    table = TopicTable(
        topics=(
            _info(0, first_seen="2026-02-01T00:00:00+00:00", last_seen="2026-03-01T00:00:00+00:00"),
            _info(1),
        ),
        centres=np.zeros((2, 2)),
    )
    out = table.with_seen(
        {
            0: ("2026-02-15T00:00:00+00:00", "2026-04-01T00:00:00+00:00"),
            1: ("2026-05-01T00:00:00+00:00", "2026-06-01T00:00:00+00:00"),
        }
    )
    assert out.topics[0].first_seen == "2026-02-01T00:00:00+00:00"
    assert out.topics[0].last_seen == "2026-04-01T00:00:00+00:00"
    assert out.topics[1].first_seen == "2026-05-01T00:00:00+00:00"
    assert out.topics[1].last_seen == "2026-06-01T00:00:00+00:00"


def test_central_members_order_by_closeness_then_index() -> None:
    """Members sort by cosine descending, ties by index.

    Returns:
        None.
    """
    vecs = unit_rows(np.array([[0.0, 1.0], [1.0, 0.0], [1.0, 1.0], [1.0, 0.0], [5.0, 5.0]]))
    labels = np.array([0, 0, 0, 0, 1])
    centre = np.array([1.0, 0.0])
    assert central_members(vecs, labels, 0, centre, 3) == [1, 3, 2]
    assert central_members(vecs, labels, 0, centre, 10) == [1, 3, 2, 0]


def test_merge_new_with_no_novel_candidate_returns_the_table_unchanged() -> None:
    """Candidates that all repeat an existing topic add nothing.

    Returns:
        None.
    """
    table = _table([[1.0, 0.0]], [0.4])
    cc = unit_rows(np.array([[1.0, 0.01]]))
    merged, mapping = merge_new(table, [_info(3)], cc, min_similarity=0.7)
    assert merged is table
    assert mapping == {}


def test_merge_new_with_no_candidates_returns_the_table_unchanged() -> None:
    """An empty candidate list adds nothing.

    Returns:
        None.
    """
    table = _table([[1.0, 0.0]], [0.4])
    merged, mapping = merge_new(table, [], np.zeros((0, 2)), min_similarity=0.7)
    assert merged is table
    assert mapping == {}
