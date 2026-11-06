"""Tests for topic reduction: groups of a run's topics, their view and the merge tree."""

from typing import Any

import numpy as np
import pytest

from docint.core.topics.model import TopicInfo, TopicTable, unit_rows
from docint.core.topics.reduce import (
    GROUPS_KEY,
    LABELS_KEY,
    check_groups,
    group_key,
    group_label,
    merge_tree,
    view_of,
)


def _info(topic_id: int, *, fit_size: int = 10, **kw: Any) -> TopicInfo:
    """Build a topic record.

    Args:
        topic_id: Topic id.
        fit_size: Members in the fit.
        **kw: Overrides for any other field.

    Returns:
        The record.
    """
    fields: dict[str, Any] = {
        "id": topic_id,
        "label": f"label {topic_id}",
        "description": f"summary {topic_id}",
        "keywords": (("shared", 0.5), (f"own{topic_id}", 0.4)),
        "origin": "build",
        "added_at": f"2026-01-0{topic_id % 9 + 1}T00:00:00+00:00",
        "fit_size": fit_size,
        "floor": 0.5 + topic_id / 100,
        "first_seen": None,
        "last_seen": None,
    }
    fields.update(kw)
    return TopicInfo(**fields)


def _table(infos: list[TopicInfo], centres: list[list[float]]) -> TopicTable:
    """Build a table with unit centres at float32 precision.

    Args:
        infos: The records, ascending id.
        centres: One centre row each.

    Returns:
        The table.
    """
    rows = unit_rows(np.asarray(centres, dtype=np.float64)).astype(np.float32).astype(np.float64)
    return TopicTable(topics=tuple(infos), centres=rows)


def _three() -> TopicTable:
    """Three topics: 0 and 2 close together, 1 apart.

    Returns:
        The table.
    """
    return _table(
        [_info(0, fit_size=30), _info(1, fit_size=10), _info(2, fit_size=10)],
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.9, 0.1, 0.0]],
    )


def test_a_run_without_groups_is_its_own_view() -> None:
    """Without groups the view is the run's own table, every topic standing for itself.

    Returns:
        None.
    """
    table = _three()
    view = view_of(table, {})
    assert view.table is table
    assert view.members == {0: (0,), 1: (1,), 2: (2,)}


def test_a_group_is_one_topic_under_its_lowest_id() -> None:
    """A group is listed once, under its lowest leaf id, in id order with the rest.

    Returns:
        None.
    """
    view = view_of(_three(), {GROUPS_KEY: [[2, 0]]})
    assert [t.id for t in view.table.topics] == [0, 1]
    assert view.members == {0: (0, 2), 1: (1,)}
    assert view.table.topic(1) == _three().topic(1)


def test_a_group_adds_up_its_leaves() -> None:
    """Fit size sums, the seen range spans every leaf, and the origin is build unless all leaves were discovered.

    Returns:
        None.
    """
    table = _table(
        [
            _info(3, fit_size=5, first_seen="2026-02-01T00:00:00+00:00", last_seen="2026-02-05T00:00:00+00:00"),
            _info(4, fit_size=7, origin="discover", first_seen="2026-01-20T00:00:00+00:00"),
            _info(5, fit_size=1, origin="discover", last_seen="2026-03-01T00:00:00+00:00"),
        ],
        [[1.0, 0.0], [0.8, 0.2], [0.7, 0.3]],
    )
    merged = view_of(table, {GROUPS_KEY: [[3, 4, 5]]}).table.topic(3)
    assert merged is not None
    assert merged.fit_size == 13
    assert (merged.first_seen, merged.last_seen) == ("2026-01-20T00:00:00+00:00", "2026-03-01T00:00:00+00:00")
    assert merged.origin == "build"
    assert merged.added_at == min(t.added_at for t in table.topics)
    discovered = view_of(table, {GROUPS_KEY: [[4, 5]]}).table.topic(4)
    assert discovered is not None and discovered.origin == "discover"


def test_a_group_without_a_written_label_carries_its_largest_leafs() -> None:
    """The largest leaf by fit size names the group, with no summary; a tie goes to the lowest id.

    Returns:
        None.
    """
    merged = view_of(_three(), {GROUPS_KEY: [[0, 2]]}).table.topic(0)
    assert merged is not None
    assert (merged.label, merged.description) == ("label 0", "")
    tied = _table([_info(0, fit_size=4), _info(1, fit_size=4)], [[1.0, 0.0], [0.0, 1.0]])
    named = view_of(tied, {GROUPS_KEY: [[0, 1]]}).table.topic(0)
    assert named is not None and named.label == "label 0"


def test_a_written_label_names_exactly_its_leaves() -> None:
    """A label written for these leaves is used; one for another set, or a malformed one, is not.

    Returns:
        None.
    """
    record = {
        GROUPS_KEY: [[0, 1, 2]],
        LABELS_KEY: {"0,1,2": {"label": "Merged", "description": "All three."}, "0,2": {"label": "Pair"}},
    }
    merged = view_of(_three(), record).table.topic(0)
    assert merged is not None
    assert (merged.label, merged.description) == ("Merged", "All three.")
    assert group_label(record, [2, 1, 0]) == ("Merged", "All three.")
    assert group_label(record, [0, 2]) is None
    assert group_label({LABELS_KEY: {"0,1": {"label": "", "description": ""}}}, [0, 1]) is None
    assert group_label({LABELS_KEY: ["not a mapping"]}, [0, 1]) is None


def test_a_groups_keywords_weigh_each_leaf_by_fit_size() -> None:
    """Scores are weighted by fit share, best first, ties alphabetical, at most ten.

    Returns:
        None.
    """
    big = _info(0, fit_size=30, keywords=(("alpha", 0.5), ("beta", 0.2)))
    small = _info(1, fit_size=10, keywords=(("beta", 0.6), ("gamma", 0.2)))
    merged = view_of(_table([big, small], [[1.0, 0.0], [0.0, 1.0]]), {GROUPS_KEY: [[0, 1]]}).table.topic(0)
    assert merged is not None
    assert merged.keywords == (("alpha", 0.375), ("beta", 0.3), ("gamma", 0.05))
    many = [_info(i, keywords=tuple((f"w{i}{j}", 0.1) for j in range(6))) for i in range(3)]
    wide = view_of(_table(many, [[1.0, 0.0], [0.9, 0.1], [0.8, 0.2]]), {GROUPS_KEY: [[0, 1, 2]]}).table.topic(0)
    assert wide is not None and len(wide.keywords) == 10
    assert [w for w, _ in wide.keywords] == sorted(w for w, _ in wide.keywords)


def test_a_groups_centre_is_its_leaves_weighted_unit_mean_in_float32() -> None:
    """The centre weighs each leaf centre by fit size and is stored at float32 precision.

    Returns:
        None.
    """
    table = _three()
    view = view_of(table, {GROUPS_KEY: [[0, 2]]})
    centre = view.table.centre(0)
    assert centre is not None
    mean = 30 * table.centres[0] + 10 * table.centres[2]
    want = (mean / np.linalg.norm(mean)).astype(np.float32).astype(np.float64)
    assert np.array_equal(centre, want)
    assert view.table.dim == table.dim


def test_a_view_adds_up_member_counts() -> None:
    """A shown topic counts the members of all its leaves.

    Returns:
        None.
    """
    view = view_of(_three(), {GROUPS_KEY: [[0, 2]]})
    assert view.sizes({0: 5, 1: 3, 2: 4, -1: 9}) == {0: 9, 1: 3}


@pytest.mark.parametrize(
    "groups",
    [[[0, 9]], [[0, 1], [1, 2]], [[0]], [[0, 0]], "0,1", [["a", 1]], 5],
)
def test_groups_this_code_did_not_write_read_as_no_reduction(groups: Any) -> None:
    """Unknown ids, overlaps, singletons or malformed values leave every topic as itself.

    Args:
        groups: The stored value.

    Returns:
        None.
    """
    view = view_of(_three(), {GROUPS_KEY: groups})
    assert view.members == {0: (0,), 1: (1,), 2: (2,)}


def test_groups_are_checked_and_put_in_order() -> None:
    """Members sort, duplicates collapse, groups order by their lowest id; bad groups raise.

    Returns:
        None.
    """
    assert check_groups([0, 1, 2, 3], [[3, 2, 2], [1, 0]]) == ((0, 1), (2, 3))
    assert check_groups([0, 1], []) == ()
    with pytest.raises(ValueError, match="two topics"):
        check_groups([0, 1], [[1, 1]])
    with pytest.raises(ValueError, match="lacks"):
        check_groups([0, 1], [[0, 5]])
    with pytest.raises(ValueError, match="two groups"):
        check_groups([0, 1, 2], [[0, 1], [1, 2]])
    assert group_key([12, 3, 7]) == "3,7,12"


def _cut(ids: list[int], merges: list[tuple[int, int, float]], n: int) -> list[list[int]]:
    """Apply the first ``len(ids) - n`` merges, as the SPA does.

    Args:
        ids: Leaf ids.
        merges: The tree.
        n: Topics to keep.

    Returns:
        The groups of two leaves or more, ascending.
    """
    clusters = {i: [i] for i in ids}
    for a, b, _ in merges[: len(ids) - n]:
        keep, gone = min(a, b), max(a, b)
        clusters[keep] = sorted(clusters[keep] + clusters.pop(gone))
    return sorted(group for group in clusters.values() if len(group) > 1)


def test_the_merge_tree_joins_the_closest_topics_first() -> None:
    """The closest pair merges first under its lower id, and similarity never rises.

    Returns:
        None.
    """
    table = _table(
        [_info(1), _info(4), _info(6), _info(9)],
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.95, 0.05, 0.0], [0.0, 0.9, 0.1]],
    )
    merges = merge_tree(table)
    assert len(merges) == 3
    assert merges[0][:2] == (1, 6)
    assert merges[1][:2] == (4, 9)
    assert {merges[2][0], merges[2][1]} == {1, 4}
    similarities = [s for _, _, s in merges]
    assert similarities == sorted(similarities, reverse=True)
    pair = unit_rows(table.centres[[0, 2]])
    assert merges[0][2] == round(float(pair[0] @ pair[1]), 6)
    assert _cut([1, 4, 6, 9], merges, 4) == []
    assert _cut([1, 4, 6, 9], merges, 3) == [[1, 6]]
    assert _cut([1, 4, 6, 9], merges, 2) == [[1, 6], [4, 9]]
    assert _cut([1, 4, 6, 9], merges, 1) == [[1, 4, 6, 9]]


def test_fewer_than_two_topics_have_no_merges() -> None:
    """One topic or none has nothing to merge.

    Returns:
        None.
    """
    assert merge_tree(_table([_info(0)], [[1.0, 0.0]])) == []
    assert merge_tree(TopicTable(topics=(), centres=np.zeros((0, 2)))) == []
