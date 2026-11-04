"""Topic reduction: a run's topics merged into groups, and the tree merges follow.

Pure: numpy, and scipy for the linkage, imported where it is used. A reduction
never rewrites a run's topics or a point's payload. Every chunk keeps the topic
it was placed in (a *leaf*), alignment and discovery keep working on the
leaves, and a group is the union of its leaves, read through a
:class:`TopicView`. The groups and their written labels live in the run's record
(``run.json``) under :data:`GROUPS_KEY` and :data:`LABELS_KEY`.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from docint.core.topics.model import TopicInfo, TopicTable, unit_rows

__all__ = [
    "GROUPS_KEY",
    "LABELS_KEY",
    "TopicView",
    "check_groups",
    "group_key",
    "group_label",
    "merge_tree",
    "view_of",
]

#: The record key holding a reduction's groups: lists of leaf ids, each sorted, two or more long.
GROUPS_KEY = "groups"
#: The record key holding written group labels, by :func:`group_key`.
LABELS_KEY = "group_labels"
#: Keywords a group keeps, as many as the fit gives a topic.
KEYWORDS_KEPT = 10


def group_key(members: Iterable[int]) -> str:
    """Name a set of leaves, the key a group's written label is stored under.

    Args:
        members: Leaf ids, in any order.

    Returns:
        The ids ascending, comma-separated, such as ``"3,7,12"``.
    """
    return ",".join(str(member) for member in sorted(members))


def check_groups(leaf_ids: Iterable[int], groups: Iterable[Iterable[int]]) -> tuple[tuple[int, ...], ...]:
    """Check a reduction's groups against a run's leaves.

    Args:
        leaf_ids: The run's topic ids.
        groups: Each group's leaf ids.

    Returns:
        Each group's ids ascending, the groups ordered by their lowest id.

    Raises:
        ValueError: For an id the run lacks, a leaf in two groups, or a group
            of fewer than two distinct leaves.
    """
    known = set(leaf_ids)
    taken: set[int] = set()
    out: list[tuple[int, ...]] = []
    for raw in groups:
        members = tuple(sorted({int(member) for member in raw}))
        if len(members) < 2:
            raise ValueError("a group needs two topics or more")
        if not known.issuperset(members):
            raise ValueError("a group names a topic the run lacks")
        if taken.intersection(members):
            raise ValueError("a topic is in two groups")
        taken.update(members)
        out.append(members)
    return tuple(sorted(out))


@dataclass(frozen=True, eq=False)
class TopicView:
    """A run's topics as a reduction presents them.

    Attributes:
        table: The topics shown: each group as one topic under its lowest leaf
            id, and every leaf in no group as itself, ascending.
        members: Each shown topic's leaf ids, ascending; ``(id,)`` for a leaf
            in no group.
    """

    table: TopicTable
    members: Mapping[int, tuple[int, ...]]

    def sizes(self, leaf_counts: Mapping[int, int]) -> dict[int, int]:
        """Add up leaf counts per shown topic.

        Args:
            leaf_counts: Leaf id to its member count.

        Returns:
            Shown topic id to the members of all its leaves.
        """
        return {tid: sum(leaf_counts.get(leaf, 0) for leaf in leaves) for tid, leaves in self.members.items()}


def group_label(record: Mapping[str, Any], members: Sequence[int]) -> tuple[str, str] | None:
    """Read a group's written label from a run's record.

    Args:
        record: The run's record.
        members: The group's leaf ids.

    Returns:
        ``(label, description)``, or ``None`` when none was written for exactly these leaves.
    """
    stored = record.get(LABELS_KEY)
    entry = stored.get(group_key(members)) if isinstance(stored, Mapping) else None
    if not isinstance(entry, Mapping):
        return None
    label, description = entry.get("label"), entry.get("description")
    if not isinstance(label, str) or not label or not isinstance(description, str):
        return None
    return label, description


def view_of(table: TopicTable, record: Mapping[str, Any]) -> TopicView:
    """Present a run's topics through the reduction its record stores.

    Args:
        table: The run's leaf topics.
        record: The run's record.

    Returns:
        The view; every leaf as itself when the record holds no usable groups.
    """
    try:
        groups = check_groups((t.id for t in table.topics), record.get(GROUPS_KEY) or ())
    except (TypeError, ValueError):
        # A record this code did not write reads as unreduced rather than failing every read.
        groups = ()
    if not groups:
        return TopicView(table=table, members={t.id: (t.id,) for t in table.topics})
    group_of = {member: group for group in groups for member in group}
    rows = {t.id: i for i, t in enumerate(table.topics)}
    infos: list[TopicInfo] = []
    centres: list[np.ndarray] = []
    members: dict[int, tuple[int, ...]] = {}
    for i, topic in enumerate(table.topics):
        group = group_of.get(topic.id)
        if group is None:
            infos.append(topic)
            centres.append(table.centres[i])
            members[topic.id] = (topic.id,)
        elif group[0] == topic.id:
            parts = [(table.topics[rows[m]], table.centres[rows[m]]) for m in group]
            infos.append(_merged_info([info for info, _ in parts], group_label(record, group)))
            centres.append(_merged_centre(parts))
            members[topic.id] = group
    merged = TopicTable(topics=tuple(infos), centres=np.array(centres).reshape(len(infos), table.centres.shape[1]))
    return TopicView(table=merged, members=members)


def _merged_info(infos: Sequence[TopicInfo], written: tuple[str, str] | None) -> TopicInfo:
    """Describe a group as one topic.

    Args:
        infos: Its leaves' records, ascending id.
        written: The group's written label and description, if any.

    Returns:
        The group's record under its lowest leaf id. Without a written label it
        carries its largest leaf's label (by fit size, ties to the lowest id) and
        no description; its keywords are its leaves' scores weighted by fit size.
    """
    largest = max(infos, key=lambda info: (info.fit_size, -info.id))
    label, description = written if written is not None else (largest.label, "")
    firsts = [info.first_seen for info in infos if info.first_seen is not None]
    lasts = [info.last_seen for info in infos if info.last_seen is not None]
    return TopicInfo(
        id=infos[0].id,
        label=label,
        description=description,
        keywords=_merged_keywords(infos),
        origin="discover" if all(info.origin == "discover" for info in infos) else "build",
        added_at=min(info.added_at for info in infos),
        fit_size=sum(info.fit_size for info in infos),
        floor=min(info.floor for info in infos),
        first_seen=min(firsts) if firsts else None,
        last_seen=max(lasts) if lasts else None,
    )


def _merged_keywords(infos: Sequence[TopicInfo]) -> tuple[tuple[str, float], ...]:
    """Merge leaves' keywords, each score weighted by its leaf's share of the group's fit size.

    An approximation: the fit's c-TF-IDF of the merged texts would need the
    texts, which docint does not keep.

    Args:
        infos: The leaves' records.

    Returns:
        At most :data:`KEYWORDS_KEPT` pairs, highest score first, ties alphabetical,
        scores rounded to 6 places as the fit reports them.
    """
    total = sum(info.fit_size for info in infos) or 1
    scores: dict[str, float] = {}
    for info in infos:
        for word, score in info.keywords:
            scores[word] = scores.get(word, 0.0) + score * info.fit_size / total
    ranked = sorted(((word, round(score, 6)) for word, score in scores.items()), key=lambda pair: (-pair[1], pair[0]))
    return tuple(ranked[:KEYWORDS_KEPT])


def _merged_centre(parts: Sequence[tuple[TopicInfo, np.ndarray]]) -> np.ndarray:
    """Compute a group's centre.

    Args:
        parts: Each leaf's record and centre.

    Returns:
        The unit mean of the leaves' centres weighted by fit size, at float32 precision.
    """
    weights = np.array([max(info.fit_size, 1) for info, _ in parts], dtype=np.float64)
    mean = (weights[:, None] * np.stack([centre for _, centre in parts])).sum(axis=0, keepdims=True)
    return unit_rows(mean)[0].astype(np.float32).astype(np.float64)


def merge_tree(table: TopicTable) -> list[tuple[int, int, float]]:
    """Order a run's topics into the merges a reduction to fewer topics follows.

    Average linkage over the cosine distance between topic centres, as
    BERTopic's ``reduce_topics`` merges by topic embeddings. Each step names
    the two clusters it joins by their lowest leaf id, and the joined cluster
    goes on under the lower of the two. Applying the first ``k - n`` steps to
    ``k`` topics leaves ``n``.

    Args:
        table: The run's leaf topics.

    Returns:
        ``(a, b, similarity)`` per step in merge order; similarity is one minus
        the average cosine distance between the two clusters' centres, rounded
        to 6 places, and never rises from one step to the next. Empty for fewer
        than two topics.
    """
    if len(table.topics) < 2:
        return []
    from scipy.cluster.hierarchy import linkage

    steps = linkage(table.centres, method="average", metric="cosine")
    names = [t.id for t in table.topics]
    out: list[tuple[int, int, float]] = []
    for left, right, distance, _size in steps.tolist():
        a, b = names[int(left)], names[int(right)]
        names.append(min(a, b))
        out.append((a, b, round(1.0 - float(distance), 6)))
    return out
