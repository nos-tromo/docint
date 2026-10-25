"""Tests for topic-fit sampling."""

import hashlib

import numpy as np

from docint.core.topics.sample import (
    TEXT_CHARS,
    Candidate,
    auto_min_topic_size,
    build_sample,
    fit_text,
    min_eligible,
    nr_topics_for,
    rank_key,
    texts_within,
    top_ranked,
)
from docint.utils.topics_client import encode_texts

LONG = "word " * 20


def _cand(point_id: str, vec: list[float], text: str = LONG) -> Candidate:
    """Build a candidate.

    Args:
        point_id: Point id.
        vec: Vector values.
        text: Chunk text.

    Returns:
        The candidate.
    """
    return Candidate(point_id=point_id, vector=np.asarray(vec, dtype=np.float64), text=text)


def test_rank_key_is_sha256_of_seed_colon_id() -> None:
    """The key is the digest of ``seed:id``.

    Returns:
        None.
    """
    assert rank_key(42, "abc") == hashlib.sha256(b"42:abc").digest()


def test_top_ranked_takes_the_smallest_keys_in_rank_order() -> None:
    """The result matches a brute-force sort.

    Returns:
        None.
    """
    ids = [f"id{i}" for i in range(200)]
    expected = [i for _, i in sorted((rank_key(1, i), i) for i in ids)][:25]
    assert top_ranked(ids, seed=1, limit=25) == expected


def test_top_ranked_ignores_input_order() -> None:
    """Shuffling the input does not change the selection.

    Returns:
        None.
    """
    ids = [f"id{i}" for i in range(100)]
    assert top_ranked(ids, seed=3, limit=10) == top_ranked(reversed(ids), seed=3, limit=10)


def test_top_ranked_changes_with_the_seed() -> None:
    """A different seed picks a different set.

    Returns:
        None.
    """
    ids = [f"id{i}" for i in range(100)]
    assert top_ranked(ids, seed=1, limit=10) != top_ranked(ids, seed=2, limit=10)


def test_fit_text_collapses_whitespace_drops_controls_and_cuts_to_2000() -> None:
    """Controls vanish, whitespace collapses, length is bounded.

    Returns:
        None.
    """
    assert fit_text("  a\x00b \n\t c\x07 ") == "ab c"
    assert len(fit_text("x" * 5000)) == TEXT_CHARS == 2000


def test_build_sample_drops_short_texts_sorts_by_id_and_keeps_the_first_duplicate() -> None:
    """Short texts drop, ids sort, the first duplicate represents.

    Returns:
        None.
    """
    sample = build_sample(
        [_cand("b", [0.0, 1.0]), _cand("a", [1.0, 0.0]), _cand("c", [1.0, 0.0]), _cand("d", [1.0, 1.0], "short")],
        min_text_chars=40,
    )
    assert [p.point_id for p in sample.points] == ["a", "b"]
    assert dict(sample.copies) == {"c": "a"}
    assert sample.n == 2


def test_negative_zero_counts_as_a_duplicate_of_zero() -> None:
    """-0.0 and 0.0 vectors are the same key.

    Returns:
        None.
    """
    sample = build_sample([_cand("a", [0.0, 1.0]), _cand("b", [-0.0, 1.0])], min_text_chars=1)
    assert dict(sample.copies) == {"b": "a"}


def test_texts_within_leaves_fitting_texts_alone() -> None:
    """Texts under the limit come back unchanged.

    Returns:
        None.
    """
    texts = ["alpha", "beta"]
    assert texts_within(texts, 10_000) == texts


def test_texts_within_cuts_every_text_to_the_limit_at_a_character_boundary() -> None:
    """Every text is a prefix and the whole encodes under the limit.

    Returns:
        None.
    """
    texts = ["漢" * 2000 for _ in range(50)]
    limit = 20_000
    out = texts_within(texts, limit)
    assert len(encode_texts(out)) <= limit
    assert all(src.startswith(cut) and cut for src, cut in zip(texts, out, strict=True))
    for cut in out:
        cut.encode("utf-8")


def test_auto_min_topic_size_clamps() -> None:
    """The auto size is n // 1000 clamped to 10..100.

    Returns:
        None.
    """
    got = [auto_min_topic_size(n) for n in (0, 9_999, 50_000, 100_000, 500_000)]
    assert got == [10, 10, 50, 100, 100]


def test_min_eligible() -> None:
    """The floor is three times the size, at least 50.

    Returns:
        None.
    """
    assert [min_eligible(n) for n in (10, 20, 100)] == [50, 60, 300]


def test_nr_topics_for() -> None:
    """Reduction counts the outlier class.

    Returns:
        None.
    """
    assert [nr_topics_for(n) for n in (0, -3, 1, 20)] == [0, 0, 2, 21]
