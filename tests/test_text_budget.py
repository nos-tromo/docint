"""Cutting a text to its allowance around the matches it carries. All text is invented."""

from __future__ import annotations

import random
import re

import pytest

from docint.utils.text_budget import (
    MARKER_RESERVE_CHARS,
    MIN_REGION_CHARS,
    allocate_shortest_first,
    allocate_with_floors,
    excerpt,
    fit_with_marker,
    place,
)

#: An invented section whose decisive sentence sits deep inside it.
FACT = "The pilot excluded every respondent recruited after March."
SECTION = ("Background filler sentence. " * 190)[:5214] + FACT + " Closing filler sentence." * 40
FACT_AT = SECTION.index(FACT)

_MARKER = re.compile(r"\[\.\.\. (\d+) of (\d+) characters not shown \.\.\.\]")


def _walk(text: str, rendered: str) -> list[tuple[int, int]]:
    """Check a rendering shows only slices of ``text`` in order, and that each marker names exactly its gap.

    Args:
        text (str): The full text.
        rendered (str): Its rendering.

    Returns:
        list[tuple[int, int]]: The shown slices, as ``(start, end)`` offsets in ``text``.
    """
    shown: list[tuple[int, int]] = []
    cursor, position = 0, 0
    for marker in _MARKER.finditer(rendered):
        piece = rendered[position : marker.start()]
        assert text[cursor : cursor + len(piece)] == piece
        if piece:
            shown.append((cursor, cursor + len(piece)))
        assert int(marker.group(2)) == len(text)
        cursor += len(piece) + int(marker.group(1))
        position = marker.end()
    tail = rendered[position:]
    assert text[cursor : cursor + len(tail)] == tail
    if tail:
        shown.append((cursor, cursor + len(tail)))
    assert cursor + len(tail) == len(text)
    return shown


@pytest.mark.parametrize(
    ("span", "size", "length", "expected"),
    [
        ((500, 520), 100, 1000, (460, 560)),
        ((500, 800), 100, 1000, (500, 600)),
        ((10, 20), 100, 1000, (0, 100)),
        ((980, 990), 100, 1000, (900, 1000)),
        ((10, 20), 100, 60, (0, 60)),
    ],
    ids=["centred", "longer-than-the-region", "clamped-at-the-start", "clamped-at-the-end", "text-shorter"],
)
def test_place_centres_a_region_on_its_span_inside_the_text(
    span: tuple[int, int], size: int, length: int, expected: tuple[int, int]
) -> None:
    """A region is centred on a short span, starts at a long one, and never leaves the text.

    Args:
        span (tuple[int, int]): The match.
        size (int): The region's size.
        length (int): The text's length.
        expected (tuple[int, int]): The region.
    """
    assert place(span[0], span[1], size, length) == expected


@pytest.mark.parametrize(
    ("anchors", "allowance"),
    [
        ([], 600),
        ([(10, 70)], 600),
        ([(0, 600)], 600),
        ([(10, 70), (300, 340)], 600),
        ([(FACT_AT, FACT_AT + len(FACT))], len(SECTION)),
        ([(FACT_AT, FACT_AT + len(FACT))], len(SECTION) + 10),
        ([(-40, -10), (len(SECTION) + 5, len(SECTION) + 50), (90, 30)], 600),
    ],
    ids=[
        "no-anchor",
        "anchor-in-the-prefix",
        "anchor-ending-at-the-allowance",
        "anchors-in-the-prefix",
        "fits",
        "fits-loosely",
        "anchors-outside-the-text",
    ],
)
def test_excerpt_is_the_prefix_cut_when_no_anchor_lies_beyond_the_prefix(
    anchors: list[tuple[int, int]], allowance: int
) -> None:
    """Without an anchor past the prefix the cut is today's, byte for byte.

    Args:
        anchors (list[tuple[int, int]]): The match spans.
        allowance (int): Characters the text may show.
    """
    assert excerpt(SECTION, anchors, allowance) == fit_with_marker(SECTION, allowance)


def test_excerpt_keeps_a_deep_match_and_marks_both_gaps() -> None:
    """The region around a match deep in the text is kept, and each side of it says what was hidden."""
    rendered, cut = excerpt(SECTION, [(FACT_AT, FACT_AT + len(FACT))], 700)

    assert cut is True
    assert FACT in rendered
    shown = _walk(SECTION, rendered)
    assert len(shown) == 1
    assert shown[0][1] - shown[0][0] == 700 - MARKER_RESERVE_CHARS
    assert len(_MARKER.findall(rendered)) == 2
    assert len(rendered) <= 700 + MARKER_RESERVE_CHARS


def test_excerpt_at_the_legible_minimum_still_keeps_its_match() -> None:
    """A passage cut to 300 characters keeps a region around its match, not its opening."""
    rendered, _ = excerpt(SECTION, [(FACT_AT, FACT_AT + len(FACT))], 300)

    assert FACT in rendered
    assert not rendered.startswith(SECTION[:20])


def test_excerpt_keeps_one_region_per_match_while_each_stays_legible() -> None:
    """Two distant matches share the allowance; neither is lost to the other."""
    early = SECTION.index("Background", 2000)
    anchors = [(early, early + 40), (FACT_AT, FACT_AT + len(FACT))]

    rendered, _ = excerpt(SECTION, anchors, 2 * (MIN_REGION_CHARS + MARKER_RESERVE_CHARS))

    shown = _walk(SECTION, rendered)
    assert len(shown) == 2
    assert FACT in rendered
    assert all(end - start == MIN_REGION_CHARS for start, end in shown)


def test_excerpt_drops_the_latest_match_when_regions_would_be_too_small() -> None:
    """One character short of two legible regions, the match later in the text goes, whatever the list order."""
    early = SECTION.index("Background", 2000)
    anchors = [(FACT_AT, FACT_AT + len(FACT)), (early, early + 40)]

    rendered, _ = excerpt(SECTION, anchors, 2 * (MIN_REGION_CHARS + MARKER_RESERVE_CHARS) - 1)

    shown = _walk(SECTION, rendered)
    assert len(shown) == 1
    assert shown[0][0] <= early < early + 40 <= shown[0][1]
    assert FACT not in rendered


def test_excerpt_falls_back_to_the_prefix_when_no_region_can_be_legible() -> None:
    """Below one legible region the cut is today's prefix again."""
    allowance = MIN_REGION_CHARS + MARKER_RESERVE_CHARS - 1

    assert excerpt(SECTION, [(FACT_AT, FACT_AT + len(FACT))], allowance) == fit_with_marker(SECTION, allowance)


def test_excerpt_merges_regions_that_overlap() -> None:
    """Nearby matches whose regions overlap read as one stretch, with no marker inside it."""
    anchors = [(FACT_AT, FACT_AT + 10), (FACT_AT + 30, FACT_AT + len(FACT))]

    rendered, _ = excerpt(SECTION, anchors, 1200)

    shown = _walk(SECTION, rendered)
    assert len(shown) == 1
    assert FACT in rendered


@pytest.mark.parametrize("length", [301, 1000, 6000, 30_000])
@pytest.mark.parametrize("allowance", [0, 1, 49, 50, 249, 250, 300, 600, 1000, 4000])
@pytest.mark.parametrize(
    "spans",
    [
        [],
        [(0.0, 0.01)],
        [(0.8, 0.82)],
        [(0.1, 0.9)],
        [(0.05, 0.06), (0.5, 0.52), (0.95, 0.97)],
        [(0.5, 0.52), (0.51, 0.6), (0.2, 0.25)],
        [(0.97, 1.0)],
    ],
    ids=["none", "at-start", "deep", "long", "three", "overlapping", "at-end"],
)
def test_excerpt_never_renders_past_its_allowance_and_one_marker(
    length: int, allowance: int, spans: list[tuple[float, float]]
) -> None:
    """The bound a prefix cut keeps holds for every placement, and every marker counts its gap exactly.

    Args:
        length (int): The text's length.
        allowance (int): Characters the text may show.
        spans (list[tuple[float, float]]): Anchors, as fractions of the text.
    """
    text = ("Invented sentence number one. " * (length // 30 + 1))[:length]
    anchors = [(int(start * length), max(int(start * length) + 1, int(end * length))) for start, end in spans]

    rendered, cut = excerpt(text, anchors, allowance)

    assert cut is (length > allowance)
    if cut:
        assert len(rendered) <= allowance + MARKER_RESERVE_CHARS
        _walk(text, rendered)
    else:
        assert rendered == text


def test_a_long_match_keeps_only_its_first_region_when_it_has_to_be_cut() -> None:
    """The known limit: a match longer than one region, such as a whole retrieved chunk, keeps its opening only."""
    first = SECTION.index("Background", 1000)
    spans = [(first, first + 600), (FACT_AT - 600, FACT_AT)]

    rendered, _ = excerpt(SECTION, spans, 2 * (MIN_REGION_CHARS + MARKER_RESERVE_CHARS))

    for start, _end in spans:
        assert SECTION[start : start + MIN_REGION_CHARS] in rendered
        assert SECTION[start : start + MIN_REGION_CHARS + 1] not in rendered


def test_allocate_with_floors_grants_every_floor_before_sharing_the_rest() -> None:
    """A text with a floor gets it first; the rest goes shortest-first to what each text still lacks."""
    assert allocate_shortest_first([1000, 1000, 100], 1200) == [550, 550, 100]

    assert allocate_with_floors([1000, 1000, 100], [750, 0, 100], 1200) == [925, 175, 100]


def test_allocate_with_floors_below_its_floors_is_todays_allocation() -> None:
    """A budget that cannot cover every floor is shared exactly as before, so a retry or a denser seed still renders."""
    lengths, floors = [1000, 1000, 100], [750, 400, 100]

    assert allocate_with_floors(lengths, floors, 1249) == allocate_shortest_first(lengths, 1249)
    assert allocate_with_floors(lengths, floors, 1250)[:2] == [750, 400]


def test_a_floor_longer_than_its_text_asks_only_for_the_text() -> None:
    """Floors count up to their text's length, both for the grant and for whether the budget covers them."""
    # Counted whole, the floors (500) would exceed the budget and fall back to [120, 140, 140].
    assert allocate_with_floors([120, 2000, 2000], [250, 250, 0], 400) == [120, 265, 15]


def test_allocate_with_floors_is_todays_allocation_when_every_floor_is_the_legible_minimum() -> None:
    """With each floor at ``min(len, 300)`` the result is exactly what shortest-first sharing gave before floors."""
    rng = random.Random(11)
    for _ in range(4000):
        lengths = [rng.choice([0, 1, 5, 120, 299, 300, 301, 450, 900, 2500, 7000]) for _ in range(rng.randrange(0, 12))]
        legible = sum(min(length, 300) for length in lengths)
        for budget in {0, max(0, legible - 1), legible, legible + 1, rng.randrange(0, sum(lengths) + 600)}:
            floors = [min(length, 300) for length in lengths]

            assert allocate_with_floors(lengths, floors, budget) == allocate_shortest_first(lengths, budget)


def test_allocate_with_floors_keeps_every_floor_and_shares_the_whole_budget() -> None:
    """Whenever the budget covers the floors, each text keeps its floor, none passes its length, nothing is wasted."""
    rng = random.Random(23)
    for _ in range(4000):
        count = rng.randrange(1, 10)
        lengths = [rng.randrange(0, 3000) for _ in range(count)]
        floors = [rng.choice([0, 250, 300, 500, 2080, 4000]) for _ in range(count)]
        needed = sum(min(length, floor) for length, floor in zip(lengths, floors, strict=True))
        budget = needed + rng.randrange(0, 4000)

        shares = allocate_with_floors(lengths, floors, budget)

        assert all(
            min(length, floor) <= share <= length for length, floor, share in zip(lengths, floors, shares, strict=True)
        )
        assert sum(shares) == min(budget, sum(lengths))
