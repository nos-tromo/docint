"""Cutting a text to its allowance around the matches it carries. All text is invented."""

from __future__ import annotations

import re

import pytest

from docint.utils.text_budget import (
    MARKER_RESERVE_CHARS,
    MIN_REGION_CHARS,
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
