"""Probes into WeasyPrint's layout for the PDF renderer tests.

The PDF exports are only as fast and as legible as the layout WeasyPrint
computes for them, so these tests drive the real engine rather than a stub:
they count the measuring pass behind auto table layout and look for text that
runs past the edge it belongs inside. Each skips where WeasyPrint's native
libraries (Pango, cairo) are not installed.

Layout boxes gain their geometry (``position_x``, ``width``) during layout, not
as declared attributes, so every box here is handled as ``Any``.
"""

from __future__ import annotations

from typing import Any

import pytest

from docint.core.state import report_render


def weasyprint_html() -> Any:
    """Return WeasyPrint's ``HTML`` class, skipping the test when the engine cannot load.

    Returns:
        Any: ``weasyprint.HTML``.
    """
    html_cls, error = report_render._load_weasyprint()
    if html_cls is None:
        pytest.skip(f"WeasyPrint is unavailable: {error}")
    return html_cls


def count_min_content_splits(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Count the line splits WeasyPrint makes while measuring min-content widths.

    ``weasyprint.layout.preferred`` imports ``split_first_line`` by name, so
    patching that binding counts exactly the measuring pass behind auto table
    layout, which takes one split per character of a cell whose text may
    break anywhere.

    Args:
        monkeypatch (pytest.MonkeyPatch): The test's monkeypatch fixture.

    Returns:
        list[int]: A one-element counter, updated while the test renders.
    """
    from weasyprint.layout import preferred

    calls = [0]
    original = preferred.split_first_line

    def counting(*args: Any, **kwargs: Any) -> Any:
        calls[0] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(preferred, "split_first_line", counting)
    return calls


def _of_kind(root: Any, kind: str) -> list[Any]:
    """Every box below ``root`` that is of WeasyPrint's box class ``kind``."""
    from weasyprint.formatting_structure import boxes

    box_class: Any = getattr(boxes, kind)
    return [box for box in root.descendants() if isinstance(box, box_class)]


def _cells(page: Any, cell_class: str) -> list[Any]:
    """The table cells on ``page`` that carry the class ``cell_class``."""
    return [
        cell
        for cell in _of_kind(page._page_box, "TableCellBox")
        if cell_class in (cell.element.get("class") or "").split()
    ]


def _overrunning(root: Any, right: float) -> list[str]:
    """Text runs below ``root`` whose right edge passes ``right``."""
    return [box.text for box in _of_kind(root, "TextBox") if box.position_x + box.width > right + 0.5]


def text_beyond_the_page(document: Any) -> list[str]:
    """Return text that runs past the right edge of its page's content area.

    Args:
        document (Any): A rendered ``weasyprint.Document``.

    Returns:
        list[str]: The offending text runs; empty when everything fits.
    """
    return [
        text
        for page in document.pages
        for text in _overrunning(page._page_box, page._page_box.content_box_x() + page._page_box.width)
    ]


def text_beyond_its_cell(document: Any, cell_class: str) -> list[str]:
    """Return text in table cells of ``cell_class`` that runs past its cell's right edge.

    Args:
        document (Any): A rendered ``weasyprint.Document``.
        cell_class (str): The class the cells carry (``<td class="…">``).

    Returns:
        list[str]: The offending text runs; empty when every cell holds its text.
    """
    return [
        text
        for page in document.pages
        for cell in _cells(page, cell_class)
        for text in _overrunning(cell, cell.content_box_x() + cell.width)
    ]


def cell_line_counts(document: Any, cell_class: str) -> list[int]:
    """Return how many lines each table cell of ``cell_class`` was laid out on.

    Args:
        document (Any): A rendered ``weasyprint.Document``.
        cell_class (str): The class the cells carry (``<td class="…">``).

    Returns:
        list[int]: One line count per cell, in document order.
    """
    return [len(_of_kind(cell, "LineBox")) for page in document.pages for cell in _cells(page, cell_class)]


def content_width_share(document: Any, cell_class: str) -> list[float]:
    """Return each ``cell_class`` cell's width as a share of its page's content width.

    Args:
        document (Any): A rendered ``weasyprint.Document``.
        cell_class (str): The class the cells carry (``<td class="…">``).

    Returns:
        list[float]: One share in ``(0, 1]`` per cell, in document order.
    """
    return [cell.width / page._page_box.width for page in document.pages for cell in _cells(page, cell_class)]


def hyphenated_text(document: Any, cell_class: str) -> list[str]:
    """Return text runs in ``cell_class`` cells that end where WeasyPrint hyphenated a word.

    Args:
        document (Any): A rendered ``weasyprint.Document``.
        cell_class (str): The class the cells carry (``<td class="…">``).

    Returns:
        list[str]: The runs ending in WeasyPrint's hyphenation character.
    """
    from weasyprint.css.properties import INITIAL_VALUES

    hyphen = INITIAL_VALUES["hyphenate_character"]
    return [
        box.text
        for page in document.pages
        for cell in _cells(page, cell_class)
        for box in _of_kind(cell, "TextBox")
        if box.text.endswith(hyphen)
    ]
