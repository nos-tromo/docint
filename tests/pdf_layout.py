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
    """Return the renderer's WeasyPrint document factory, skipping the test when the engine cannot load.

    Returns:
        Any: A callable taking ``string=`` like ``weasyprint.HTML``.
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


def _classes(box: Any) -> list[str]:
    """The class names of the element ``box`` was laid out for."""
    if box.element is None:
        return []
    return str(box.element.get("class") or "").split()


def rows_split_across_pages(document: Any, css_class: str) -> list[str]:
    """Return the text of table rows marked ``css_class`` that were laid out on more than one page.

    A row is marked when it, or one of its own cells, carries the class.

    Args:
        document (Any): A rendered ``weasyprint.Document``.
        css_class (str): The row or cell class to look for.

    Returns:
        list[str]: The start of each split row's text; empty when every row stayed whole.
    """
    pages_of: dict[int, set[int]] = {}
    text_of: dict[int, str] = {}
    for number, page in enumerate(document.pages):
        for row in _of_kind(page._page_box, "TableRowBox"):
            if css_class not in _classes(row) and not any(css_class in _classes(cell) for cell in row.children):
                continue
            pages_of.setdefault(id(row.element), set()).add(number)
            text_of[id(row.element)] = " ".join("".join(row.element.itertext()).split())[:60]
    return [text_of[key] for key, pages in pages_of.items() if len(pages) > 1]


def rows_alone_at_page_foot(document: Any, css_class: str) -> list[str]:
    """Return the text of rows marked ``css_class`` that end a page with no table row beneath them.

    Args:
        document (Any): A rendered ``weasyprint.Document``.
        css_class (str): The row class to look for (a finding's ``f-head`` band).

    Returns:
        list[str]: The stranded rows' text; empty when each such row has a row under it.
    """
    stranded: list[str] = []
    for page in document.pages:
        rows = [row for row in _of_kind(page._page_box, "TableRowBox") if row.element is not None]
        for row in rows:
            if css_class in _classes(row) and not any(other.position_y > row.position_y + 0.5 for other in rows):
                stranded.append(" ".join("".join(row.element.itertext()).split())[:60])
    return stranded


def pages_showing_a_finding_without_its_band(document: Any) -> list[int]:
    """Return the 1-based pages that show part of a finding table but not its header band.

    Args:
        document (Any): A rendered ``weasyprint.Document``.

    Returns:
        list[int]: The offending page numbers; empty when every page names the findings it shows.
    """
    missing: list[int] = []
    for number, page in enumerate(document.pages, start=1):
        for table in _of_kind(page._page_box, "TableBox"):
            if "finding" not in _classes(table):
                continue
            if not any("f-head" in _classes(row) for row in _of_kind(table, "TableRowBox")):
                missing.append(number)
    return missing


def images_drawn(document: Any) -> int:
    """Return how many images the rendered document actually draws.

    Args:
        document (Any): A rendered ``weasyprint.Document``.

    Returns:
        int: The number of image boxes across all pages.
    """
    return sum(
        len(_of_kind(page._page_box, "InlineReplacedBox")) + len(_of_kind(page._page_box, "BlockReplacedBox"))
        for page in document.pages
    )


def contents_entries(document: Any) -> dict[str, tuple[str, int | None]]:
    """Return what each contents link prints as its page, and the page its target starts on.

    Read from WeasyPrint's laid-out boxes, not from the PDF's text: extraction
    order puts a floated page number wherever the PDF happens to draw it.

    Args:
        document (Any): A rendered ``weasyprint.Document`` with a contents block.

    Returns:
        dict[str, tuple[str, int | None]]: ``{target id: (the number the entry
            prints, the 1-based page the target starts on)}``.
    """
    printed: dict[str, str] = {}
    for box in _of_kind(document.pages[0]._page_box, "TextBox"):
        if box.element_tag == "a::after":
            target = str(box.element.get("href") or "").lstrip("#")
            printed[target] = printed.get(target, "") + box.text
    starts: dict[str, int] = {}
    for number, page in enumerate(document.pages, start=1):
        for box in _of_kind(page._page_box, "BlockBox"):
            anchor = box.element.get("id") if box.element is not None else None
            if anchor in printed and anchor not in starts:
                starts[anchor] = number
    return {target: (text, starts.get(target)) for target, text in printed.items()}
