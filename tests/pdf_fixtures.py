"""Hand-rolled minimal PDF builder for pipeline tests.

The pipeline tests need real, deterministic PDF bytes (docling-parse and
pypdfium2 parse actual files — mocking their object models would test the
mocks). No PDF-authoring library is in the dev dependencies, so this module
writes the handful of PDF syntax the tests need: pages of a given size,
Helvetica / Helvetica-Bold text runs at explicit coordinates and sizes, and
uncompressed RGB image XObjects placed via a ``cm``/``Do`` pair. All text is
synthetic.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field


@dataclass(frozen=True)
class TextRun:
    """One line of text drawn at ``(x, y)`` (bottom-left origin, points).

    Attributes:
        text: ASCII text to draw (parentheses/backslashes are escaped).
        x: Left edge of the text baseline start.
        y: Baseline y position.
        size: Font size in points.
        bold: Use Helvetica-Bold instead of Helvetica.
        rotate90: Draw the run rotated 90 degrees counter-clockwise (text runs upward).
    """

    text: str
    x: float
    y: float
    size: float = 11.0
    bold: bool = False
    rotate90: bool = False


@dataclass(frozen=True)
class ImageBox:
    """One embedded RGB image drawn into the rectangle ``(x, y, x+w, y+h)``.

    Attributes:
        x: Left edge on the page.
        y: Bottom edge on the page.
        w: Placed width in points.
        h: Placed height in points.
        pixels: Image dimensions ``(width, height)`` in pixels.
        rgb: Solid fill colour of the image.
    """

    x: float
    y: float
    w: float
    h: float
    pixels: tuple[int, int] = (2, 2)
    rgb: tuple[int, int, int] = (255, 0, 0)


@dataclass
class PageSpec:
    """Content of one page.

    Attributes:
        runs: Text runs to draw.
        images: Images to place.
        width: Page width in points.
        height: Page height in points.
    """

    runs: list[TextRun] = field(default_factory=list)
    images: list[ImageBox] = field(default_factory=list)
    width: float = 612.0
    height: float = 792.0


def _escape(text: str) -> bytes:
    """Escape a string for a PDF literal string."""
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)").encode("latin-1")


def build_pdf(pages: list[PageSpec]) -> bytes:
    """Serialise ``pages`` into a valid single-revision PDF 1.4 file.

    Args:
        pages: Page specifications, in order.

    Returns:
        bytes: The PDF file contents.
    """
    objects: list[bytes] = []

    def add(obj: bytes) -> int:
        """Append an object body and return its 1-based object number."""
        objects.append(obj)
        return len(objects)

    # Reserve object numbers 1 (catalog) and 2 (pages tree); fill later.
    add(b"")
    add(b"")
    font_regular = add(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    font_bold = add(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>")

    page_ids: list[int] = []
    for spec in pages:
        content = io.BytesIO()
        for run in spec.runs:
            font = "/F2" if run.bold else "/F1"
            # Tm sets the text matrix: identity for upright text, a 90-degree
            # rotation for vertical text (as in an arXiv-style side stamp).
            matrix = b"0 1 -1 0" if run.rotate90 else b"1 0 0 1"
            content.write(
                b"BT %s %g Tf %s %g %g Tm (%s) Tj ET\n"
                % (font.encode(), run.size, matrix, run.x, run.y, _escape(run.text))
            )
        xobjects: list[tuple[str, int]] = []
        for idx, img in enumerate(spec.images):
            pw, ph = img.pixels
            data = bytes(img.rgb) * (pw * ph)
            obj_id = add(
                b"<< /Type /XObject /Subtype /Image /Width %d /Height %d /ColorSpace /DeviceRGB "
                b"/BitsPerComponent 8 /Length %d >>\nstream\n" % (pw, ph, len(data)) + data + b"\nendstream"
            )
            name = f"/Im{idx}"
            xobjects.append((name, obj_id))
            content.write(b"q %g 0 0 %g %g %g cm %s Do Q\n" % (img.w, img.h, img.x, img.y, name.encode()))
        stream = content.getvalue()
        content_id = add(b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream")

        xobj_dict = b""
        if xobjects:
            xobj_dict = b" /XObject << " + b" ".join(b"%s %d 0 R" % (n.encode(), i) for n, i in xobjects) + b" >>"
        page_id = add(
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 %g %g] "
            b"/Resources << /Font << /F1 %d 0 R /F2 %d 0 R >>%s >> /Contents %d 0 R >>"
            % (spec.width, spec.height, font_regular, font_bold, xobj_dict, content_id)
        )
        page_ids.append(page_id)

    objects[0] = b"<< /Type /Catalog /Pages 2 0 R >>"
    kids = b" ".join(b"%d 0 R" % pid for pid in page_ids)
    objects[1] = b"<< /Type /Pages /Kids [%s] /Count %d >>" % (kids, len(page_ids))

    out = io.BytesIO()
    out.write(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets: list[int] = []
    for number, body in enumerate(objects, start=1):
        offsets.append(out.tell())
        out.write(b"%d 0 obj\n" % number + body + b"\nendobj\n")
    xref = out.tell()
    out.write(b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1))
    for off in offsets:
        out.write(b"%010d 00000 n \n" % off)
    out.write(b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objects) + 1, xref))
    return out.getvalue()


def two_column_page() -> PageSpec:
    """A page whose text is laid out in two columns (left then right)."""
    runs = []
    for i in range(3):
        runs.append(TextRun(f"Left column line {i + 1}", x=60, y=700 - 14 * i))
    for i in range(3):
        runs.append(TextRun(f"Right column line {i + 1}", x=330, y=700 - 14 * i))
    return PageSpec(runs=runs)


def report_pages(count: int = 3, *, running_head: str = "Quarterly Review 2031") -> list[PageSpec]:
    """A multi-page document with a running head, a footer and page numbers.

    Each page carries the same head and footer text plus its own page number,
    with a few lines of body prose in between.

    Args:
        count: Number of pages.
        running_head: Text repeated at the top of every page.

    Returns:
        list[PageSpec]: One spec per page.
    """
    pages: list[PageSpec] = []
    for page_no in range(1, count + 1):
        runs = [
            TextRun(running_head, x=60, y=760, size=9),
            TextRun(f"Body line one of page {page_no} with enough words to read as prose.", x=60, y=700),
            TextRun(f"Body line two of page {page_no} continuing the same paragraph here.", x=60, y=686),
            TextRun(f"Body line three of page {page_no} rounding the paragraph off nicely.", x=60, y=672),
            TextRun("Confidential draft", x=60, y=40, size=8),
            TextRun(str(page_no), x=300, y=26, size=9),
        ]
        pages.append(PageSpec(runs=runs))
    return pages


TABLE_ROWS: tuple[tuple[str, ...], ...] = (
    ("Model", "Accuracy", "F1"),
    ("Alpha", "89.3", "88.1"),
    ("Beta", "91.0", "90.5"),
    ("Gamma", "87.2", "86.4"),
)


def table_page(*, caption: str | None = "Table 1: Results summary") -> PageSpec:
    """A page holding a column-aligned table, optionally captioned.

    Args:
        caption: Caption line above the table, or ``None`` for a bare grid.

    Returns:
        PageSpec: The page specification.
    """
    runs: list[TextRun] = []
    top = 700.0
    if caption is not None:
        runs.append(TextRun(caption, x=60, y=top))
        top -= 20
    for row, values in enumerate(TABLE_ROWS):
        for col, text in enumerate(values):
            runs.append(TextRun(text, x=60 + col * 120, y=top - row * 14))
    runs.append(TextRun("Following prose paragraph that is clearly not part of the table.", x=60, y=560))
    return PageSpec(runs=runs)


def wrapped_header_table_page(*, caption: str | None = "Table 1: Complexity by layer type") -> PageSpec:
    """A table whose second header cell wraps onto a second line.

    Mirrors a common academic-table shape: a column heading too long for its
    column, so it occupies two lines while the other columns occupy one.

    Args:
        caption: Caption line above the table, or ``None``.

    Returns:
        PageSpec: The page specification.
    """
    runs: list[TextRun] = []
    top = 700.0
    if caption is not None:
        runs.append(TextRun(caption, x=60, y=top))
        top -= 24
    runs.append(TextRun("Layer Type", x=60, y=top))
    runs.append(TextRun("Complexity", x=240, y=top))
    runs.append(TextRun("Path Length", x=400, y=top))
    runs.append(TextRun("per Layer", x=240, y=top - 12))  # wrapped header cell
    body = [
        ("Self-Attention", "O(n2 d)", "O(1)"),
        ("Recurrent", "O(n d2)", "O(n)"),
        ("Convolutional", "O(k n d2)", "O(1)"),
    ]
    for row, values in enumerate(body):
        y = top - 30 - row * 14
        for col, (text, x) in enumerate(zip(values, (60, 240, 400), strict=True)):
            runs.append(TextRun(text, x=x, y=y))
            del col
    return PageSpec(runs=runs)


def irregular_table_page() -> PageSpec:
    """A captioned table whose structure cannot be recovered as a clean grid.

    Mirrors the academic tables that defeat geometric validation: the caption
    wraps onto a second line of prose, the header has two levels (a group
    heading spanning two sub-columns), one data cell is split into several runs
    the way mathematical notation is, and one row has a missing cell. The rows
    are still rows, so the text must come out row by row.

    Returns:
        PageSpec: The page specification.
    """
    runs = [
        TextRun("Table 2: Scores and cost on both corpora", x=60, y=700),
        TextRun("Values are averages over three runs; lower cost is better.", x=60, y=688),
        # Two-level header: a group heading over two sub-columns.
        TextRun("Score", x=250, y=660),
        TextRun("Cost", x=420, y=660),
        TextRun("Model", x=60, y=646),
        TextRun("EN-DE", x=230, y=646),
        TextRun("EN-FR", x=310, y=646),
        TextRun("EN-DE", x=400, y=646),
        TextRun("EN-FR", x=470, y=646),
        # Data rows: one with a math-style split cell, one missing a value.
        TextRun("Alpha", x=60, y=628),
        TextRun("23.8", x=230, y=628),
        TextRun("39.2", x=310, y=628),
        TextRun("2", x=400, y=628),
        TextRun(". 3", x=410, y=630),
        TextRun("10", x=424, y=628),
        TextRun("19", x=438, y=632),
        TextRun("1.4", x=470, y=628),
        TextRun("Beta", x=60, y=614),
        TextRun("24.6", x=230, y=614),
        TextRun("41.0", x=310, y=614),
        TextRun("9.6", x=470, y=614),
        TextRun("Gamma", x=60, y=600),
        TextRun("26.4", x=230, y=600),
        TextRun("41.8", x=310, y=600),
        TextRun("3.3", x=400, y=600),
        TextRun("9.8", x=470, y=600),
        # Prose well below the table.
        TextRun("The following paragraph is ordinary body text, far below the table.", x=60, y=520),
    ]
    return PageSpec(runs=runs)


def math_caption_table_page() -> PageSpec:
    """A captioned table whose caption wraps into a line broken up by inline maths.

    Mirrors the academic pattern where the caption's second line reads
    "for different layer types. n is the sequence length, d is ..." — the
    italic symbols split it into many short runs, so it does not look like
    prose cell by cell, yet it is prose and must not join the table.

    Returns:
        PageSpec: The page specification.
    """
    runs = [
        TextRun("Table 1: Maximum path lengths and per-layer complexity", x=60, y=700),
        # Caption continuation, chopped up by inline maths symbols.
        TextRun("for different layer types.", x=60, y=688),
        TextRun("n", x=210, y=688),
        TextRun("is the sequence length and", x=230, y=688),
        TextRun("d", x=385, y=688),
        TextRun("the dimension.", x=405, y=688),
        # The table itself.
        TextRun("Layer Type", x=80, y=660),
        TextRun("Complexity", x=250, y=660),
        TextRun("Path Length", x=400, y=660),
        TextRun("Self-Attention", x=80, y=646),
        TextRun("O(n2 d)", x=250, y=646),
        TextRun("O(1)", x=400, y=646),
        TextRun("Recurrent", x=80, y=632),
        TextRun("O(n d2)", x=250, y=632),
        TextRun("O(n)", x=400, y=632),
        TextRun("Convolutional", x=80, y=618),
        TextRun("O(k n d2)", x=250, y=618),
        TextRun("O(1)", x=400, y=618),
    ]
    return PageSpec(runs=runs)
