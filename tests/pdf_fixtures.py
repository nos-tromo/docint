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
    """

    text: str
    x: float
    y: float
    size: float = 11.0
    bold: bool = False


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
            content.write(
                b"BT %s %g Tf %g %g Td (%s) Tj ET\n" % (font.encode(), run.size, run.x, run.y, _escape(run.text))
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
