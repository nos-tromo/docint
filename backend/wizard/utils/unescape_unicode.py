import codecs
import html
import json
import unicodedata
from typing import Any


def unescape_unicode(s: Any) -> str:
    """
    Unescape a string that may contain Unicode escape sequences, HTML entities,
    or be doubly-escaped as JSON.

    Args:
        s (Any): The input string to unescape. It can be a string, bytes, or None.

    Returns:
        str: The unescaped string. If the input is None, returns an empty string.
        If the input is a bytes object, it will be decoded to a string using UTF-8.
        If the input is a string that appears to be doubly-escaped JSON, it will
        be parsed and unescaped accordingly. If the input is not a valid escape sequence,
        it will be returned as-is.
        If the input contains HTML entities, they will be decoded.
        The final string will be normalized to NFC form.
    """
    if s is None:
        return ""
    if isinstance(s, bytes):
        s = s.decode("utf-8", "replace")
    s = str(s)

    # 1) Try JSON round-trip if it looks doubly-escaped JSON content
    # (e.g., a value that was dumped as a JSON string literal)
    try:
        # json.loads will interpret \uXXXX and standard escapes
        if s.startswith('"') and s.endswith('"'):
            s = json.loads(s)
    except Exception:
        pass

    # 2) Interpret backslash escapes like \u00e4, \n, etc.
    try:
        s = codecs.decode(s, "unicode_escape")
    except Exception:
        # If it wasn't really escaped, leave it as-is
        pass

    # 3) Decode any HTML entities (&auml; &amp; etc.)
    s = html.unescape(s)

    # 4) Normalize to NFC (helps with accent composition)
    s = unicodedata.normalize("NFC", s)
    return s
