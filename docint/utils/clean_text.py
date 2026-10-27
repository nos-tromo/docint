"""Lightweight text normalization helpers shared across readers and retrieval."""

import re


def basic_clean(text: str) -> str:
    r"""Normalize whitespace and line breaks, keeping paragraph gaps.

    Line endings become ``\n``, trailing whitespace is stripped from every
    line, runs of three or more newlines collapse to one blank line, and runs
    of spaces/tabs collapse to a single space. A paragraph gap therefore
    survives as exactly ``\n\n``, the separator the node parsers split
    paragraphs on.

    Args:
        text (str): The text to clean.

    Returns:
        str: The cleaned text.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")  # normalize newlines
    text = re.sub(r"[^\S\n]+\n", "\n", text)  # strip trailing whitespace, not the newline itself
    text = re.sub(r"\n{3,}", "\n\n", text)  # collapse multiple newlines
    text = re.sub(r"[ \t]{2,}", " ", text)  # collapse multiple spaces/tabs
    return text.strip()
