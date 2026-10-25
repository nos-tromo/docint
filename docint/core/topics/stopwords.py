"""Stop words kept out of topic keywords.

The lists are the NLTK ``english`` and ``german`` corpora that llama-index-core
ships inside its wheel, so nothing is fetched at runtime.
"""

import hashlib
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path

import llama_index.core

from docint.utils.topics_client import MAX_STOP_WORD_CHARS

STOP_WORD_LANGUAGES = ("english", "german")


def stop_words_dir() -> Path:
    """Return the directory holding the bundled stop-word lists.

    Returns:
        The ``stopwords`` corpus directory inside llama-index-core.
    """
    return Path(llama_index.core.__file__).parent / "_static/nltk_cache/corpora/stopwords"


@lru_cache(maxsize=1)
def load_stop_words() -> tuple[str, ...]:
    """Load the merged stop words.

    Returns:
        Sorted unique words, with blanks and words the service would refuse dropped.
    """
    words: set[str] = set()
    for language in STOP_WORD_LANGUAGES:
        text = (stop_words_dir() / language).read_text(encoding="utf-8")
        words.update(word for word in (line.strip() for line in text.splitlines()) if word)
    return tuple(sorted(word for word in words if len(word) <= MAX_STOP_WORD_CHARS))


def stop_words_digest(words: Sequence[str]) -> str:
    """Return a digest of a stop-word list, for recording which list a run used.

    Args:
        words: The words, in order.

    Returns:
        The hex sha256 of the words joined by newlines.
    """
    return hashlib.sha256("\n".join(words).encode("utf-8")).hexdigest()
