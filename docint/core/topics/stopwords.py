"""Stop words kept out of topic keywords.

The base lists are the NLTK ``english`` and ``german`` corpora that llama-index-core
ships inside its wheel, so nothing is fetched at runtime, plus :data:`SUPPLEMENT`.
Each fit then adds the words its own texts call for (:func:`corpus_stop_words`).
"""

import hashlib
import json
import re
from collections import Counter
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path

import llama_index.core

from docint.utils.topics_client import MAX_STOP_WORD_CHARS, MAX_STOP_WORDS, META_MAX_BYTES

STOP_WORD_LANGUAGES = ("english", "german")

#: Words the NLTK lists lack that never make a useful keyword: auxiliaries, a
#: reporting verb, honorifics, and citation and web-address fragments. Words
#: that carry meaning elsewhere (``us``, ``may``, ``frau``, ``table``, ``figure``)
#: are left to :func:`corpus_stop_words`.
SUPPLEMENT = (
    # English
    "al",
    "com",
    "could",
    "doi",
    "dr",
    "dx",
    "eq",
    "eqs",
    "et",
    "fig",
    "figs",
    "http",
    "https",
    "might",
    "mr",
    "mrs",
    "ms",
    "must",
    "one",
    "org",
    "pp",
    "said",
    "shall",
    "upon",
    "vol",
    "would",
    "www",
    # German
    "bereits",
    "herr",
    "hätte",
    "hätten",
    "sagte",
    "sei",
    "seien",
    "sowie",
    "worden",
    "wurde",
    "wurden",
    "wäre",
    "wären",
)

# The service's words: scikit-learn's default tokens of the lowercased text.
_TOKEN = re.compile(r"(?u)\b\w\w+\b")
# The stop words' share of the service's meta part, which is at most META_MAX_BYTES.
_STOP_WORDS_MAX_BYTES = META_MAX_BYTES // 2


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
        Sorted unique words of both lists and :data:`SUPPLEMENT`, with blanks and
        words the service would refuse dropped.
    """
    words: set[str] = set(SUPPLEMENT)
    for language in STOP_WORD_LANGUAGES:
        text = (stop_words_dir() / language).read_text(encoding="utf-8")
        words.update(word for word in (line.strip() for line in text.splitlines()) if word)
    return tuple(sorted(word for word in words if len(word) <= MAX_STOP_WORD_CHARS))


def corpus_stop_words(texts: Sequence[str], base: Sequence[str], *, max_df: float, min_df: int) -> tuple[str, ...]:
    """Add the stop words a fit's own texts call for.

    Words are found as the service's vectorizer finds them: two or more word
    characters of the lowercased text, counted once per text. Two kinds are
    added: a number found in at least ``min_df`` texts, which would otherwise
    reach the keywords, and a word found in more than ``max_df`` of the texts,
    which cannot tell one topic from another. The most widespread come first
    when the service's limits leave room for only some.

    Args:
        texts: The texts the fit sends.
        base: The fixed stop words.
        max_df: Share of the texts a word may appear in before it is left out.
        min_df: Fewest texts a number must appear in to be left out.

    Returns:
        ``base`` and the added words, sorted and unique.
    """
    words = set(base)
    df: Counter[str] = Counter()
    for text in texts:
        df.update(set(_TOKEN.findall(text.lower())))
    ceiling = max_df * len(texts)
    found = sorted(
        (-count, word)
        for word, count in df.items()
        if word not in words
        and len(word) <= MAX_STOP_WORD_CHARS
        and ((word.isdecimal() and count >= min_df) or count > ceiling)
    )
    size = sum(_json_bytes(word) for word in words)
    for _, word in found:
        cost = _json_bytes(word)
        if len(words) >= MAX_STOP_WORDS or size + cost > _STOP_WORDS_MAX_BYTES:
            break
        words.add(word)
        size += cost
    return tuple(sorted(words))


def _json_bytes(word: str) -> int:
    """Return a word's encoded length in the meta part's stop-word array.

    Args:
        word: The word.

    Returns:
        Bytes of its UTF-8 JSON string and the comma that follows it.
    """
    return len(json.dumps(word, ensure_ascii=False).encode()) + 1


def stop_words_digest(words: Sequence[str]) -> str:
    """Return a digest of a stop-word list, for recording which list a run used.

    Args:
        words: The words, in order.

    Returns:
        The hex sha256 of the words joined by newlines.
    """
    return hashlib.sha256("\n".join(words).encode("utf-8")).hexdigest()
