"""Tests for the topic keyword stop words."""

from docint.core.topics.stopwords import STOP_WORD_LANGUAGES, load_stop_words, stop_words_digest, stop_words_dir
from docint.utils.topics_client import MAX_STOP_WORD_CHARS, MAX_STOP_WORDS


def test_the_bundled_lists_exist() -> None:
    """Both bundled lists are present in the installed wheel.

    Returns:
        None.
    """
    for language in STOP_WORD_LANGUAGES:
        assert (stop_words_dir() / language).is_file()


def test_load_stop_words_is_sorted_unique_and_within_the_service_bounds() -> None:
    """The merged list fits the service's limits and holds both languages.

    Returns:
        None.
    """
    words = load_stop_words()
    assert list(words) == sorted(set(words))
    assert len(words) <= MAX_STOP_WORDS
    assert all(0 < len(word) <= MAX_STOP_WORD_CHARS for word in words)
    assert "the" in words
    assert "und" in words


def test_the_digest_is_stable_and_order_sensitive() -> None:
    """The digest repeats for equal input and changes with order.

    Returns:
        None.
    """
    assert stop_words_digest(["a", "b"]) == stop_words_digest(["a", "b"])
    assert stop_words_digest(["a", "b"]) != stop_words_digest(["b", "a"])
    assert len(stop_words_digest(["a"])) == 64
