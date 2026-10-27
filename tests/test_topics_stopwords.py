"""Tests for the topic keyword stop words."""

import json

from docint.core.topics.stopwords import (
    STOP_WORD_LANGUAGES,
    SUPPLEMENT,
    corpus_stop_words,
    load_stop_words,
    stop_words_digest,
    stop_words_dir,
)
from docint.utils.topics_client import MAX_STOP_WORD_CHARS, MAX_STOP_WORDS, META_MAX_BYTES


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


def test_the_supplement_covers_function_words_the_bundled_lists_lack() -> None:
    """Auxiliaries, a reporting verb, honorifics and citation fragments are stop words.

    Returns:
        None.
    """
    words = set(load_stop_words())
    for word in ("would", "could", "said", "mr", "upon", "et", "al", "doi", "wurde", "worden", "sei", "sowie"):
        assert word in words
    assert set(SUPPLEMENT) <= words
    assert len(SUPPLEMENT) == len(set(SUPPLEMENT))


def test_the_supplement_leaves_out_words_that_carry_meaning_elsewhere() -> None:
    """Words that can be content (a country, a month, a person, a document part) are not stop words.

    Returns:
        None.
    """
    words = set(load_stop_words())
    for word in ("us", "may", "frau", "table", "figure"):
        assert word not in words


def test_corpus_stop_words_tokenize_as_the_service_does() -> None:
    """Lowercased runs of two or more word characters, counted once per text.

    Returns:
        None.
    """
    texts = ["The archivist's ledger, 1,000,000 marks.", "İstanbul a_b ledger", "ledger ledger ledger"]
    added = set(corpus_stop_words(texts, (), max_df=0.5, min_df=1))
    # "ledger" is in all three texts (once each, however often it repeats), so over half of them.
    assert "ledger" in added
    # A possessive splits at the apostrophe; "1,000,000" yields the decimal token "000".
    assert "000" in added
    assert "archivist" not in added  # in one text of three
    words = corpus_stop_words(["İstanbul a_b x"] * 3, (), max_df=0.5, min_df=1)
    assert "stanbul" in words  # lowercasing "İ" leaves a combining dot that is not a word character
    assert "a_b" in words  # the underscore is a word character
    assert "x" not in words  # single characters are never tokens


def test_corpus_stop_words_cut_strictly_above_the_share() -> None:
    """A word in exactly ``max_df`` of the texts stays; one in more is added.

    Returns:
        None.
    """
    texts = ["alpha beta", "alpha beta", "alpha gamma", "delta gamma"]
    words = corpus_stop_words(texts, (), max_df=0.5, min_df=2)
    assert "alpha" in words  # 3 of 4 texts
    assert "beta" not in words and "gamma" not in words  # exactly 2 of 4
    assert "delta" not in words


def test_corpus_stop_words_add_numbers_found_in_at_least_min_df_texts() -> None:
    """Decimal tokens reach the stop words once they could reach the vocabulary.

    Returns:
        None.
    """
    texts = ["alpha 2024 10²", "beta 2024 7", "gamma 31", "delta"]
    words = corpus_stop_words(texts, (), max_df=0.9, min_df=2)
    assert "2024" in words
    assert "31" not in words  # one text only, so the vocabulary drops it anyway
    assert "10²" not in words and "10" not in words  # a superscript is not a decimal digit


def test_corpus_stop_words_keep_the_base_and_skip_overlong_words() -> None:
    """The result holds the base, sorted and unique, and never a word the service would refuse.

    Returns:
        None.
    """
    long = "z" * (MAX_STOP_WORD_CHARS + 1)
    texts = [f"{long} shared"] * 4
    words = corpus_stop_words(texts, ("the", "and", "the"), max_df=0.5, min_df=2)
    assert list(words) == sorted(set(words))
    assert {"the", "and", "shared"} <= set(words)
    assert long not in words
    assert corpus_stop_words([], ("the",), max_df=0.5, min_df=2) == ("the",)


def test_corpus_stop_words_stay_within_the_service_limits() -> None:
    """The most widespread words win when the count or byte limit leaves room for only some.

    Returns:
        None.
    """
    base = tuple(f"base{i:05d}" for i in range(MAX_STOP_WORDS - 2))
    texts = [f"common{i}" for i in range(3)] + ["2024 2025 2026"] * 3 + ["2024"]
    words = corpus_stop_words(texts, base, max_df=0.9, min_df=2)
    assert len(words) == MAX_STOP_WORDS
    assert "2024" in words  # in four texts, ahead of the numbers in three
    # 9,000 sixty-digit numbers fit the count limit but not the byte share of the meta part.
    many = [" ".join(f"{i:060d}" for i in range(9_000))] * 2
    words = corpus_stop_words(many, (), max_df=0.9, min_df=2)
    sent = json.dumps(list(words), ensure_ascii=False, separators=(",", ":")).encode()  # as the client encodes it
    assert len(sent) <= META_MAX_BYTES // 2 + 1
    assert 0 < len(words) < 9_000
