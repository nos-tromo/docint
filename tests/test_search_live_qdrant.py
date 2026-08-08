"""Integration tests pinning the Qdrant behaviours full-text search relies on.

Skipped unless a Qdrant is reachable at ``QDRANT_URL`` (default
``http://localhost:6333``). These assertions are about the server, not our
code: the design depends on prefix matching, non-ASCII case folding, mid-word
non-matching, and AND across keywords all behaving exactly as pinned here.

The non-ASCII case-folding assertion earns its keep on its own — that
assumption was *false* at design time (un-indexed ``MatchText`` only case-folds
ASCII), and only a real server revealed it. A mocked client would have happily
confirmed the bug.
"""

from __future__ import annotations

import os
import time
import uuid
from collections.abc import Iterator

import pytest
from qdrant_client import QdrantClient, models

from docint.core.search.fulltext import build_search_filter
from docint.core.search.index import SEARCH_TEXT_FIELD, search_index_params, write_search_text

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")


def _reachable() -> bool:
    """Return whether a Qdrant server answers at ``QDRANT_URL``."""
    try:
        QdrantClient(url=QDRANT_URL, timeout=2).get_collections()
    except Exception:
        return False
    return True


pytestmark = pytest.mark.skipif(not _reachable(), reason="no Qdrant reachable")


@pytest.fixture
def collection() -> Iterator[tuple[QdrantClient, str]]:
    """Create a throwaway collection with the search index, then drop it."""
    client = QdrantClient(url=QDRANT_URL, timeout=30)
    name = f"zz_test_search_{uuid.uuid4().hex[:8]}"
    client.create_collection(
        name,
        vectors_config=models.VectorParams(size=2, distance=models.Distance.COSINE),
    )
    try:
        client.create_payload_index(
            collection_name=name,
            field_name=SEARCH_TEXT_FIELD,
            field_schema=search_index_params(),
        )
        yield client, name
    finally:
        client.delete_collection(name)


def _seed(
    client: QdrantClient,
    name: str,
    rows: dict[int, str],
    hier: dict[int, str] | None = None,
) -> None:
    """Upsert synthetic points and write their search text.

    Args:
        client (QdrantClient): Live client.
        name (str): Collection name.
        rows (dict[int, str]): ``{point_id: text}``.
        hier (dict[int, str] | None): Optional ``{point_id: docint_hier_type}``.
    """
    hier = hier or {}
    client.upsert(
        name,
        points=[
            models.PointStruct(
                id=pid,
                vector=[0.1, 0.2],
                payload={"docint_hier_type": hier[pid]} if pid in hier else {},
            )
            for pid in rows
        ],
        wait=True,
    )
    write_search_text(client, name, dict(rows), wait=True)
    time.sleep(1.0)  # let the payload index catch up


def _ids(client: QdrantClient, name: str, keywords: list[str]) -> list[int]:
    """Return the ids matching a keyword search, sorted.

    Args:
        client (QdrantClient): Live client.
        name (str): Collection name.
        keywords (list[str]): Keywords that must all match.

    Returns:
        list[int]: Matching point ids.
    """
    flt = build_search_filter(keywords)
    return sorted(int(point.id) for point in client.scroll(name, scroll_filter=flt, limit=50)[0])


def test_prefix_matching_finds_a_compound_from_its_head(collection: tuple[QdrantClient, str]) -> None:
    """German compounds are head-final, so the head is what gets typed."""
    client, name = collection
    _seed(client, name, {1: "Der Parteitag begann", 2: "Die Partei erklärte", 3: "Etwas anderes"})

    assert _ids(client, name, ["Partei"]) == [1, 2]


def test_matching_is_case_insensitive_for_non_ascii_text(collection: tuple[QdrantClient, str]) -> None:
    """The lowercase index is what makes this true — un-indexed MatchText is not.

    This is the assumption that was wrong at design time; it is the single
    reason the payload index is mandatory rather than an optimisation.
    """
    client, name = collection
    _seed(client, name, {1: "Treffen in Königswinter", 2: "Nichts dergleichen"})

    assert _ids(client, name, ["könig"]) == [1]
    assert _ids(client, name, ["KÖNIG"]) == [1]


def test_mid_word_fragments_do_not_match(collection: tuple[QdrantClient, str]) -> None:
    """Prefix, not substring — a deliberate limit, pinned so it cannot drift."""
    client, name = collection
    _seed(client, name, {1: "Der Parteitag begann"})

    assert _ids(client, name, ["tag"]) == []


def test_multiple_keywords_are_anded_regardless_of_order(collection: tuple[QdrantClient, str]) -> None:
    """Two keywords list only the chunks where both appear."""
    client, name = collection
    _seed(
        client,
        name,
        {1: "Die Konferenz in Berlin", 2: "Eine Konferenz in Hamburg", 3: "Berlin ohne Anlass"},
    )

    assert _ids(client, name, ["Berlin", "Konferenz"]) == [1]
    assert _ids(client, name, ["Konferenz", "Berlin"]) == [1]
    assert _ids(client, name, ["Berlin", "Warschau"]) == []


def test_coarse_parent_chunks_are_excluded_but_untagged_ones_are_not(
    collection: tuple[QdrantClient, str],
) -> None:
    """Excluding coarse must not also exclude non-hierarchical collections.

    Requiring ``docint_hier_type == "fine"`` would return nothing at all for a
    collection ingested without hierarchical chunking, which tags no point.
    """
    client, name = collection
    _seed(
        client,
        name,
        {1: "Parteitag Bericht", 2: "Parteitag Bericht", 3: "Parteitag Bericht"},
        hier={1: "fine", 2: "coarse"},
    )

    assert _ids(client, name, ["Parteitag"]) == [1, 3]
