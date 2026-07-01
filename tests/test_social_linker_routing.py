"""Routing tests for SocialLinker: image CLIP path, video Nextext path, manifest caching."""

from pathlib import Path
from typing import Any

import pandas as pd

from docint.core.ingest.images_service import IngestContext
from docint.core.ingest.social_linker import SocialLinker
from docint.utils.nextext_client import NextextResult


class _FakeImageService:
    """In-memory image-service stub that records calls without touching Qdrant."""

    def __init__(self) -> None:
        """Initialise with empty tracking lists."""
        self.images: list[Any] = []
        self.keyframe_calls: list[dict[str, Any]] = []

    def ingest_image(self, asset: Any, *, context: IngestContext) -> Any:
        """Record the asset and return None.

        Args:
            asset: The image asset to record.
            context: Ingestion context (ignored).

        Returns:
            None.
        """
        self.images.append(asset)
        return None

    def ingest_keyframe_set(
        self,
        frames: list[bytes],
        *,
        context: IngestContext,
        source_doc_id: str | None,
        extra_metadata: dict[str, Any] | None = None,
        dedup_cosine: float = 0.95,
    ) -> list[Any]:
        """Record the keyframe call and return an empty list.

        Args:
            frames: Keyframe bytes (recorded but not stored).
            context: Ingestion context (ignored).
            source_doc_id: Posting UUID stamped on each point.
            extra_metadata: Optional extra payload fields.
            dedup_cosine: Cosine similarity threshold (ignored).

        Returns:
            An empty list (no records stored in the stub).
        """
        self.keyframe_calls.append({"frames": frames, "source_doc_id": source_doc_id, "dedup_cosine": dedup_cosine})
        return []


class _FakeNextext:
    """Nextext stub that returns a fixed transcript + one keyframe."""

    def process_media(self, file_path: Path) -> NextextResult:
        """Return a fixed completed result regardless of the input file.

        Args:
            file_path: Path to the media file (ignored).

        Returns:
            A completed NextextResult with one transcript segment and one keyframe.
        """
        return NextextResult(
            status="completed",
            transcript_jsonl=b'{"text":"spoken","start_seconds":0,"end_seconds":1}\n',
            keyframes=[b"\xff\xd8\xff0"],
        )


def _write_export(root: Path) -> None:
    """Write a minimal social export tree under *root* for testing.

    The fixture includes a ``comments.csv`` that contains both ``UUID`` and
    ``Posting ID`` columns to guard against subset-collision with the postings
    profile detection — it must NOT be misdetected as the postings table.

    Args:
        root: Temporary directory in which to create the export.
    """
    (root / "tables").mkdir(parents=True)
    (root / "media").mkdir()
    # Full 25-column postings profile — exact-match required by _find_tables.
    postings_cols = [
        "UUID",
        "Posting ID",
        "URL",
        "Date last updated",
        "Timestamp",
        "Timezone",
        "Crawled at",
        "Postings Connections",
        "Network Posting ID",
        "Location",
        "Author ID",
        "Author",
        "Vanity Name",
        "Co-Author",
        "Quoted User",
        "Expected Reactions",
        "Collected Reactions",
        "Expected Comments",
        "Collected Comments",
        "Network",
        "Posted in Group",
        "Task",
        "Text Content",
        "Filename",
        "Tags",
    ]
    postings_data = {col: ["", ""] for col in postings_cols}
    postings_data["UUID"] = ["u1", "u2"]
    postings_data["Posting ID"] = ["P_1", "P_2"]
    postings_data["Text Content"] = ["a", "b"]
    pd.DataFrame(postings_data).to_csv(root / "tables" / "postings.csv", index=False)
    pd.DataFrame({"Media ID": ["P_1_0", "P_2_0"], "Exported media filename": ["pic.jpg", "clip.mp4"]}).to_csv(
        root / "tables" / "media.csv", index=False
    )
    # comments.csv contains UUID + Posting ID but is NOT the full postings header set;
    # it must NOT be misdetected as the postings table (guards subset-collision regression).
    pd.DataFrame({"UUID": ["c1"], "Posting ID": ["P_1"], "Text Content": ["comment text"]}).to_csv(
        root / "tables" / "comments.csv", index=False
    )
    (root / "media" / "pic.jpg").write_bytes(b"\xff\xd8\xff")
    (root / "media" / "clip.mp4").write_bytes(b"video")


def test_run_routes_image_and_video_and_links(tmp_path: Path) -> None:
    """Image goes to CLIP path; video goes to Nextext; both are linked to their posting UUID."""
    _write_export(tmp_path)
    img = _FakeImageService()
    linker = SocialLinker(image_service=img, nextext_client=_FakeNextext(), target_collection="c")
    result = linker.run(tmp_path)

    # The image went through the CLIP path with the posting UUID.
    assert len(img.images) == 1
    assert img.images[0].source_doc_id == "u1"
    # The video produced keyframes (linked to u2) and a transcript Document.
    assert img.keyframe_calls and img.keyframe_calls[0]["source_doc_id"] == "u2"
    assert len(result.transcript_documents) == 1
    assert result.transcript_documents[0].metadata["posting_uuid"] == "u2"
    # media.csv + both media files are consumed (excluded from the generic sweep).
    consumed_names = {p.name for p in result.consumed_paths}
    assert {"media.csv", "pic.jpg", "clip.mp4"}.issubset(consumed_names)
    # postings.csv is NOT consumed (the sweep ingests it as text nodes).
    assert "postings.csv" not in consumed_names


class _CountingNextext:
    """Nextext stub that counts how many times it is called."""

    def __init__(self) -> None:
        """Initialise the call counter to zero."""
        self.calls = 0

    def process_media(self, file_path: Path) -> NextextResult:
        """Increment the call counter and return a completed result.

        Args:
            file_path: Path to the media file (ignored).

        Returns:
            A completed NextextResult with one transcript segment and no keyframes.
        """
        self.calls += 1
        return NextextResult(
            status="completed",
            transcript_jsonl=b'{"text":"x","start_seconds":0,"end_seconds":1}\n',
            keyframes=[],
        )


class _FakeManifest:
    """In-memory manifest stub with an optional pre-seeded cache entry."""

    def __init__(self, cached: str | None = None) -> None:
        """Initialise with an optional cached transcript string.

        Args:
            cached: Pre-seeded transcript JSONL string, or None for a cold cache.
        """
        self._cached = cached
        self.saved: list[tuple[str, str, str]] = []
        self.lookup_calls: int = 0

    def get_nextext_transcript(self, collection: str, file_hash: str) -> str | None:
        """Return the pre-seeded cached transcript (ignores collection/hash).

        Args:
            collection: Collection name (ignored in stub).
            file_hash: Media file hash (ignored in stub).

        Returns:
            The pre-seeded transcript string, or None.
        """
        self.lookup_calls += 1
        return self._cached

    def cache_nextext_transcript(self, collection: str, file_hash: str, jsonl: str) -> None:
        """Record a cache-write call.

        Args:
            collection: Collection name.
            file_hash: Media file hash.
            jsonl: Transcript JSONL string to persist.
        """
        self.saved.append((collection, file_hash, jsonl))


def test_cached_transcript_skips_nextext(tmp_path: Path) -> None:
    """A manifest cache hit must prevent the Nextext job from being submitted."""
    _write_export(tmp_path)
    nx = _CountingNextext()
    manifest = _FakeManifest(cached='{"text":"cached","start_seconds":0,"end_seconds":1}\n')
    result = SocialLinker(
        image_service=_FakeImageService(), nextext_client=nx, target_collection="c", manifest=manifest
    ).run(tmp_path)
    assert nx.calls == 0  # cache hit -> Nextext job not submitted
    assert manifest.lookup_calls >= 1  # manifest was consulted for the cache lookup
    assert any(d.metadata.get("posting_uuid") == "u2" for d in result.transcript_documents)


def test_cache_miss_persists_transcript(tmp_path: Path) -> None:
    """A manifest cache miss must call Nextext once and persist the result."""
    _write_export(tmp_path)
    nx = _CountingNextext()
    manifest = _FakeManifest(cached=None)
    SocialLinker(image_service=_FakeImageService(), nextext_client=nx, target_collection="c", manifest=manifest).run(
        tmp_path
    )
    assert nx.calls == 1
    assert manifest.saved and manifest.saved[0][0] == "c"


def test_configured_keyframe_dedup_cosine_reaches_image_service(tmp_path: Path) -> None:
    """The linker's configured ``keyframe_dedup_cosine`` must be forwarded to ``ingest_keyframe_set``.

    Regression guard for the cosine threshold being silently dropped on the
    way to the image service (it previously always fell back to that
    method's hardcoded default, so ``KEYFRAME_DEDUP_COSINE`` had no effect).
    """
    _write_export(tmp_path)
    img = _FakeImageService()
    linker = SocialLinker(
        image_service=img,
        nextext_client=_FakeNextext(),
        target_collection="c",
        keyframe_dedup_cosine=0.5,
    )
    linker.run(tmp_path)

    assert img.keyframe_calls
    assert img.keyframe_calls[0]["dedup_cosine"] == 0.5
