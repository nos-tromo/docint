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
        keyframe_source_type: str = "social_media_keyframe",
        link_field: str | None = "posting_uuid",
    ) -> list[Any]:
        """Record the keyframe call and return an empty list.

        Mirrors the real ``ImageIngestionService.ingest_keyframe_set`` signature
        (``keyframe_source_type``/``link_field`` added for the standalone media
        path) so this stub keeps accepting whatever the production call site
        passes, including when it now passes those two explicitly at their
        historical default values.

        Args:
            frames: Keyframe bytes (recorded but not stored).
            context: Ingestion context (ignored).
            source_doc_id: Posting UUID stamped on each point.
            extra_metadata: Optional extra payload fields.
            dedup_cosine: Cosine similarity threshold (ignored).
            keyframe_source_type: ``source_type`` payload value (recorded, not applied).
            link_field: Payload key aliasing ``source_doc_id`` (recorded, not applied).

        Returns:
            An empty list (no records stored in the stub).
        """
        self.keyframe_calls.append(
            {
                "frames": frames,
                "source_doc_id": source_doc_id,
                "extra_metadata": extra_metadata,
                "dedup_cosine": dedup_cosine,
                "keyframe_source_type": keyframe_source_type,
                "link_field": link_field,
            }
        )
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

    Under the flat single-directory contract, ``postings.csv``, ``media.csv``,
    and every referenced media file live directly in *root* — there is no
    ``tables/``/``media/`` split.

    The fixture includes a ``comments.csv`` that contains both ``UUID`` and
    ``Posting ID`` columns to guard against subset-collision with the postings
    profile detection — it must NOT be misdetected as the postings table.

    Args:
        root: Temporary directory in which to create the export.
    """
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
    postings_data["Network"] = ["Facebook", "Facebook"]
    postings_data["Author"] = ["Jane Poster", "Jane Poster"]
    postings_data["URL"] = ["https://fb.example/p1", "https://fb.example/p2"]
    postings_data["Timestamp"] = ["2023-01-01 10:00", "2023-02-02 11:00"]
    pd.DataFrame(postings_data).to_csv(root / "postings.csv", index=False)
    pd.DataFrame({"Media ID": ["P_1_0", "P_2_0"], "Exported media filename": ["pic.jpg", "clip.mp4"]}).to_csv(
        root / "media.csv", index=False
    )
    # comments.csv contains UUID + Posting ID but is NOT the full postings header set;
    # it must NOT be misdetected as the postings table (guards subset-collision regression).
    pd.DataFrame({"UUID": ["c1"], "Posting ID": ["P_1"], "Text Content": ["comment text"]}).to_csv(
        root / "comments.csv", index=False
    )
    (root / "pic.jpg").write_bytes(b"\xff\xd8\xff")
    (root / "clip.mp4").write_bytes(b"video")


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


def test_run_stamps_posting_reference_metadata(tmp_path: Path) -> None:
    """Derived media artifacts carry the parent posting's reference fields, additively.

    The image asset and the keyframe call must carry the ``posting_*`` fields
    plus a ready-made nested ``reference_metadata`` block; the transcript
    segment must merge them into its own ``reference_metadata`` WITHOUT
    dropping the Nextext identity (``network: nextext`` /
    ``type: transcript_segment``).
    """
    _write_export(tmp_path)
    img = _FakeImageService()
    linker = SocialLinker(image_service=img, nextext_client=_FakeNextext(), target_collection="c")
    result = linker.run(tmp_path)

    image_extra = img.images[0].extra_metadata
    assert image_extra["posting_network"] == "Facebook"
    assert image_extra["posting_author"] == "Jane Poster"
    assert image_extra["posting_url"] == "https://fb.example/p1"
    assert image_extra["posting_timestamp"] == "2023-01-01 10:00"
    assert image_extra["posting_text"] == "a"
    assert image_extra["reference_metadata"]["type"] == "image"
    assert image_extra["reference_metadata"]["posting_uuid"] == "u1"
    assert image_extra["reference_metadata"]["posting_network"] == "Facebook"

    keyframe_extra = img.keyframe_calls[0]["extra_metadata"]
    assert keyframe_extra["posting_network"] == "Facebook"
    assert keyframe_extra["posting_url"] == "https://fb.example/p2"
    assert keyframe_extra["reference_metadata"]["type"] == "keyframe"
    assert keyframe_extra["reference_metadata"]["posting_uuid"] == "u2"

    segment_ref = result.transcript_documents[0].metadata["reference_metadata"]
    # Nextext identity preserved (additive merge, nothing dropped).
    assert segment_ref["network"] == "nextext"
    assert segment_ref["type"] == "transcript_segment"
    assert segment_ref["posting_uuid"] == "u2"
    assert segment_ref["posting_network"] == "Facebook"
    assert segment_ref["posting_author"] == "Jane Poster"
    assert segment_ref["posting_url"] == "https://fb.example/p2"
    assert segment_ref["posting_text"] == "b"


def test_build_posting_reference_index_requires_postings_profile() -> None:
    """Header drift away from the postings profile degrades to link-ids-only."""
    from docint.core.ingest.social_linker import build_posting_reference_index

    df = pd.DataFrame({"UUID": ["u1"], "Posting ID": ["P_1"], "Something Else": ["x"]})
    assert build_posting_reference_index(df) == {}


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


_SEMICOLON_POSTINGS_COLUMNS = [
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


def _write_semicolon_postings(root: Path, media_rows: dict[str, str]) -> None:
    """Write a semicolon-delimited, BOM-prefixed postings + media manifest pair.

    Mirrors :func:`_write_export`'s full 25-column postings profile (postings
    ``u1``/``P_1`` and ``u2``/``P_2``) but serializes both tables with ``;``
    as the delimiter and a UTF-8 BOM, matching real social-platform exports,
    so tests can exercise delimiter sniffing end to end. Each test supplies
    its own media manifest rows. Both CSVs are written directly in *root*,
    matching the flat single-directory contract; callers are responsible for
    placing any referenced media files directly in *root* as well.

    Args:
        root: Temporary directory in which to create the export.
        media_rows: Mapping of ``Media ID`` to ``Exported media filename``
            for the media manifest.
    """
    postings_data = {col: ["", ""] for col in _SEMICOLON_POSTINGS_COLUMNS}
    postings_data["UUID"] = ["u1", "u2"]
    postings_data["Posting ID"] = ["P_1", "P_2"]
    postings_data["Text Content"] = ["a", "b"]
    pd.DataFrame(postings_data).to_csv(root / "postings.csv", index=False, sep=";", encoding="utf-8-sig")
    pd.DataFrame(
        {
            "Media ID": list(media_rows.keys()),
            "Exported media filename": list(media_rows.values()),
        }
    ).to_csv(root / "media.csv", index=False, sep=";", encoding="utf-8-sig")


def test_run_detects_semicolon_delimited_export(tmp_path: Path) -> None:
    """A semicolon-delimited, BOM-prefixed export is still detected and linked.

    Regression guard for the delimiter bug: plain ``pd.read_csv`` defaults to
    a comma separator, so a ``;``-delimited header collapsed into a single
    column and both ``is_media_manifest`` and the postings-profile exact
    match failed, making the linker silently no-op on real social exports
    (which are semicolon-delimited with a UTF-8 BOM).
    """
    _write_semicolon_postings(tmp_path, {"P_1_0": "pic.jpg"})
    (tmp_path / "pic.jpg").write_bytes(b"\xff\xd8\xff")
    img = _FakeImageService()
    result = SocialLinker(image_service=img, nextext_client=_FakeNextext(), target_collection="c").run(tmp_path)

    assert len(img.images) == 1
    assert img.images[0].source_doc_id == "u1"
    consumed_names = {p.name for p in result.consumed_paths}
    assert {"media.csv", "pic.jpg"}.issubset(consumed_names)


def test_run_links_only_present_media(tmp_path: Path) -> None:
    """Only manifest rows whose media file exists in the batch are ingested.

    Mirrors a full manifest that references files never copied into the
    batch (a common real-export shape): the row with no matching file must
    be skipped rather than erroring, while the two present rows still
    resolve and route.
    """
    _write_semicolon_postings(
        tmp_path,
        {"P_1_0": "pic.jpg", "P_2_0": "clip.mp4", "P_1_1": "missing.jpg"},
    )
    (tmp_path / "pic.jpg").write_bytes(b"\xff\xd8\xff")
    (tmp_path / "clip.mp4").write_bytes(b"video")
    img = _FakeImageService()
    result = SocialLinker(image_service=img, nextext_client=_FakeNextext(), target_collection="c").run(tmp_path)

    assert len(img.images) == 1
    assert img.images[0].source_doc_id == "u1"
    assert img.keyframe_calls and img.keyframe_calls[0]["source_doc_id"] == "u2"
    consumed_names = {p.name for p in result.consumed_paths}
    assert {"pic.jpg", "clip.mp4"}.issubset(consumed_names)
    assert "missing.jpg" not in consumed_names


def test_run_skips_absolute_or_traversal_media_reference(tmp_path: Path) -> None:
    """An absolute or ``../`` manifest filename collapses to its basename and is not found.

    Regression guard, updated for the flat single-directory model: resolution
    now only ever looks up ``Path(filename).name`` inside the manifest's own
    directory — there is no path-branch handling and thus nothing that needs
    a containment check. An absolute path and a ``../`` traversal both
    collapse to the same basename (``secret.jpg``); since the real file lives
    outside the batch directory and no ``secret.jpg`` exists directly inside
    it, both rows are skipped rather than ingested.
    """
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir(parents=True)
    outside_file = outside_dir / "secret.jpg"
    outside_file.write_bytes(b"\xff\xd8\xff")

    batch = tmp_path / "batch"
    batch.mkdir()
    postings_data = {col: ["", ""] for col in _SEMICOLON_POSTINGS_COLUMNS}
    postings_data["UUID"] = ["u1", "u2"]
    postings_data["Posting ID"] = ["P_1", "P_2"]
    postings_data["Text Content"] = ["a", "b"]
    pd.DataFrame(postings_data).to_csv(batch / "postings.csv", index=False, sep=";", encoding="utf-8-sig")
    # One row escapes via an absolute path, the other via a "../" traversal;
    # both point at the same real file living outside the batch directory.
    pd.DataFrame(
        {
            "Media ID": ["P_1_0", "P_2_0"],
            "Exported media filename": [str(outside_file.resolve()), "../outside/secret.jpg"],
        }
    ).to_csv(batch / "media.csv", index=False, sep=";", encoding="utf-8-sig")

    img = _FakeImageService()
    result = SocialLinker(image_service=img, nextext_client=_FakeNextext(), target_collection="c").run(batch)

    assert img.images == []
    assert not img.keyframe_calls
    assert not result.transcript_documents
