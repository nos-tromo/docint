"""Routing tests for SocialLinker: image CLIP path, video Nextext path, manifest caching."""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from docint.core.ingest.images_service import IngestContext
from docint.core.ingest.social_linker import SocialLinker
from docint.utils.nextext_client import NextextKeyframe, NextextResult


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
        frame_times: Sequence[float | None] | None = None,
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
            frame_times: Per-frame sampling times (recorded, not applied).

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
                "frame_times": frame_times,
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
            keyframes=[NextextKeyframe(jpeg=b"\xff\xd8\xff0", index=0, time_sec=2.0)],
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


def test_build_posting_reference_index_requires_a_known_social_profile() -> None:
    """Header drift away from both social profiles degrades to link-ids-only."""
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


def _write_album_export(root: Path) -> None:
    """Write a keyless album export with media nested in subdirectories.

    Mirrors the default multimedia batch output: the two tables sit at the
    root while the files live under ``dir/photos`` and ``dir/videos``. Channel
    ``9900112233`` publishes one three-message album recorded as three media
    rows but a single posting, filed under the group's last message id — so the
    manifest names a known posting for only one of the three rows, and the
    other two can be attached only by album inference.

    Args:
        root: Temporary directory in which to create the export.
    """
    postings_data: dict[str, list[str]] = {col: ["", ""] for col in _SEMICOLON_POSTINGS_COLUMNS}
    postings_data["UUID"] = ["u1", "u2"]
    postings_data["Posting ID"] = ["990011223303", "990011223309"]
    postings_data["Author ID"] = ["9900112233", "9900112233"]
    postings_data["Timestamp"] = ["2026-03-04 21:30:56+00", "2026-03-05 08:00:00+00"]
    postings_data["Text Content"] = ["album post", "later post"]
    postings_data["Network"] = ["Telegram", "Telegram"]
    postings_data["Author"] = ["Jane Poster", "Jane Poster"]
    pd.DataFrame(postings_data).to_csv(root / "postings.csv", index=False)
    pd.DataFrame(
        {
            "Media ID": ["990011223301", "990011223302", "990011223303"],
            "Network ID": ["990011223301", "990011223302", "990011223303"],
            "Exported media filename": ["shot.jpg", "clip.mp4", "last.jpg"],
            "Timestamp": [
                "2026-03-04 21:30:55+00",
                "2026-03-04 21:30:56+00",
                "2026-03-04 21:30:56+00",
            ],
        }
    ).to_csv(root / "media.csv", index=False)
    photos = root / "dir" / "photos"
    videos = root / "dir" / "videos"
    photos.mkdir(parents=True)
    videos.mkdir(parents=True)
    (photos / "shot.jpg").write_bytes(b"\xff\xd8\xff")
    (photos / "last.jpg").write_bytes(b"\xff\xd8\xff")
    (videos / "clip.mp4").write_bytes(b"video")


def test_run_links_nested_media_including_album_members(tmp_path: Path) -> None:
    """A nested, keyless album export links every media row to its posting.

    Covers both halves of the Telegram failure at once: the files sit in
    subdirectories rather than beside the manifest, and two of the three rows
    name no posting of their own.
    """
    _write_album_export(tmp_path)

    img = _FakeImageService()
    result = SocialLinker(image_service=img, nextext_client=_FakeNextext(), target_collection="c").run(tmp_path)

    assert [asset.source_doc_id for asset in img.images] == ["u1", "u1"]
    assert img.keyframe_calls[0]["source_doc_id"] == "u1"
    consumed_names = {path.name for path in result.consumed_paths}
    assert {"media.csv", "shot.jpg", "last.jpg", "clip.mp4"}.issubset(consumed_names)
    assert all(segment.metadata["posting_uuid"] == "u1" for segment in result.transcript_documents)


def test_run_leaves_album_members_unlinked_when_disabled(tmp_path: Path) -> None:
    """``album_link_enabled=False`` restores manifest-key-only linking.

    The operator keeps a way back to the declared-key behaviour; only the row
    whose own ``Media ID`` names a posting survives.
    """
    _write_album_export(tmp_path)

    img = _FakeImageService()
    result = SocialLinker(
        image_service=img,
        nextext_client=_FakeNextext(),
        target_collection="c",
        album_link_enabled=False,
    ).run(tmp_path)

    assert [asset.source_doc_id for asset in img.images] == ["u1"]
    assert not img.keyframe_calls
    consumed_names = {path.name for path in result.consumed_paths}
    assert "last.jpg" in consumed_names
    assert "shot.jpg" not in consumed_names


def test_timestamp_link_can_be_switched_off() -> None:
    """``SOCIAL_TIMESTAMP_LINK_ENABLED=false`` leaves the fallback inert."""
    linker = SocialLinker(
        image_service=None,
        nextext_client=None,
        target_collection=None,
        timestamp_link_enabled=False,
    )
    assert linker.timestamp_link_enabled is False


#: The messages profile's exact header list, as a chat-style export writes it.
_MESSAGES_COLUMNS: list[str] = [
    "UUID",
    "Chat ID",
    "Sender",
    "Timestamp",
    "Text",
    "Tags",
    "URL",
    "Chat Group",
    "Answers Count",
    "Reply To",
    "Network",
]


def _write_messages_export(
    root: Path,
    *,
    media_author: str = "Jane Poster",
    filename: str = "pic.jpg",
) -> None:
    """Write a chat-style export: a messages-schema table plus a media manifest.

    Serialized like :func:`_write_semicolon_postings` (``;`` + UTF-8 BOM). The
    manifest's ids name no message, so only the stamp links the row — or the
    repeated text, when *media_author* names someone else.

    Args:
        root: Temporary directory in which to create the export.
        media_author: ``Author`` on the manifest row; a value other than the
            message's ``Sender`` produces the shared-post shape.
        filename: Basename written into ``Exported media filename``.
    """
    pd.DataFrame(
        {
            "UUID": ["u1"],
            "Chat ID": ["4400000000000000001"],
            "Sender": ["Jane Poster"],
            "Timestamp": ["2026-03-04 21:30:56+00"],
            "Text": ["A short invented post about nothing at all."],
            "Tags": [""],
            "URL": ["https://social.invalid/janeposter/status/4400000000000000001"],
            "Chat Group": [""],
            "Answers Count": ["0"],
            "Reply To": [""],
            "Network": ["ChatNet"],
        },
        columns=_MESSAGES_COLUMNS,
    ).to_csv(root / "messages.csv", index=False, sep=";", encoding="utf-8-sig")
    pd.DataFrame(
        {
            "Media ID": ["7700000000000000009"],
            "Network ID": ["7700000000000000009"],
            "Author": [media_author],
            "Network": ["ChatNet"],
            "Timestamp": ["2026-03-04 21:30:56+00"],
            "Title": ["A short invented post about nothing at all."],
            "Exported media filename": [filename],
        }
    ).to_csv(root / "media.csv", index=False, sep=";", encoding="utf-8-sig")


def test_find_tables_accepts_a_messages_schema_as_the_postings_table(tmp_path: Path) -> None:
    """A chat-style export carries its postings in the messages schema."""
    _write_messages_export(tmp_path)
    linker = SocialLinker(image_service=_FakeImageService(), nextext_client=_FakeNextext(), target_collection="c")

    postings_csv, media_csv = linker._find_tables(tmp_path)

    assert postings_csv is not None
    assert postings_csv.name == "messages.csv"
    assert media_csv is not None
    assert media_csv.name == "media.csv"


def test_postings_profile_wins_over_a_messages_table(tmp_path: Path) -> None:
    """A real postings table is the authority; the messages one only stands in.

    The messages file is named to sort first, so a first-match-wins sweep fails.
    """
    _write_semicolon_postings(tmp_path, {"P_1_0": "pic.jpg"})
    _write_messages_export(tmp_path)
    (tmp_path / "messages.csv").rename(tmp_path / "chats.csv")
    linker = SocialLinker(image_service=_FakeImageService(), nextext_client=_FakeNextext(), target_collection="c")

    postings_csv, _ = linker._find_tables(tmp_path)

    assert postings_csv is not None
    assert postings_csv.name == "postings.csv"


def test_run_links_media_for_a_messages_style_export(tmp_path: Path) -> None:
    """A chat-style export links end to end, keyed by the message's own id.

    Regression: requiring the exact postings profile made such exports no-op.
    """
    _write_messages_export(tmp_path)
    (tmp_path / "pic.jpg").write_bytes(b"\xff\xd8\xff")
    img = _FakeImageService()

    result = SocialLinker(image_service=img, nextext_client=_FakeNextext(), target_collection="c").run(tmp_path)

    assert len(img.images) == 1
    assert img.images[0].source_doc_id == "u1"
    assert img.images[0].extra_metadata["posting_id"] == "4400000000000000001"
    consumed_names = {p.name for p in result.consumed_paths}
    assert {"media.csv", "pic.jpg"}.issubset(consumed_names)
    assert "messages.csv" not in consumed_names


def test_run_stamps_posting_reference_metadata_from_a_messages_table(tmp_path: Path) -> None:
    """Derived artifacts carry the message's own reference fields."""
    _write_messages_export(tmp_path)
    (tmp_path / "pic.jpg").write_bytes(b"\xff\xd8\xff")
    img = _FakeImageService()

    SocialLinker(image_service=img, nextext_client=_FakeNextext(), target_collection="c").run(tmp_path)

    metadata = img.images[0].extra_metadata
    assert metadata["posting_author"] == "Jane Poster"
    assert metadata["posting_text"] == "A short invented post about nothing at all."
    assert metadata["posting_network"] == "ChatNet"
    assert metadata["posting_url"] == "https://social.invalid/janeposter/status/4400000000000000001"
    assert metadata["posting_timestamp"] == "2026-03-04 21:30:56+00"


def test_build_posting_reference_index_reads_a_messages_frame() -> None:
    """A messages table is keyed by its own id column, not by ``Posting ID``."""
    from docint.core.ingest.social_linker import build_posting_reference_index

    df = pd.DataFrame(
        {
            "UUID": ["u1"],
            "Chat ID": ["4400000000000000001"],
            "Sender": ["Jane Poster"],
            "Timestamp": ["2026-03-04 21:30:56+00"],
            "Text": ["A short invented post about nothing at all."],
            "Tags": [""],
            "URL": ["https://social.invalid/janeposter/status/4400000000000000001"],
            "Chat Group": [""],
            "Answers Count": ["0"],
            "Reply To": [""],
            "Network": ["ChatNet"],
        },
        columns=_MESSAGES_COLUMNS,
    )

    index = build_posting_reference_index(df)

    assert list(index) == ["4400000000000000001"]
    assert index["4400000000000000001"]["posting_author"] == "Jane Poster"


def test_run_links_a_shared_post_by_text_when_the_author_differs(tmp_path: Path) -> None:
    """A shared post names the original author, so only its text can link it.

    The manifest records the writer while the row is the sharer's, so the
    author-scoped stamp rule refuses it and only the text rule is left.
    """
    _write_messages_export(tmp_path, media_author="Original Author")
    (tmp_path / "pic.jpg").write_bytes(b"\xff\xd8\xff")
    img = _FakeImageService()

    SocialLinker(image_service=img, nextext_client=_FakeNextext(), target_collection="c").run(tmp_path)

    assert len(img.images) == 1
    assert img.images[0].source_doc_id == "u1"


def test_text_link_can_be_switched_off(tmp_path: Path) -> None:
    """``SOCIAL_TEXT_LINK_ENABLED=false`` leaves a shared post unlinked."""
    _write_messages_export(tmp_path, media_author="Original Author")
    (tmp_path / "pic.jpg").write_bytes(b"\xff\xd8\xff")
    img = _FakeImageService()

    SocialLinker(
        image_service=img,
        nextext_client=_FakeNextext(),
        target_collection="c",
        text_link_enabled=False,
    ).run(tmp_path)

    assert img.images == []


def test_derived_artifacts_name_the_media_file_they_came_from(tmp_path: Path) -> None:
    """A keyframe and a transcript segment must name the clip, not the transient JSONL.

    Without this the only identity a social video artifact carries is a posting
    UUID and a manifest media id, so neither an extract nor a report can say
    which attachment an analyst is looking at. The standalone path has always
    stamped these; the social path did not.
    """
    _write_export(tmp_path)
    img = _FakeImageService()

    result = SocialLinker(image_service=img, nextext_client=_FakeNextext(), target_collection="c").run(tmp_path)

    keyframe_extra = img.keyframe_calls[0]["extra_metadata"]
    assert keyframe_extra["source_file"] == "clip.mp4"
    assert keyframe_extra["source_path"].endswith("clip.mp4")
    assert keyframe_extra["reference_metadata"]["source_file"] == "clip.mp4"
    assert keyframe_extra["media_file_hash"] == keyframe_extra["reference_metadata"]["media_file_hash"]

    segment = result.transcript_documents[0].metadata
    assert segment["source_file"] == "clip.mp4"
    assert segment["file_name"] == "clip.mp4"
    assert segment["reference_metadata"]["source_file"] == "clip.mp4"


def test_a_transcript_segment_keeps_the_transcript_hash(tmp_path: Path) -> None:
    """``file_hash`` must stay the parsed transcript's, not the clip's.

    The pipeline skips documents whose ``file_hash`` is already in the
    collection. Stamping the media hash here would make every segment of an
    already-ingested clip look new, and a re-ingest would duplicate the whole
    transcript. The clip's own hash rides along as ``media_file_hash``.
    """
    _write_export(tmp_path)
    img = _FakeImageService()

    result = SocialLinker(image_service=img, nextext_client=_FakeNextext(), target_collection="c").run(tmp_path)

    segment = result.transcript_documents[0].metadata
    assert segment["media_file_hash"]
    assert segment["file_hash"] != segment["media_file_hash"]
