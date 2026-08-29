"""Tests for social-media linker join core logic."""

from pathlib import Path

import pandas as pd

from docint.core.ingest.social_linker import (
    _derive_posting_id,
    _infer_album_posting_id,
    build_posting_album_index,
    build_posting_index,
    build_posting_time_index,
    build_posting_url_index,
    resolve_media_rows,
    strip_counter,
)


def test_strip_counter_drops_trailing_numeric_segment() -> None:
    """Test that strip_counter removes trailing _<digits> segment."""
    assert strip_counter("2603434334845655437_44657421320_0") == "2603434334845655437_44657421320"
    assert strip_counter("2603434334845655437_44657421320_12") == "2603434334845655437_44657421320"


def test_resolve_matches_by_known_posting_id_in_flat_dir(tmp_path: Path) -> None:
    """Test matching by known posting ID with the file directly in the manifest dir."""
    img = tmp_path / "a.jpg"
    img.write_bytes(b"\xff\xd8\xff")

    postings = pd.DataFrame({"Posting ID": ["P_1"], "UUID": ["uuid-1"]})
    # ORPHAN_0 is skipped as an orphan before file resolution.
    # P_1_1 exercises the missing-file skip branch (known posting, file not on disk).
    media = pd.DataFrame(
        {
            "Media ID": ["P_1_0", "ORPHAN_0", "P_1_1"],
            "Exported media filename": ["a.jpg", "ignored.jpg", "nope.jpg"],
        }
    )

    links = resolve_media_rows(media, build_posting_index(postings), tmp_path)
    assert len(links) == 1
    assert links[0].posting_uuid == "uuid-1"
    assert links[0].media_id == "P_1_0"
    assert links[0].path == img


def test_resolve_does_not_find_file_in_different_directory(tmp_path: Path) -> None:
    """A basename that exists only outside the batch tree is not resolved.

    Resolution recurses, but only ever under the root it is given: ``x.jpg``
    lives in a sibling of the batch directory passed to ``resolve_media_rows``,
    so no amount of recursion can reach it and the row must be skipped rather
    than ingested. Containment comes from looking up basenames inside the batch
    tree, not from refusing to descend into its subdirectories.
    """
    other_dir = tmp_path / "other"
    other_dir.mkdir()
    (other_dir / "x.jpg").write_bytes(b"\xff\xd8\xff")

    batch = tmp_path / "batch"
    batch.mkdir()

    postings = pd.DataFrame({"Posting ID": ["P_1"], "UUID": ["uuid-1"]})
    media = pd.DataFrame({"Media ID": ["P_1_0"], "Exported media filename": ["x.jpg"]})

    links = resolve_media_rows(media, build_posting_index(postings), batch)
    assert links == []


def test_resolve_media_rows_aggregates_skips_for_large_manifest(tmp_path: Path) -> None:
    """A large manifest with only a few present files logs ONE summary, not per row.

    Guards robustness for the real-world drop-in shape: a full media.csv (tens of
    thousands of rows) placed in a batch that physically contains only a handful of
    the referenced files. Per-row skip logging would flood; resolution must stay quiet.
    """
    from loguru import logger

    (tmp_path / "a.jpg").write_bytes(b"\xff\xd8\xff")

    postings = pd.DataFrame({"Posting ID": ["P_1"], "UUID": ["uuid-1"]})
    n_orphan, n_missing = 500, 300
    media = pd.DataFrame(
        {
            "Media ID": (
                ["P_1_0"]  # linkable: known posting + file present
                + [f"ORPHAN_{i}_0" for i in range(n_orphan)]  # unknown posting
                + [f"P_1_{i}" for i in range(1, n_missing + 1)]  # known posting, file absent
            ),
            "Exported media filename": (["a.jpg"] + ["x.jpg"] * n_orphan + ["missing.jpg"] * n_missing),
        }
    )

    lines: list[str] = []
    sink_id = logger.add(lambda message: lines.append(str(message)), level="DEBUG", format="{level}|{message}")
    try:
        links = resolve_media_rows(media, build_posting_index(postings), tmp_path)
    finally:
        logger.remove(sink_id)

    assert len(links) == 1  # only the present, posting-matched file links
    # Robustness: exactly one aggregated summary line, not ~800 per-row lines.
    assert len(lines) == 1
    assert lines[0].startswith("INFO|")
    assert "media linked" in lines[0]


def test_derive_posting_id_prefers_network_id() -> None:
    """The Network ID column is the join key when it names a known posting."""
    posting_uuids = {"POST_1": "uuid-1"}
    # strip_counter(Media ID) would yield the wrong id here; Network ID is correct.
    assert _derive_posting_id("POST_1", "ALBUM_9_POST_1_ACCT", posting_uuids) == "POST_1"


def test_derive_posting_id_falls_back_to_media_id_then_strip_counter() -> None:
    """Falls back to the raw Media ID, then strip_counter(Media ID); None if neither."""
    posting_uuids = {"POST_1": "uuid-1"}
    assert _derive_posting_id("", "POST_1", posting_uuids) == "POST_1"  # raw Media ID == Posting ID
    assert _derive_posting_id("", "POST_1_0", posting_uuids) == "POST_1"  # <Posting ID>_<counter>
    assert _derive_posting_id("NOPE", "ALSO_NOPE", posting_uuids) is None


def test_resolve_media_rows_links_via_network_id(tmp_path: Path) -> None:
    """A row whose Network ID (not strip_counter) matches a posting links correctly."""
    (tmp_path / "clip.jpg").write_bytes(b"\xff\xd8\xff")
    postings = pd.DataFrame({"Posting ID": ["3745_779"], "UUID": ["uuid-1"]})
    # Media ID is <album>_<posting>_<account>: strip_counter -> "18525_3745" (wrong);
    # the true parent Posting ID is carried in Network ID.
    media = pd.DataFrame(
        {
            "Media ID": ["18525_3745_779"],
            "Network ID": ["3745_779"],
            "Exported media filename": ["clip.jpg"],
        }
    )
    links = resolve_media_rows(media, build_posting_index(postings), tmp_path)
    assert len(links) == 1
    assert links[0].posting_uuid == "uuid-1"
    assert links[0].posting_id == "3745_779"
    assert links[0].media_id == "18525_3745_779"


def test_resolve_finds_media_in_nested_subdirectories(tmp_path: Path) -> None:
    """Media nested below the manifest resolve; the default batch layout works.

    A multimedia export drops ``postings.csv`` / ``media.csv`` at the root and
    its files under ``dir/photos`` / ``dir/videos``. Resolution must reach them.
    """
    photos = tmp_path / "dir" / "photos"
    videos = tmp_path / "dir" / "videos"
    photos.mkdir(parents=True)
    videos.mkdir(parents=True)
    (photos / "shot.jpg").write_bytes(b"\xff\xd8\xff")
    (videos / "clip.mp4").write_bytes(b"video")

    postings = pd.DataFrame({"Posting ID": ["P_1", "P_2"], "UUID": ["uuid-1", "uuid-2"]})
    media = pd.DataFrame(
        {
            "Media ID": ["P_1_0", "P_2_0"],
            "Exported media filename": ["shot.jpg", "clip.mp4"],
        }
    )

    links = resolve_media_rows(media, build_posting_index(postings), tmp_path)
    assert {link.path for link in links} == {photos / "shot.jpg", videos / "clip.mp4"}


def test_resolve_skips_ambiguous_basename_across_subdirectories(tmp_path: Path) -> None:
    """A basename occurring in two subdirectories is refused, not guessed at.

    The manifest carries a basename and nothing else, so nothing in the data
    says which of the two files is the evidence. Attaching the wrong picture to
    a posting is worse than attaching none.
    """
    first = tmp_path / "dir" / "photos"
    second = tmp_path / "dir" / "extra"
    first.mkdir(parents=True)
    second.mkdir(parents=True)
    (first / "dup.jpg").write_bytes(b"\xff\xd8\xff")
    (second / "dup.jpg").write_bytes(b"\xff\xd8\xff")

    postings = pd.DataFrame({"Posting ID": ["P_1"], "UUID": ["uuid-1"]})
    media = pd.DataFrame({"Media ID": ["P_1_0"], "Exported media filename": ["dup.jpg"]})

    assert resolve_media_rows(media, build_posting_index(postings), tmp_path) == []


def test_resolve_prefers_the_manifest_directory_on_basename_clash(tmp_path: Path) -> None:
    """A copy beside the manifest breaks a duplicate-basename tie."""
    nested = tmp_path / "dir" / "photos"
    nested.mkdir(parents=True)
    (nested / "dup.jpg").write_bytes(b"\xff\xd8\xff")
    beside = tmp_path / "dup.jpg"
    beside.write_bytes(b"\xff\xd8\xff")

    postings = pd.DataFrame({"Posting ID": ["P_1"], "UUID": ["uuid-1"]})
    media = pd.DataFrame({"Media ID": ["P_1_0"], "Exported media filename": ["dup.jpg"]})

    links = resolve_media_rows(media, build_posting_index(postings), tmp_path, manifest_dir=tmp_path)
    assert [link.path for link in links] == [beside]


def _album_export() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return a (postings, media) pair shaped like a keyless album export.

    Channel ``9900112233`` posts one album of three messages (``01``-``03``)
    recorded as three media rows but a single posting, filed under the group's
    last message id — so only message ``03`` names a posting outright.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: The postings and media tables.
    """
    postings = pd.DataFrame(
        {
            "UUID": ["uuid-1"],
            "Posting ID": ["990011223303"],
            "Author ID": ["9900112233"],
            "Timestamp": ["2026-03-04 21:30:56+00"],
        }
    )
    media = pd.DataFrame(
        {
            "Media ID": ["990011223301", "990011223302", "990011223303"],
            "Network ID": ["990011223301", "990011223302", "990011223303"],
            "Exported media filename": ["a.jpg", "b.jpg", "c.jpg"],
            "Timestamp": ["2026-03-04 21:30:55+00", "2026-03-04 21:30:56+00", "2026-03-04 21:30:56+00"],
        }
    )
    return postings, media


def test_album_inference_links_members_that_name_no_posting(tmp_path: Path) -> None:
    """Album members link to the posting closing their group.

    Only the album's last message is itself a posting; the earlier two carry
    their own message ids and would otherwise be dropped as orphans.
    """
    for name in ("a.jpg", "b.jpg", "c.jpg"):
        (tmp_path / name).write_bytes(b"\xff\xd8\xff")
    postings, media = _album_export()

    links = resolve_media_rows(
        media,
        build_posting_index(postings),
        tmp_path,
        albums=build_posting_album_index(postings),
    )
    assert len(links) == 3
    assert {link.posting_uuid for link in links} == {"uuid-1"}


def test_album_inference_is_refused_when_timestamps_disagree(tmp_path: Path) -> None:
    """The timestamp guard rejects a candidate that is not the media's own post.

    With the owning posting absent from an export, the next one along would be
    picked purely by message order; requiring the stamps to agree leaves those
    rows unlinked instead of mis-attributed.
    """
    for name in ("a.jpg", "b.jpg", "c.jpg"):
        (tmp_path / name).write_bytes(b"\xff\xd8\xff")
    postings, media = _album_export()
    postings["Timestamp"] = ["2026-03-04 22:30:56+00"]  # an hour after the media

    links = resolve_media_rows(
        media,
        build_posting_index(postings),
        tmp_path,
        albums=build_posting_album_index(postings),
    )
    # Only the row whose own Media ID names the posting survives, via the manifest key.
    assert [link.media_id for link in links] == ["990011223303"]


def test_album_inference_is_off_unless_an_index_is_supplied(tmp_path: Path) -> None:
    """Omitting the album index leaves the manifest-key-only behaviour intact."""
    for name in ("a.jpg", "b.jpg", "c.jpg"):
        (tmp_path / name).write_bytes(b"\xff\xd8\xff")
    postings, media = _album_export()

    links = resolve_media_rows(media, build_posting_index(postings), tmp_path)
    assert [link.media_id for link in links] == ["990011223303"]


def test_manifest_key_wins_over_a_conflicting_album_inference(tmp_path: Path) -> None:
    """A working manifest key is never overridden by the album fallback.

    An export can be album-shaped (``Posting ID`` == ``<Author ID><message no>``)
    *and* still carry a real ``Network ID`` key. The inference would then attach the
    row to the first posting at or above its own message number — an earlier, wrong
    posting — so the fallback must run only after the declared key has failed.
    """
    (tmp_path / "a.jpg").write_bytes(b"\xff\xd8\xff")
    postings = pd.DataFrame(
        {
            "UUID": ["uuid-early", "uuid-keyed"],
            "Posting ID": ["990011223303", "990011223309"],
            "Author ID": ["9900112233", "9900112233"],
            "Timestamp": ["2026-03-04 21:30:56+00", "2026-03-04 21:30:56+00"],
        }
    )
    media = pd.DataFrame(
        {
            "Media ID": ["990011223301"],
            "Network ID": ["990011223309"],
            "Exported media filename": ["a.jpg"],
            "Timestamp": ["2026-03-04 21:30:56+00"],
        }
    )
    albums = build_posting_album_index(postings)
    # Guard the guard: the inference really would fire here, and really would disagree.
    # Without this the test could pass for the wrong reason on a table it cannot decompose.
    assert _infer_album_posting_id("990011223301", pd.Timestamp("2026-03-04 21:30:56+00"), albums, 5.0) == (
        "990011223303"
    )

    links = resolve_media_rows(media, build_posting_index(postings), tmp_path, albums=albums)

    assert [link.posting_id for link in links] == ["990011223309"]
    assert [link.posting_uuid for link in links] == ["uuid-keyed"]


def test_album_index_is_empty_for_exports_carrying_a_join_key() -> None:
    """A Meta-style export yields no album index, so the inference cannot fire.

    Its ``Posting ID`` is ``<postingId>_<accountId>``, carrying the ``Author ID``
    as a suffix rather than a prefix, so no channel decomposes — which is what
    keeps exports that already join correctly untouched by the fallback.
    """
    postings = pd.DataFrame(
        {
            "UUID": ["uuid-1"],
            "Posting ID": ["3745000000000000001_77503905789"],
            "Author ID": ["77503905789"],
            "Timestamp": ["2026-03-04 21:30:56+00"],
        }
    )
    assert build_posting_album_index(postings) == {}


def test_album_index_ignores_postings_with_unparseable_timestamps() -> None:
    """A malformed stamp costs that row its inference, not the whole index."""
    postings = pd.DataFrame(
        {
            "UUID": ["uuid-1", "uuid-2"],
            "Posting ID": ["990011223303", "990011223309"],
            "Author ID": ["9900112233", "9900112233"],
            "Timestamp": ["2026-03-04 21:30:56+00", "not a timestamp"],
        }
    )
    index = build_posting_album_index(postings)
    assert [entry[1] for entry in index["9900112233"]] == ["990011223303"]


# ---------------------------------------------------------------------------
# URL-derived key + author/timestamp fallbacks (key-less Meta-style exports)
# ---------------------------------------------------------------------------


def _keyless_export() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return a (postings, media) pair shaped like a key-less Meta-style export.

    The posting carries a crawler-minted UUID as its ``Posting ID`` while the
    manifest's media id is the bare network id, so no candidate the manifest-key
    path tries can ever name the posting. Its ``URL`` still ends in that id, and
    its author/network/timestamp still match the media row.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: The postings and media tables.
    """
    postings = pd.DataFrame(
        {
            "UUID": ["uuid-reel"],
            "Posting ID": ["3f4c1a90-0000-4000-8000-000000000001"],
            "URL": ["https://example.invalid/reel/2830654730615424/"],
            "Author ID": ["100007940942252"],
            "Author": ["authorname"],
            "Network": ["ExampleNet"],
            "Timestamp": ["2026-03-30 17:38:44+00"],
        }
    )
    media = pd.DataFrame(
        {
            "Media ID": ["2830654730615424"],
            "Network ID": ["2830654730615424"],
            "Exported media filename": ["a.jpg"],
            "Author": ["authorname"],
            "Network": ["ExampleNet"],
            "Timestamp": ["2026-03-30 17:38:44+00"],
        }
    )
    return postings, media


def test_url_index_maps_numeric_path_segments_to_their_posting() -> None:
    """A posting URL's numeric id becomes a join key for the manifest's media id."""
    postings = pd.DataFrame(
        {
            "UUID": ["uuid-1", "uuid-2"],
            "Posting ID": ["p-1", "p-2"],
            "URL": [
                "https://example.invalid/reel/2830654730615424/",
                "https://example.invalid/@name/video/7490165774523354390",
            ],
        }
    )
    assert build_posting_url_index(postings) == {
        "2830654730615424": "p-1",
        "7490165774523354390": "p-2",
    }


def test_url_index_ignores_short_and_non_numeric_segments() -> None:
    """Only long all-digit segments are keys; slugs and short ids are not.

    A short numeric segment (a version number, a page index) is not a network
    id, and matching one would attach media to an arbitrary posting.
    """
    postings = pd.DataFrame(
        {
            "UUID": ["uuid-1"],
            "Posting ID": ["p-1"],
            "URL": ["https://example.invalid/v2/name/posts/pfbid0abc123/"],
        }
    )
    assert build_posting_url_index(postings) == {}


def test_url_index_drops_ids_claimed_by_two_postings() -> None:
    """An id appearing in two postings' URLs is ambiguous and is not indexed."""
    postings = pd.DataFrame(
        {
            "UUID": ["uuid-1", "uuid-2"],
            "Posting ID": ["p-1", "p-2"],
            "URL": [
                "https://example.invalid/reel/2830654730615424/",
                "https://example.invalid/story/2830654730615424/",
            ],
        }
    )
    assert build_posting_url_index(postings) == {}


def test_url_fallback_links_media_whose_posting_id_is_a_uuid(tmp_path: Path) -> None:
    """Bare-digit media ids reach a UUID-keyed posting through its URL."""
    (tmp_path / "a.jpg").write_bytes(b"\xff\xd8\xff")
    postings, media = _keyless_export()

    links = resolve_media_rows(
        media,
        build_posting_index(postings),
        tmp_path,
        url_index=build_posting_url_index(postings),
    )

    assert [link.posting_uuid for link in links] == ["uuid-reel"]
    assert [link.posting_id for link in links] == ["3f4c1a90-0000-4000-8000-000000000001"]


def test_url_fallback_is_off_unless_an_index_is_supplied(tmp_path: Path) -> None:
    """Omitting the URL index leaves the manifest-key-only behaviour intact."""
    (tmp_path / "a.jpg").write_bytes(b"\xff\xd8\xff")
    postings, media = _keyless_export()

    assert resolve_media_rows(media, build_posting_index(postings), tmp_path) == []


def test_manifest_key_wins_over_a_conflicting_url_match(tmp_path: Path) -> None:
    """A working manifest key is never overridden by the URL fallback."""
    (tmp_path / "a.jpg").write_bytes(b"\xff\xd8\xff")
    postings = pd.DataFrame(
        {
            "UUID": ["uuid-url", "uuid-keyed"],
            "Posting ID": ["3f4c1a90-0000-4000-8000-000000000001", "2830654730615424"],
            "URL": ["https://example.invalid/reel/2830654730615424/", ""],
        }
    )
    media = pd.DataFrame(
        {
            "Media ID": ["2830654730615424"],
            "Network ID": ["2830654730615424"],
            "Exported media filename": ["a.jpg"],
        }
    )
    url_index = build_posting_url_index(postings)
    # Guard the guard: the URL path really would fire here, and really would disagree.
    assert url_index == {"2830654730615424": "3f4c1a90-0000-4000-8000-000000000001"}

    links = resolve_media_rows(media, build_posting_index(postings), tmp_path, url_index=url_index)

    assert [link.posting_uuid for link in links] == ["uuid-keyed"]


def test_time_index_groups_by_network_and_author_case_insensitively() -> None:
    """Channel keys fold case, so ``ExampleNet``/``examplenet`` are one channel."""
    postings = pd.DataFrame(
        {
            "UUID": ["uuid-1", "uuid-2"],
            "Posting ID": ["p-1", "p-2"],
            "Author": ["AuthorName", "authorname"],
            "Network": ["ExampleNet", "examplenet"],
            "Timestamp": ["2026-03-30 17:38:44+00", "2026-03-30 18:00:00+00"],
        }
    )

    index = build_posting_time_index(postings)

    assert list(index) == [("examplenet", "authorname")]
    assert [posting_id for _, posting_id in index[("examplenet", "authorname")]] == ["p-1", "p-2"]


def test_time_index_skips_rows_with_no_stamp_or_no_author() -> None:
    """Rows that cannot anchor a match are left out of the index entirely."""
    postings = pd.DataFrame(
        {
            "UUID": ["uuid-1", "uuid-2", "uuid-3"],
            "Posting ID": ["p-1", "p-2", "p-3"],
            "Author": ["authorname", "", "authorname"],
            "Network": ["ExampleNet", "ExampleNet", "ExampleNet"],
            "Timestamp": ["not a date", "2026-03-30 17:38:44+00", "2026-03-30 17:38:44+00"],
        }
    )

    assert build_posting_time_index(postings) == {
        ("examplenet", "authorname"): [(pd.Timestamp("2026-03-30 17:38:44+00"), "p-3")]
    }


def test_timestamp_fallback_links_a_media_row_to_its_own_posting(tmp_path: Path) -> None:
    """Same author, same network, agreeing stamps: the row links."""
    (tmp_path / "a.jpg").write_bytes(b"\xff\xd8\xff")
    postings, media = _keyless_export()
    postings["URL"] = [""]  # force the timestamp path, not the URL one

    links = resolve_media_rows(
        media,
        build_posting_index(postings),
        tmp_path,
        time_index=build_posting_time_index(postings),
    )

    assert [link.posting_uuid for link in links] == ["uuid-reel"]


def test_timestamp_fallback_is_refused_outside_the_tolerance(tmp_path: Path) -> None:
    """The tolerance is the guard: a posting a minute away is not the parent."""
    (tmp_path / "a.jpg").write_bytes(b"\xff\xd8\xff")
    postings, media = _keyless_export()
    postings["URL"] = [""]
    media["Timestamp"] = ["2026-03-30 17:39:44+00"]  # a minute after the posting

    links = resolve_media_rows(
        media,
        build_posting_index(postings),
        tmp_path,
        time_index=build_posting_time_index(postings),
        album_tolerance_s=5.0,
    )

    assert links == []


def test_timestamp_fallback_is_refused_when_two_postings_are_in_range(tmp_path: Path) -> None:
    """Two candidates inside the window is ambiguity, not a link.

    Picking either would be a coin flip presented to the investigator as
    provenance, so the row stays unlinked.
    """
    (tmp_path / "a.jpg").write_bytes(b"\xff\xd8\xff")
    postings, media = _keyless_export()
    second = pd.DataFrame(
        {
            "UUID": ["uuid-other"],
            "Posting ID": ["3f4c1a90-0000-4000-8000-000000000002"],
            "URL": [""],
            "Author ID": ["100007940942252"],
            "Author": ["authorname"],
            "Network": ["ExampleNet"],
            "Timestamp": ["2026-03-30 17:38:46+00"],
        }
    )
    postings["URL"] = [""]
    postings = pd.concat([postings, second], ignore_index=True)

    links = resolve_media_rows(
        media,
        build_posting_index(postings),
        tmp_path,
        time_index=build_posting_time_index(postings),
    )

    assert links == []


def test_timestamp_fallback_is_refused_across_authors(tmp_path: Path) -> None:
    """A stamp match on a different author's posting is not a link."""
    (tmp_path / "a.jpg").write_bytes(b"\xff\xd8\xff")
    postings, media = _keyless_export()
    postings["URL"] = [""]
    media["Author"] = ["someone-else"]

    links = resolve_media_rows(
        media,
        build_posting_index(postings),
        tmp_path,
        time_index=build_posting_time_index(postings),
    )

    assert links == []


def test_timestamp_fallback_is_off_unless_an_index_is_supplied(tmp_path: Path) -> None:
    """Omitting the time index leaves the manifest-key-only behaviour intact."""
    (tmp_path / "a.jpg").write_bytes(b"\xff\xd8\xff")
    postings, media = _keyless_export()
    postings["URL"] = [""]

    assert resolve_media_rows(media, build_posting_index(postings), tmp_path) == []


def test_url_match_wins_over_a_conflicting_timestamp_match(tmp_path: Path) -> None:
    """An exact URL id beats a corroborated-but-inferred timestamp match."""
    (tmp_path / "a.jpg").write_bytes(b"\xff\xd8\xff")
    postings = pd.DataFrame(
        {
            "UUID": ["uuid-url", "uuid-timestamp"],
            "Posting ID": ["p-url", "p-timestamp"],
            "URL": ["https://example.invalid/reel/2830654730615424/", ""],
            "Author": ["authorname", "authorname"],
            "Network": ["ExampleNet", "ExampleNet"],
            "Timestamp": ["2026-03-30 19:00:00+00", "2026-03-30 17:38:44+00"],
        }
    )
    media = pd.DataFrame(
        {
            "Media ID": ["2830654730615424"],
            "Network ID": ["2830654730615424"],
            "Exported media filename": ["a.jpg"],
            "Author": ["authorname"],
            "Network": ["ExampleNet"],
            "Timestamp": ["2026-03-30 17:38:44+00"],
        }
    )

    links = resolve_media_rows(
        media,
        build_posting_index(postings),
        tmp_path,
        url_index=build_posting_url_index(postings),
        time_index=build_posting_time_index(postings),
    )

    assert [link.posting_uuid for link in links] == ["uuid-url"]
