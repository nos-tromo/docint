"""Tests for social-media linker join core logic."""

from pathlib import Path

import pandas as pd

from docint.core.ingest.social_linker import (
    _derive_posting_id,
    _infer_album_posting_id,
    _infer_stamp_posting_id,
    _infer_text_posting_id,
    build_network_posting_index,
    build_posting_album_index,
    build_posting_index,
    build_posting_stamp_index,
    build_posting_text_index,
    normalize_postings_frame,
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


def test_network_posting_index_maps_the_networks_own_id_to_the_posting() -> None:
    """A posting is findable by the id its own network uses, not just ``Posting ID``.

    Some exports mint an internal ``Posting ID`` (a UUID) while the id the media
    manifest carries is the network's own — recorded in ``Network Posting ID``.
    """
    postings = pd.DataFrame(
        {
            "UUID": ["uuid-1"],
            "Posting ID": ["1929d6fa-da9c-586b-912b-86a33371d93e"],
            "Network Posting ID": ["900622622300677"],
            "Author ID": ["100007940942252"],
            "URL": ["https://www.example.invalid/posts/abc"],
        }
    )
    assert build_network_posting_index(postings) == {"900622622300677": "1929d6fa-da9c-586b-912b-86a33371d93e"}


def test_network_posting_index_harvests_the_id_embedded_in_a_url() -> None:
    """A reel-style posting carries its network id only in its ``URL``.

    ``Network Posting ID`` is empty for those rows, so the numeric id in the
    permalink is the sole way the manifest's media id can name the posting.
    """
    postings = pd.DataFrame(
        {
            "UUID": ["uuid-1"],
            "Posting ID": ["1929d6fa-da9c-586b-912b-86a33371d93e"],
            "Network Posting ID": [""],
            "Author ID": ["100007940942252"],
            "URL": ["https://www.example.invalid/reel/900622622300677/"],
        }
    )
    assert build_network_posting_index(postings) == {"900622622300677": "1929d6fa-da9c-586b-912b-86a33371d93e"}


def test_network_posting_index_drops_an_id_naming_two_postings() -> None:
    """An id that two postings advertise identifies neither.

    Nothing in the data says which one owns a media row, so the id is dropped
    rather than resolved to whichever posting happened to be read last.
    """
    postings = pd.DataFrame(
        {
            "UUID": ["uuid-1", "uuid-2"],
            "Posting ID": ["internal-1", "internal-2"],
            "Network Posting ID": ["900622622300677", "900622622300677"],
            "Author ID": ["100007940942252", "100007940942252"],
            "URL": ["", ""],
        }
    )
    assert build_network_posting_index(postings) == {}


def test_network_posting_index_drops_the_account_id() -> None:
    """A permalink carries the account id beside the posting id.

    The account appears in every one of its postings' URLs, so treating it as a
    posting id would attach media to an arbitrary posting of that account.
    """
    postings = pd.DataFrame(
        {
            "UUID": ["uuid-1"],
            "Posting ID": ["internal-1"],
            "Network Posting ID": [""],
            "Author ID": ["100007940942252"],
            "URL": ["https://www.example.invalid/100007940942252/posts/900622622300677"],
        }
    )
    assert build_network_posting_index(postings) == {"900622622300677": "internal-1"}


def _network_id_export() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return a (postings, media) pair whose key is the network's own id.

    ``Posting ID`` is internal to the crawler, so the manifest's media id can
    only reach the posting through its network-level id.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: The postings and media tables.
    """
    postings = pd.DataFrame(
        {
            "UUID": ["uuid-1"],
            "Posting ID": ["1929d6fa-da9c-586b-912b-86a33371d93e"],
            "Network Posting ID": [""],
            "Author ID": ["100007940942252"],
            "URL": ["https://www.example.invalid/reel/900622622300677/"],
            "Timestamp": ["2026-03-04 21:30:56+00"],
        }
    )
    media = pd.DataFrame(
        {
            "Media ID": ["900622622300677"],
            "Network ID": ["900622622300677"],
            "Exported media filename": ["a.jpg"],
            "Timestamp": ["2026-03-04 21:30:56+00"],
        }
    )
    return postings, media


def test_resolve_links_through_the_network_posting_id(tmp_path: Path) -> None:
    """A media row naming the network's posting id resolves to that posting."""
    (tmp_path / "a.jpg").write_bytes(b"\xff\xd8\xff")
    postings, media = _network_id_export()

    links = resolve_media_rows(
        media,
        build_posting_index(postings),
        tmp_path,
        network_index=build_network_posting_index(postings),
    )

    assert [link.posting_uuid for link in links] == ["uuid-1"]


def test_resolve_ignores_the_network_index_without_one(tmp_path: Path) -> None:
    """Omitting the index leaves the manifest-key-only behaviour intact."""
    (tmp_path / "a.jpg").write_bytes(b"\xff\xd8\xff")
    postings, media = _network_id_export()

    assert resolve_media_rows(media, build_posting_index(postings), tmp_path) == []


def test_manifest_key_wins_over_the_network_posting_index(tmp_path: Path) -> None:
    """A working manifest key is never overridden by the network-id index.

    An export can name a posting outright *and* advertise a network-level id that
    another posting also matches; the declared key is the stronger statement, so
    the index is consulted only once that key has failed.
    """
    (tmp_path / "a.jpg").write_bytes(b"\xff\xd8\xff")
    postings = pd.DataFrame(
        {
            "UUID": ["uuid-declared", "uuid-network"],
            "Posting ID": ["900622622300677", "internal-2"],
            "Network Posting ID": ["", "900622622300677"],
            "Author ID": ["100007940942252", "100007940942252"],
            "URL": ["", ""],
            "Timestamp": ["2026-03-04 21:30:56+00", "2026-03-04 21:30:56+00"],
        }
    )
    media = pd.DataFrame(
        {
            "Media ID": ["900622622300677"],
            "Network ID": ["900622622300677"],
            "Exported media filename": ["a.jpg"],
            "Timestamp": ["2026-03-04 21:30:56+00"],
        }
    )
    index = build_network_posting_index(postings)
    # Guard the guard: the index really does offer a different answer here.
    assert index == {"900622622300677": "internal-2"}

    links = resolve_media_rows(media, build_posting_index(postings), tmp_path, network_index=index)

    assert [link.posting_uuid for link in links] == ["uuid-declared"]


def _keyless_export() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return a (postings, media) pair with no reachable media->posting key.

    ``Posting ID`` is internal, the permalink carries no numeric id, and the
    manifest's media id names a photo rather than a post -- so only the stamp
    the two share can associate them.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: The postings and media tables.
    """
    postings = pd.DataFrame(
        {
            "UUID": ["uuid-1"],
            "Posting ID": ["1929d6fa-da9c-586b-912b-86a33371d93e"],
            "Network Posting ID": ["pfbid02D4AdkWqDmfcEjyCdxhWPfNWLHinwFZ"],
            "Author ID": ["100007940942252"],
            "Author": ["authorname"],
            "Network": ["Facebook"],
            "URL": ["https://www.example.invalid/authorname/posts/pfbid02D4AdkWqDmfcEjyCdxhWPfNWLHinwFZ"],
            "Timestamp": ["2026-03-04 21:30:56+00"],
        }
    )
    media = pd.DataFrame(
        {
            "Media ID": ["847656814058061"],
            "Network ID": ["847656814058061"],
            "Author": ["authorname"],
            "Network": ["Facebook"],
            "Exported media filename": ["a.jpg"],
            "Timestamp": ["2026-03-04 21:30:56+00"],
        }
    )
    return postings, media


def test_stamp_inference_links_a_keyless_row_to_its_only_stamp_mate(tmp_path: Path) -> None:
    """A row no key can reach attaches to the one posting sharing its stamp."""
    (tmp_path / "a.jpg").write_bytes(b"\xff\xd8\xff")
    postings, media = _keyless_export()

    links = resolve_media_rows(
        media,
        build_posting_index(postings),
        tmp_path,
        stamps=build_posting_stamp_index(postings),
    )

    assert [link.posting_uuid for link in links] == ["uuid-1"]


def test_stamp_inference_refuses_two_postings_at_one_instant(tmp_path: Path) -> None:
    """Two postings by one author at the same instant identify neither."""
    (tmp_path / "a.jpg").write_bytes(b"\xff\xd8\xff")
    postings, media = _keyless_export()
    postings = pd.concat([postings, postings.assign(UUID="uuid-2", **{"Posting ID": "internal-2"})])

    links = resolve_media_rows(
        media, build_posting_index(postings), tmp_path, stamps=build_posting_stamp_index(postings)
    )

    assert links == []


def test_stamp_inference_refuses_an_instant_no_posting_shares(tmp_path: Path) -> None:
    """A partial export is missing the parent; the row stays unlinked.

    This is the common case for a manifest slice whose postings were crawled
    separately, and it must not be papered over with a neighbouring post.
    """
    (tmp_path / "a.jpg").write_bytes(b"\xff\xd8\xff")
    postings, media = _keyless_export()
    media["Timestamp"] = ["2026-03-04 23:59:59+00"]

    links = resolve_media_rows(
        media, build_posting_index(postings), tmp_path, stamps=build_posting_stamp_index(postings)
    )

    assert links == []


def test_album_inference_wins_over_the_timestamp_fallback(tmp_path: Path) -> None:
    """Album ordering is the stronger statement, so it is consulted first.

    The fallback knows only that two rows share an instant; album inference
    knows the message numbering as well, so where both answer it is the one to
    believe.
    """
    for name in ("a.jpg", "b.jpg", "c.jpg"):
        (tmp_path / name).write_bytes(b"\xff\xd8\xff")
    postings, media = _album_export()
    postings["Network"] = ["Telegram"]
    postings["Author"] = ["authorname"]
    decoy = postings.assign(
        UUID="uuid-decoy",
        **{"Posting ID": "990011223309", "Timestamp": "2026-03-04 21:30:55+00"},
    )
    postings = pd.concat([postings, decoy], ignore_index=True)
    media["Network"] = ["Telegram"] * len(media)
    media["Author"] = ["authorname"] * len(media)
    stamps = build_posting_stamp_index(postings)
    # Guard the guard: the fallback really would answer differently here.
    assert _infer_stamp_posting_id(media.iloc[0], pd.Timestamp("2026-03-04 21:30:55+00"), stamps) == "990011223309"

    links = resolve_media_rows(
        media,
        build_posting_index(postings),
        tmp_path,
        albums=build_posting_album_index(postings),
        stamps=stamps,
    )

    assert {link.posting_uuid for link in links} == {"uuid-1"}


def _messages_export() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return a (messages, media) pair in the chat-style export shape.

    The manifest's ids name no message, so only the stamp or the text links them.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: The messages and media tables.
    """
    messages = pd.DataFrame(
        {
            "UUID": ["uuid-1"],
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
        }
    )
    media = pd.DataFrame(
        {
            "Media ID": ["7700000000000000009"],
            "Network ID": ["7700000000000000009"],
            "Author": ["Jane Poster"],
            "Network": ["ChatNet"],
            "Timestamp": ["2026-03-04 21:30:56+00"],
            "Title": ["A short invented post about nothing at all."],
            "Exported media filename": ["a.jpg"],
        }
    )
    return messages, media


def test_normalize_maps_the_messages_vocabulary_to_the_postings_one() -> None:
    """A messages table is renamed into the vocabulary the join rules read."""
    messages, _ = _messages_export()

    normalized = normalize_postings_frame(messages)

    assert normalized["Posting ID"].tolist() == ["4400000000000000001"]
    assert normalized["Author"].tolist() == ["Jane Poster"]
    assert normalized["Text Content"].tolist() == ["A short invented post about nothing at all."]
    assert "Chat ID" not in normalized.columns


def test_normalize_leaves_a_postings_table_untouched() -> None:
    """A real postings table already speaks the vocabulary and is returned as-is."""
    postings, _ = _keyless_export()

    assert normalize_postings_frame(postings) is postings


def test_normalize_ignores_a_table_matching_no_profile() -> None:
    """A foreign table that merely carries a Text column is never rewritten."""
    foreign = pd.DataFrame({"Text": ["hello"], "Something Else": ["x"]})

    assert normalize_postings_frame(foreign) is foreign


def test_stamp_inference_links_a_messages_row_through_the_renamed_columns(tmp_path: Path) -> None:
    """The stamp rule reaches a chat-style export once its columns are normalized."""
    (tmp_path / "a.jpg").write_bytes(b"\xff\xd8\xff")
    messages, media = _messages_export()
    join_df = normalize_postings_frame(messages)

    links = resolve_media_rows(
        media,
        build_posting_index(join_df),
        tmp_path,
        stamps=build_posting_stamp_index(join_df),
    )

    assert [link.posting_uuid for link in links] == ["uuid-1"]
    assert [link.posting_id for link in links] == ["4400000000000000001"]


def test_text_match_links_a_row_whose_title_names_one_posting(tmp_path: Path) -> None:
    """A shared post names another author, so only the repeated text can link it."""
    (tmp_path / "a.jpg").write_bytes(b"\xff\xd8\xff")
    messages, media = _messages_export()
    # The manifest records the original author; the export's row is the sharer's.
    media["Author"] = ["Original Author"]
    join_df = normalize_postings_frame(messages)
    stamps = build_posting_stamp_index(join_df)
    # Guard the guard: the author-scoped stamp rule really cannot reach this row.
    assert _infer_stamp_posting_id(media.iloc[0], pd.Timestamp("2026-03-04 21:30:56+00"), stamps) is None

    links = resolve_media_rows(
        media,
        build_posting_index(join_df),
        tmp_path,
        stamps=stamps,
        texts=build_posting_text_index(join_df),
    )

    assert [link.posting_uuid for link in links] == ["uuid-1"]


def test_text_match_refuses_two_postings_carrying_the_same_text(tmp_path: Path) -> None:
    """One text advertised by two postings identifies neither."""
    (tmp_path / "a.jpg").write_bytes(b"\xff\xd8\xff")
    messages, media = _messages_export()
    media["Author"] = ["Original Author"]
    messages = pd.concat(
        [messages, messages.assign(UUID="uuid-2", **{"Chat ID": "4400000000000000002"})],
        ignore_index=True,
    )
    join_df = normalize_postings_frame(messages)

    links = resolve_media_rows(
        media,
        build_posting_index(join_df),
        tmp_path,
        texts=build_posting_text_index(join_df),
    )

    assert links == []


def test_text_match_refuses_a_title_no_posting_carries(tmp_path: Path) -> None:
    """A partial export is missing the parent; the row stays unlinked."""
    (tmp_path / "a.jpg").write_bytes(b"\xff\xd8\xff")
    messages, media = _messages_export()
    media["Author"] = ["Original Author"]
    media["Title"] = ["A different invented post nobody in this export wrote."]
    join_df = normalize_postings_frame(messages)

    links = resolve_media_rows(
        media,
        build_posting_index(join_df),
        tmp_path,
        texts=build_posting_text_index(join_df),
    )

    assert links == []


def test_text_index_skips_postings_with_no_text() -> None:
    """An empty text is shared by every media-only post and names none of them."""
    messages, _ = _messages_export()
    messages["Text"] = [""]

    assert build_posting_text_index(normalize_postings_frame(messages)) == {}


def test_text_match_is_scoped_to_the_network() -> None:
    """One network's text says nothing about another's posting of the same words."""
    messages, media = _messages_export()
    media["Network"] = ["OtherNet"]
    texts = build_posting_text_index(normalize_postings_frame(messages))

    assert _infer_text_posting_id(media.iloc[0], texts) is None


def test_text_match_is_off_unless_an_index_is_supplied(tmp_path: Path) -> None:
    """Without the index the rule never runs, which is the kill switch's shape."""
    (tmp_path / "a.jpg").write_bytes(b"\xff\xd8\xff")
    messages, media = _messages_export()
    media["Author"] = ["Original Author"]
    join_df = normalize_postings_frame(messages)

    links = resolve_media_rows(media, build_posting_index(join_df), tmp_path)

    assert links == []


def test_timestamp_link_wins_over_the_text_match(tmp_path: Path) -> None:
    """The stamp is the stronger statement, so it is consulted first.

    Same words are shared by design by a quote or re-post; a shared stamp means
    the author agrees too.
    """
    (tmp_path / "a.jpg").write_bytes(b"\xff\xd8\xff")
    messages, media = _messages_export()
    decoy = messages.assign(
        UUID="uuid-decoy",
        **{"Chat ID": "4400000000000000002", "Sender": "Someone Else"},
    )
    messages = pd.concat([messages, decoy], ignore_index=True)
    # The decoy repeats the text, so the text rule alone answers ambiguously; give
    # it a single unambiguous answer that disagrees with the stamp rule's.
    messages.loc[0, "Text"] = "The words only the stamp rule's posting carries."
    join_df = normalize_postings_frame(messages)
    texts = build_posting_text_index(join_df)
    stamps = build_posting_stamp_index(join_df)
    media["Title"] = ["A short invented post about nothing at all."]
    # Guard the guard: the text rule really would answer differently here.
    assert _infer_text_posting_id(media.iloc[0], texts) == "4400000000000000002"

    links = resolve_media_rows(
        media,
        build_posting_index(join_df),
        tmp_path,
        stamps=stamps,
        texts=texts,
    )

    assert [link.posting_uuid for link in links] == ["uuid-1"]
