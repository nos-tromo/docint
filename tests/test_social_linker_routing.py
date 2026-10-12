"""Tests for SocialLinker over ``me-dossier/1`` exports: nodes, links, routing, claims.

Every fixture is synthetic: invented handles, ids and texts only.
"""

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from _pytest.logging import LogCaptureFixture
from llama_index.core import Document
from typing_extensions import override

from docint.core.ingest.images_service import IngestContext
from docint.core.ingest.preprocess import collection_of_key
from docint.core.ingest.social_linker import SocialLinker, SocialLinkResult, build_posting_album_index
from docint.core.summary.units import is_social_payload
from docint.utils.hashing import compute_file_hash
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
        on_frame: Any = None,
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
            on_frame: Per-frame progress sink (accepted, not called — the stub
                stores nothing, and the task closes the tally either way).

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


#: The profile folder of the standard fixture, named the way the exporter names them.
_FB = "jane.poster - facebook"


def _author(name: str = "Jane Poster", vanity: str = "jane.poster", platform_id: str = "42") -> dict[str, Any]:
    """Return an invented author record.

    Args:
        name: Display name.
        vanity: Handle.
        platform_id: The network's own account id.

    Returns:
        dict[str, Any]: The author as a dossier writes it.
    """
    return {"id": f"acct-{platform_id}", "name": name, "vanity": vanity, "platformId": platform_id}


def _posting(
    uuid: str,
    platform_id: str,
    text: str | None,
    *,
    published_at: str = "2023-01-01T10:00:00+00:00",
    network: str = "facebook",
    author: dict[str, Any] | None = None,
    media_ids: Sequence[str] = (),
    comments: Sequence[dict[str, Any]] = (),
) -> dict[str, Any]:
    """Return an invented posting record.

    Args:
        uuid: The export's own posting id.
        platform_id: The network's own posting id.
        text: The posting text.
        published_at: ISO-8601 publication time.
        network: The network key.
        author: The author record; :func:`_author` when omitted.
        media_ids: Media the posting names (the posting side of the link).
        comments: Received comments nested under the posting.

    Returns:
        dict[str, Any]: The posting as a dossier writes it.
    """
    return {
        "id": uuid,
        "platformId": platform_id,
        "publishedAt": published_at,
        "text": text,
        "url": f"https://social.invalid/{platform_id}",
        "network": network,
        "author": author or _author(),
        "mediaIds": list(media_ids),
        "comments": list(comments),
    }


def _comment(
    uuid: str, platform_id: str, text: str | None, *, posting: str, parent: str | None = None
) -> dict[str, Any]:
    """Return an invented comment record.

    Args:
        uuid: The export's own comment id.
        platform_id: The network's own comment id.
        text: The comment text.
        posting: The id of the posting it was left on.
        parent: The id of the comment it answers, if any.

    Returns:
        dict[str, Any]: The comment as a dossier nests it under its posting.
    """
    return {
        "id": uuid,
        "platformId": platform_id,
        "publishedAt": "2023-01-01T11:00:00+00:00",
        "text": text,
        "url": None,
        "network": "facebook",
        "author": _author("Joe Commenter", "joe.commenter", "77"),
        "postingId": posting,
        "parentCommentId": parent,
    }


def _media(
    uuid: str,
    platform_id: str,
    file: str | None,
    *,
    published_at: str = "2023-01-01T10:00:00+00:00",
    status: str = "complete",
    attached_to: Sequence[str] = (),
) -> dict[str, Any]:
    """Return an invented media record.

    Args:
        uuid: The export's own media id.
        platform_id: The network's own media id.
        file: The file path relative to the export root, ``None`` when not shipped.
        published_at: ISO-8601 publication time.
        status: ``complete`` or ``missing``.
        attached_to: Postings the media names (the media side of the link).

    Returns:
        dict[str, Any]: The media item as a dossier writes it.
    """
    return {
        "id": uuid,
        "platformId": platform_id,
        "publishedAt": published_at,
        "file": file,
        "status": status,
        "attachedTo": {"postings": [{"id": posting, "inFile": True} for posting in attached_to], "albumId": None},
    }


def _write_dossier(
    root: Path,
    folder: str,
    *,
    postings: Sequence[dict[str, Any]] = (),
    media: Sequence[dict[str, Any]] = (),
    files: dict[str, bytes] | None = None,
    **fields: Any,
) -> Path:
    """Write ``<root>/<folder>/dossier.json`` plus the media files it ships.

    Args:
        root: The export root.
        folder: The profile folder.
        postings: The posting records.
        media: The media records.
        files: Media bytes by path relative to the profile folder.
        **fields: Top-level fields to add or override (e.g. ``schema``, ``messages``).

    Returns:
        Path: The dossier file.
    """
    profile = root / folder
    profile.mkdir(parents=True, exist_ok=True)
    dossier = {"schema": "me-dossier/1", "postings": list(postings), "media": list(media), **fields}
    path = profile / "dossier.json"
    path.write_text(json.dumps(dossier), encoding="utf-8")
    for relative, data in (files or {}).items():
        (profile / relative).parent.mkdir(parents=True, exist_ok=True)
        (profile / relative).write_bytes(data)
    return path


def _write_export(root: Path) -> None:
    """Write a minimal ``me-dossier/1`` export whose root is *root*.

    Posting ``u1`` names its photo in ``mediaIds``; clip ``m2`` names its
    posting ``u2`` only in ``attachedTo`` — one link from each side. The
    export's bookkeeping sits where the exporter writes it.

    Args:
        root: Temporary directory in which to create the export.
    """
    _write_dossier(
        root,
        _FB,
        postings=[
            _posting("u1", "P_1", "a", media_ids=["m1"]),
            _posting("u2", "P_2", "b", published_at="2023-02-02T11:00:00+00:00"),
        ],
        media=[
            _media("m1", "M_1", f"{_FB}/media/photos/pic.jpg"),
            _media("m2", "M_2", f"{_FB}/media/videos/clip.mp4", attached_to=["u2"]),
        ],
        files={"media/photos/pic.jpg": b"\xff\xd8\xff", "media/videos/clip.mp4": b"video"},
    )
    (root / _FB / "progress.json").write_text('{"schema": "me-dossier-progress/1"}', encoding="utf-8")
    (root / "_run.json").write_text("{}", encoding="utf-8")


def _linker(**kwargs: Any) -> SocialLinker:
    """Return a linker over the in-memory stubs, overridable per test."""
    kwargs.setdefault("image_service", _FakeImageService())
    kwargs.setdefault("nextext_client", _FakeNextext())
    kwargs.setdefault("target_collection", "c")
    return SocialLinker(**kwargs)


def _transcripts(result: SocialLinkResult) -> list[Document]:
    """Return the transcript segments among a run's Documents."""
    return [doc for doc in result.documents if doc.metadata.get("docint_doc_kind") == "transcript_segment"]


def _rows(result: SocialLinkResult, style: str) -> list[Document]:
    """Return a run's posting (``postings``) or comment (``comments``) Documents."""
    return [doc for doc in result.documents if (doc.metadata.get("table") or {}).get("style") == style]


def _linked(img: _FakeImageService) -> dict[str, Any]:
    """Return ``{media file stem: posting uuid}`` for every image routed to CLIP."""
    return {Path(asset.image_path).stem: asset.source_doc_id for asset in img.images}


def test_run_routes_image_and_video_and_links(tmp_path: Path) -> None:
    """Image goes to CLIP, video to Nextext; each linked to its posting from either side of the link."""
    _write_export(tmp_path)
    img = _FakeImageService()
    result = _linker(image_service=img).run(tmp_path)

    assert _linked(img) == {"pic": "u1"}
    assert img.keyframe_calls and img.keyframe_calls[0]["source_doc_id"] == "u2"
    assert [segment.metadata["posting_uuid"] for segment in _transcripts(result)] == ["u2"]
    # The dossier, its bookkeeping and both media files are the linker's; no generic reader sees them.
    consumed_names = {p.name for p in result.consumed_paths}
    assert {"dossier.json", "progress.json", "_run.json", "pic.jpg", "clip.mp4"}.issubset(consumed_names)


def test_run_stamps_posting_reference_metadata(tmp_path: Path) -> None:
    """Derived media artifacts carry the parent posting's reference fields, additively.

    The image's payload is pinned exactly — it is the shape every downstream
    reader already knows. The transcript segment must merge the fields into its
    own ``reference_metadata`` WITHOUT dropping the Nextext identity
    (``network: nextext`` / ``type: transcript_segment``).
    """
    _write_export(tmp_path)
    img = _FakeImageService()
    result = _linker(image_service=img).run(tmp_path)

    link_ids = {"posting_uuid": "u1", "posting_id": "P_1", "media_id": "M_1"}
    posting_ref = {
        "posting_network": "facebook",
        "posting_author": "Jane Poster",
        "posting_author_id": "42",
        "posting_vanity": "jane.poster",
        "posting_timestamp": "2023-01-01T10:00:00+00:00",
        "posting_url": "https://social.invalid/P_1",
        "posting_text": "a",
    }
    assert img.images[0].extra_metadata == {
        **link_ids,
        "source_type": "social_media",
        **posting_ref,
        "reference_metadata": {"type": "image", **link_ids, **posting_ref},
    }

    keyframe_extra = img.keyframe_calls[0]["extra_metadata"]
    assert keyframe_extra["posting_network"] == "facebook"
    assert keyframe_extra["posting_url"] == "https://social.invalid/P_2"
    assert keyframe_extra["reference_metadata"]["type"] == "keyframe"
    assert keyframe_extra["reference_metadata"]["posting_uuid"] == "u2"

    segment_ref = _transcripts(result)[0].metadata["reference_metadata"]
    assert segment_ref["network"] == "nextext"
    assert segment_ref["type"] == "transcript_segment"
    assert segment_ref["posting_uuid"] == "u2"
    assert segment_ref["posting_author"] == "Jane Poster"
    assert segment_ref["posting_text"] == "b"


def test_postings_become_table_rows_carrying_their_reference_metadata(tmp_path: Path) -> None:
    """A posting is one row-shaped Document, so every social reader downstream takes it unchanged.

    ``source: "table"`` is load-bearing: node routing, social detection, the
    collection profile and extract posting units all key on it.
    """
    _write_export(tmp_path)
    result = _linker().run(tmp_path)

    posting = _rows(result, "postings")[0]
    assert posting.text == "a"
    assert posting.metadata["reference_metadata"] == {
        "network": "facebook",
        "type": "posting",
        "uuid": "u1",
        "url": "https://social.invalid/P_1",
        "timestamp": "2023-01-01T10:00:00+00:00",
        "author": "Jane Poster",
        "author_id": "42",
        "vanity": "jane.poster",
        "text": "a",
        "text_id": "P_1",
        "anchor_text": None,
        "parent_text": None,
    }
    assert posting.metadata["source"] == "table"
    assert posting.metadata["table"] == {"style": "postings", "row_index": 0, "n_rows": 2}
    assert posting.metadata["file_name"] == f"{_FB}/dossier.json"
    assert posting.metadata["file_hash"] == compute_file_hash(tmp_path / _FB / "dossier.json")
    assert is_social_payload(posting.metadata)


def test_comments_become_rows_naming_their_posting_and_parent(tmp_path: Path) -> None:
    """A received comment is a comment row: the post it answers and the comment it replies to ride along."""
    comments = [
        _comment("c1", "C_1", "first", posting="u1"),
        _comment("c2", "C_2", "a reply", posting="u1", parent="c1"),
        _comment("c3", "C_3", None, posting="u1"),
    ]
    _write_dossier(tmp_path, _FB, postings=[_posting("u1", "P_1", "post body", comments=comments)])

    result = _linker().run(tmp_path)

    rows = _rows(result, "comments")
    assert [row.text for row in rows] == ["first", "a reply"]
    assert rows[1].metadata["reference_metadata"] == {
        "network": "facebook",
        "type": "comment",
        "uuid": "c2",
        "url": None,
        "timestamp": "2023-01-01T11:00:00+00:00",
        "author": "Joe Commenter",
        "author_id": "77",
        "vanity": "joe.commenter",
        "text": "a reply",
        "text_id": "C_2",
        "anchor_text": "post body",
        "parent_text": "first",
    }
    assert rows[1].metadata["table"] == {"style": "comments", "row_index": 2, "n_rows": 4}
    assert is_social_payload(rows[1].metadata)


def test_a_posting_without_text_gets_no_row_but_its_media_still_link(tmp_path: Path) -> None:
    """An empty text makes no node, as an empty table row never did; its photo keeps the posting's identity."""
    _write_dossier(
        tmp_path,
        _FB,
        postings=[_posting("u1", "P_1", "  ", media_ids=["m1"])],
        media=[_media("m1", "M_1", f"{_FB}/media/photos/pic.jpg")],
        files={"media/photos/pic.jpg": b"\xff\xd8\xff"},
    )
    img = _FakeImageService()

    result = _linker(image_service=img).run(tmp_path)

    assert _rows(result, "postings") == []
    assert _linked(img) == {"pic": "u1"}


def test_each_profile_names_its_rows_by_its_own_folder(tmp_path: Path) -> None:
    """Every profile's file is called dossier.json; the documents listing keys by name, so rows must differ."""
    _write_dossier(tmp_path, "jane.poster - facebook", postings=[_posting("u1", "P_1", "a")])
    _write_dossier(tmp_path, "jane.poster - instagram", postings=[_posting("u2", "P_2", "b", network="instagram")])

    result = _linker().run(tmp_path)

    assert sorted(row.metadata["file_name"] for row in _rows(result, "postings")) == [
        "jane.poster - facebook/dossier.json",
        "jane.poster - instagram/dossier.json",
    ]


def test_a_media_item_two_postings_name_is_linked_to_both(tmp_path: Path) -> None:
    """The export may attach one file to several posts; each gets the link (the image service dedupes)."""
    _write_dossier(
        tmp_path,
        _FB,
        postings=[_posting("u1", "P_1", "a", media_ids=["m1"]), _posting("u2", "P_2", "b")],
        media=[_media("m1", "M_1", f"{_FB}/media/photos/pic.jpg", attached_to=["u2"])],
        files={"media/photos/pic.jpg": b"\xff\xd8\xff"},
    )
    img = _FakeImageService()

    _linker(image_service=img).run(tmp_path)

    assert sorted(asset.source_doc_id for asset in img.images) == ["u1", "u2"]


def test_media_the_export_or_the_upload_lacks_is_counted_and_left_unclaimed(
    tmp_path: Path, loguru_caplog_info: LogCaptureFixture
) -> None:
    """A media item the crawler never fetched, one the upload left out, and one whose name is ambiguous."""
    _write_dossier(
        tmp_path,
        _FB,
        postings=[_posting("u1", "P_1", "a", media_ids=["m1", "m2", "m3"])],
        media=[
            _media("m1", "M_1", None, status="missing"),
            _media("m2", "M_2", f"{_FB}/media/photos/not-uploaded.jpg"),
            _media("m3", "M_3", f"{_FB}/media/photos/twice.jpg"),
        ],
        files={"media/photos/twice.jpg": b"\xff\xd8\xff", "media/copy/twice.jpg": b"\xff\xd8\xff"},
    )
    img = _FakeImageService()

    result = _linker(image_service=img).run(tmp_path)

    assert img.images == []
    assert "twice.jpg" not in {path.name for path in result.consumed_paths}
    log = loguru_caplog_info.text
    assert "1 missing from the export" in log
    assert "1 with no local file" in log
    assert "1 with an ambiguous filename" in log


def test_a_media_path_cannot_reach_outside_its_profile_folder(tmp_path: Path) -> None:
    """Only the basename is looked up, and only inside the dossier's own folder."""
    outside = tmp_path / "secret.jpg"
    outside.write_bytes(b"\xff\xd8\xff")
    _write_dossier(
        tmp_path,
        _FB,
        postings=[_posting("u1", "P_1", "a", media_ids=["m1", "m2", "m3"])],
        media=[
            _media("m1", "M_1", "../secret.jpg"),
            _media("m2", "M_2", str(outside.resolve())),
            _media("m3", "M_3", f"{_FB}/media/photos/inside.jpg"),
        ],
        files={"media/photos/inside.jpg": b"\xff\xd8\xff"},
    )
    img = _FakeImageService()

    result = _linker(image_service=img).run(tmp_path)

    assert _linked(img) == {"inside": "u1"}
    assert outside not in result.consumed_paths


#: A Telegram channel's own id; its postings and photos are numbered ``<channel><message>``.
_CHANNEL = "100200"
_TG = "jane.channel - telegram"


def _tg_posting(uuid: str, message: int, text: str, published_at: str) -> dict[str, Any]:
    """Return an invented Telegram posting filed under message number *message*."""
    return _posting(
        uuid,
        f"{_CHANNEL}{message}",
        text,
        published_at=published_at,
        network="telegram",
        author=_author("Jane Channel", "jane.channel", _CHANNEL),
    )


def _tg_photo(uuid: str, message: int, published_at: str, *, attached_to: Sequence[str] = ()) -> dict[str, Any]:
    """Return an invented Telegram photo carrying its own message number."""
    return _media(
        uuid,
        f"{_CHANNEL}{message}",
        f"{_TG}/media/photos/{uuid}.jpg",
        published_at=published_at,
        attached_to=attached_to,
    )


def _write_album_export(root: Path) -> None:
    """Write a channel whose photos the export links for one of six.

    - messages 5-7: a three-photo album; the text sits on the last message, 7;
    - message 8: explicitly attached to the later post 12, though by message
      order and time alone it would read as part of post 9;
    - message 9: a single-photo post — the photo carries the post's own id;
    - message 10: an hour off the next posting at or above it (12);
    - message 20: no posting at or above it — the owner is absent from the export.

    Args:
        root: The export root.
    """
    photos = [
        _tg_photo("m5", 5, "2026-03-04T21:30:55+01:00"),
        _tg_photo("m6", 6, "2026-03-04T21:30:56+01:00"),
        _tg_photo("m7", 7, "2026-03-04T21:30:56+01:00"),
        _tg_photo("m8", 8, "2026-03-05T08:00:00+01:00", attached_to=["u12"]),
        _tg_photo("m9", 9, "2026-03-05T08:00:00+01:00"),
        _tg_photo("m10", 10, "2026-03-05T09:00:00+01:00"),
        _tg_photo("m20", 20, "2026-03-06T08:00:00+01:00"),
    ]
    _write_dossier(
        root,
        _TG,
        postings=[
            _tg_posting("u7", 7, "album text", "2026-03-04T21:30:56+01:00"),
            _tg_posting("u9", 9, "single photo post", "2026-03-05T08:00:00+01:00"),
            _tg_posting("u12", 12, "later post", "2026-03-05T10:00:00+01:00"),
        ],
        media=photos,
        files={f"media/photos/{photo['id']}.jpg": photo["id"].encode() for photo in photos},
    )


def test_album_members_link_to_the_text_post_filed_under_the_last_message(
    tmp_path: Path, loguru_caplog_info: LogCaptureFixture
) -> None:
    """Telegram photos carry only their own message id; the album rule is what joins them.

    An explicit link always wins over the rule, and a photo whose timestamp
    disagrees, or whose owner is absent, is left for the standalone path
    rather than attributed to a neighbouring post.
    """
    _write_album_export(tmp_path)
    img = _FakeImageService()

    result = _linker(image_service=img).run(tmp_path)

    assert _linked(img) == {"m5": "u7", "m6": "u7", "m7": "u7", "m8": "u12", "m9": "u9"}
    consumed_names = {path.name for path in result.consumed_paths}
    assert "m10.jpg" not in consumed_names
    assert "m20.jpg" not in consumed_names
    assert "5 media linked (1 explicitly, 4 by album inference)" in loguru_caplog_info.text
    assert "2 with no posting" in loguru_caplog_info.text


def test_album_rule_can_be_switched_off(tmp_path: Path) -> None:
    """``SOCIAL_ALBUM_LINK_ENABLED=false`` keeps only the links the export itself declares."""
    _write_album_export(tmp_path)
    img = _FakeImageService()

    result = _linker(image_service=img, album_link_enabled=False).run(tmp_path)

    assert _linked(img) == {"m8": "u12"}
    assert "m9.jpg" not in {path.name for path in result.consumed_paths}


def test_a_photo_naming_an_absent_posting_is_not_reattached_by_the_album_rule(tmp_path: Path) -> None:
    """The export's own claim outranks the rule even when the post it names was not exported.

    The photo would otherwise album-link to the next post along, which the
    export says is not its owner.
    """
    photo = _tg_photo("m6", 6, "2026-03-04T21:30:56+01:00", attached_to=["not-exported"])
    _write_dossier(
        tmp_path,
        _TG,
        postings=[_tg_posting("u7", 7, "album text", "2026-03-04T21:30:56+01:00")],
        media=[photo],
        files={"media/photos/m6.jpg": b"\xff\xd8\xff"},
    )
    img = _FakeImageService()

    _linker(image_service=img).run(tmp_path)

    assert img.images == []


def test_a_dossier_saved_with_a_byte_order_mark_is_read(tmp_path: Path) -> None:
    """Exporters write a UTF-8 BOM (the export's own CSVs carry one); it must not make a dossier unreadable."""
    path = _write_dossier(tmp_path, _FB, postings=[_posting("u1", "P_1", "a")])
    path.write_bytes(b"\xef\xbb\xbf" + path.read_bytes())

    result = _linker().run(tmp_path)

    assert [row.text for row in _rows(result, "postings")] == ["a"]


def test_album_index_ignores_ids_that_do_not_decompose_by_channel() -> None:
    """Instagram's ``<post>_<account>`` and Facebook's opaque ids never start with the author's id."""
    postings = [
        _posting("u1", "3000000000000000001_42", "ig", network="instagram"),
        _posting("u2", "UzpfSVNDOjAwMDAwMDAwMDE=", "fb"),
    ]

    assert build_posting_album_index(postings) == {}


def test_bookkeeping_outside_the_batch_is_never_claimed(tmp_path: Path) -> None:
    """A profile folder uploaded on its own has no export root inside the batch to tidy."""
    _write_dossier(tmp_path, _FB, postings=[_posting("u1", "P_1", "a")])
    (tmp_path / "_run.json").write_text("{}", encoding="utf-8")

    result = _linker().run(tmp_path / _FB)

    assert {path.name for path in result.consumed_paths} == {"dossier.json"}


def test_an_unreadable_or_foreign_dossier_is_claimed_and_skipped(
    tmp_path: Path, loguru_caplog: LogCaptureFixture
) -> None:
    """Neither reaches the generic JSON reader, and neither costs the rest of the export."""
    (tmp_path / "broken").mkdir()
    (tmp_path / "broken" / "dossier.json").write_text("{not json", encoding="utf-8")
    _write_dossier(tmp_path, "future", postings=[_posting("u9", "P_9", "z")], schema="me-dossier/2")
    _write_dossier(tmp_path, _FB, postings=[_posting("u1", "P_1", "a")])

    result = _linker().run(tmp_path)

    assert [row.metadata["reference_metadata"]["uuid"] for row in _rows(result, "postings")] == ["u1"]
    assert {path.parent.name for path in result.consumed_paths if path.name == "dossier.json"} == {
        "broken",
        "future",
        _FB,
    }
    assert "broken/dossier.json" in loguru_caplog.text
    assert "me-dossier/2" in loguru_caplog.text


def test_sections_without_a_sample_are_reported_not_dropped_silently(
    tmp_path: Path, loguru_caplog: LogCaptureFixture
) -> None:
    """Messages and own comments have no known shape yet, so they are skipped out loud."""
    _write_dossier(tmp_path, _FB, messages=[{"id": "x"}], ownComments=[{"id": "y"}, {"id": "z"}])

    _linker().run(tmp_path)

    assert "1 messages and 2 own comments are not ingested" in loguru_caplog.text


class _FailingImageService(_FakeImageService):
    """Image service whose every store fails, as with the embedding endpoint down."""

    @override
    def ingest_image(self, asset: Any, *, context: IngestContext) -> Any:
        """Fail the store.

        Args:
            asset: The image asset (ignored).
            context: Ingestion context (ignored).

        Raises:
            RuntimeError: Always.
        """
        raise RuntimeError("embedding endpoint down")


def test_a_routing_failure_keeps_the_postings(tmp_path: Path, loguru_caplog: LogCaptureFixture) -> None:
    """Media routing failing must not cost the text: the rows stay, the media go to the standalone passes."""
    _write_export(tmp_path)

    result = _linker(image_service=_FailingImageService()).run(tmp_path)

    assert [row.text for row in _rows(result, "postings")] == ["a", "b"]
    consumed_names = {path.name for path in result.consumed_paths}
    assert "dossier.json" in consumed_names
    assert not {"pic.jpg", "clip.mp4"} & consumed_names
    assert "embedding endpoint down" in loguru_caplog.text


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
    assert any(d.metadata.get("posting_uuid") == "u2" for d in _transcripts(result))


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


def test_derived_artifacts_name_the_media_file_they_came_from(tmp_path: Path) -> None:
    """A keyframe and a transcript segment must name the clip, not the transient JSONL.

    Without this the only identity a social video artifact carries is a posting
    UUID and a network media id, so neither an extract nor a report can say
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

    segment = _transcripts(result)[0].metadata
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

    segment = _transcripts(result)[0].metadata
    assert segment["media_file_hash"]
    assert segment["file_hash"] != segment["media_file_hash"]


def test_linked_images_go_through_the_pool_after_joining_their_loose_task(tmp_path: Path, recording_pool: Any) -> None:
    """Each linked image is a pool task keyed by posting that first waits for the file's loose task.

    The pool may already be storing the same file loose (the upload handler
    submitted it before the manifest arrived); the social call must run after
    that, not instead of it, so the loose point is relinked rather than lost.
    """
    _write_export(tmp_path)
    service = _FakeImageService()

    SocialLinker(
        image_service=service, nextext_client=_CountingNextext(), target_collection="c", pool=recording_pool
    ).run(tmp_path)

    assert service.images, "the export's image was linked"
    link_keys = [key for key in recording_pool.keys if key.startswith("image-link#c#")]
    assert len(link_keys) == len(service.images)  # submitted once per image, joined by future
    # kind#collection#hash:posting — the posting rides inside the last
    # component, which is why the key's own separator has to be one no
    # collection name can contain.
    assert all(collection_of_key(key) == "c" for key in link_keys)
    assert len([j for j in recording_pool.joins if j.startswith("image#c#")]) == len(service.images)


def test_linked_images_report_their_own_progress(tmp_path: Path, recording_pool: Any) -> None:
    """An export of a few thousand images spent this whole pass saying nothing."""
    _write_export(tmp_path)
    reported: list[str] = []

    SocialLinker(
        image_service=_FakeImageService(),
        nextext_client=_CountingNextext(),
        target_collection="c",
        pool=recording_pool,
        progress_callback=reported.append,
    ).run(tmp_path)

    linked = [line for line in reported if line.startswith("Linking images:")]
    assert linked and linked[-1].endswith(f"{len(linked)}/{len(linked)} images linked")


def test_a_linked_clips_keyframes_reach_the_pools_tally(tmp_path: Path) -> None:
    """A social export's clips run through the same pool, so their frames report the same way."""
    from docint.core.ingest.preprocess import PreprocessPool, StageProgress

    _write_export(tmp_path)
    pool = PreprocessPool(max_workers=1)
    linker = SocialLinker(
        image_service=_FakeImageService(), nextext_client=_FakeNextext(), target_collection="c", pool=pool
    )

    try:
        linker.run(tmp_path)

        assert StageProgress("keyframes", 1, 1, 0) in pool.progress.snapshot("c")
    finally:
        pool.shutdown()
