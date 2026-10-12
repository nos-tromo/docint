"""Tests for ``/sources/thumbnail``: the picture behind an image finding.

A finding read off a video keyframe or a PDF figure names, as its file, the
clip or the document it was cut from, so ``/sources/preview`` serves an .mp4
or a PDF that no ``<img>`` can show. The thumbnail route addresses the image
itself, by its ``image_id``, and answers with the stored thumbnail on the
``_images`` companion — the one copy of a keyframe's pixels that exists.
"""

import base64
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient, models

import docint.core.api as api_module

_JPEG = b"\xff\xd8\xff\xe0 stored thumbnail bytes"
_PNG = b"\x89PNG\r\n\x1a\n original image bytes"


class _Rag:
    """The two seams the route reads: the client and the companion's name."""

    def __init__(self, client: QdrantClient) -> None:
        """Wrap *client*.

        Args:
            client (QdrantClient): The in-memory client.
        """
        self.qdrant_client = client

    def _image_collection_name(self, collection: str | None = None) -> str:
        """Name the ``_images`` companion the way the engine does.

        Args:
            collection (str | None): The physical collection.

        Returns:
            str: The companion's name.
        """
        return f"{collection}_images"


def _store(client: QdrantClient, collection: str, points: dict[int, dict[str, Any]]) -> None:
    """Create *collection* and upsert *points* into it.

    Args:
        client (QdrantClient): The in-memory client.
        collection (str): Collection name.
        points (dict[int, dict[str, Any]]): ``{point_id: payload}``.
    """
    client.create_collection(collection, vectors_config=models.VectorParams(size=2, distance=models.Distance.COSINE))
    if points:
        client.upsert(
            collection,
            points=[models.PointStruct(id=pid, vector=[0.1, 0.2], payload=payload) for pid, payload in points.items()],
        )


@pytest.fixture
def qdrant(monkeypatch: pytest.MonkeyPatch) -> QdrantClient:
    """An in-memory store holding an empty main collection ``docs``, wired into the API.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.

    Returns:
        QdrantClient: The client, for tests to add the companion to.
    """
    client = QdrantClient(location=":memory:")
    _store(client, "docs", {})
    monkeypatch.setattr(api_module, "rag", _Rag(client))
    monkeypatch.setattr(api_module, "_require_owned_collection", lambda collection, principal: collection)
    return client


def _thumbnail(image_id: str) -> Any:
    """Request the picture of *image_id* in collection ``docs``.

    Args:
        image_id (str): The image's content hash.

    Returns:
        Any: The HTTP response.
    """
    # No lifespan: the stand-in RAG carries none of the startup probes.
    return TestClient(api_module.app).get(
        "/sources/thumbnail",
        params={"collection": "docs", "image_id": image_id},
        headers={"X-Auth-User": "alice"},
    )


def test_a_keyframe_is_served_from_its_stored_thumbnail(qdrant: QdrantClient) -> None:
    """A keyframe's only pixels are its stored thumbnail; its clip is never asked for.

    Args:
        qdrant (QdrantClient): The wired in-memory store.
    """
    _store(
        qdrant,
        "docs_images",
        {
            1: {
                "image_id": "frame-1",
                "source_type": "video_keyframe",
                "media_file_hash": "clip-hash",
                "source_path": "batch/talk.mp4",
                "thumbnail_b64": base64.b64encode(_JPEG).decode("ascii"),
                "thumbnail_mime": "image/jpeg",
            }
        },
    )

    resp = _thumbnail("frame-1")

    assert resp.status_code == 200, resp.text
    assert resp.headers["content-type"] == "image/jpeg"
    assert resp.content == _JPEG


def test_a_stored_thumbnail_is_never_served_as_anything_but_an_image(qdrant: QdrantClient) -> None:
    """A thumbnail whose stored type is not an image's goes out as the JPEG it is.

    Args:
        qdrant (QdrantClient): The wired in-memory store.
    """
    _store(
        qdrant,
        "docs_images",
        {
            1: {
                "image_id": "img-1",
                "thumbnail_b64": base64.b64encode(_JPEG).decode("ascii"),
                "thumbnail_mime": "text/html",
            }
        },
    )

    resp = _thumbnail("img-1")

    assert resp.status_code == 200, resp.text
    assert resp.headers["content-type"] == "image/jpeg"


def test_an_image_stored_before_thumbnails_existed_falls_back_to_its_own_file(
    qdrant: QdrantClient, tmp_path: Path
) -> None:
    """A point written before thumbnails were stored is shown from the image file it was read from.

    Args:
        qdrant (QdrantClient): The wired in-memory store.
        tmp_path (Path): Pytest temp directory.
    """
    poster = tmp_path / "poster.png"
    poster.write_bytes(_PNG)
    _store(
        qdrant,
        "docs_images",
        {
            1: {
                "image_id": "still-1",
                "source_type": "social_media",
                "file_path": str(poster),
                "source_path": str(poster),
            }
        },
    )

    resp = _thumbnail("still-1")

    assert resp.status_code == 200, resp.text
    assert resp.headers["content-type"] == "image/png"
    assert resp.content == _PNG


@pytest.mark.parametrize("name", ["talk.mp4", "report.pdf"])
def test_the_fallback_never_serves_the_clip_or_document_an_image_was_cut_from(
    qdrant: QdrantClient, tmp_path: Path, name: str
) -> None:
    """Without a stored thumbnail, a keyframe or figure resolves only to its clip or document, which is refused.

    Args:
        qdrant (QdrantClient): The wired in-memory store.
        tmp_path (Path): Pytest temp directory.
        name (str): The source file the image was cut from.
    """
    source = tmp_path / name
    source.write_bytes(b"not an image")
    _store(qdrant, "docs_images", {1: {"image_id": "cut-1", "source_path": str(source)}})

    resp = _thumbnail("cut-1")

    assert resp.status_code == 404
    assert resp.json() == {"detail": "Image not found"}


def test_an_unreadable_thumbnail_without_a_file_answers_404(qdrant: QdrantClient) -> None:
    """A corrupt stored thumbnail is a missing picture, not a server error.

    Args:
        qdrant (QdrantClient): The wired in-memory store.
    """
    _store(qdrant, "docs_images", {1: {"image_id": "bad-1", "thumbnail_b64": "@@@not-base64"}})

    resp = _thumbnail("bad-1")

    assert resp.status_code == 404
    assert resp.json() == {"detail": "Image not found"}


def test_a_collection_without_an_image_companion_answers_404(qdrant: QdrantClient) -> None:
    """No ``_images`` companion means no picture, not a server error.

    Args:
        qdrant (QdrantClient): The wired in-memory store.
    """
    resp = _thumbnail("anything")

    assert resp.status_code == 404
    assert resp.json() == {"detail": "Image not found"}
