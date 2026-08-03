"""Tests that the source preview serves display-hostile text types as plain text.

``/sources/preview`` exists to *show* a source. Browsers refuse to display
``text/markdown`` and ``text/csv`` — they hand both to the download manager,
in a new tab and inside the preview dialog alike — so serving those types by
their registered mimetype turns every preview of an .md or .csv source into a
surprise download. Declaring them ``text/plain`` renders them as readable
text; the session-ZIP endpoint remains the download path.
"""

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

import docint.core.api as api_module


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    """TestClient with the collection gate and principal stubbed out.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.

    Yields:
        TestClient: Client whose preview requests reach the file-serving code.
    """
    monkeypatch.setattr(api_module, "_require_owned_collection", lambda collection, principal: collection)
    with TestClient(api_module.app) as test_client:
        yield test_client


def _preview(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    filename: str,
    body: str = "col_a,col_b\n1,2\n",
) -> Any:
    """Request a preview of a temp file with the given name.

    Args:
        client (TestClient): The API client.
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        tmp_path (Path): Pytest temp directory.
        filename (str): Name (and so extension) of the served file.
        body (str): File content.

    Returns:
        Any: The HTTP response.
    """
    src = tmp_path / filename
    src.write_text(body, encoding="utf-8")
    monkeypatch.setattr(api_module, "_resolve_source_file_path", lambda collection, file_hash, **_kw: src)
    return client.get(
        "/sources/preview",
        params={"collection": "docs", "file_hash": "h"},
        headers={"X-Auth-User": "alice"},
    )


@pytest.mark.parametrize("filename", ["notes.md", "table.csv", "table.CSV"])
def test_download_only_text_types_preview_as_plain_text(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, filename: str
) -> None:
    """Markdown and CSV render as readable text instead of downloading."""
    resp = _preview(client, monkeypatch, tmp_path, filename)

    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/plain")
    assert "charset=utf-8" in resp.headers["content-type"]


def test_pdf_previews_keep_their_real_media_type(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The browser's PDF viewer needs the real type; only text types are coerced."""
    resp = _preview(client, monkeypatch, tmp_path, "doc.pdf", body="%PDF-1.4")

    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/pdf"


def test_json_previews_keep_their_real_media_type(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Browsers display application/json natively; no coercion needed."""
    resp = _preview(client, monkeypatch, tmp_path, "data.json", body="{}")

    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/json"
