"""HTTP client for the remote CLIP image+text embedding service.

Mirrors :mod:`docint.utils.ner_client`. This replaces the in-process
``CLIPImageEmbeddingBackend`` previously defined in
``docint.core.ingest.images_service``. The full vllm-service stack
exposes the endpoints as router pass-throughs:

* ``http://vllm-router:4000/clip/embed_image``
* ``http://vllm-router:4000/clip/embed_text``
* ``http://vllm-router:4000/clip/dimension``

The standalone ``clip-only`` deployment shape (CPU, pairs with the
``ner-only`` and ``rerank-only`` profiles for non-CUDA dev hosts)
exposes the same endpoints directly at ``http://clip-embed:8000/clip/*``
with no Bearer auth.

The factory's return shape matches the
:class:`docint.core.ingest.images_service.ImageEmbeddingBackend`
Protocol (``embed``, ``embed_text``, ``dimension``) so call sites in
the ingestion pipeline swap in place without further changes.
"""

from __future__ import annotations

import base64
from typing import cast

import httpx
from loguru import logger

from docint.utils.env_cfg import CLIPClientConfig, load_clip_client_env


def _build_client(cfg: CLIPClientConfig) -> httpx.Client:
    """Construct the shared ``httpx.Client`` used for CLIP calls."""
    headers: dict[str, str] = {}
    if cfg.api_key:
        headers["Authorization"] = f"Bearer {cfg.api_key}"
    return httpx.Client(
        base_url=cfg.api_base,
        timeout=cfg.timeout,
        headers=headers,
    )


class RemoteCLIPBackend:
    """Adapter that satisfies ``ImageEmbeddingBackend`` over HTTP.

    Attributes match the Protocol exactly: ``dimension`` is fetched once
    at construction time via ``GET /clip/dimension`` (cheap probe — no
    embed cost) so call sites can size Qdrant collections without an
    extra round trip. ``embed`` and ``embed_text`` issue per-call POSTs.

    Failure semantics mirror the legacy in-process backend: any HTTP /
    transport / payload error raises so the caller's
    ``fail_on_embedding_error`` toggle can decide whether to skip the
    asset or abort the ingestion run.
    """

    def __init__(self, cfg: CLIPClientConfig | None = None) -> None:
        """Create the client and probe the remote model's dimension.

        Args:
            cfg: Override client configuration. When ``None``, reads
                from the environment via
                :func:`docint.utils.env_cfg.load_clip_client_env`.

        Raises:
            httpx.HTTPError: If the dimension probe fails. Caller
                decides whether to swallow via
                ``fail_on_embedding_error``.
        """
        effective_cfg = cfg if cfg is not None else load_clip_client_env()
        self._cfg = effective_cfg
        self._client = _build_client(effective_cfg)
        response = self._client.get("/clip/dimension")
        response.raise_for_status()
        self._dimension = int(response.json()["dimension"])
        logger.info(
            "Remote CLIP backend ready: api_base={} auth={} dimension={}",
            effective_cfg.api_base,
            "bearer" if effective_cfg.api_key else "none",
            self._dimension,
        )

    @property
    def dimension(self) -> int:
        """Return the CLIP projection dimensionality probed at construction."""
        return self._dimension

    def embed(self, image_bytes: bytes) -> list[float]:
        """Embed raw image bytes via the remote CLIP image tower.

        Args:
            image_bytes: Raw bytes of the image to embed (any format
                Pillow can decode on the server side).

        Returns:
            L2-normalized embedding vector of length :attr:`dimension`.

        Raises:
            httpx.HTTPError: Network, timeout, or non-2xx response.
            ValueError: Server returned a malformed payload.
        """
        payload = {"image_b64": base64.b64encode(image_bytes).decode("ascii")}
        response = self._client.post("/clip/embed_image", json=payload)
        response.raise_for_status()
        data = response.json()
        embedding = data.get("embedding") if isinstance(data, dict) else None
        if not isinstance(embedding, list):
            raise ValueError("remote /clip/embed_image returned no embedding")
        return cast(list[float], [float(x) for x in embedding])

    def embed_text(self, text: str) -> list[float]:
        """Embed a text query via the remote CLIP text tower.

        Args:
            text: The query string.

        Returns:
            L2-normalized embedding vector of length :attr:`dimension`.

        Raises:
            httpx.HTTPError: Network, timeout, or non-2xx response.
            ValueError: Server returned a malformed payload.
        """
        response = self._client.post("/clip/embed_text", json={"text": text})
        response.raise_for_status()
        data = response.json()
        embedding = data.get("embedding") if isinstance(data, dict) else None
        if not isinstance(embedding, list):
            raise ValueError("remote /clip/embed_text returned no embedding")
        return cast(list[float], [float(x) for x in embedding])
