"""Shared fakes for the topics tests: in-memory Qdrant helpers and a scripted fit."""

from __future__ import annotations

import uuid
from collections import Counter
from collections.abc import Sequence
from typing import Any

import numpy as np
from qdrant_client import QdrantClient, models

from docint.utils.topics_client import FitParams, FitResult, FitTopic

DIM = 8
SPARSE_NAME = "text-sparse-new"
DENSE_NAME = "text-dense"


def make_collection(client: QdrantClient, name: str, *, named: bool = True, dim: int = DIM) -> None:
    """Create a collection shaped like a docint one.

    Args:
        client: Qdrant client.
        name: Collection name.
        named: Hybrid shape (named dense plus sparse) when true, else one unnamed vector.
        dim: Dense vector size.
    """
    if named:
        client.create_collection(
            name,
            vectors_config={DENSE_NAME: models.VectorParams(size=dim, distance=models.Distance.COSINE)},
            sparse_vectors_config={SPARSE_NAME: models.SparseVectorParams()},
        )
    else:
        client.create_collection(name, vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE))


def add_points(
    client: QdrantClient,
    name: str,
    rows: Sequence[tuple[str | int, Sequence[float] | np.ndarray, dict[str, Any]]],
    *,
    named: bool = True,
) -> None:
    """Upsert points.

    Args:
        client: Qdrant client.
        name: Collection name.
        rows: ``(id, vector, payload)`` triples.
        named: Whether the collection stores its dense vector under a name.
    """
    points = [
        models.PointStruct(
            id=pid,
            vector={DENSE_NAME: [float(x) for x in vec]} if named else [float(x) for x in vec],
            payload=payload,
        )
        for pid, vec, payload in rows
    ]
    client.upsert(name, points=points, wait=True)


def blob_vectors(seed: int, centres: np.ndarray, per: int, noise: float) -> np.ndarray:
    """Draw ``per`` noisy vectors around each centre.

    Args:
        seed: Random seed.
        centres: Centre matrix, one row per blob.
        per: Vectors per blob.
        noise: Gaussian noise scale.

    Returns:
        A ``(len(centres) * per, dim)`` float32 matrix, blob by blob.
    """
    rng = np.random.default_rng(seed)
    out = [c + rng.normal(0.0, noise, size=(per, centres.shape[1])) for c in centres]
    return np.vstack(out).astype(np.float32)


def uuid_ids(n: int) -> list[str]:
    """Return ``n`` deterministic UUID strings.

    Args:
        n: How many ids.

    Returns:
        The ids.
    """
    return [str(uuid.UUID(int=i + 1)) for i in range(n)]


class RecordingClient:
    """Delegates to a client and records the name of every method called."""

    def __init__(self, inner: Any) -> None:
        """Wrap a client.

        Args:
            inner: The client to delegate to.
        """
        self._inner = inner
        self.calls: list[str] = []

    def __getattr__(self, name: str) -> Any:
        """Delegate an attribute, recording callables when they are invoked.

        Args:
            name: Attribute name.

        Returns:
            The attribute, wrapped when callable.
        """
        attr = getattr(self._inner, name)
        if not callable(attr):
            return attr

        def _call(*args: Any, **kwargs: Any) -> Any:
            self.calls.append(name)
            return attr(*args, **kwargs)

        return _call


class FakeFit:
    """A scripted stand-in for the fit service call."""

    def __init__(self, centres: np.ndarray, *, radius: float = 0.5, raise_first: Exception | None = None) -> None:
        """Script the fit.

        Args:
            centres: One row per topic; a row is labelled by its nearest centre.
            radius: Largest ``1 - cos`` distance still assigned to a centre.
            raise_first: Exception raised by the first call only.
        """
        unit = centres / np.linalg.norm(centres, axis=1, keepdims=True)
        self.centres = unit.astype(np.float64)
        self.radius = radius
        self.raise_first = raise_first
        self.requests: list[tuple[list[str], np.ndarray, FitParams]] = []

    def __call__(self, texts: Sequence[str], vectors: np.ndarray, params: FitParams) -> FitResult:
        """Label the rows.

        Args:
            texts: One text per row.
            vectors: Row matrix.
            params: Fit settings.

        Returns:
            The scripted result.

        Raises:
            Exception: The scripted failure, once.
        """
        self.requests.append((list(texts), np.asarray(vectors), params))
        if self.raise_first is not None:
            exc, self.raise_first = self.raise_first, None
            raise exc
        x = np.asarray(vectors, dtype=np.float64)
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        sims = (x / np.where(norms == 0, 1.0, norms)) @ self.centres.T
        best = sims.argmax(axis=1)
        labels = [int(b) if 1.0 - sims[i, b] <= self.radius else -1 for i, b in enumerate(best)]
        topics = []
        for tid in sorted({lab for lab in labels if lab >= 0}):
            tokens: Counter[str] = Counter()
            for text, lab in zip(texts, labels, strict=True):
                if lab == tid:
                    tokens.update(text.lower().split())
            total = sum(tokens.values()) or 1
            kws = tuple((w, c / total) for w, c in tokens.most_common(3))
            topics.append(FitTopic(id=tid, size=labels.count(tid), keywords=kws))
        return FitResult(
            labels=tuple(labels),
            topics=tuple(topics),
            outliers=labels.count(-1),
            versions={"bertopic": "fake"},
            runtime={},
            timings={"fit_s": 0.0},
        )
