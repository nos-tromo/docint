from __future__ import annotations

import time
import uuid
from typing import Any, Callable, TypeVar

import httpx
from llama_index.core.storage.kvstore.types import DEFAULT_COLLECTION, BaseKVStore
from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException
from qdrant_client.http import models as rest


T = TypeVar("T")


def _generate_id(key: str, collection: str) -> str:
    """Generate a deterministic UUID from the key and collection.

    Args:
        key (str): The key.
        collection (str): The collection name.

    Returns:
        str: The generated UUID as a string.
    """
    namespace = uuid.uuid5(uuid.NAMESPACE_DNS, collection)
    return str(uuid.uuid5(namespace, key))


class QdrantKVStore(BaseKVStore):
    """A Key-Value Store implementation backed by Qdrant.
    It uses a dedicated Qdrant collection to store key-value pairs as points.
    """

    def __init__(
        self,
        client: QdrantClient,
        collection_name: str,
        batch_size: int = 100,
        max_retries: int = 3,
        retry_backoff_seconds: float = 0.25,
        retry_backoff_max_seconds: float = 2.0,
    ) -> None:
        """Initialize the QdrantKVStore.

        Args:
            client (QdrantClient): The Qdrant client instance.
            collection_name (str): The name of the Qdrant collection to use.
            batch_size (int, optional): The batch size for upsert operations. Defaults to 100.
            max_retries (int, optional): Number of retries for transient transport
                failures. Defaults to 3.
            retry_backoff_seconds (float, optional): Initial backoff in seconds between
                retries. Defaults to 0.25.
            retry_backoff_max_seconds (float, optional): Maximum backoff in seconds.
                Defaults to 2.0.
        """
        self.client = client
        self.collection_name = collection_name
        self.batch_size = batch_size
        self.max_retries = max(0, max_retries)
        self.retry_backoff_seconds = max(0.0, retry_backoff_seconds)
        self.retry_backoff_max_seconds = max(0.0, retry_backoff_max_seconds)

        # Create collection if it doesn't exist
        if not self.client.collection_exists(self.collection_name):
            logger.info(
                "Creating Qdrant collection for KV Store: {}", self.collection_name
            )
            self.client.create_collection(
                collection_name=self.collection_name,
                # Dummy vector config - required by Qdrant
                vectors_config=rest.VectorParams(
                    size=1,
                    distance=rest.Distance.DOT,
                ),
            )

    @staticmethod
    def _is_retryable_exception(exc: Exception) -> bool:
        """Return True when *exc* looks like a transient transport failure.

        Uses the ``qdrant_client`` and ``httpx`` exception hierarchies to
        classify errors structurally rather than by parsing message strings.

        Args:
            exc: Exception raised by Qdrant/http transports.

        Returns:
            bool: Whether this exception should be retried.
        """
        if isinstance(exc, (ConnectionError, ConnectionResetError, TimeoutError)):
            return True
        if isinstance(exc, (httpx.NetworkError, httpx.TimeoutException)):
            return True
        if isinstance(exc, ResponseHandlingException):
            cause = exc.__cause__
            if cause is not None:
                return QdrantKVStore._is_retryable_exception(cause)
            return True
        return False

    def _execute_with_retry(self, operation: str, fn: Callable[[], T]) -> T:
        """Execute *fn* with bounded retries for transient Qdrant errors.

        Args:
            operation: Human-readable operation name for logs.
            fn: Callable that performs the underlying operation.

        Returns:
            T: Result from *fn*.

        Raises:
            Exception: Re-raises the last exception on exhaustion or non-retryable
                failures.
        """
        attempts = max(1, self.max_retries + 1)
        attempt = 1
        last_exc: Exception | None = None
        while attempt <= attempts:
            try:
                return fn()
            except Exception as exc:
                last_exc = exc
                if not self._is_retryable_exception(exc) or attempt >= attempts:
                    raise
                delay = min(
                    self.retry_backoff_seconds * (2 ** (attempt - 1)),
                    self.retry_backoff_max_seconds,
                )
                logger.warning(
                    "Qdrant KV operation '{}' failed on attempt {}/{} with retryable "
                    "error: {}. Retrying in {:.2f}s",
                    operation,
                    attempt,
                    attempts,
                    exc,
                    delay,
                )
                if delay > 0:
                    time.sleep(delay)
                attempt += 1

        if last_exc is not None:
            raise last_exc
        raise RuntimeError("Retry loop exited unexpectedly")

    def put(
        self,
        key: str,
        val: dict,
        collection: str = DEFAULT_COLLECTION,
    ) -> None:
        """Store a key-value pair in the Qdrant collection.

        Args:
            key (str): The key.
            val (dict): The value to store.
            collection (str, optional): The collection name. Defaults to DEFAULT_COLLECTION.
        """
        logger.debug("Putting single item into collection '{}'", collection)
        point_id = _generate_id(key, collection)
        payload = {
            "key": key,
            "collection": collection,
            "val": val,
        }
        self._execute_with_retry(
            "put",
            lambda: self.client.upsert(
                collection_name=self.collection_name,
                points=[
                    rest.PointStruct(
                        id=point_id,
                        vector=[0.0],
                        payload=payload,
                    )
                ],
                wait=True,
            ),
        )

    def put_all(
        self,
        kv_pairs: list[tuple[str, dict[str, Any]]],
        collection: str = DEFAULT_COLLECTION,
        batch_size: int | None = None,
    ) -> None:
        """Store multiple key-value pairs in the Qdrant collection.

        Args:
            kv_pairs (list[tuple[str, dict[str, Any]]]): list of key-value pairs.
            collection (str, optional): The collection name. Defaults to DEFAULT_COLLECTION.
            batch_size (int, optional): The batch size for upsert. Defaults to stored batch_size.
        """
        batch_size = batch_size or self.batch_size
        points = []
        for key, val in kv_pairs:
            point_id = _generate_id(key, collection)
            payload = {
                "key": key,
                "collection": collection,
                "val": val,
            }
            points.append(
                rest.PointStruct(
                    id=point_id,
                    vector=[0.0],
                    payload=payload,
                )
            )

        # Batch upsert
        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            logger.debug(
                "Upserting batch {} of size {}", i // batch_size + 1, len(batch)
            )

            def _upsert_batch() -> Any:
                return self.client.upsert(
                    collection_name=self.collection_name,
                    points=batch,
                    wait=True,
                )

            self._execute_with_retry(
                "put_all.batch_upsert",
                _upsert_batch,
            )

    def get(
        self,
        key: str,
        collection: str = DEFAULT_COLLECTION,
    ) -> dict | None:
        """Retrieve a value by key from the Qdrant collection.

        Args:
            key (str): The key.
            collection (str, optional): The collection name. Defaults to DEFAULT_COLLECTION.

        Returns:
            dict | None: The retrieved value or None if not found.
        """
        point_id = _generate_id(key, collection)
        res = self._execute_with_retry(
            "get",
            lambda: self.client.retrieve(
                collection_name=self.collection_name,
                ids=[point_id],
                with_payload=True,
            ),
        )
        if res and res[0].payload:
            val = res[0].payload.get("val")
            return val if isinstance(val, dict) else None
        return None

    def get_all(self, collection: str = DEFAULT_COLLECTION) -> dict[str, Any]:
        """Retrieve all key-value pairs from the specified collection.

        Args:
            collection (str, optional): The collection name. Defaults to DEFAULT_COLLECTION.

        Returns:
            dict[str, Any]: All key-value pairs in the collection.
        """
        # Qdrant scroll
        data = {}
        offset = None
        filter_ = rest.Filter(
            must=[
                rest.FieldCondition(
                    key="collection",
                    match=rest.MatchValue(value=collection),
                )
            ]
        )

        while True:

            def _scroll_once() -> Any:
                return self.client.scroll(
                    collection_name=self.collection_name,
                    scroll_filter=filter_,
                    limit=self.batch_size,
                    offset=offset,
                    with_payload=True,
                )

            points, offset = self._execute_with_retry(
                "get_all.scroll",
                _scroll_once,
            )
            for point in points:
                if point.payload:
                    key = point.payload.get("key")
                    val = point.payload.get("val")
                    if key and val:
                        data[key] = val

            if offset is None:
                break
        return data

    def delete(self, key: str, collection: str = DEFAULT_COLLECTION) -> bool:
        """Delete a key-value pair from the Qdrant collection.

        Args:
            key (str): The key.
            collection (str, optional): The collection name. Defaults to DEFAULT_COLLECTION.

        Returns:
            bool: True if the key-value pair was deleted successfully.
        """
        point_id = _generate_id(key, collection)
        self._execute_with_retry(
            "delete",
            lambda: self.client.delete(
                collection_name=self.collection_name,
                points_selector=[point_id],
            ),
        )
        return True

    async def aput(
        self,
        key: str,
        val: dict,
        collection: str = DEFAULT_COLLECTION,
    ) -> None:
        """Store a key-value pair in the Qdrant collection.

        Args:
            key (str): The key.
            val (dict): The value.
            collection (str, optional): The collection name. Defaults to DEFAULT_COLLECTION.
        """
        return self.put(key, val, collection)

    async def aput_all(
        self,
        kv_pairs: list[tuple[str, dict[str, Any]]],
        collection: str = DEFAULT_COLLECTION,
        batch_size: int | None = None,
    ) -> None:
        """Store multiple key-value pairs in the Qdrant collection.

        Args:
            kv_pairs (list[tuple[str, dict[str, Any]]]): The key-value pairs to store.
            collection (str, optional): The collection name. Defaults to DEFAULT_COLLECTION.
            batch_size (int, optional): The batch size. Defaults to stored batch_size.
        """
        return self.put_all(kv_pairs, collection, batch_size)

    async def aget(
        self,
        key: str,
        collection: str = DEFAULT_COLLECTION,
    ) -> dict | None:
        return self.get(key, collection)

    async def aget_all(self, collection: str = DEFAULT_COLLECTION) -> dict[str, Any]:
        return self.get_all(collection)

    async def adelete(self, key: str, collection: str = DEFAULT_COLLECTION) -> bool:
        return self.delete(key, collection)
