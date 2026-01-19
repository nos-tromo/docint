from __future__ import annotations

import uuid
from typing import Any

from llama_index.core.storage.kvstore.types import DEFAULT_COLLECTION, BaseKVStore
from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest


def _generate_id(key: str, collection: str) -> str:
    """
    Generate a deterministic UUID from the key and collection.

    Args:
        key (str): The key.
        collection (str): The collection name.

    Returns:
        str: The generated UUID as a string.
    """
    namespace = uuid.uuid5(uuid.NAMESPACE_DNS, collection)
    return str(uuid.uuid5(namespace, key))


class QdrantKVStore(BaseKVStore):
    """
    A Key-Value Store implementation backed by Qdrant.
    It uses a dedicated Qdrant collection to store key-value pairs as points.
    """

    def __init__(
        self,
        client: QdrantClient,
        collection_name: str,
        batch_size: int = 100,
    ) -> None:
        """
        Initialize the QdrantKVStore.

        Args:
            client (QdrantClient): The Qdrant client instance.
            collection_name (str): The name of the Qdrant collection to use.
            batch_size (int, optional): The batch size for upsert operations. Defaults to 100.
        """
        self.client = client
        self.collection_name = collection_name
        self.batch_size = batch_size

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

    def put(
        self,
        key: str,
        val: dict,
        collection: str = DEFAULT_COLLECTION,
    ) -> None:
        """
        Store a key-value pair in the Qdrant collection.

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
        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                rest.PointStruct(
                    id=point_id,
                    vector=[0.0],
                    payload=payload,
                )
            ],
            wait=True,
        )

    def put_all(
        self,
        kv_pairs: list[tuple[str, dict[str, Any]]],
        collection: str = DEFAULT_COLLECTION,
        batch_size: int | None = None,
    ) -> None:
        """
        Store multiple key-value pairs in the Qdrant collection.

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
            self.client.upsert(
                collection_name=self.collection_name,
                points=batch,
                wait=True,
            )

    def get(
        self,
        key: str,
        collection: str = DEFAULT_COLLECTION,
    ) -> dict | None:
        """
        Retrieve a value by key from the Qdrant collection.

        Args:
            key (str): The key.
            collection (str, optional): The collection name. Defaults to DEFAULT_COLLECTION.

        Returns:
            dict | None: The retrieved value or None if not found.
        """
        point_id = _generate_id(key, collection)
        res = self.client.retrieve(
            collection_name=self.collection_name,
            ids=[point_id],
            with_payload=True,
        )
        if res and res[0].payload:
            val = res[0].payload.get("val")
            return val if isinstance(val, dict) else None
        return None

    def get_all(self, collection: str = DEFAULT_COLLECTION) -> dict[str, Any]:
        """
        Retrieve all key-value pairs from the specified collection.

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
            points, offset = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=filter_,
                limit=self.batch_size,
                offset=offset,
                with_payload=True,
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
        """
        Delete a key-value pair from the Qdrant collection.

        Args:
            key (str): The key.
            collection (str, optional): The collection name. Defaults to DEFAULT_COLLECTION.

        Returns:
            bool: True if the key-value pair was deleted successfully.
        """
        point_id = _generate_id(key, collection)
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=[point_id],
        )
        return True

    async def aput(
        self,
        key: str,
        val: dict,
        collection: str = DEFAULT_COLLECTION,
    ) -> None:
        """
        Store a key-value pair in the Qdrant collection.

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
        """
        Store multiple key-value pairs in the Qdrant collection.

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
