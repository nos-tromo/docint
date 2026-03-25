"""Neo4j-backed graph ingestion, retrieval, and analysis services."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Iterable

from loguru import logger
from neo4j import GraphDatabase
from neo4j.exceptions import Neo4jError

from docint.core.graph.entity_resolution import (
    extract_domain,
    parse_tag_values,
    resolve_author_identity,
    resolve_entity_identity,
)
from docint.core.graph.models import (
    GraphCandidate,
    GraphPathResult,
    GraphSourceRecord,
    GraphTraversalResult,
)
from docint.core.graph.query_planner import GraphQueryPlan
from docint.utils.env_cfg import GraphStoreConfig


@dataclass(slots=True)
class Neo4jGraphService:
    """Thin graph service wrapper that centralizes Cypher access."""

    config: GraphStoreConfig
    _driver: Any = None

    @property
    def enabled(self) -> bool:
        """Return whether the graph service is enabled.

        Returns:
            bool: True if the graph service is enabled, False otherwise.
        """

        return bool(self.config.enabled)

    @property
    def driver(self) -> Any:
        """Lazily construct the Neo4j driver.

        Returns:
            Any: The Neo4j driver instance.
        """

        if self._driver is None:
            self._driver = GraphDatabase.driver(
                self.config.uri,
                auth=(self.config.username, self.config.password),
            )
        return self._driver

    def close(self) -> None:
        """Close the Neo4j driver if it has been opened."""

        if self._driver is not None:
            self._driver.close()
            self._driver = None

    def _run_write(
        self, query: str, parameters: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Execute a write query and return normalized row dictionaries.

        Args:
            query: The Cypher query string to execute.
            parameters: A dictionary of parameters to pass to the query.

        Returns:
            list[dict[str, Any]]: A list of dictionaries representing the query results, or
                an empty list if the query failed.

        Raises:
            Neo4jError: If the query execution fails due to a Neo4j error.
        """

        if not self.enabled:
            return []
        try:
            with self.driver.session(database=self.config.database) as session:
                result = session.run(query, parameters)
                return [record.data() for record in result]
        except Neo4jError as exc:
            logger.warning("Neo4j write failed: {}", exc)
            return []

    def _run_read(self, query: str, parameters: dict[str, Any]) -> list[dict[str, Any]]:
        """Execute a read query and return normalized row dictionaries.

        Args:
            query: The Cypher query string to execute.
            parameters: A dictionary of parameters to pass to the query.

        Returns:
            list[dict[str, Any]]: A list of dictionaries representing the query results, or
                an empty list if the query failed.

        Raises:
            Neo4jError: If the query execution fails due to a Neo4j error.
        """

        if not self.enabled:
            return []
        try:
            with self.driver.session(database=self.config.database) as session:
                result = session.run(query, parameters)
                return [record.data() for record in result]
        except Neo4jError as exc:
            logger.warning("Neo4j read failed: {}", exc)
            return []

    def ingest_source_records(
        self,
        *,
        collection: str,
        records: Iterable[GraphSourceRecord],
    ) -> dict[str, int]:
        """Upsert graph facts for a batch of source records.

        Args:
            collection: The name of the collection to which the records belong.
            records: An iterable of GraphSourceRecord objects to ingest.

        Returns:
            dict[str, int]: A dictionary containing the count of ingested records.
        """

        total = 0
        for record in records:
            total += 1
            self._upsert_source_record(collection=collection, record=record)
        return {"records": total}

    def _upsert_source_record(
        self, *, collection: str, record: GraphSourceRecord
    ) -> None:
        """Upsert a single source record and its core graph neighborhood.

        Args:
            collection: The name of the collection to which the record belongs.
            record: A GraphSourceRecord object to upsert.
        """

        params = {
            "collection": collection,
            "node_id": record.node_id,
            "source_kind": record.source_kind,
            "record_kind": record.record_kind,
            "text": record.text,
            "filename": record.filename,
            "file_hash": record.file_hash,
            "text_id": record.text_id,
            "thread_id": record.thread_id,
            "parent_record_id": record.parent_record_id,
            "timestamp": record.timestamp,
            "page": record.page,
            "row": record.row,
            "search_blob": record.search_blob,
            "metadata": json.dumps(record.metadata),
        }
        self._run_write(
            """
            MERGE (c:Collection {name: $collection})
            MERGE (s:SourceRecord {node_id: $node_id})
            SET s.source_kind = $source_kind,
                s.record_kind = $record_kind,
                s.text = $text,
                s.filename = $filename,
                s.file_hash = $file_hash,
                s.text_id = $text_id,
                s.thread_id = $thread_id,
                s.parent_record_id = $parent_record_id,
                s.timestamp = $timestamp,
                s.page = $page,
                s.row = $row,
                s.search_blob = $search_blob,
                s.metadata_json = $metadata
            MERGE (s)-[:IN_COLLECTION]->(c)
            FOREACH (_ IN CASE WHEN $filename IS NULL THEN [] ELSE [1] END |
                MERGE (d:Document {collection: $collection, filename: $filename})
                MERGE (s)-[:FROM_DOCUMENT]->(d)
            )
            MERGE (p:Provenance {node_id: $node_id})
            SET p.file_hash = $file_hash,
                p.text_id = $text_id,
                p.page = $page,
                p.row = $row
            MERGE (s)-[:EVIDENCED_BY]->(p)
            """,
            params,
        )

        if record.platform:
            self._run_write(
                """
                MATCH (s:SourceRecord {node_id: $node_id})
                MERGE (p:Platform {name: $platform})
                MERGE (s)-[:POSTED_ON]->(p)
                """,
                {"node_id": record.node_id, "platform": record.platform},
            )

        author_identity = resolve_author_identity(
            author=record.author,
            author_id=record.author_id,
            collection=collection,
        )
        if author_identity is not None:
            self._run_write(
                """
                MATCH (s:SourceRecord {node_id: $node_id})
                MERGE (a:Author {author_key: $author_key})
                SET a.display_name = $author_name,
                    a.scope = $scope,
                    a.auto_merged = $auto_merged
                MERGE (s)-[:AUTHORED_BY]->(a)
                """,
                {
                    "node_id": record.node_id,
                    "author_key": author_identity.canonical_key,
                    "author_name": author_identity.canonical_name,
                    "scope": author_identity.scope,
                    "auto_merged": author_identity.auto_merged,
                },
            )

        if record.thread_id:
            self._run_write(
                """
                MATCH (s:SourceRecord {node_id: $node_id})
                MERGE (t:Thread {thread_id: $thread_id})
                MERGE (s)-[:BELONGS_TO_THREAD]->(t)
                """,
                {"node_id": record.node_id, "thread_id": record.thread_id},
            )
        if record.parent_record_id:
            self._run_write(
                """
                MATCH (s:SourceRecord {node_id: $node_id})
                MERGE (p:SourceRecord {node_id: $parent_record_id})
                MERGE (s)-[:REPLIES_TO]->(p)
                """,
                {
                    "node_id": record.node_id,
                    "parent_record_id": record.parent_record_id,
                },
            )

        if record.url:
            self._run_write(
                """
                MATCH (s:SourceRecord {node_id: $node_id})
                MERGE (u:URL {url: $url})
                MERGE (s)-[:LINKS_TO]->(u)
                """,
                {"node_id": record.node_id, "url": record.url},
            )
        domain = record.domain or extract_domain(record.url)
        if domain:
            self._run_write(
                """
                MATCH (s:SourceRecord {node_id: $node_id})
                MERGE (d:Domain {name: $domain})
                MERGE (s)-[:LINKS_TO_DOMAIN]->(d)
                """,
                {"node_id": record.node_id, "domain": domain},
            )

        for tag in parse_tag_values(record.tags):
            self._run_write(
                """
                MATCH (s:SourceRecord {node_id: $node_id})
                MERGE (t:Tag {name: $tag})
                MERGE (s)-[:TAGGED_WITH]->(t)
                """,
                {"node_id": record.node_id, "tag": tag},
            )

        if record.timestamp:
            time_bucket = str(record.timestamp)[:13]
            self._run_write(
                """
                MATCH (s:SourceRecord {node_id: $node_id})
                MERGE (t:TimeBucket {bucket: $bucket})
                MERGE (s)-[:IN_TIME_BUCKET]->(t)
                """,
                {"node_id": record.node_id, "bucket": time_bucket},
            )

        entity_mentions: list[dict[str, Any]] = list(record.entities or [])
        seen_entity_keys: set[str] = set()
        for entity in entity_mentions:
            text = str(entity.get("text") or "").strip()
            entity_type = str(entity.get("type") or "Unlabeled").strip()
            if not text:
                continue
            resolved = resolve_entity_identity(
                text=text,
                entity_type=entity_type,
                collection=collection,
                score=entity.get("score"),
                min_confidence=self.config.resolution_min_confidence,
            )
            if resolved.canonical_key in seen_entity_keys:
                continue
            seen_entity_keys.add(resolved.canonical_key)
            self._run_write(
                """
                MATCH (s:SourceRecord {node_id: $node_id})
                MERGE (e:Entity {entity_key: $entity_key})
                SET e.canonical_text = $canonical_text,
                    e.entity_type = $entity_type,
                    e.scope = $scope,
                    e.auto_merged = $auto_merged
                MERGE (a:EntityAlias {alias_key: $alias_key})
                SET a.alias = $alias,
                    a.entity_type = $entity_type
                MERGE (a)-[:ALIAS_OF]->(e)
                MERGE (s)-[:MENTIONS]->(e)
                FOREACH (_ IN CASE WHEN $candidate_key IS NULL THEN [] ELSE [1] END |
                    MERGE (ce:Entity {entity_key: $candidate_key})
                    SET ce.canonical_text = $canonical_text,
                        ce.entity_type = $entity_type,
                        ce.scope = 'global_candidate'
                    MERGE (e)-[:CANDIDATE_MATCH]->(ce)
                )
                """,
                {
                    "node_id": record.node_id,
                    "entity_key": resolved.canonical_key,
                    "canonical_text": resolved.canonical_name,
                    "entity_type": entity_type or "Unlabeled",
                    "scope": resolved.scope,
                    "auto_merged": resolved.auto_merged,
                    "alias_key": f"{entity_type.casefold()}::{text.casefold()}",
                    "alias": text,
                    "candidate_key": resolved.candidate_key,
                },
            )

        entity_keys = list(seen_entity_keys)
        for left_idx, left_key in enumerate(entity_keys):
            for right_key in entity_keys[left_idx + 1 :]:
                self._run_write(
                    """
                    MATCH (l:Entity {entity_key: $left_key})
                    MATCH (r:Entity {entity_key: $right_key})
                    MERGE (l)-[rel:CO_OCCURS_WITH]-(r)
                    ON CREATE SET rel.weight = 1
                    ON MATCH SET rel.weight = rel.weight + 1
                    """,
                    {"left_key": left_key, "right_key": right_key},
                )

    def search_entities(
        self,
        *,
        collection: str,
        q: str,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Search canonical entities and aliases within a collection.

        Args:
            collection: The name of the collection to search within.
            q: The raw search query string to match against entity texts and aliases.
            limit: The maximum number of results to return.

        Returns:
            list[dict[str, Any]]: A list of dictionaries representing matching entities,
                including their canonical text, type, alias, and mention count.
        """

        needle = str(q or "").strip().casefold()
        if not needle:
            return []
        rows = self._run_read(
            """
            MATCH (:Collection {name: $collection})<-[:IN_COLLECTION]-(:SourceRecord)-[:MENTIONS]->(e:Entity)
            OPTIONAL MATCH (a:EntityAlias)-[:ALIAS_OF]->(e)
            WHERE toLower(e.canonical_text) CONTAINS $needle OR toLower(a.alias) CONTAINS $needle
            RETURN e.entity_key AS entity_key,
                   e.canonical_text AS text,
                   e.entity_type AS type,
                   max(a.alias) AS alias,
                   count(DISTINCT e) AS mentions
            ORDER BY mentions DESC, text ASC
            LIMIT $limit
            """,
            {"collection": collection, "needle": needle, "limit": max(1, int(limit))},
        )
        return rows

    def retrieve_candidates(
        self,
        *,
        collection: str,
        plan: GraphQueryPlan,
        limit: int = 12,
    ) -> GraphTraversalResult:
        """Retrieve source-record candidates from direct and graph-expanded matches.

        Args:
            collection: The name of the collection to search within.
            plan: A GraphQueryPlan object containing the retrieval mode, seeds, and path terms.
            limit: The maximum number of candidate records to return.

        Returns:
            GraphTraversalResult: An object containing the list of candidate records and a trace of the retrieval process.
        """

        needle = " ".join(seed.value for seed in plan.seeds).strip() or ""
        if not needle:
            needle = ""
        seed_rows = self._run_read(
            """
            MATCH (:Collection {name: $collection})<-[:IN_COLLECTION]-(s:SourceRecord)
            WHERE $needle = ''
               OR toLower(s.search_blob) CONTAINS $needle
            RETURN s.node_id AS node_id,
                   CASE
                     WHEN toLower(coalesce(s.text_id, '')) = $needle THEN 1.0
                     WHEN toLower(coalesce(s.search_blob, '')) CONTAINS $needle THEN 0.85
                     ELSE 0.5
                   END AS exact_score,
                   0.0 AS graph_score,
                   'source' AS matched_on
            ORDER BY exact_score DESC
            LIMIT $limit
            """,
            {
                "collection": collection,
                "needle": needle.casefold(),
                "limit": max(1, int(limit)),
            },
        )
        entity_rows = []
        if needle:
            entity_rows = self._run_read(
                """
                MATCH (:Collection {name: $collection})<-[:IN_COLLECTION]-(s:SourceRecord)-[:MENTIONS]->(e:Entity)
                OPTIONAL MATCH (a:EntityAlias)-[:ALIAS_OF]->(e)
                WHERE toLower(e.canonical_text) CONTAINS $needle OR toLower(a.alias) CONTAINS $needle
                RETURN s.node_id AS node_id,
                       0.65 AS exact_score,
                       0.70 AS graph_score,
                       'entity' AS matched_on
                LIMIT $limit
                """,
                {
                    "collection": collection,
                    "needle": needle.casefold(),
                    "limit": max(1, int(limit)),
                },
            )

        seed_ids = [
            row["node_id"] for row in [*seed_rows, *entity_rows] if row.get("node_id")
        ]
        traversed_rows: list[dict[str, Any]] = []
        if seed_ids:
            traversed_rows = self._run_read(
                """
                MATCH (:Collection {name: $collection})<-[:IN_COLLECTION]-(seed:SourceRecord)
                WHERE seed.node_id IN $seed_ids
                MATCH p=(seed)-[:MENTIONS|AUTHORED_BY|BELONGS_TO_THREAD|LINKS_TO|LINKS_TO_DOMAIN|TAGGED_WITH|POSTED_ON|IN_TIME_BUCKET*1..2]-(nbr:SourceRecord)-[:IN_COLLECTION]->(:Collection {name: $collection})
                RETURN nbr.node_id AS node_id,
                       0.15 AS exact_score,
                       max(1.0 / length(p)) AS graph_score,
                       'traversal' AS matched_on
                ORDER BY graph_score DESC
                LIMIT $limit
                """,
                {
                    "collection": collection,
                    "seed_ids": seed_ids,
                    "limit": max(1, int(limit)),
                },
            )

        merged: dict[str, GraphCandidate] = {}
        for row in [*seed_rows, *entity_rows, *traversed_rows]:
            node_id = str(row.get("node_id") or "").strip()
            if not node_id:
                continue
            candidate = GraphCandidate(
                node_id=node_id,
                exact_score=float(row.get("exact_score") or 0.0),
                graph_score=float(row.get("graph_score") or 0.0),
                matched_on=str(row.get("matched_on") or "graph"),
            )
            existing = merged.get(node_id)
            if existing is None:
                merged[node_id] = candidate
                continue
            merged[node_id] = GraphCandidate(
                node_id=node_id,
                exact_score=max(existing.exact_score, candidate.exact_score),
                graph_score=max(existing.graph_score, candidate.graph_score),
                matched_on=existing.matched_on
                if existing.exact_score + existing.graph_score
                >= candidate.exact_score + candidate.graph_score
                else candidate.matched_on,
            )

        ranked = sorted(
            merged.values(),
            key=lambda item: (item.exact_score + item.graph_score, item.node_id),
            reverse=True,
        )[: max(1, int(limit))]
        return GraphTraversalResult(
            candidates=ranked,
            trace={
                "query_mode": plan.mode,
                "seed_count": len(plan.seeds),
                "seed_node_count": len(seed_ids),
                "candidate_count": len(ranked),
            },
        )

    def get_neighborhood(
        self,
        *,
        collection: str,
        entity: str,
        hops: int = 2,
        limit: int = 25,
    ) -> dict[str, Any]:
        """Return a compact neighborhood payload around one entity.

        Args:
            collection: The name of the collection to search within.
            entity: The raw entity text to find the neighborhood around.
            hops: The maximum number of hops for neighborhood expansion.
            limit: The maximum number of neighboring nodes to return.

        Returns:
            dict[str, Any]: A dictionary containing the center entity and a list of neighboring nodes with their
                labels, properties, and hop distance.
        """

        rows = self._run_read(
            """
            MATCH (a:EntityAlias)-[:ALIAS_OF]->(e:Entity)
            WHERE toLower(a.alias) = $entity OR toLower(e.canonical_text) = $entity
            MATCH p=(e)-[*1..$hops]-(n)
            RETURN e.canonical_text AS center,
                   labels(n) AS labels,
                   properties(n) AS properties,
                   length(p) AS hops
            LIMIT $limit
            """,
            {
                "entity": str(entity or "").strip().casefold(),
                "hops": max(1, int(hops)),
                "limit": max(1, int(limit)),
            },
        )
        return {"center": entity, "neighbors": rows}

    def find_path(
        self,
        *,
        collection: str,
        source: str,
        target: str,
        max_hops: int = 6,
    ) -> GraphPathResult:
        """Return a shortest path between two entities or aliases.

        Args:
            collection: The name of the collection to search within.
            source: The raw text of the source entity or alias.
            target: The raw text of the target entity or alias.
            max_hops: The maximum number of hops to consider for path finding.

        Returns:
            GraphPathResult: An object containing the source and target texts, lists of nodes and relationships
                in the found path, and a trace of the path finding process.
        """

        rows = self._run_read(
            """
            MATCH (sa:EntityAlias)-[:ALIAS_OF]->(start:Entity)
            MATCH (ta:EntityAlias)-[:ALIAS_OF]->(end:Entity)
            WHERE (toLower(sa.alias) = $source OR toLower(start.canonical_text) = $source)
              AND (toLower(ta.alias) = $target OR toLower(end.canonical_text) = $target)
            MATCH p = shortestPath((start)-[*..6]-(end))
            RETURN [node IN nodes(p) | {labels: labels(node), properties: properties(node)}] AS nodes,
                   [rel IN relationships(p) | {type: type(rel), properties: properties(rel)}] AS relationships
            LIMIT 1
            """,
            {
                "collection": collection,
                "source": str(source or "").strip().casefold(),
                "target": str(target or "").strip().casefold(),
                "max_hops": max(1, int(max_hops)),
            },
        )
        first = rows[0] if rows else {}
        return GraphPathResult(
            source=source,
            target=target,
            nodes=list(first.get("nodes") or []),
            relationships=list(first.get("relationships") or []),
            trace={"max_hops": max_hops, "found": bool(rows)},
        )

    def get_collection_stats(self, *, collection: str) -> dict[str, Any]:
        """Return simple graph stats for the active collection.

        Args:
            collection: The name of the collection to retrieve stats for.

        Returns:
            dict[str, Any]: A dictionary containing the number of source records and entities in the collection.
        """

        rows = self._run_read(
            """
            MATCH (c:Collection {name: $collection})
            OPTIONAL MATCH (c)<-[:IN_COLLECTION]-(s:SourceRecord)
            OPTIONAL MATCH (s)-[:MENTIONS]->(e:Entity)
            RETURN count(DISTINCT s) AS source_records,
                   count(DISTINCT e) AS entities
            """,
            {"collection": collection},
        )
        return rows[0] if rows else {"source_records": 0, "entities": 0}

    def delete_collection(self, *, collection: str) -> None:
        """Delete graph nodes scoped to one collection.

        Args:
            collection: The name of the collection to delete.
        """

        self._run_write(
            """
            MATCH (c:Collection {name: $collection})
            OPTIONAL MATCH (c)<-[:IN_COLLECTION]-(s:SourceRecord)
            DETACH DELETE s, c
            """,
            {"collection": collection},
        )
