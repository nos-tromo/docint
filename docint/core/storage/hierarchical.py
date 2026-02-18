from __future__ import annotations

import uuid
from copy import deepcopy
from typing import Sequence, cast

from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.node_parser import NodeParser, SentenceSplitter
from llama_index.core.schema import BaseNode, Document, NodeRelationship

from docint.utils.env_cfg import load_ingestion_env


class HierarchicalNodeParser(NodeParser):
    """
    Splits documents into a hierarchy of nodes:
    - Level 0: Document (Implicit)
    - Level 1: Coarse Chunks (Sections/Paragraphs/Large blocks)
    - Level 2: Fine Chunks (Sentence-based)

    Fine chunks are linked to their parent coarse chunk via metadata.
    """

    _ingestion_config = load_ingestion_env()
    coarse_chunk_size: int = _ingestion_config.coarse_chunk_size
    fine_chunk_size: int = _ingestion_config.fine_chunk_size
    fine_chunk_overlap: int = _ingestion_config.fine_chunk_overlap
    _coarse_splitter: SentenceSplitter = PrivateAttr()
    _fine_splitter: SentenceSplitter = PrivateAttr()

    def __init__(
        self,
        coarse_chunk_size: int = 8192,
        fine_chunk_size: int = 1024,
        fine_chunk_overlap: int = 0,
        **kwargs,
    ):
        """
        Initialize the HierarchicalNodeParser.

        Args:
            coarse_chunk_size (int): Size of coarse chunks (Level 1).
            fine_chunk_size (int): Size of fine chunks (Level 2).
            fine_chunk_overlap (int): Overlap for fine chunks.
        """
        super().__init__(**kwargs)
        self._coarse_splitter = SentenceSplitter(
            chunk_size=coarse_chunk_size, chunk_overlap=0
        )
        self._fine_splitter = SentenceSplitter(
            chunk_size=fine_chunk_size, chunk_overlap=fine_chunk_overlap
        )

    def _parse_nodes(
        self, nodes: Sequence[BaseNode], show_progress: bool = False, **kwargs
    ) -> list[BaseNode]:
        """
        Parse nodes into hierarchical chunks.

        If the input nodes are Documents (Level 0), we first create Coarse Chunks (Level 1),
        then Fine Chunks (Level 2).
        If the input nodes are already "chunks" (e.g. from MarkdownNodeParser), treat them as
        Coarse Chunks (Level 1) and split into Fine Chunks (Level 2).

        Args:
            nodes (Sequence[BaseNode]): Input nodes to parse.
            show_progress (bool): Whether to show progress.

        Returns:
            list[BaseNode]: Hierarchically chunked nodes.
        """
        all_nodes: list[BaseNode] = []

        for node in nodes:
            # Determine if we should treat this as a Document (Level 0) or Coarse Chunk (Level 1)
            # Typically, if it's a Document, it has no parent (or source points to itself).
            # If it's a Node from another parser, it might be Level 1.

            # Strategy:
            # 1. If the node is large, split into Coarse Chunks.
            # 2. If the node is already "coarse" (e.g. from Markdown parser), use it directly.

            # However, to be consistent, let's assume `nodes` passed here are what we want to be parents of the fine chunks.
            # But `_create_nodes` in ingestion pipeline handles different document types.
            # For "plain text", we get a Document. We should split it into Coarse.
            # For "markdown", we might get Nodes (sections). We should treat as Coarse (or split if huge).

            # Let's handle two cases:
            # A) Input is a Document (Level 0) -> Split to Coarse -> Split to Fine
            # B) Input is a Pre-chunked Node (Level 1 candidates) -> Split to Fine

            is_document = isinstance(node, Document)

            coarse_candidates: list[BaseNode] = []

            if is_document:
                # Level 0 -> Level 1
                # Use coarse splitter to get Level 1 chunks
                doc_node = cast(Document, node)
                coarse_candidates = self._coarse_splitter.get_nodes_from_documents(
                    [doc_node]
                )
            else:
                # Already a node (Level 1 candidate).
                # Check if it's too big? If so, split it further?
                # For simplicity, let's treat it as a Coarse Chunk, but ensure it respects coarse_chunk_size if possible.
                # If we re-split a Node, we lose its original identity if we aren't careful.
                # But here we assume the incoming nodes (e.g. Markdown sections) describe the "structure".
                # If a section is huge, we probably WANT to split it.

                # Check length
                text_len = len(node.get_content())
                if text_len > self.coarse_chunk_size:
                    # It's too big, split it mechanism
                    # But we must preserve the metadata of the incoming node (e.g. header path)
                    # SentenceSplitter works on Documents usually.
                    # We can wrap the node's text in a temporary Document or use `get_nodes_from_documents`.

                    # Create a temp doc to split
                    temp_doc = Document(text=node.get_content(), metadata=node.metadata)
                    coarse_candidates = self._coarse_splitter.get_nodes_from_documents(
                        [temp_doc]
                    )
                else:
                    # It's fine as a coarse chunk
                    coarse_candidates = [node]

            # Now process coarse candidates -> Fine chunks
            for coarse_node in coarse_candidates:
                # Assign ID if missing
                if not coarse_node.node_id:
                    coarse_node.node_id = str(uuid.uuid4())

                # Mark as coarse
                coarse_node.metadata["hier.level"] = 1
                coarse_node.metadata["docint_hier_type"] = "coarse"

                # Ensure parent (Document) linkage is set if available
                # If input was Document, SentneceSplitter sets ref_doc_id.
                # If input was Node, we might have lost connection if we re-split.

                # Add to result list
                all_nodes.append(coarse_node)

                # Level 1 -> Level 2
                # We split the *content* of the coarse chunk
                # We must ensure fine chunks link back to THIS coarse_node

                # Create a temp doc for splitting to ensure easy usage of SentenceSplitter
                temp_coarse_doc = Document(
                    text=coarse_node.get_content(),
                    metadata=deepcopy(coarse_node.metadata),
                )

                fine_nodes = self._fine_splitter.get_nodes_from_documents(
                    [temp_coarse_doc]
                )

                for fine_node in fine_nodes:
                    fine_node.metadata["hier.level"] = 2
                    fine_node.metadata["docint_hier_type"] = "fine"
                    fine_node.metadata["hier.parent_id"] = coarse_node.node_id

                    # Also keep track of ancestors if possible?
                    # coarse_node might have ref_doc_id.
                    if coarse_node.ref_doc_id:
                        fine_node.metadata["hier.doc_id"] = coarse_node.ref_doc_id

                    # Ensure relationships
                    fine_node.relationships[NodeRelationship.PARENT] = (
                        coarse_node.as_related_node_info()
                    )

                    # Add to result
                    all_nodes.append(fine_node)

        return all_nodes
