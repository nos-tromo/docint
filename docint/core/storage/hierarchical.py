"""Hierarchical node storage maintaining parent/child relationships in Qdrant."""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from copy import deepcopy
from typing import Any, cast

from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.node_parser import NodeParser, SentenceSplitter
from llama_index.core.schema import BaseNode, Document, NodeRelationship
from typing_extensions import override

from docint.utils.env_cfg import load_ingestion_env


class HierarchicalNodeParser(NodeParser):
    """Splits documents into a hierarchy of nodes.

    - Level 0: Document (Implicit)
    - Level 1: Coarse Chunks (Sections/Paragraphs/Large blocks)
    - Level 2: Fine Chunks (Sentence-based).

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
        **kwargs: Any,
    ) -> None:
        """Initialize the HierarchicalNodeParser.

        Args:
            coarse_chunk_size (int): Size of coarse chunks (Level 1).
            fine_chunk_size (int): Size of fine chunks (Level 2).
            fine_chunk_overlap (int): Overlap for fine chunks.
            **kwargs: Forwarded to the ``NodeParser`` base class.
        """
        super().__init__(**kwargs)
        # ``basic_clean`` normalizes paragraph gaps to a single blank line
        # (``\n\n``), but llama-index's SentenceSplitter defaults to a triple
        # newline as the paragraph separator — so without an override it
        # cannot see paragraph boundaries in cleaned text and is forced to
        # split mid-paragraph.
        self._coarse_splitter = SentenceSplitter(
            chunk_size=coarse_chunk_size,
            chunk_overlap=0,
            paragraph_separator="\n\n",
        )
        self._fine_splitter = SentenceSplitter(
            chunk_size=fine_chunk_size,
            chunk_overlap=fine_chunk_overlap,
            paragraph_separator="\n\n",
        )

    @staticmethod
    def _splitter_input_doc(text: str, metadata: dict[str, Any]) -> Document:
        """Wrap *text* / *metadata* for a metadata-aware splitter, hiding the metadata.

        llama-index's :class:`SentenceSplitter` is a ``MetadataAwareTextSplitter``:
        it shrinks each chunk's token budget by the node's rendered metadata (the
        longer of ``MetadataMode.EMBED`` / ``MetadataMode.LLM``), on the assumption
        that the metadata is embedded alongside the text. docint does not embed
        structural metadata — :mod:`docint.utils.embed_chunking` excludes every key
        from ``MetadataMode.EMBED`` and the chat path curates ``MetadataMode.LLM``
        via ``LLM_VISIBLE_METADATA_KEYS`` — so that reservation is
        counter-productive: a PDF coarse unit's layout metadata (``block_ids`` /
        ``image_ids`` / ``bbox_refs`` …) can render to more tokens than
        ``fine_chunk_size``, which makes the splitter raise *"Metadata length (N)
        is longer than chunk size (M)"* and abort ingestion instead of splitting
        the text.

        Excluding every key from both render modes makes the splitter operate on
        the text alone. llama-index copies the exclusion lists onto each produced
        split, so the children also render text-only in embed mode — matching the
        sub-node contract in :mod:`docint.utils.embed_chunking`. The metadata
        *dict* is untouched, so citations and image/table linking keep working.

        Args:
            text (str): Coarse-unit text to be split.
            metadata (dict[str, Any]): Metadata to carry onto the produced nodes,
                rendered into neither the embed nor the LLM payload.

        Returns:
            Document: A document whose metadata is hidden from both render modes.
        """
        keys = list(metadata.keys())
        return Document(
            text=text,
            metadata=metadata,
            excluded_embed_metadata_keys=keys,
            excluded_llm_metadata_keys=keys,
        )

    @override
    def _parse_nodes(self, nodes: Sequence[BaseNode], show_progress: bool = False, **kwargs: Any) -> list[BaseNode]:
        """Parse nodes into hierarchical chunks.

        If the input nodes are Documents (Level 0), we first create Coarse Chunks (Level 1),
        then Fine Chunks (Level 2).
        If the input nodes are already "chunks" (e.g. from MarkdownNodeParser), treat them as
        Coarse Chunks (Level 1) and split into Fine Chunks (Level 2).

        Args:
            nodes (Sequence[BaseNode]): Input nodes to parse.
            show_progress (bool): Whether to show progress.
            **kwargs: Forwarded to llama-index's parse_nodes helpers; unused locally.

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

            # However, to be consistent, let's assume `nodes` passed here are what we want to
            # be parents of the fine chunks.
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
                coarse_candidates = self._coarse_splitter.get_nodes_from_documents([doc_node])
            else:
                # Already a node (Level 1 candidate).
                # Check if it's too big? If so, split it further?
                # For simplicity, treat it as a Coarse Chunk but respect coarse_chunk_size.
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

                    # Create a temp doc to split (metadata hidden from the
                    # metadata-aware splitter so it splits on text alone).
                    temp_doc = self._splitter_input_doc(node.get_content(), node.metadata)
                    coarse_candidates = self._coarse_splitter.get_nodes_from_documents([temp_doc])
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

                # Create a temp doc for splitting to ensure easy usage of
                # SentenceSplitter (metadata hidden from the metadata-aware
                # splitter so it splits on text alone, never reserving budget
                # for layout metadata docint does not embed).
                temp_coarse_doc = self._splitter_input_doc(
                    coarse_node.get_content(),
                    deepcopy(coarse_node.metadata),
                )

                fine_nodes = self._fine_splitter.get_nodes_from_documents([temp_coarse_doc])

                for fine_node in fine_nodes:
                    fine_node.metadata["hier.level"] = 2
                    fine_node.metadata["docint_hier_type"] = "fine"
                    fine_node.metadata["hier.parent_id"] = coarse_node.node_id

                    # Also keep track of ancestors if possible?
                    # coarse_node might have ref_doc_id.
                    if coarse_node.ref_doc_id:
                        fine_node.metadata["hier.doc_id"] = coarse_node.ref_doc_id

                    # Ensure relationships
                    fine_node.relationships[NodeRelationship.PARENT] = coarse_node.as_related_node_info()

                    # ``hier.parent_id`` / ``hier.doc_id`` are added after the
                    # split, so refresh the exclusion lists to cover them too:
                    # every metadata key stays a locator on the dict but is kept
                    # out of the embed/LLM payloads (text-only embedding, like
                    # the embed_chunking sub-node contract).
                    fine_keys = list(fine_node.metadata.keys())
                    fine_node.excluded_embed_metadata_keys = fine_keys
                    fine_node.excluded_llm_metadata_keys = fine_keys

                    # Add to result
                    all_nodes.append(fine_node)

        return all_nodes
