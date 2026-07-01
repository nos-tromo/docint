"""Unit tests for LinkFollowingPostprocessor."""

from types import SimpleNamespace

from llama_index.core.schema import NodeWithScore, TextNode

from docint.core.rag import LinkFollowingPostprocessor


def test_postprocessor_appends_siblings_once() -> None:
    """Siblings are appended once and deduplicated across repeated calls."""
    sibling = NodeWithScore(node=TextNode(id_="s1", text="spoken", metadata={"posting_uuid": "u1"}), score=None)

    rag = SimpleNamespace(
        _fetch_posting_entity_nodes=lambda uuid, *, exclude_node_ids: [sibling] if uuid == "u1" else []
    )
    pp = LinkFollowingPostprocessor(rag=rag)  # type: ignore[arg-type]

    hit = NodeWithScore(
        node=TextNode(id_="p1", text="post text", metadata={"reference_metadata": {"uuid": "u1"}}), score=0.9
    )
    out = pp._postprocess_nodes([hit], None)
    ids = [n.node.node_id for n in out]
    assert "p1" in ids and "s1" in ids
    # Idempotent: a second sibling for an already-included id is not duplicated.
    out2 = pp._postprocess_nodes(out, None)
    assert out2.count(sibling) <= 1
