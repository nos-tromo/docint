"""Tests for the retrieval target: which evidence may answer a turn.

``all`` fuses text and imagery, ``documents`` drops the image lane, and
``visual`` answers from the image companion alone. The target is a separate
field from ``retrieval_mode`` (session routing on the request, a different
vocabulary on the response) precisely so neither can be mistaken for the
other, and a pinned scope still outranks it.
"""

import types
from typing import Any, cast

import pytest
from llama_index.core import Response

import docint.core.rag as rag_module
from docint.core.rag import RAG
from docint.core.retrieval.visual import VisualRetriever


@pytest.fixture
def engine_capture(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Capture the query engine's constructor arguments.

    Args:
        monkeypatch (pytest.MonkeyPatch): Patching fixture.

    Returns:
        dict[str, Any]: The captured kwargs, filled on ``build_query_engine``.
    """
    captured: dict[str, Any] = {}
    monkeypatch.setattr(RAG, "_infer_collection_profile", lambda self: {"is_social_table": False})
    monkeypatch.setattr(RAG, "_build_response_synthesizer", lambda self, **kwargs: kwargs)
    monkeypatch.setattr(
        rag_module.RetrieverQueryEngine,
        "from_args",
        staticmethod(lambda **kwargs: captured.update(kwargs) or kwargs),
    )
    return captured


def _rag(monkeypatch: pytest.MonkeyPatch) -> RAG:
    """Build a RAG instance with the Qdrant-facing parts stubbed out.

    Args:
        monkeypatch (pytest.MonkeyPatch): Patching fixture.

    Returns:
        RAG: The instance.
    """
    rag = RAG(qdrant_collection="testbatch")
    rag.index = cast(Any, types.SimpleNamespace(docstore=object(), as_retriever=lambda **_kwargs: object()))
    monkeypatch.setattr(RAG, "_ensure_visual_indexes_once", lambda self, collection: None)
    monkeypatch.setattr(RAG, "_collection_exists", lambda self, name: True)
    return rag


def test_the_visual_target_retrieves_from_the_companion(
    monkeypatch: pytest.MonkeyPatch, engine_capture: dict[str, Any]
) -> None:
    """Imagery is the whole evidence set, not a lane beside the text."""
    rag = _rag(monkeypatch)

    rag.build_query_engine(retrieval_target="visual")

    assert isinstance(engine_capture["retriever"], VisualRetriever)


def test_the_visual_chain_keeps_only_what_means_something_without_text(
    monkeypatch: pytest.MonkeyPatch, engine_capture: dict[str, Any]
) -> None:
    """Keep only the postprocessors that mean something without text.

    Parent context reads a docstore with no companion nodes in it, the
    diversity cap would collapse a clip's consecutive keyframes, and
    link-following would pull posting prose into a set meant to be pixels.
    """
    rag = _rag(monkeypatch)

    rag.build_query_engine(retrieval_target="visual")

    names = [type(processor).__name__ for processor in engine_capture["node_postprocessors"]]
    assert names == [
        "LazyRerankerPostprocessor",
        "ImageRelevanceFloorPostprocessor",
        "CitationNumberingPostprocessor",
    ]


def test_the_visual_rerank_keeps_at_least_the_images_the_answer_may_see(
    monkeypatch: pytest.MonkeyPatch, engine_capture: dict[str, Any]
) -> None:
    """Rerank at least as deep as the answer may look.

    The default cut is smaller than the attach cap, so it would otherwise
    drop evidence the answer is allowed to see.
    """
    rag = _rag(monkeypatch)
    rag.rerank_top_n = 2

    rag.build_query_engine(retrieval_target="visual")

    assert engine_capture["node_postprocessors"][0].top_n >= rag._visual_answer_max_images()


def test_the_visual_target_asks_for_the_visual_synthesizer(
    monkeypatch: pytest.MonkeyPatch, engine_capture: dict[str, Any]
) -> None:
    """Only the visual target puts pixels in front of the model."""
    rag = _rag(monkeypatch)

    rag.build_query_engine(retrieval_target="visual")

    assert engine_capture["response_synthesizer"]["visual"] is True


def test_the_documents_target_drops_the_image_lane(monkeypatch: pytest.MonkeyPatch) -> None:
    """Documents means documents: no image lane at all."""
    rag = _rag(monkeypatch)
    seen: dict[str, Any] = {}

    def _build_image_lane(self: RAG, **kwargs: Any) -> None:
        """Record that the lane was asked for.

        Args:
            self (RAG): The instance.
            **kwargs (Any): Lane arguments.

        Returns:
            None: Never a lane.
        """
        seen["asked"] = True
        return None

    monkeypatch.setattr(RAG, "_build_image_lane", _build_image_lane)

    rag._build_retriever(include_images=False)

    assert "asked" not in seen


def test_the_all_target_keeps_the_image_lane(monkeypatch: pytest.MonkeyPatch) -> None:
    """The default target is unchanged by this feature."""
    rag = _rag(monkeypatch)
    seen: dict[str, Any] = {}

    def _build_image_lane(self: RAG, **kwargs: Any) -> None:
        """Record that the lane was asked for.

        Args:
            self (RAG): The instance.
            **kwargs (Any): Lane arguments.

        Returns:
            None: Never a lane.
        """
        seen["asked"] = True
        return None

    monkeypatch.setattr(RAG, "_build_image_lane", _build_image_lane)

    rag._build_retriever()

    assert seen["asked"] is True


def test_a_scope_outranks_the_target(monkeypatch: pytest.MonkeyPatch, engine_capture: dict[str, Any]) -> None:
    """Hand-picked chunks are hand-picked whatever the target says."""
    rag = _rag(monkeypatch)

    rag.build_query_engine(scoped_node_ids=["c1"], retrieval_target="visual")

    assert isinstance(engine_capture["retriever"], rag_module._ScopedRetriever)


def test_the_visual_target_refuses_filters_that_reached_neither_lane(monkeypatch: pytest.MonkeyPatch) -> None:
    """Refuse filters that reached neither lane.

    Answering a filtered visual question from unfiltered imagery is exactly
    the silent wrong answer this target exists to avoid.
    """
    rag = _rag(monkeypatch)

    with pytest.raises(ValueError, match="filters are active"):
        rag._build_visual_retriever(
            metadata_filter_rules=None,
            metadata_filters_active=True,
            vector_store_kwargs=None,
        )


def test_the_visual_target_compiles_rules_that_arrived_uncompiled(monkeypatch: pytest.MonkeyPatch) -> None:
    """A stateless caller may pass rules without a compiled filter beside them."""
    rag = _rag(monkeypatch)
    compiled: dict[str, Any] = {}

    monkeypatch.setattr(
        rag_module,
        "build_qdrant_filter",
        lambda rules: (compiled.setdefault("rules", rules) and None) or "compiled",
    )

    rag._build_visual_retriever(
        metadata_filter_rules=[{"field": "source_type", "operator": "eq", "value": "video_keyframe"}],
        metadata_filters_active=True,
        vector_store_kwargs=None,
    )

    assert compiled["rules"][0]["value"] == "video_keyframe"


def test_a_non_default_target_never_reuses_the_cached_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    """Build a fresh engine for a non-default target.

    The cached engine is the ``all`` one; reusing it would answer a visual
    turn from text.
    """
    rag = _rag(monkeypatch)
    rag.query_engine = cast(Any, "cached-all-engine")
    built: dict[str, Any] = {}

    def _build(self: RAG, **kwargs: Any) -> Any:
        """Record the target the engine was built for.

        Args:
            self (RAG): The instance.
            **kwargs (Any): Engine arguments.

        Returns:
            Any: A marker engine that answers with an empty response.
        """
        built["target"] = kwargs.get("retrieval_target")
        return types.SimpleNamespace(query=lambda _q: Response(response="", source_nodes=[]))

    monkeypatch.setattr(RAG, "build_query_engine", _build)
    monkeypatch.setattr(RAG, "_normalize_response_data", lambda self, *args, **kwargs: {"response": ""})
    monkeypatch.setattr(RAG, "_resolve_runtime_retrieval_settings", lambda self, **kwargs: _RETRIEVAL_SETTINGS)

    rag.run_query("what is at the gate", retrieval_target="visual")

    assert built["target"] == "visual"


_RETRIEVAL_SETTINGS: dict[str, Any] = {
    "label": "default",
    "parent_context_enabled": False,
    "vector_store_query_mode": types.SimpleNamespace(value="default"),
}


def test_a_visual_turn_reports_what_the_model_was_shown(monkeypatch: pytest.MonkeyPatch) -> None:
    """Report what the model was shown.

    Zero attached images is a degraded visual turn, and must not read as an
    ordinary one.
    """
    rag = _rag(monkeypatch)
    monkeypatch.setattr(
        RAG,
        "build_query_engine",
        lambda self, **kwargs: types.SimpleNamespace(query=lambda _q: Response(response="", source_nodes=[])),
    )
    monkeypatch.setattr(RAG, "_normalize_response_data", lambda self, *args, **kwargs: {"response": ""})
    monkeypatch.setattr(RAG, "_resolve_runtime_retrieval_settings", lambda self, **kwargs: _RETRIEVAL_SETTINGS)

    result = rag.run_query("what is at the gate", retrieval_target="visual")

    assert result["retrieval_target"] == "visual"
    assert result["visual"] == {"images_attached": 0}


def test_a_documents_turn_reports_no_visual_block(monkeypatch: pytest.MonkeyPatch) -> None:
    """Absent is what every non-visual turn looks like."""
    rag = _rag(monkeypatch)
    monkeypatch.setattr(
        RAG,
        "build_query_engine",
        lambda self, **kwargs: types.SimpleNamespace(query=lambda _q: Response(response="", source_nodes=[])),
    )
    monkeypatch.setattr(RAG, "_normalize_response_data", lambda self, *args, **kwargs: {"response": ""})
    monkeypatch.setattr(RAG, "_resolve_runtime_retrieval_settings", lambda self, **kwargs: _RETRIEVAL_SETTINGS)

    result = rag.run_query("what is in the report", retrieval_target="documents")

    assert result["retrieval_target"] == "documents"
    assert result.get("visual") is None
