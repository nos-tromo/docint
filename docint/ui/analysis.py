"""Analysis page: collection summarisation and NER overview."""

import json
import time
from typing import Any

import pandas as pd
import requests
import streamlit as st

from docint.ui.components import (
    aggregate_ner,
    entity_density_by_document,
    filter_entities,
    render_ner_overview,
    render_response_validation,
    render_source_item,
    render_summary_diagnostics,
)
from docint.ui.state import BACKEND_HOST


def render_analysis() -> None:
    """Render the analysis interface."""
    st.header("📈 Analysis")

    collection = st.session_state.selected_collection
    if not collection:
        st.info("Select a collection from the sidebar to view analysis.")
        return

    st.caption(f"📁 {collection}")

    ner_cache = _get_ner_cache()
    summary_cache = _get_summary_cache()

    if collection not in ner_cache:
        ner_result = _load_ner_analysis(collection, refresh=True)
    else:
        ner_result = ner_cache.get(collection, _empty_ner_result(collection))

    summary_state = summary_cache.setdefault(
        collection, _empty_summary_result(collection)
    )

    if st.button("🔄 Refresh NER analysis", width="stretch"):
        ner_result = _load_ner_analysis(collection, refresh=True)

    if ner_result.get("error"):
        st.warning(str(ner_result["error"]))

    summary_state = _render_summary_section(collection)
    _render_analysis_result(ner_result, collection, summary_state=summary_state)


def _update_summary_metadata(result: dict[str, Any], data: dict[str, Any]) -> None:
    """Update summary-related metadata fields in analysis state.

    Args:
        result: Mutable analysis-result state.
        data: Parsed SSE event payload.
    """
    if "validation_checked" in data:
        result["validation_checked"] = data.get("validation_checked")
    if "validation_mismatch" in data:
        result["validation_mismatch"] = data.get("validation_mismatch")
    if "validation_reason" in data:
        result["validation_reason"] = data.get("validation_reason")

    diagnostics = data.get("summary_diagnostics")
    if isinstance(diagnostics, dict):
        result["summary_diagnostics"] = diagnostics


def _empty_ner_result(collection: str) -> dict[str, Any]:
    """Build default NER analysis state for a collection.

    Args:
        collection: Currently selected collection name.

    Returns:
        Default state payload for NER analysis data and metadata.
    """
    return {
        "collection": collection,
        "ner": None,
        "ner_graph": {"nodes": [], "edges": [], "meta": {}},
        "hate_speech": [],
        "loaded": False,
        "error": None,
        "errors": {},
        "last_loaded_at": None,
    }


def _empty_summary_result(collection: str) -> dict[str, Any]:
    """Build default summary state for a collection.

    Args:
        collection: Currently selected collection name.

    Returns:
        Default state payload for optional summary output.
    """
    return {
        "collection": collection,
        "generated": False,
        "summary": "",
        "sources": [],
        "validation_checked": None,
        "validation_mismatch": None,
        "validation_reason": None,
        "summary_diagnostics": None,
        "error": None,
    }


def _get_ner_cache() -> dict[str, dict[str, Any]]:
    """Return mutable NER cache map keyed by collection.

    Returns:
        Mutable dictionary representing the NER cache.
    """
    cache = st.session_state.get("analysis_ner_by_collection")
    if not isinstance(cache, dict):
        cache = {}
        st.session_state.analysis_ner_by_collection = cache
    return cache


def _get_summary_cache() -> dict[str, dict[str, Any]]:
    """Return mutable summary cache map keyed by collection."""
    cache = st.session_state.get("analysis_summary_by_collection")
    if not isinstance(cache, dict):
        cache = {}
        st.session_state.analysis_summary_by_collection = cache
    return cache


def _fetch_ner_analysis(collection: str, *, refresh: bool = False) -> dict[str, Any]:
    """Fetch NER + graph + hate-speech payloads for a collection.

    Args:
        collection: Currently selected collection name.
        refresh: Whether to bypass backend NER cache.

    Returns:
        NER payload with load/error metadata.
    """
    _ = collection
    result = _empty_ner_result(collection)
    result["loaded"] = True
    errors: dict[str, str] = {}

    try:
        params = {"refresh": "true"} if refresh else None
        ner_resp = requests.get(
            f"{BACKEND_HOST}/collections/ner",
            params=params,
            timeout=120,
        )
        if ner_resp.status_code == 200:
            result["ner"] = ner_resp.json().get("sources", [])
        else:
            errors["ner"] = (
                f"NER request failed ({ner_resp.status_code}): {ner_resp.text.strip()}"
            )
            result["ner"] = []
    except Exception as exc:
        errors["ner"] = f"NER request failed: {exc}"
        result["ner"] = []

    try:
        graph_resp = requests.get(
            f"{BACKEND_HOST}/collections/ner/graph",
            params={"top_k_nodes": 75, "min_edge_weight": 2},
            timeout=120,
        )
        if graph_resp.status_code == 200:
            result["ner_graph"] = graph_resp.json()
        else:
            errors["ner_graph"] = (
                f"Entity graph request failed ({graph_resp.status_code}): "
                f"{graph_resp.text.strip()}"
            )
    except Exception as exc:
        errors["ner_graph"] = f"Entity graph request failed: {exc}"

    try:
        hate_resp = requests.get(
            f"{BACKEND_HOST}/collections/hate-speech",
            timeout=120,
        )
        if hate_resp.status_code == 200:
            result["hate_speech"] = hate_resp.json().get("results", [])
        else:
            errors["hate_speech"] = (
                f"Hate-speech request failed ({hate_resp.status_code}): "
                f"{hate_resp.text.strip()}"
            )
            result["hate_speech"] = []
    except Exception as exc:
        errors["hate_speech"] = f"Hate-speech request failed: {exc}"
        result["hate_speech"] = []

    result["errors"] = errors
    if errors:
        result["error"] = " ; ".join(errors.values())
    result["last_loaded_at"] = time.time()
    return result


def _load_ner_analysis(collection: str, *, refresh: bool) -> dict[str, Any]:
    """Load NER analysis data and persist it in session cache.

    Args:
        collection: Currently selected collection name.
        refresh: Whether to bypass backend NER cache.

    Returns:
        Loaded NER analysis payload.
    """
    with st.spinner("Loading entity and hate-speech analysis…"):
        result = _fetch_ner_analysis(collection, refresh=refresh)
    _get_ner_cache()[collection] = result
    return result


def _generate_summary(collection: str) -> None:
    """Generate summary for the selected collection and store it in cache.

    Args:
        collection: Currently selected collection name.
    """
    summary_placeholder = st.empty()
    full_summary = ""
    current_sources: list[dict[str, Any]] = []
    summary_state = _empty_summary_result(collection)

    with st.spinner("Generating summary…"):
        try:
            resp = requests.post(
                f"{BACKEND_HOST}/summarize/stream", stream=True, timeout=600
            )
            if resp.status_code == 200:
                for line in resp.iter_lines():
                    if not line:
                        continue
                    decoded = line.decode("utf-8")
                    if decoded.startswith("data: "):
                        data = json.loads(decoded[6:])
                        if "token" in data:
                            full_summary += data["token"]
                            summary_placeholder.markdown(full_summary + "▌")
                        elif "sources" in data:
                            current_sources = data["sources"]
                        _update_summary_metadata(summary_state, data)
                        if "error" in data:
                            summary_state["error"] = str(data["error"])
                summary_placeholder.markdown(full_summary)
                summary_state["summary"] = full_summary
                summary_state["sources"] = current_sources
                summary_state["generated"] = True
            else:
                summary_state["error"] = (
                    f"Summarisation failed ({resp.status_code}): {resp.text}"
                )
        except Exception as e:
            summary_state["error"] = f"Error: {e}"

    _get_summary_cache()[collection] = summary_state
    st.rerun()


def _render_summary_section(collection: str) -> dict[str, Any]:
    """Render manual summary controls and current summary output.

    Args:
        collection: Currently selected collection name.

    Returns:
        Summary state for the selected collection.
    """
    summary_cache = _get_summary_cache()
    summary_state = summary_cache.setdefault(
        collection, _empty_summary_result(collection)
    )

    st.markdown(f"### Summary of '{collection}'")
    st.caption("Summary generation uses LLM tokens.")
    button_label = (
        "Generate summary"
        if not summary_state.get("generated")
        else "Regenerate summary"
    )
    if st.button(button_label, type="primary", width="stretch"):
        _generate_summary(collection)

    if summary_state.get("error"):
        st.error(str(summary_state["error"]))

    if not summary_state.get("generated"):
        st.info("Summary has not been generated yet.")
        return summary_state

    st.markdown(str(summary_state.get("summary") or ""))
    render_response_validation(
        validation_checked=summary_state.get("validation_checked"),
        validation_mismatch=summary_state.get("validation_mismatch"),
        validation_reason=summary_state.get("validation_reason"),
    )
    render_summary_diagnostics(summary_state.get("summary_diagnostics"))

    sources = list(summary_state.get("sources") or [])
    if sources:
        with st.expander("Sources used for summary"):
            for src in sources:
                render_source_item(src, collection)

    return summary_state


def _entity_related_chunks(
    entity: str,
    sources: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return source chunks that mention a selected entity.

    Args:
        entity: The selected entity text to match.
        sources: List of source dictionaries containing chunk and entity information.

    Returns:
        A list of source dictionaries that contain the selected entity.
    """
    needle = str(entity or "").strip().lower()
    if not needle:
        return []

    matches: list[dict[str, Any]] = []
    for src in sources:
        entities = src.get("entities")
        if not isinstance(entities, list):
            continue
        if not any(
            isinstance(ent, dict)
            and (
                str(ent.get("text") or ent.get("name") or "").strip().lower() == needle
                or str(ent.get("key") or "").split("::", maxsplit=1)[0].strip().lower()
                == needle
            )
            for ent in entities
        ):
            continue

        chunk_text = str(
            src.get("chunk_text") or src.get("text") or src.get("preview_text") or ""
        ).strip()
        matches.append(
            {
                "chunk_id": str(src.get("chunk_id") or "").strip(),
                "chunk_text": chunk_text,
                "filename": str(src.get("filename") or "Unknown"),
                "page": src.get("page"),
                "row": src.get("row"),
            }
        )
    return matches


def _entity_chunks_to_txt(entity: str, chunks: list[dict[str, Any]]) -> str:
    """Build downloadable text payload for entity-linked chunks.

    Args:
        entity: The selected entity text.
        chunks: List of source dictionaries containing chunk information.

    Returns:
        A formatted string containing metadata and text of chunks related to the entity.
    """
    lines = [f"Entity: {entity}", ""]
    for idx, chunk in enumerate(chunks, start=1):
        lines.append(
            f"[{idx}] {chunk.get('filename')} page={chunk.get('page')} row={chunk.get('row')} "
            f"chunk_id={chunk.get('chunk_id')}"
        )
        lines.append(str(chunk.get("chunk_text") or ""))
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def _hate_speech_chunks_to_txt(chunks: list[dict[str, Any]]) -> str:
    """Build downloadable text payload for hate-speech findings.

    Args:
        chunks: List of source dictionaries containing hate-speech finding information.

    Returns:
        A formatted string containing metadata and text of chunks flagged for hate-speech.
    """
    lines = ["Flagged hate-speech chunks", ""]
    for idx, chunk in enumerate(chunks, start=1):
        lines.append(
            f"[{idx}] source={chunk.get('source_ref')} page={chunk.get('page')} "
            f"chunk_id={chunk.get('chunk_id')}"
        )
        lines.append(
            f"category={chunk.get('category')} confidence={chunk.get('confidence')}"
        )
        lines.append(f"reason={chunk.get('reason')}")
        lines.append(str(chunk.get("chunk_text") or ""))
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def _dot_escape(value: str) -> str:
    """Escape text for safe Graphviz DOT labels and identifiers.

    Args:
        value: Raw text value.

    Returns:
        Escaped text compatible with double-quoted DOT strings.
    """
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _graph_connected_subgraph(graph: dict[str, Any]) -> dict[str, Any]:
    """Return only nodes that participate in at least one edge.

    Args:
        graph: Full graph payload with ``nodes`` and ``edges``.

    Returns:
        Graph payload restricted to edge-connected nodes. If no edges are present,
        returns an empty node/edge payload for rendering.
    """
    nodes = list(graph.get("nodes") or [])
    edges = list(graph.get("edges") or [])
    if not edges:
        return {
            "nodes": [],
            "edges": [],
            "meta": dict(graph.get("meta") or {}),
        }

    connected_ids = {
        str(edge.get("source") or "") for edge in edges if str(edge.get("source") or "")
    }
    connected_ids.update(
        str(edge.get("target") or "") for edge in edges if str(edge.get("target") or "")
    )
    filtered_nodes = [
        node for node in nodes if str(node.get("id") or "") in connected_ids
    ]
    return {
        "nodes": filtered_nodes,
        "edges": edges,
        "meta": dict(graph.get("meta") or {}),
    }


def _graphviz_dot(graph: dict[str, Any], selected_entity: str | None = None) -> str:
    """Build a Graphviz DOT representation for entity graph rendering.

    Args:
        graph: A dictionary containing 'nodes' and 'edges' for the entity graph.
        selected_entity: Optional text of the currently selected entity for styling.

    Returns:
        A string in DOT format representing the graph, suitable for rendering with Graphviz.
    """
    nodes = list(graph.get("nodes") or [])
    edges = list(graph.get("edges") or [])
    selected = str(selected_entity or "").strip().lower()

    lines = ["graph Entities {"]
    lines.append('  graph [overlap=false, splines=true, bgcolor="transparent"];')
    lines.append('  node [shape=ellipse, style=filled, fillcolor="#E3F2FD"];')
    lines.append('  edge [color="#90A4AE"];')
    for node in nodes:
        node_id = str(node.get("id") or "")
        label = str(node.get("text") or node_id or "Unknown")
        mentions = int(node.get("mentions", 0) or 0)
        fill = "#90CAF9" if label.lower() == selected else "#E3F2FD"
        escaped_id = _dot_escape(node_id)
        escaped_label = _dot_escape(label)
        lines.append(
            f'  "{escaped_id}" [label="{escaped_label}\\n({mentions})", fillcolor="{fill}"];'
        )
    for edge in edges:
        source = str(edge.get("source") or "")
        target = str(edge.get("target") or "")
        if not source or not target:
            continue
        weight = int(edge.get("weight", 0) or 0)
        label = str(edge.get("label") or edge.get("kind") or "related")
        escaped_source = _dot_escape(source)
        escaped_target = _dot_escape(target)
        escaped_label = _dot_escape(label)
        lines.append(
            f'  "{escaped_source}" -- "{escaped_target}" [label="{escaped_label} ({weight})", penwidth="{max(1, min(weight, 5))}"];'
        )
    lines.append("}")
    return "\n".join(lines)


def _filter_graph_for_display(
    graph: dict[str, Any],
    *,
    top_k_nodes: int,
    min_edge_weight: int,
) -> dict[str, Any]:
    """Apply local graph filters for visualization controls.

    Args:
        graph: Full graph payload.
        top_k_nodes: Maximum number of nodes to include.
        min_edge_weight: Minimum edge weight threshold.

    Returns:
        Filtered graph payload.
    """
    nodes = list(graph.get("nodes") or [])
    edges = list(graph.get("edges") or [])
    if not nodes:
        return {"nodes": [], "edges": [], "meta": dict(graph.get("meta") or {})}

    ranked_nodes = sorted(
        nodes,
        key=lambda node: (
            -int(node.get("mentions", 0) or 0),
            str(node.get("text") or "").lower(),
        ),
    )
    selected_nodes = ranked_nodes[: max(1, int(top_k_nodes))]
    selected_node_ids = {str(node.get("id") or "") for node in selected_nodes}
    filtered_edges = [
        edge
        for edge in edges
        if int(edge.get("weight", 0) or 0) >= int(min_edge_weight)
        and str(edge.get("source") or "") in selected_node_ids
        and str(edge.get("target") or "") in selected_node_ids
    ]

    return {
        "nodes": selected_nodes,
        "edges": filtered_edges,
        "meta": {
            **dict(graph.get("meta") or {}),
            "node_count": len(selected_nodes),
            "edge_count": len(filtered_edges),
        },
    }


def _render_entities_tab(result: dict[str, Any], collection: str) -> None:
    """Render entity graph and entity-to-chunk drill-down UI.

    Args:
        result: The analysis result dictionary containing NER data.
        collection: The name of the currently selected collection.
    """
    ner_sources = result.get("ner") or []
    if not ner_sources:
        st.info("No entities or relations found in this collection.")
        return

    entities, _ = aggregate_ner(ner_sources)
    graph = result.get("ner_graph") or {}

    st.markdown("#### Entity graph")
    ctrl_cols = st.columns([1, 1])
    min_edge_weight = ctrl_cols[0].number_input(
        "Min edge weight",
        min_value=1,
        max_value=25,
        value=2,
        step=1,
        key=f"analysis_entity_min_edge_{collection}",
    )
    top_k_nodes = ctrl_cols[1].number_input(
        "Top nodes",
        min_value=10,
        max_value=500,
        value=75,
        step=5,
        key=f"analysis_entity_top_k_{collection}",
    )
    filtered_graph = _filter_graph_for_display(
        graph,
        top_k_nodes=int(top_k_nodes),
        min_edge_weight=int(min_edge_weight),
    )

    nodes = list(filtered_graph.get("nodes") or [])
    edges = list(filtered_graph.get("edges") or [])
    node_options = sorted(
        {
            str(n.get("text") or "").strip()
            for n in nodes
            if str(n.get("text") or "").strip()
        }
    )
    if not node_options:
        node_options = sorted({str(e.get("text") or "") for e in entities})
    if not node_options:
        st.caption("No entities available for chunk drill-down.")
        return
    selected_entity = st.selectbox(
        "Select entity to inspect related chunks",
        options=node_options,
        index=0,
        key=f"analysis_entity_selected_{collection}",
    )

    if nodes and not edges:
        st.info(
            "No graph edges match the current threshold. "
            "Lower `Min edge weight` and refresh."
        )

    graph_for_render = _graph_connected_subgraph(filtered_graph)
    if graph_for_render.get("nodes") and graph_for_render.get("edges"):
        st.graphviz_chart(
            _graphviz_dot(graph_for_render, selected_entity), width="stretch"
        )
    else:
        st.caption("No graph data available.")

    if selected_entity:
        chunks = _entity_related_chunks(selected_entity, ner_sources)
        st.markdown(f"#### Chunks related to **{selected_entity}**")
        if chunks:
            for idx, chunk in enumerate(chunks, start=1):
                label = (
                    f"{chunk.get('filename')} "
                    f"(page={chunk.get('page')}, row={chunk.get('row')}, id={chunk.get('chunk_id') or 'n/a'})"
                )
                with st.expander(f"Chunk {idx}: {label}"):
                    chunk_text = str(chunk.get("chunk_text") or "").strip()
                    if chunk_text:
                        st.write(chunk_text)
                    else:
                        st.caption("Chunk text unavailable for this record.")
            st.download_button(
                label="📥 Download related chunks (.txt)",
                data=_entity_chunks_to_txt(selected_entity, chunks),
                file_name=f"entity_{selected_entity}_{collection}.txt".replace(
                    " ", "_"
                ),
                mime="text/plain",
            )
        else:
            st.caption("No chunks were matched for the selected entity.")


def _render_hate_speech_tab(result: dict[str, Any], collection: str) -> None:
    """Render hate-speech findings with expandable chunk details.

    Displays a numbered summary table followed by expanders whose labels
    reference the corresponding table row so users can cross-reference
    findings easily.

    Args:
        result: The analysis result dictionary containing hate-speech findings.
        collection: The name of the currently selected collection.
    """
    findings = list(result.get("hate_speech") or [])
    st.markdown("#### Hate-speech findings")
    if not findings:
        st.info("No hate-speech flags detected for this collection.")
        return

    st.dataframe(
        {
            "#": list(range(1, len(findings) + 1)),
            "Category": [row.get("category") for row in findings],
            "Confidence": [row.get("confidence") for row in findings],
            "Reason": [row.get("reason") for row in findings],
            "Source": [row.get("source_ref") for row in findings],
            "Chunk ID": [row.get("chunk_id") for row in findings],
        },
        width="stretch",
        hide_index=True,
    )
    st.caption("Expand a row below to view the flagged chunk text.")
    for idx, row in enumerate(findings, start=1):
        category = row.get("category") or "unknown"
        confidence = row.get("confidence") or "unknown"
        reason = row.get("reason") or ""
        with st.expander(
            f"Row {idx} — {category.capitalize()}: {reason} (confidence: {confidence})"
        ):
            source = row.get("source_ref") or "Unknown source"
            page = row.get("page")
            chunk_id = row.get("chunk_id") or "n/a"
            reason = row.get("reason") or ""
            chunk_text = (row.get("chunk_text") or "").strip()
            st.markdown(
                f"**Source:** {source}  \n"
                f"**Page:** {page if page is not None else 'n/a'}  \n"
                f"**Chunk ID:** {chunk_id}  \n"
                f"**Reason:** {reason}  \n"
                f"**Text:** {chunk_text if chunk_text else 'n/a'}"
            )

    st.download_button(
        label="📥 Download flagged chunks (.txt)",
        data=_hate_speech_chunks_to_txt(findings),
        file_name=f"hate_speech_{collection}.txt",
        mime="text/plain",
    )


def _render_analysis_result(
    result: dict[str, Any],
    collection: str,
    *,
    summary_state: dict[str, Any] | None = None,
) -> None:
    """Render NER analysis output for a collection.

    Args:
        result: NER analysis result dictionary.
        collection: The name of the currently selected collection.
        summary_state: Optional summary state used when composing downloadable text.
    """
    summary_payload = summary_state or {}
    analysis_text = f"COLLECTION: {result['collection']}\n\n"
    if summary_payload.get("generated"):
        analysis_text += f"SUMMARY:\n{summary_payload.get('summary')}\n\n"
    if summary_payload.get("generated") and (
        summary_payload.get("validation_checked") is not None
        or summary_payload.get("validation_mismatch") is not None
        or summary_payload.get("validation_reason")
    ):
        analysis_text += "VALIDATION:\n"
        analysis_text += (
            f"checked={summary_payload.get('validation_checked')}\n"
            f"mismatch={summary_payload.get('validation_mismatch')}\n"
        )
        if summary_payload.get("validation_reason"):
            analysis_text += f"reason={summary_payload['validation_reason']}\n"
        analysis_text += "\n"
    diagnostics = summary_payload.get("summary_diagnostics")
    if isinstance(diagnostics, dict):
        analysis_text += "SUMMARY_DIAGNOSTICS:\n"
        analysis_text += (
            f"total_documents={diagnostics.get('total_documents')}\n"
            f"covered_documents={diagnostics.get('covered_documents')}\n"
            f"coverage_ratio={diagnostics.get('coverage_ratio')}\n"
            f"coverage_target={diagnostics.get('coverage_target')}\n\n"
        )
        uncovered_docs = diagnostics.get("uncovered_documents")
        if isinstance(uncovered_docs, list):
            uncovered_text = ", ".join(str(item) for item in uncovered_docs)
            analysis_text += f"uncovered_documents={uncovered_text}\n\n"

    st.markdown("---")
    tabs = st.tabs(["Overview", "Entities", "Hate Speech"])

    with tabs[0]:
        st.markdown("### Collection-wide Information Extraction")
        ner_sources = result.get("ner") or []
        if ner_sources:
            render_ner_overview(ner_sources)
            entities, relations = aggregate_ner(ner_sources)
            density_rows = entity_density_by_document(ner_sources)

            controls = st.columns([2, 2, 2, 2])
            query = controls[0].text_input(
                "Search entity",
                value="",
                key=f"analysis_entity_query_{collection}",
            )
            type_options = ["All"] + sorted(
                {
                    str(ent.get("type") or "Unlabeled")
                    for ent in entities
                    if str(ent.get("type") or "").strip()
                }
            )
            selected_type = controls[1].selectbox(
                "Type",
                options=type_options,
                key=f"analysis_entity_type_{collection}",
            )
            sort_label = controls[2].radio(
                "Sort",
                options=["Mentions", "Best score"],
                horizontal=True,
                key=f"analysis_entity_sort_{collection}",
            )
            min_mentions = controls[3].number_input(
                "Min mentions",
                min_value=1,
                max_value=1000,
                value=1,
                step=1,
                key=f"analysis_entity_min_mentions_{collection}",
            )

            filtered_entities = filter_entities(
                entities,
                query=query,
                entity_type=None if selected_type == "All" else selected_type,
                min_mentions=int(min_mentions),
                sort_by="score" if sort_label == "Best score" else "mentions",
            )
            st.dataframe(
                {
                    "Entity": [e["text"] for e in filtered_entities],
                    "Type": [e.get("type") or "Unlabeled" for e in filtered_entities],
                    "Mentions": [e["count"] for e in filtered_entities],
                    "Best score": [e.get("best_score") for e in filtered_entities],
                    "Sources": [", ".join(e["files"]) for e in filtered_entities],
                },
                width="stretch",
                hide_index=True,
            )
            if density_rows:
                st.markdown("#### Entity Density by Document")
                st.dataframe(density_rows, width="stretch", hide_index=True)

            entities_csv = pd.DataFrame(
                [
                    {
                        "entity": e["text"],
                        "type": e.get("type") or "Unlabeled",
                        "mentions": e["count"],
                        "best_score": e.get("best_score"),
                        "sources": ", ".join(e["files"]),
                    }
                    for e in filtered_entities
                ]
            ).to_csv(index=False)
            relations_csv = pd.DataFrame(
                [
                    {
                        "head": r["head"],
                        "label": r.get("label") or "rel",
                        "tail": r["tail"],
                        "mentions": r["count"],
                        "best_score": r.get("best_score"),
                        "sources": ", ".join(r["files"]),
                    }
                    for r in relations
                ]
            ).to_csv(index=False)

            d1, d2 = st.columns(2)
            d1.download_button(
                label="📥 Download entities (.csv)",
                data=entities_csv,
                file_name=f"entities_{collection}.csv",
                mime="text/csv",
            )
            d2.download_button(
                label="📥 Download relations (.csv)",
                data=relations_csv,
                file_name=f"relations_{collection}.csv",
                mime="text/csv",
            )

            analysis_text += "ENTITIES:\n"
            for ent in entities:
                analysis_text += (
                    f"- {ent['text']} ({ent.get('type', 'Unlabeled')}): "
                    f"{ent['count']} mentions in {', '.join(ent['files'])}\n"
                )
            analysis_text += "\nRELATIONS:\n"
            for rel in relations:
                analysis_text += (
                    f"- {rel['head']} --[{rel.get('label', 'rel')}]--> "
                    f"{rel['tail']}: {rel['count']} mentions in "
                    f"{', '.join(rel['files'])}\n"
                )
        else:
            st.info("No entities or relations found in this collection.")

    with tabs[1]:
        _render_entities_tab(result, collection)

    with tabs[2]:
        _render_hate_speech_tab(result, collection)

    st.download_button(
        label="📥 Download analysis (.txt)",
        data=analysis_text,
        file_name=f"analysis_{collection}.txt",
        mime="text/plain",
    )
