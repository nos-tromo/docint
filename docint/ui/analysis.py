"""Analysis page: collection summarisation and NER overview."""

import json
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
    render_summary_diagnostics,
    render_source_item,
)
from docint.ui.state import BACKEND_HOST


def render_analysis() -> None:
    """Render the analysis interface."""
    st.header("📈 Analysis")

    collection = st.session_state.selected_collection
    if not collection:
        st.info("Select a collection from the sidebar to run analysis.")
        return

    st.caption(f"📁 {collection}")

    if st.button("🔬 Run Analysis", type="primary", use_container_width=True):
        _execute_analysis(collection)

    result = st.session_state.analysis_result
    if result and result.get("collection") == collection:
        _render_analysis_result(result, collection)


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


def _execute_analysis(collection: str) -> None:
    """Run summarization and IE lookups, storing results in session state.

    Args:
        collection: Currently selected collection name.
    """
    st.session_state.analysis_result = {
        "summary": "",
        "sources": [],
        "ie": None,
        "ie_graph": {"nodes": [], "edges": [], "meta": {}},
        "hate_speech": [],
        "collection": collection,
        "validation_checked": None,
        "validation_mismatch": None,
        "validation_reason": None,
        "summary_diagnostics": None,
    }
    summary_placeholder = st.empty()
    full_summary = ""
    current_sources: list[dict[str, Any]] = []

    st.markdown(f"### Summary of '{collection}'")

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
                        _update_summary_metadata(st.session_state.analysis_result, data)
                        if "error" in data:
                            st.error(f"Error: {data['error']}")
                summary_placeholder.markdown(full_summary)
                st.session_state.analysis_result["summary"] = full_summary
                st.session_state.analysis_result["sources"] = current_sources
            else:
                st.error(f"Summarisation failed: {resp.text}")
        except Exception as e:
            st.error(f"Error: {e}")

    with st.spinner("Aggregating entities, graph, and hate-speech flags…"):
        try:
            ner_resp = requests.get(f"{BACKEND_HOST}/collections/ner", timeout=120)
            if ner_resp.status_code == 200:
                st.session_state.analysis_result["ie"] = ner_resp.json().get(
                    "sources", []
                )
            else:
                st.session_state.analysis_result["ie"] = []
        except Exception:
            st.session_state.analysis_result["ie"] = []

        try:
            graph_resp = requests.get(
                f"{BACKEND_HOST}/collections/ner/graph",
                params={"top_k_nodes": 75, "min_edge_weight": 2},
                timeout=120,
            )
            if graph_resp.status_code == 200:
                st.session_state.analysis_result["ie_graph"] = graph_resp.json()
        except Exception:
            st.session_state.analysis_result["ie_graph"] = {
                "nodes": [],
                "edges": [],
                "meta": {},
            }

        try:
            hate_resp = requests.get(
                f"{BACKEND_HOST}/collections/hate-speech",
                timeout=120,
            )
            if hate_resp.status_code == 200:
                st.session_state.analysis_result["hate_speech"] = hate_resp.json().get(
                    "results", []
                )
            else:
                st.session_state.analysis_result["hate_speech"] = []
        except Exception:
            st.session_state.analysis_result["hate_speech"] = []

    st.rerun()


def _entity_related_chunks(
    entity: str,
    sources: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return source chunks that mention a selected entity."""
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
            and str(ent.get("text") or "").strip().lower() == needle
            for ent in entities
        ):
            continue

        chunk_text = str(src.get("chunk_text") or "").strip()
        if not chunk_text:
            continue
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
    """Build downloadable text payload for entity-linked chunks."""
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
    """Build downloadable text payload for hate-speech findings."""
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


def _graphviz_dot(graph: dict[str, Any], selected_entity: str | None = None) -> str:
    """Build a Graphviz DOT representation for entity graph rendering."""
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
        lines.append(
            f'  "{node_id}" [label="{label}\\n({mentions})", fillcolor="{fill}"];'
        )
    for edge in edges:
        source = str(edge.get("source") or "")
        target = str(edge.get("target") or "")
        if not source or not target:
            continue
        weight = int(edge.get("weight", 0) or 0)
        label = str(edge.get("label") or edge.get("kind") or "related")
        lines.append(
            f'  "{source}" -- "{target}" [label="{label} ({weight})", penwidth="{max(1, min(weight, 5))}"];'
        )
    lines.append("}")
    return "\n".join(lines)


def _render_entities_tab(result: dict[str, Any], collection: str) -> None:
    """Render entity graph and entity-to-chunk drill-down UI."""
    ner_sources = result.get("ie") or []
    if not ner_sources:
        st.info("No entities or relations found in this collection.")
        return

    entities, _ = aggregate_ner(ner_sources)
    graph = result.get("ie_graph") or {}

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
    if st.button("Refresh graph", key=f"analysis_entity_graph_refresh_{collection}"):
        try:
            graph_resp = requests.get(
                f"{BACKEND_HOST}/collections/ner/graph",
                params={
                    "top_k_nodes": int(top_k_nodes),
                    "min_edge_weight": int(min_edge_weight),
                },
                timeout=120,
            )
            if graph_resp.status_code == 200:
                graph = graph_resp.json()
                st.session_state.analysis_result["ie_graph"] = graph
            else:
                st.warning("Unable to refresh graph.")
        except Exception:
            st.warning("Unable to refresh graph.")

    nodes = list(graph.get("nodes") or [])
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

    if nodes:
        st.graphviz_chart(
            _graphviz_dot(graph, selected_entity), use_container_width=True
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
                    st.write(chunk.get("chunk_text"))
            st.download_button(
                label="📥 Download related chunks (.txt)",
                data=_entity_chunks_to_txt(selected_entity, chunks),
                file_name=f"entity_{selected_entity}_{collection}.txt".replace(
                    " ", "_"
                ),
                mime="text/plain",
            )
        else:
            st.caption("No chunk text was available for the selected entity.")


def _render_hate_speech_tab(result: dict[str, Any], collection: str) -> None:
    """Render hate-speech findings with expandable chunk details."""
    findings = list(result.get("hate_speech") or [])
    st.markdown("#### Hate-speech findings")
    if not findings:
        st.info("No hate-speech flags detected for this collection.")
        return

    st.dataframe(
        {
            "Source": [row.get("source_ref") for row in findings],
            "Category": [row.get("category") for row in findings],
            "Confidence": [row.get("confidence") for row in findings],
            "Reason": [row.get("reason") for row in findings],
            "Chunk ID": [row.get("chunk_id") for row in findings],
        },
        use_container_width=True,
        hide_index=True,
    )
    for idx, row in enumerate(findings, start=1):
        with st.expander(
            f"{idx}. {row.get('source_ref') or 'Unknown source'} "
            f"(confidence: {row.get('confidence')})"
        ):
            st.caption(
                f"Category: {row.get('category')} · Page: {row.get('page')} · "
                f"Chunk ID: {row.get('chunk_id')}"
            )
            st.write(row.get("chunk_text") or "")

    st.download_button(
        label="📥 Download flagged chunks (.txt)",
        data=_hate_speech_chunks_to_txt(findings),
        file_name=f"hate_speech_{collection}.txt",
        mime="text/plain",
    )


def _render_analysis_result(result: dict[str, Any], collection: str) -> None:
    """Render a previously computed analysis result."""
    st.markdown(f"### Summary of '{result['collection']}'")
    st.markdown(result["summary"])
    render_response_validation(
        validation_checked=result.get("validation_checked"),
        validation_mismatch=result.get("validation_mismatch"),
        validation_reason=result.get("validation_reason"),
    )
    render_summary_diagnostics(result.get("summary_diagnostics"))

    sources = result.get("sources", [])
    analysis_text = f"COLLECTION: {result['collection']}\n\n"
    analysis_text += f"SUMMARY:\n{result['summary']}\n\n"
    if (
        result.get("validation_checked") is not None
        or result.get("validation_mismatch") is not None
        or result.get("validation_reason")
    ):
        analysis_text += "VALIDATION:\n"
        analysis_text += (
            f"checked={result.get('validation_checked')}\n"
            f"mismatch={result.get('validation_mismatch')}\n"
        )
        if result.get("validation_reason"):
            analysis_text += f"reason={result['validation_reason']}\n"
        analysis_text += "\n"
    diagnostics = result.get("summary_diagnostics")
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

    if sources:
        with st.expander("Sources used for summary"):
            for src in sources:
                render_source_item(src, collection)

    st.markdown("---")
    tabs = st.tabs(["Overview", "Entities", "Hate Speech"])

    with tabs[0]:
        st.markdown("### Collection-wide Information Extraction")
        ner_sources = result.get("ie") or []
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
                use_container_width=True,
                hide_index=True,
            )
            if density_rows:
                st.markdown("#### Entity Density by Document")
                st.dataframe(density_rows, use_container_width=True, hide_index=True)

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
