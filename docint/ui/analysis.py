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

    # ── Action ────────────────────────────────────────────────
    if st.button("🔬 Run Analysis", type="primary", use_container_width=True):
        _execute_analysis(collection)

    # ── Stored result ─────────────────────────────────────────
    result = st.session_state.analysis_result
    if result and result.get("collection") == collection:
        _render_analysis_result(result, collection)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _execute_analysis(collection: str) -> None:
    """Run summarisation and NER extraction, storing results in session state.

    Args:
        collection: Currently selected collection name.
    """
    st.session_state.analysis_result = {
        "summary": "",
        "sources": [],
        "ie": None,
        "collection": collection,
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
                        elif "error" in data:
                            st.error(f"Error: {data['error']}")
                summary_placeholder.markdown(full_summary)
                st.session_state.analysis_result["summary"] = full_summary
                st.session_state.analysis_result["sources"] = current_sources
            else:
                st.error(f"Summarisation failed: {resp.text}")
        except Exception as e:
            st.error(f"Error: {e}")

    with st.spinner("Aggregating entities and relations…"):
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

    st.rerun()


def _render_analysis_result(result: dict[str, Any], collection: str) -> None:
    """Render a previously computed analysis result.

    Args:
        result: Stored analysis dict with keys ``summary``, ``sources``,
            ``ie``, ``collection``.
        collection: Currently selected collection name.
    """
    st.markdown(f"### Summary of '{result['collection']}'")
    st.markdown(result["summary"])

    sources = result.get("sources", [])
    analysis_text = f"COLLECTION: {result['collection']}\n\n"
    analysis_text += f"SUMMARY:\n{result['summary']}\n\n"

    if sources:
        with st.expander("Sources used for summary"):
            for src in sources:
                render_source_item(src, collection)

    # ── NER overview ──────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Collection-wide Information Extraction")

    ner_sources = result.get("ie") or []
    if ner_sources:
        render_ner_overview(ner_sources)

        entities, relations = aggregate_ner(ner_sources)
        density_rows = entity_density_by_document(ner_sources)

        st.markdown("#### Explore Entities")
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

        if filtered_entities:
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
        else:
            st.caption("No entities match the current filters.")

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

    st.download_button(
        label="📥 Download analysis (.txt)",
        data=analysis_text,
        file_name=f"analysis_{collection}.txt",
        mime="text/plain",
    )
