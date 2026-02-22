"""
Analysis page: collection summarisation and NER overview.
"""

import json
from typing import Any

import requests
import streamlit as st

from docint.ui.components import (
    aggregate_ner,
    render_ner_overview,
    render_source_item,
)
from docint.ui.state import BACKEND_HOST


def render_analysis() -> None:
    """
    Render the analysis interface.
    """
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
    """
    Run summarisation and NER extraction, storing results in session state.

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
            ner_resp = requests.get(f"{BACKEND_HOST}/collections/ie", timeout=120)
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
    """
    Render a previously computed analysis result.

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
