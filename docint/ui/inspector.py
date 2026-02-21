"""Inspector page: document listing and detail viewer."""

from typing import Any

import pandas as pd
import requests
import streamlit as st

from docint.ui.state import BACKEND_HOST, BACKEND_PUBLIC_HOST


def render_inspector() -> None:
    """Render the collection inspector."""
    st.header("🔍 Inspector")

    collection = st.session_state.selected_collection
    if not collection:
        st.info("Select a collection from the sidebar to inspect documents.")
        return

    st.caption(f"📁 {collection}")

    # ── Load action ───────────────────────────────────────────
    if st.button("🔄 Load Documents", type="primary", use_container_width=True):
        with st.spinner("Fetching document list…"):
            try:
                resp = requests.get(f"{BACKEND_HOST}/collections/documents", timeout=30)
                if resp.status_code == 200:
                    st.session_state.inspector_docs = resp.json().get("documents", [])
                else:
                    st.error(f"Failed to fetch documents: {resp.text}")
            except Exception as e:
                st.error(f"Error fetching documents: {e}")

    docs: list[dict[str, Any]] = st.session_state.get("inspector_docs", [])
    if not docs:
        st.info("Click **Load Documents** to view the documents in this collection.")
        return

    # ── Overview ──────────────────────────────────────────────
    with st.container(border=True):
        st.metric("Total Documents", len(docs))

    st.markdown("")  # spacer

    # ── Table ────────────────────────────────────────────────
    display_data = _build_display_data(docs)
    st.dataframe(display_data, use_container_width=True, hide_index=True)

    csv_data = pd.DataFrame(display_data).to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download overview (.csv)",
        data=csv_data,
        file_name=f"inspector_{collection}.csv",
        mime="text/csv",
    )

    # ── Detail expanders ──────────────────────────────────────
    for doc in docs:
        mimetype = doc.get("mimetype") or "unknown"
        with st.expander(f"📄 {doc['filename']} ({mimetype})"):
            c1, c2, c3 = st.columns(3)

            if "max_rows" in doc:
                c1.metric("Rows", doc["max_rows"])
            elif "max_duration" in doc:
                c1.metric("Duration", _format_duration(doc["max_duration"]))
            else:
                c1.metric("Pages", doc.get("page_count", 0))

            c2.metric("Nodes", doc.get("node_count", 0))
            c3.metric("Mimetype", mimetype)

            if doc.get("entity_types"):
                st.caption(f"Entities: {', '.join(doc['entity_types'])}")
            if doc.get("file_hash"):
                st.caption(f"File Hash: {doc['file_hash']}")
                link = (
                    f"{BACKEND_PUBLIC_HOST}/sources/preview"
                    f"?collection={collection}"
                    f"&file_hash={doc['file_hash']}"
                )
                st.markdown(f"[View Original File]({link})")


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _format_duration(seconds: float | int) -> str:
    """Format a duration in seconds to a human-readable string.

    Args:
        seconds: Duration in seconds.

    Returns:
        Formatted duration string.
    """
    total = int(seconds)
    hours = total // 3600
    mins = (total % 3600) // 60
    secs = total % 60
    if hours > 0:
        return f"{hours}h {mins}m {secs}s"
    return f"{mins}m {secs}s"


def _build_display_data(docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build a table-ready list of dicts from raw document payloads.

    Args:
        docs: Raw document dicts from the backend.

    Returns:
        List of dicts suitable for ``st.dataframe``.
    """
    display: list[dict[str, Any]] = []
    for d in docs:
        entry: dict[str, Any] = {
            "Filename": d["filename"],
            "Nodes": d.get("node_count", 0),
            "Type": d.get("mimetype") or "—",
        }
        if "max_rows" in d:
            entry["Length"] = f"{d['max_rows']} rows"
        elif "max_duration" in d:
            entry["Length"] = _format_duration(d["max_duration"])
        elif d.get("page_count", 0) > 0:
            entry["Length"] = f"{d['page_count']} pages"
        else:
            entry["Length"] = "—"
        display.append(entry)
    return display
