"""
Dashboard page: KPI metrics, quick-start guide, and recent activity.
"""

from typing import Any

import requests
import streamlit as st

from docint.ui.state import BACKEND_HOST, navigate_to


def render_dashboard() -> None:
    """
    Render the main dashboard page.
    """
    st.header("📊 Dashboard")

    collections: list[str] = st.session_state.get("_cached_collections", [])
    sessions: list[dict[str, Any]] = st.session_state.get("_cached_sessions", [])
    collection: str = st.session_state.selected_collection
    backend_online: bool = st.session_state.get("_backend_online", False)

    # Attempt a lightweight backend health-check if we have no cached data
    if not backend_online and not collections:
        try:
            r = requests.get(f"{BACKEND_HOST}/collections/list", timeout=5)
            backend_online = r.status_code == 200
        except Exception:
            backend_online = False

    # Fetch document count for the selected collection
    doc_count: int | None = None
    docs: list[dict[str, Any]] = []
    if collection:
        docs = _fetch_documents(collection)
        doc_count = len(docs) if docs is not None else None

    # ── KPI cards ──────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        with st.container(border=True):
            st.metric(
                "Backend",
                "Online" if backend_online else "Offline",
                help="Backend API reachability",
            )
    with c2:
        with st.container(border=True):
            st.metric("Collections", len(collections))
    with c3:
        with st.container(border=True):
            st.metric(
                "Documents",
                doc_count if doc_count is not None else "—",
                help="Documents in the current collection"
                if collection
                else "Select a collection first",
            )
    with c4:
        with st.container(border=True):
            st.metric("Sessions", len(sessions))

    st.markdown("")  # spacer

    # ── Empty state: no collection selected ────────────────────
    if not collection:
        _render_welcome(sessions)
        return

    # ── Collection overview + recent sessions ──────────────────
    col_left, col_right = st.columns([3, 2], gap="large")

    with col_left:
        st.subheader(f"📁 {collection}")

        # Document-type distribution chart
        if docs:
            type_counts: dict[str, int] = {}
            for d in docs:
                mt = d.get("mimetype") or "unknown"
                type_counts[mt] = type_counts.get(mt, 0) + 1
            if type_counts:
                st.caption("Documents by type")
                st.bar_chart(type_counts)
        else:
            st.info(
                "No documents found in this collection. "
                "Head to **Ingest** to upload files."
            )

        # Last ingestion summary
        ingest_summary = st.session_state.get("ingest_summary")
        if ingest_summary:
            st.caption("Last ingestion")
            m1, m2, m3 = st.columns(3)
            m1.metric("Total", ingest_summary.get("total", 0))
            m2.metric("Succeeded", ingest_summary.get("done", 0))
            m3.metric("Errors", ingest_summary.get("errors", 0))

        # TODO: Wire real latency / token-usage metrics when available.
        # Placeholder row — shown only as guidance for future work.
        st.caption("Performance")
        p1, p2 = st.columns(2)
        with p1:
            with st.container(border=True):
                st.metric("Avg. latency", "Not available")
        with p2:
            with st.container(border=True):
                st.metric("Token usage", "Not available")

    with col_right:
        _render_recent_sessions(sessions)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _fetch_documents(collection: str) -> list[dict[str, Any]]:
    """
    Fetch documents for *collection* from the backend.

    Args:
        collection: Collection name.

    Returns:
        List of document dicts, or empty list on failure.
    """
    try:
        resp = requests.get(f"{BACKEND_HOST}/collections/documents", timeout=10)
        if resp.status_code == 200:
            return resp.json().get("documents", [])
    except Exception:
        pass
    return []


def _render_welcome(sessions: list[dict[str, Any]]) -> None:
    """
    Render the first-visit / no-collection welcome block.

    Args:
        sessions: List of existing sessions (may be empty).
    """
    col_left, col_right = st.columns([3, 2], gap="large")

    with col_left:
        st.subheader("Welcome to DocInt")
        st.markdown(
            """
Get started in three easy steps:

1. **📥 Ingest** — Upload documents to create a searchable collection.
2. **💬 Chat** — Ask questions and receive answers grounded in your data.
3. **📈 Analyse** — Generate summaries and extract entities & relations.
            """
        )

        if st.button("📥  Go to Ingest", type="primary"):
            navigate_to("Ingest")

    with col_right:
        _render_recent_sessions(sessions)


def _render_recent_sessions(sessions: list[dict[str, Any]]) -> None:
    """
    Show the five most-recent chat sessions.

    Args:
        sessions: Full sessions list (will be sliced to five).
    """
    st.subheader("Recent Sessions")
    if not sessions:
        st.info("No sessions yet. Start chatting to create your first session.")
        return

    for s in sessions[:5]:
        with st.container(border=True):
            sc1, sc2 = st.columns([3, 1])
            with sc1:
                st.markdown(f"**{s['title']}**")
                if s.get("collection"):
                    st.caption(f"📁 {s['collection']}")
            with sc2:
                if st.button("Resume", key=f"dash_sess_{s['id']}"):
                    st.session_state.session_id = s["id"]
                    try:
                        h = requests.get(
                            f"{BACKEND_HOST}/sessions/{s['id']}/history",
                            timeout=10,
                        )
                        if h.status_code == 200:
                            st.session_state.messages = h.json().get("messages", [])
                    except Exception:
                        pass
                    navigate_to("Chat")
