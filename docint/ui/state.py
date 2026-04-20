"""Centralized session-state initialization and backend configuration."""

import streamlit as st

from docint.utils.env_cfg import load_host_env

host_cfg = load_host_env()

BACKEND_HOST: str = host_cfg.backend_host
BACKEND_PUBLIC_HOST: str = host_cfg.backend_public_host or BACKEND_HOST
"""Public-facing backend URL (used for download / preview links shown to the user)."""

PAGES: list[str] = [
    "Dashboard",
    "Ingest",
    "Chat",
    "Analysis",
    "Inspector",
]
"""Ordered list of page names for sidebar navigation."""

PAGE_ICONS: dict[str, str] = {
    "Dashboard": "📊",
    "Ingest": "📥",
    "Chat": "💬",
    "Analysis": "📈",
    "Inspector": "🔍",
}
"""Emoji icon for each page."""


def init_session_state() -> None:
    """Initialise all session-state keys with sane defaults.

    Must be called once, before any widget is rendered, so that every key
    referenced elsewhere already exists.
    """
    defaults: dict[str, object] = {
        "current_page": "Dashboard",
        "messages": [],
        "session_id": None,
        "selected_collection": "",
        "preview_url": None,
        "chat_running": False,
        "chat_pending_prompt": None,
        "chat_query_mode": "answer",
        "chat_retrieval_mode": "stateless",
        "chat_filter_enabled": False,
        "chat_filter_scope": "All content",
        "chat_filter_mime_pattern": "",
        "chat_filter_date_field": "None",
        "chat_filter_start_date": "",
        "chat_filter_end_date": "",
        "chat_filter_hate_speech_only": False,
        "chat_filter_custom_count": 0,
        "analysis_result": None,
        "analysis_ner_by_collection": {},
        "analysis_summary_by_collection": {},
        "ingest_summary": None,
        "inspector_docs": [],
        # Sidebar-fetched caches (written by sidebar, read by dashboard)
        "_cached_collections": [],
        "_cached_sessions": [],
        "_backend_online": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def navigate_to(page: str) -> None:
    """Set the active page and trigger a rerun.

    Args:
        page (str): Target page name (must be one of ``PAGES``).
    """
    st.session_state.current_page = page
    st.rerun()
