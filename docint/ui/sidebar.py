"""Sidebar: navigation, collection selector, and chat-session management."""

import requests
import streamlit as st
from loguru import logger

from docint.ui.state import PAGE_ICONS, PAGES
from docint.utils.env_cfg import load_host_env

BACKEND_HOST = load_host_env().backend_host


def render_sidebar() -> None:
    """Render the full sidebar: branding, nav, collection control, sessions."""
    with st.sidebar:
        # ── Branding ──────────────────────────────────────────────
        st.markdown("## 🔍 docint")
        st.caption("Document Intelligence")

        st.divider()

        # ── Navigation ───────────────────────────────────────────
        selected_page = st.radio(
            "Navigation",
            options=PAGES,
            format_func=lambda p: f"{PAGE_ICONS.get(p, '')}  {p}",
            index=PAGES.index(st.session_state.current_page)
            if st.session_state.current_page in PAGES
            else 0,
            label_visibility="collapsed",
        )
        if selected_page != st.session_state.current_page:
            st.session_state.current_page = selected_page
            st.rerun()

        st.divider()

        # ── Collection selector ───────────────────────────────────
        _render_collection_selector()

        st.divider()

        # ── Chat sessions ────────────────────────────────────────
        _render_chat_sessions()


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _render_collection_selector() -> None:
    """Render the collection dropdown and delete popover."""
    st.markdown("##### 📁 Collection")

    try:
        resp = requests.get(f"{BACKEND_HOST}/collections/list", timeout=10)
        if resp.status_code == 200:
            cols = [
                c
                for c in resp.json()
                if not c.endswith("_dockv") and not c.endswith("_images")
            ]
        else:
            logger.error("Failed to fetch collections: {}", resp.text)
            st.error(f"Failed to fetch collections: {resp.text}")
            cols = []
    except Exception as e:
        st.error(f"Backend offline: {e}")
        cols = []

    # Persist for dashboard consumption
    st.session_state._cached_collections = cols
    st.session_state._backend_online = len(cols) > 0 or st.session_state.get(
        "_backend_online", False
    )

    if not cols:
        st.caption("No collections available.")
        return

    # Determine default index
    index = 0
    if st.session_state.selected_collection in cols:
        index = cols.index(st.session_state.selected_collection)

    selected = st.selectbox(
        label="Collection",
        options=[c.strip() for c in cols],
        index=index,
        label_visibility="collapsed",
    )

    # Delete popover
    if selected:
        with st.popover("🗑️ Delete collection"):
            st.write(f"Are you sure you want to delete collection **{selected}**?")
            st.warning("This action cannot be undone.")
            if st.button("Yes, delete", type="primary"):
                try:
                    r = requests.delete(
                        f"{BACKEND_HOST}/collections/{selected}", timeout=10
                    )
                    if r.status_code == 200:
                        if st.session_state.selected_collection == selected:
                            st.session_state.selected_collection = ""
                            st.session_state.messages = []
                            st.session_state.session_id = None
                        st.success(f"Collection **{selected}** deleted.")
                        st.rerun()
                    else:
                        st.error(f"Failed to delete: {r.text}")
                except Exception as e:
                    st.error(f"Error deleting collection: {e}")

    # Handle collection switch
    if selected and selected != st.session_state.selected_collection:
        try:
            r = requests.post(
                f"{BACKEND_HOST}/collections/select",
                json={"name": selected},
                timeout=10,
            )
            if r.status_code == 200:
                st.session_state.selected_collection = selected
                st.session_state.messages = []
                st.session_state.session_id = None
                st.rerun()
            else:
                logger.error("Failed to select collection: {}", r.text)
                st.error(f"Failed to select collection: {r.text}")
        except Exception as e:
            st.error(f"Backend error: {e}")


def _render_chat_sessions() -> None:
    """Render chat session list with new/delete controls."""
    st.markdown("##### 💬 Chat Sessions")

    # New chat button
    if st.button(
        "➕ New Chat",
        width="stretch",
        type="primary" if st.session_state.session_id is None else "secondary",
    ):
        st.session_state.session_id = None
        st.session_state.messages = []
        st.session_state.current_page = "Chat"
        st.rerun()

    try:
        resp = requests.get(f"{BACKEND_HOST}/sessions/list", timeout=10)
        if resp.status_code != 200:
            st.error("Failed to load sessions")
            return
        sessions: list[dict] = resp.json().get("sessions", [])
    except Exception as e:
        logger.warning("Failed to fetch sessions: {}", e)
        sessions = []

    # Persist for dashboard consumption
    st.session_state._cached_sessions = sessions

    if not sessions:
        st.caption("No sessions yet.")
        return

    for s in sessions:
        b_type = "primary" if s["id"] == st.session_state.session_id else "secondary"
        if st.button(
            s["title"],
            key=f"sess_{s['id']}",
            width="stretch",
            type=b_type,
        ):
            if st.session_state.session_id != s["id"]:
                # Switch collection if the session belongs to a different one
                clicked_col = s.get("collection")
                if clicked_col and clicked_col != st.session_state.selected_collection:
                    try:
                        r = requests.post(
                            f"{BACKEND_HOST}/collections/select",
                            json={"name": clicked_col},
                            timeout=10,
                        )
                        if r.status_code == 200:
                            st.session_state.selected_collection = clicked_col
                        else:
                            st.error(f"Failed to switch to collection '{clicked_col}'")
                    except Exception as e:
                        st.error(f"Collection switch error: {e}")

                st.session_state.session_id = s["id"]

                # Load history
                try:
                    h_resp = requests.get(
                        f"{BACKEND_HOST}/sessions/{s['id']}/history",
                        timeout=10,
                    )
                    if h_resp.status_code == 200:
                        st.session_state.messages = h_resp.json().get("messages", [])
                except Exception:
                    pass

                st.session_state.current_page = "Chat"
                st.rerun()

    # Delete active session
    if st.session_state.session_id is not None:
        st.divider()
        if st.button(
            "🗑️ Delete Current Chat",
            type="secondary",
            width="stretch",
        ):
            try:
                requests.delete(
                    f"{BACKEND_HOST}/sessions/{st.session_state.session_id}",
                    timeout=10,
                )
            except Exception:
                pass
            st.session_state.session_id = None
            st.session_state.messages = []
            st.rerun()
