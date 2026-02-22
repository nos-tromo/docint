"""
DocInt Streamlit application entry-point.

This module wires together the sidebar, page router, and shared chrome.
All page rendering logic lives in ``docint.ui.*`` sub-modules.

Public re-exports
-----------------
The following symbols are re-exported for backward-compatibility and
direct use by tests:

* ``_format_score``
* ``_normalize_entities``
* ``_normalize_relations``
* ``_source_label``
* ``_aggregate_ner``
"""

import sys
from pathlib import Path
from typing import Callable

import streamlit as st
from loguru import logger
from streamlit.runtime import exists
from streamlit.web import cli as st_cli

from docint.ui.analysis import render_analysis
from docint.ui.chat import render_chat
from docint.ui.components import (
    aggregate_ner,
    format_score,
    normalize_entities,
    normalize_relations,
    source_label,
)
from docint.ui.dashboard import render_dashboard
from docint.ui.ingest import render_ingestion
from docint.ui.inspector import render_inspector
from docint.ui.sidebar import render_sidebar
from docint.ui.state import init_session_state
from docint.ui.theme import apply_custom_css, configure_page, render_footer
from docint.utils.env_cfg import set_offline_env
from docint.utils.logging_cfg import setup_logging

# ---------------------------------------------------------------------------
# Backward-compatible aliases used by tests (e.g. test_app_ner.py)
# ---------------------------------------------------------------------------
_format_score = format_score
_normalize_entities = normalize_entities
_normalize_relations = normalize_relations
_source_label = source_label
_aggregate_ner = aggregate_ner

# Page registry: maps page name -> render callable
_PAGE_RENDERERS: dict[str, Callable[[], None]] = {
    "Dashboard": render_dashboard,
    "Chat": render_chat,
    "Ingest": render_ingestion,
    "Analysis": render_analysis,
    "Inspector": render_inspector,
}


# ---------------------------------------------------------------------------
# Application setup & main loop
# ---------------------------------------------------------------------------


def setup_app() -> None:
    """
    Initialise page config, CSS, logging, and session state.

    Must be the first function called during each script run.
    """
    set_offline_env()
    setup_logging()
    configure_page()
    apply_custom_css()
    init_session_state()


def main() -> None:
    """
    Main Streamlit app entry-point: sidebar -> active page -> footer.
    """
    setup_app()
    render_sidebar()

    # -- Active-collection status bar --
    collection = st.session_state.selected_collection
    if collection:
        st.caption(f"Currently using collection: **{collection}**")

    # -- Route to the selected page --
    page = st.session_state.get("current_page", "Dashboard")
    renderer = _PAGE_RENDERERS.get(page, render_dashboard)
    renderer()

    render_footer()


# ---- Streamlit CLI wrapper ----------------------------------------------- #
def run() -> None:
    """
    CLI entry-point for the Streamlit app.

    Re-writes ``sys.argv`` so that ``streamlit run app.py <extra-args>`` is
    invoked transparently when the user runs ``uv run docint``.
    """
    app_path = Path(__file__).resolve()
    sys.argv = ["streamlit", "run", str(app_path)] + sys.argv[1:]
    sys.exit(st_cli.main())


if __name__ == "__main__":
    try:
        if exists():
            main()
        else:
            run()
    except ImportError as e:
        logger.exception(f"Failed to run the Streamlit app: {e}")
        run()
