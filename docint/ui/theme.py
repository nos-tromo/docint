"""Page configuration, custom CSS, and shared chrome (header / footer)."""

import streamlit as st


def configure_page() -> None:
    """
    Set the Streamlit page config.  Must be the **first** Streamlit call.
    """
    st.set_page_config(
        page_title="DocInt · Document Intelligence",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def apply_custom_css() -> None:
    """
    Inject lightweight CSS for consistent spacing and typography.
    """
    st.markdown(
        """
        <style>
        /* Metric cards: subtle background */
        div[data-testid="stMetric"] {
            padding: 0.25rem 0;
        }

        /* Expander rounded corners */
        div[data-testid="stExpander"] {
            border-radius: 0.5rem;
        }

        /* Footer */
        .docint-footer {
            text-align: center;
            color: #888;
            padding: 1rem 0 0.5rem;
            font-size: 0.85rem;
        }
        .docint-footer a {
            color: inherit;
            text-decoration: none;
            font-weight: 600;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_footer() -> None:
    """
    Render a compact footer with a GitHub link.
    """
    st.divider()
    st.markdown(
        '<div class="docint-footer">'
        "Powered by "
        '<a href="https://github.com/nos-tromo/docint" target="_blank">DocInt</a>'
        "</div>",
        unsafe_allow_html=True,
    )
