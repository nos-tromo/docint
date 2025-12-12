import json
import sys
from pathlib import Path

import requests
import streamlit as st
from loguru import logger
from streamlit.runtime import exists
from streamlit.web import cli as st_cli

from docint.utils.env_cfg import load_host_env, set_offline_env
from docint.utils.logging_cfg import setup_logging

BACKEND_HOST = load_host_env().backend_host


def setup_app():
    """
    Initialize application state and configuration.
    """
    set_offline_env()
    setup_logging()
    st.set_page_config(page_title="DocInt", layout="wide")
    st.title("Document Intelligence")

    # Custom CSS to spread tabs evenly
    st.markdown(
        """
        <style>
            .stTabs [data-baseweb="tab-list"] {
                gap: 2px;
            }
            .stTabs [data-baseweb="tab"] {
                height: 50px;
                white-space: pre-wrap;
                background-color: transparent;
                border-radius: 4px 4px 0px 0px;
                gap: 1px;
                padding-top: 10px;
                padding-bottom: 10px;
                flex: 1;
            }
            .stTabs [data-baseweb="tab"] p {
                font-size: 1.1rem !important;
            }
        </style>
    """,
        unsafe_allow_html=True,
    )

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "session_id" not in st.session_state:
        st.session_state.session_id = None
    if "selected_collection" not in st.session_state:
        st.session_state.selected_collection = ""
    if "preview_url" not in st.session_state:
        st.session_state.preview_url = None


def render_sidebar():
    """
    Render the sidebar controls.
    """
    with st.sidebar:
        # Add top padding so sidebar content starts lower
        st.markdown("<div style='height:61px'></div>", unsafe_allow_html=True)

        # Collection Selection
        try:
            resp = requests.get(f"{BACKEND_HOST}/collections/list")
            if resp.status_code == 200:
                cols = resp.json()
                # If we have a selected collection in state, try to keep it selected
                index = 0
                if st.session_state.selected_collection in cols:
                    index = cols.index(st.session_state.selected_collection)

                selected = st.selectbox(
                    "Collection",
                    [col.strip() for col in cols],
                    index=index if cols else None,
                )

                if selected and selected != st.session_state.selected_collection:
                    # Notify backend of selection
                    r = requests.post(
                        f"{BACKEND_HOST}/collections/select", json={"name": selected}
                    )
                    if r.status_code == 200:
                        st.session_state.selected_collection = selected
                        st.session_state.messages = []  # Clear chat on switch
                        st.session_state.session_id = None
                        st.rerun()
                    else:
                        logger.error(f"Failed to select collection: {r.text}")
                        st.error(f"Failed to select collection: {r.text}")
            else:
                logger.error(f"Failed to fetch collections: {resp.text}")
                st.error(f"Failed to fetch collections: {resp.text}")
                cols = []

        except Exception as e:
            st.error(f"Backend offline: {e}")
            cols = []

        st.divider()

        # Session Management
        st.subheader("Chat History")

        # New Chat Button
        if st.button(
            "➕ New Chat",
            use_container_width=True,
            type="primary" if st.session_state.session_id is None else "secondary",
        ):
            st.session_state.session_id = None
            st.session_state.messages = []
            st.rerun()

        try:
            resp = requests.get(f"{BACKEND_HOST}/sessions/list")
            if resp.status_code == 200:
                sessions = resp.json().get("sessions", [])

                for s in sessions:
                    # Determine button type based on active session
                    b_type = (
                        "primary"
                        if s["id"] == st.session_state.session_id
                        else "secondary"
                    )

                    if st.button(
                        s["title"],
                        key=f"sess_{s['id']}",
                        use_container_width=True,
                        type=b_type,
                    ):
                        if st.session_state.session_id != s["id"]:
                            st.session_state.session_id = s["id"]
                            # Load History
                            h_resp = requests.get(
                                f"{BACKEND_HOST}/sessions/{s['id']}/history"
                            )
                            if h_resp.status_code == 200:
                                st.session_state.messages = h_resp.json().get(
                                    "messages", []
                                )
                            st.rerun()

                # Delete button for active session
                if st.session_state.session_id is not None:
                    st.divider()
                    if st.button(
                        "🗑️ Delete Current Chat",
                        type="secondary",
                        use_container_width=True,
                    ):
                        requests.delete(
                            f"{BACKEND_HOST}/sessions/{st.session_state.session_id}"
                        )
                        st.session_state.session_id = None
                        st.session_state.messages = []
                        st.rerun()

            else:
                st.error("Failed to load sessions")

        except Exception as e:
            logger.warning(f"Failed to fetch sessions: {e}")


def render_ingestion():
    """
    Render the ingestion interface.
    """
    st.header("Ingestion")

    # New collection input
    new_col = st.text_input("New Collection Name")
    target_col = new_col if new_col else st.session_state.selected_collection

    uploaded_files = st.file_uploader("Upload Documents", accept_multiple_files=True)

    with st.expander("Advanced Options"):
        table_row_limit = st.number_input(
            "Table Row Limit", min_value=0, value=0, help="0 for no limit"
        )
        table_row_filter = st.text_input("Table Row Filter", help="Pandas query string")

    if uploaded_files and st.button("Upload & Ingest"):
        if not target_col:
            logger.error("No target collection specified.")
            st.error("Please select or enter a collection name.")
        else:
            with st.status("Processing...", expanded=True) as status:
                files = [("files", (f.name, f, f.type)) for f in uploaded_files]
                data = {
                    "collection": target_col,
                    "hybrid": "True",
                }
                if table_row_limit > 0:
                    data["table_row_limit"] = str(table_row_limit)
                if table_row_filter:
                    data["table_row_filter"] = table_row_filter

                try:
                    # Use stream=True to handle SSE if possible, or just wait for response
                    # Since requests doesn't parse SSE automatically, we'll just read lines
                    response = requests.post(
                        f"{BACKEND_HOST}/ingest/upload",
                        data=data,
                        files=files,
                        stream=True,
                    )

                    if response.status_code == 200:
                        for line in response.iter_lines():
                            if line:
                                decoded_line = line.decode("utf-8")
                                if decoded_line.startswith("data: "):
                                    event_data = json.loads(decoded_line[6:])
                                    # We could update status here based on event type
                                    # But for now, just logging or simple updates
                                    if "filename" in event_data:
                                        logger.info(
                                            f"Processed {event_data['filename']}"
                                        )
                                        status.write(
                                            f"Processed {event_data['filename']}"
                                        )
                                    if "message" in event_data:  # Error
                                        logger.error(f"Error: {event_data['message']}")
                                        status.write(f"Error: {event_data['message']}")

                        status.update(
                            label="Ingestion complete!",
                            state="complete",
                            expanded=False,
                        )
                        st.success("Ingestion complete!")
                        # Refresh collections list if new one was created
                        if new_col:
                            st.rerun()
                    else:
                        status.update(label="Ingestion failed", state="error")
                        logger.error(f"Ingestion failed: {response.text}")
                        st.error(f"Ingestion failed: {response.text}")
                except Exception as e:
                    status.update(label="Error", state="error")
                    logger.error(f"Error during ingestion: {e}")
                    st.error(f"Error during ingestion: {e}")


def render_analysis():
    """
    Render the analysis interface.
    """
    st.header("Analysis")
    st.info("Generate summaries and insights from your collection.")

    if st.button("Summarize Collection"):
        if not st.session_state.selected_collection:
            logger.error("No collection selected.")
            st.error("No collection selected.")
        else:
            with st.spinner("Generating summary..."):
                try:
                    resp = requests.post(f"{BACKEND_HOST}/summarize")
                    if resp.status_code == 200:
                        data = resp.json()
                        st.markdown(
                            f"### Summary of {st.session_state.selected_collection}"
                        )
                        st.markdown(data["summary"])

                        if data.get("sources"):
                            with st.expander("Sources used for summary"):
                                for src in data["sources"]:
                                    st.markdown(f"- {src.get('filename')}")
                    else:
                        logger.error(f"Summarization failed: {resp.text}")
                        st.error(f"Summarization failed: {resp.text}")
                except Exception as e:
                    logger.error(f"Error: {e}")
                    st.error(f"Error: {e}")


def render_chat() -> None:
    """
    Render the main chat interface.
    """
    st.header("Chat with your Documents")

    if not st.session_state.selected_collection:
        st.info("Please select or create a collection from the sidebar to start.")
        return

    st.caption(f"Current Collection: {st.session_state.selected_collection}")

    # Display history
    for i, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

            # Display reasoning if available
            if "reasoning" in msg and msg["reasoning"]:
                with st.expander("Reasoning"):
                    st.markdown(msg["reasoning"])

            if "sources" in msg and msg["sources"]:
                with st.expander("View Sources"):
                    for j, src in enumerate(msg["sources"]):
                        # Construct preview
                        loc = ""
                        if src.get("page"):
                            loc += f" (Page {src['page']})"
                        if src.get("row"):
                            loc += f" (Row {src['row']})"

                        score = ""
                        if src.get("score"):
                            score = f" - Score: {src['score']:.2f}"

                        st.markdown(f"**{src.get('filename')}{loc}**{score}")
                        st.caption(src.get("preview_text", ""))

                        if src.get("file_hash"):
                            link = f"{BACKEND_HOST}/sources/preview?collection={st.session_state.selected_collection}&file_hash={src['file_hash']}"
                            st.markdown(
                                f'<a href="{link}" target="_blank">Download/View Original</a>',
                                unsafe_allow_html=True,
                            )
                        st.divider()

    # Chat Input
    if prompt := st.chat_input("Ask a question..."):
        # 1. User message
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        # 2. Bot response
        with st.chat_message("assistant"):
            # Placeholder for answer
            answer_placeholder = st.empty()
            full_answer = ""
            sources = []
            reasoning = None

            payload = {
                "question": prompt,
                "session_id": st.session_state.session_id,
            }
            try:
                with requests.post(
                    f"{BACKEND_HOST}/stream_query", json=payload, stream=True
                ) as resp:
                    if resp.status_code == 200:
                        lines = resp.iter_lines()
                        first_line = None

                        # Show spinner while waiting for the first chunk
                        with st.spinner("Thinking..."):
                            try:
                                first_line = next(lines)
                            except StopIteration:
                                pass

                        def process_line(line):
                            nonlocal full_answer, sources, reasoning
                            if line:
                                decoded_line = line.decode("utf-8")
                                if decoded_line.startswith("data: "):
                                    data_str = decoded_line[6:]
                                    try:
                                        data = json.loads(data_str)
                                        if "token" in data:
                                            full_answer += data["token"]
                                            answer_placeholder.markdown(
                                                full_answer + "▌"
                                            )
                                        elif "sources" in data:
                                            # End of stream metadata
                                            sources = data.get("sources", [])
                                            reasoning = data.get("reasoning")
                                            st.session_state.session_id = data.get(
                                                "session_id"
                                            )
                                        elif "error" in data:
                                            st.error(f"Stream error: {data['error']}")
                                    except json.JSONDecodeError:
                                        pass

                        if first_line:
                            process_line(first_line)

                        for line in lines:
                            process_line(line)

                        answer_placeholder.markdown(full_answer)

                        if reasoning:
                            with st.expander("Reasoning"):
                                st.markdown(reasoning)

                        if sources:
                            with st.expander("View Sources"):
                                for j, src in enumerate(sources):
                                    loc = ""
                                    if src.get("page"):
                                        loc += f" (Page {src['page']})"
                                    if src.get("row"):
                                        loc += f" (Row {src['row']})"

                                    score = ""
                                    if src.get("score"):
                                        score = f" - Score: {src['score']:.2f}"

                                    st.markdown(
                                        f"**{src.get('filename')}{loc}**{score}"
                                    )
                                    st.caption(src.get("preview_text", ""))
                                    if src.get("file_hash"):
                                        link = f"{BACKEND_HOST}/sources/preview?collection={st.session_state.selected_collection}&file_hash={src['file_hash']}"
                                        st.markdown(
                                            f'<a href="{link}" target="_blank">Download/View Original</a>',
                                            unsafe_allow_html=True,
                                        )
                                    st.divider()

                        # 3. Save bot message
                        msg_entry = {
                            "role": "assistant",
                            "content": full_answer,
                            "sources": sources,
                        }
                        if reasoning:
                            msg_entry["reasoning"] = reasoning
                        st.session_state.messages.append(msg_entry)
                        st.rerun()
                    else:
                        logger.error(f"Query failed: {resp.text}")
                        st.error(f"Query failed: {resp.text}")
            except Exception as e:
                logger.error(f"Error: {e}")
                st.error(f"Error: {e}")


def main():
    setup_app()
    render_sidebar()

    tab_chat, tab_ingest, tab_analysis = st.tabs(
        ["💬 Chat", "📥 Ingest", "📊 Analysis"]
    )

    with tab_chat:
        render_chat()

    with tab_ingest:
        render_ingestion()

    with tab_analysis:
        render_analysis()


# ---- Streamlit CLI wrapper ----------------------------------------------- #
def run() -> None:
    """
    CLI entry point for the Streamlit app. This function is used to run the app from the command
    line. It sets up the command line arguments as if the user typed them. For example: `streamlit
    run app.py <any extra args>`.
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
