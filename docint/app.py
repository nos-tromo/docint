import json

import requests
import streamlit as st

from docint.utils.env_cfg import set_offline_env
from docint.utils.logging_cfg import setup_logging

# --- Application Setup ---
set_offline_env()
setup_logging()


# Configuration
API_URL = "http://localhost:8000"

st.set_page_config(page_title="DocInt", layout="wide")

# --- Session State Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "selected_collection" not in st.session_state:
    st.session_state.selected_collection = ""

# --- Sidebar: Controls ---
with st.sidebar:
    st.header("Configuration")

    # 1. Collection Selection
    try:
        resp = requests.get(f"{API_URL}/collections/list")
        if resp.status_code == 200:
            cols = resp.json()
            # If we have a selected collection in state, try to keep it selected
            index = 0
            if st.session_state.selected_collection in cols:
                index = cols.index(st.session_state.selected_collection)

            selected = st.selectbox("Collection", cols, index=index if cols else None)

            if selected and selected != st.session_state.selected_collection:
                # Notify backend of selection
                r = requests.post(
                    f"{API_URL}/collections/select", json={"name": selected}
                )
                if r.status_code == 200:
                    st.session_state.selected_collection = selected
                    st.session_state.messages = []  # Clear chat on switch
                    st.session_state.session_id = None
                    st.rerun()
                else:
                    st.error(f"Failed to select collection: {r.text}")
        else:
            st.error(f"Failed to fetch collections: {resp.text}")
            cols = []

    except Exception as e:
        st.error(f"Backend offline: {e}")
        cols = []

    st.divider()

    # 2. Ingestion
    st.subheader("Ingestion")

    # New collection input
    new_col = st.text_input("New Collection Name")
    target_col = new_col if new_col else st.session_state.selected_collection

    uploaded_files = st.file_uploader("Upload Documents", accept_multiple_files=True)

    with st.expander("Advanced Options"):
        hybrid_search = st.checkbox("Hybrid Search", value=True)
        table_row_limit = st.number_input(
            "Table Row Limit", min_value=0, value=0, help="0 for no limit"
        )
        table_row_filter = st.text_input("Table Row Filter", help="Pandas query string")

    if uploaded_files and st.button("Upload & Ingest"):
        if not target_col:
            st.error("Please select or enter a collection name.")
        else:
            with st.status("Processing...", expanded=True) as status:
                files = [("files", (f.name, f, f.type)) for f in uploaded_files]
                data = {
                    "collection": target_col,
                    "hybrid": str(hybrid_search),
                }
                if table_row_limit > 0:
                    data["table_row_limit"] = str(table_row_limit)
                if table_row_filter:
                    data["table_row_filter"] = table_row_filter

                try:
                    # Use stream=True to handle SSE if possible, or just wait for response
                    # Since requests doesn't parse SSE automatically, we'll just read lines
                    response = requests.post(
                        f"{API_URL}/ingest/upload", data=data, files=files, stream=True
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
                                        status.write(
                                            f"Processed {event_data['filename']}"
                                        )
                                    if "message" in event_data:  # Error
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
                        st.error(f"Ingestion failed: {response.text}")
                except Exception as e:
                    status.update(label="Error", state="error")
                    st.error(f"Error during ingestion: {e}")

    st.divider()
    if st.button("Summarize Collection"):
        if not st.session_state.selected_collection:
            st.error("No collection selected.")
        else:
            with st.spinner("Generating summary..."):
                try:
                    resp = requests.post(f"{API_URL}/summarize")
                    if resp.status_code == 200:
                        data = resp.json()
                        st.session_state.messages.append(
                            {
                                "role": "assistant",
                                "content": f"**Summary:** {data['summary']}",
                                "sources": data.get("sources", []),
                            }
                        )
                        st.rerun()
                    else:
                        st.error(f"Summarization failed: {resp.text}")
                except Exception as e:
                    st.error(f"Error: {e}")

# --- Main Chat Interface ---
st.title("Document Intelligence")

if not st.session_state.selected_collection:
    st.info("Please select or create a collection from the sidebar to start.")
else:
    st.caption(f"Current Collection: {st.session_state.selected_collection}")

    # Display history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

            # Display reasoning if available (custom field we might add)
            if "reasoning" in msg and msg["reasoning"]:
                with st.expander("Reasoning"):
                    st.markdown(msg["reasoning"])

            if "sources" in msg and msg["sources"]:
                with st.expander("View Sources"):
                    for src in msg["sources"]:
                        # Construct preview
                        loc = ""
                        if src.get("page"):
                            loc += f" (Page {src['page']})"
                        if src.get("row"):
                            loc += f" (Row {src['row']})"

                        st.markdown(f"**{src.get('filename')}{loc}**")
                        st.caption(src.get("preview_text", ""))

                        # If we had a way to serve the file preview, we'd link it here
                        # But Streamlit runs on a different port, so we'd need to proxy or link to backend
                        if src.get("file_hash"):
                            link = f"{API_URL}/sources/preview?collection={st.session_state.selected_collection}&file_hash={src['file_hash']}"
                            st.markdown(f"[Download/View Original]({link})")
                        st.divider()

    # Chat Input
    if prompt := st.chat_input("Ask a question..."):
        # 1. User message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 2. Bot response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                payload = {
                    "question": prompt,
                    "session_id": st.session_state.session_id,
                }
                try:
                    resp = requests.post(f"{API_URL}/query", json=payload)
                    if resp.status_code == 200:
                        data = resp.json()

                        answer = data["answer"]
                        sources = data.get("sources", [])
                        reasoning = data.get(
                            "reasoning"
                        )  # If backend returns it (we added it recently!)

                        st.session_state.session_id = data.get("session_id")

                        st.markdown(answer)

                        if reasoning:
                            with st.expander("Reasoning"):
                                st.markdown(reasoning)

                        if sources:
                            with st.expander("View Sources"):
                                for src in sources:
                                    loc = ""
                                    if src.get("page"):
                                        loc += f" (Page {src['page']})"
                                    if src.get("row"):
                                        loc += f" (Row {src['row']})"

                                    st.markdown(f"**{src.get('filename')}{loc}**")
                                    st.caption(src.get("preview_text", ""))
                                    if src.get("file_hash"):
                                        link = f"{API_URL}/sources/preview?collection={st.session_state.selected_collection}&file_hash={src['file_hash']}"
                                        st.markdown(f"[Download/View Original]({link})")
                                    st.divider()

                        # 3. Save bot message
                        msg_entry = {
                            "role": "assistant",
                            "content": answer,
                            "sources": sources,
                        }
                        if reasoning:
                            msg_entry["reasoning"] = reasoning
                        st.session_state.messages.append(msg_entry)
                    else:
                        st.error(f"Query failed: {resp.text}")
                except Exception as e:
                    st.error(f"Error: {e}")
