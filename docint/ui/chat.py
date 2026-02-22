"""
Chat page: streaming Q&A with source rendering and NER.
"""

import json
from typing import Any

import requests
import streamlit as st
from loguru import logger

from docint.ui.components import render_ner_overview, render_source_item
from docint.ui.state import BACKEND_HOST


def render_chat() -> None:
    """
    Render the chat interface.
    This includes the message history, chat input, and handling of streaming responses from the backend."""
    st.header("💬 Chat")

    collection = st.session_state.selected_collection
    if not collection:
        st.info("Select or create a collection from the sidebar to start chatting.")
        return

    st.caption(f"📁 {collection}")

    # ── Download current chat ────────────────────────────────
    if st.session_state.messages:
        chat_text = ""
        for msg in st.session_state.messages:
            chat_text += f"{msg['role'].upper()}: {msg['content']}\n\n"
            if msg.get("reasoning"):
                chat_text += f"REASONING: {msg['reasoning']}\n\n"

        st.download_button(
            label="📥 Download chat (.txt)",
            data=chat_text,
            file_name=(f"chat_{st.session_state.session_id or 'session'}.txt"),
            mime="text/plain",
        )

    # ── Message history ──────────────────────────────────────
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

            if msg.get("reasoning"):
                with st.expander("Reasoning"):
                    st.markdown(msg["reasoning"])

            if msg.get("sources"):
                with st.expander("View Sources"):
                    for src in msg["sources"]:
                        render_source_item(src, collection)
                    st.markdown("**Information Extraction Overview**")
                    render_ner_overview(msg["sources"])

    # ── Callbacks ────────────────────────────────────────────

    def _stop_generation() -> None:
        """
        Stop the current chat generation.
        """
        st.session_state.chat_running = False
        if st.session_state.get("current_answer"):
            st.session_state.messages.append(
                {"role": "assistant", "content": st.session_state.current_answer}
            )
            del st.session_state.current_answer

    def _start_chat() -> None:
        """
        Start a new chat turn by sending the user's query to the backend and handling the streaming response.
        """
        st.session_state.chat_running = True
        st.session_state.current_answer = ""

    # ── Chat input ───────────────────────────────────────────
    if prompt := st.chat_input(
        "Ask a question about your documents…", on_submit=_start_chat
    ):
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            # Stop button
            c1, c2 = st.columns([0.85, 0.15])
            with c2:
                if st.session_state.get("chat_running"):
                    st.button(
                        "⏹️ Stop",
                        on_click=_stop_generation,
                        help="Stop generation",
                    )

            with c1:
                answer_placeholder = st.empty()

            full_answer = ""
            sources: list[dict[str, Any]] = []
            reasoning: str | None = None

            payload = {
                "question": prompt,
                "session_id": st.session_state.session_id,
            }
            try:
                with requests.post(
                    f"{BACKEND_HOST}/stream_query",
                    json=payload,
                    stream=True,
                    timeout=600,
                ) as resp:
                    if resp.status_code == 200:
                        lines = resp.iter_lines()
                        first_line = None

                        with c1:
                            with st.spinner("Thinking…"):
                                try:
                                    first_line = next(lines)
                                except StopIteration:
                                    pass

                        def _process_line(line: bytes) -> None:
                            """
                            Process a single SSE line.

                            Args:
                                line: Raw bytes from the streaming response.
                            """
                            nonlocal full_answer, sources, reasoning
                            if not line:
                                return
                            decoded = line.decode("utf-8")
                            if not decoded.startswith("data: "):
                                return
                            try:
                                data = json.loads(decoded[6:])
                                if "token" in data:
                                    full_answer += data["token"]
                                    st.session_state.current_answer = full_answer
                                    answer_placeholder.markdown(full_answer + "▌")
                                elif "sources" in data:
                                    sources = data.get("sources", [])
                                    reasoning = data.get("reasoning")
                                    st.session_state.session_id = data.get("session_id")
                                elif "error" in data:
                                    st.error(f"Stream error: {data['error']}")
                            except json.JSONDecodeError:
                                pass

                        if first_line:
                            _process_line(first_line)

                        for line in lines:
                            if not st.session_state.get("chat_running"):
                                break
                            _process_line(line)

                        answer_placeholder.markdown(full_answer)

                        if not full_answer and not reasoning:
                            st.warning("Received empty response from the backend.")
                        elif not full_answer and reasoning:
                            st.info("No text response generated (reasoning only).")

                        if reasoning:
                            with st.expander("Reasoning"):
                                st.markdown(reasoning)

                        if sources:
                            with st.expander("View Sources"):
                                for src in sources:
                                    render_source_item(src, collection)
                            st.markdown("**Information Extraction Overview**")
                            render_ner_overview(sources)

                        # Persist message
                        msg_entry: dict[str, Any] = {
                            "role": "assistant",
                            "content": full_answer,
                            "sources": sources,
                        }
                        if reasoning:
                            msg_entry["reasoning"] = reasoning
                        st.session_state.messages.append(msg_entry)
                        st.session_state.chat_running = False
                        st.rerun()
                    else:
                        st.session_state.chat_running = False
                        logger.error("Query failed: {}", resp.text)
                        st.error(f"Query failed: {resp.text}")
            except Exception as e:
                st.session_state.chat_running = False
                logger.error("Error: {}", e)
                st.error(f"Error: {e}")
