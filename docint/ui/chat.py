"""Chat page: streaming Q&A with source rendering and NER."""

import json
from typing import Any

import requests
import streamlit as st
from loguru import logger

from docint.ui.components import (
    reference_metadata_text_block,
    render_ner_overview,
    render_response_validation,
    render_source_item,
)
from docint.utils.env_cfg import load_host_env

BACKEND_HOST = load_host_env().backend_host


def _format_graph_debug_summary(graph_debug: dict[str, Any] | None) -> str | None:
    """Build a compact text line for GraphRAG debug metadata.

    Args:
        graph_debug: Optional GraphRAG debug payload.

    Returns:
        A compact summary string or ``None`` when unavailable.
    """
    if not isinstance(graph_debug, dict):
        return None
    enabled = graph_debug.get("enabled")
    applied = graph_debug.get("applied")
    reason = graph_debug.get("reason")
    anchors = graph_debug.get("anchor_entities")
    neighbors = graph_debug.get("neighbor_entities")
    return (
        f"enabled={enabled}, applied={applied}, reason={reason}, "
        f"anchors={anchors}, neighbors={neighbors}"
    )


def render_chat() -> None:
    """Render the chat interface.
    This includes the message history, chat input, and handling of streaming responses from the backend.
    """
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
            if (
                msg.get("validation_checked") is not None
                or msg.get("validation_mismatch") is not None
                or msg.get("validation_reason")
            ):
                chat_text += "VALIDATION: "
                chat_text += (
                    f"checked={msg.get('validation_checked')}, "
                    f"mismatch={msg.get('validation_mismatch')}"
                )
                if msg.get("validation_reason"):
                    chat_text += f", reason={msg['validation_reason']}"
                chat_text += "\n\n"
            graph_line = _format_graph_debug_summary(msg.get("graph_debug"))
            if graph_line:
                chat_text += f"GRAPHRAG: {graph_line}\n\n"
            if msg.get("sources"):
                chat_text += "SOURCES:\n"
                for idx, src in enumerate(msg["sources"], start=1):
                    chat_text += (
                        f"[{idx}] {src.get('filename')} "
                        f"page={src.get('page')} row={src.get('row')} "
                        f"score={src.get('score')}\n"
                    )
                    preview = str(
                        src.get("preview_text") or src.get("text") or ""
                    ).strip()
                    if preview:
                        chat_text += f"{preview}\n"
                    metadata_block = reference_metadata_text_block(src)
                    if metadata_block:
                        chat_text += f"{metadata_block}\n"
                    chat_text += "\n"

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

            render_response_validation(
                validation_checked=msg.get("validation_checked"),
                validation_mismatch=msg.get("validation_mismatch"),
                validation_reason=msg.get("validation_reason"),
            )
            if isinstance(msg.get("graph_debug"), dict):
                with st.expander("GraphRAG Debug"):
                    st.json(msg["graph_debug"])

            if msg.get("sources"):
                with st.expander("View Sources"):
                    for src in msg["sources"]:
                        render_source_item(src, collection)
                    st.markdown("**Information Extraction Overview**")
                    render_ner_overview(msg["sources"])

    # ── Callbacks ────────────────────────────────────────────

    def _stop_generation() -> None:
        """Stop the current chat generation."""
        st.session_state.chat_running = False
        if st.session_state.get("current_answer"):
            st.session_state.messages.append(
                {"role": "assistant", "content": st.session_state.current_answer}
            )
            del st.session_state.current_answer

    def _start_chat() -> None:
        """Start a new chat turn by sending the user's query to the backend and handling the streaming response."""
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
            validation_checked: bool | None = None
            validation_mismatch: bool | None = None
            validation_reason: str | None = None
            graph_debug: dict[str, Any] | None = None

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
                            """Process a single SSE line.

                            Args:
                                line: Raw bytes from the streaming response.
                            """
                            nonlocal full_answer
                            nonlocal sources
                            nonlocal reasoning
                            nonlocal validation_checked
                            nonlocal validation_mismatch
                            nonlocal validation_reason
                            nonlocal graph_debug
                            if not line:
                                return
                            decoded = line.decode("utf-8")
                            if not decoded.startswith("data: "):
                                return
                            try:
                                data = json.loads(decoded[6:])
                                if "validation_checked" in data:
                                    validation_checked = data.get("validation_checked")
                                if "validation_mismatch" in data:
                                    validation_mismatch = data.get(
                                        "validation_mismatch"
                                    )
                                if "validation_reason" in data:
                                    validation_reason = data.get("validation_reason")
                                if isinstance(data.get("graph_debug"), dict):
                                    graph_debug = data.get("graph_debug")
                                if data.get("session_id"):
                                    st.session_state.session_id = data.get("session_id")
                                if "token" in data:
                                    full_answer += data["token"]
                                    st.session_state.current_answer = full_answer
                                    answer_placeholder.markdown(full_answer + "▌")
                                elif "sources" in data:
                                    sources = data.get("sources", [])
                                    reasoning = data.get("reasoning")
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

                        render_response_validation(
                            validation_checked=validation_checked,
                            validation_mismatch=validation_mismatch,
                            validation_reason=validation_reason,
                        )
                        if graph_debug is not None:
                            with st.expander("GraphRAG Debug"):
                                st.json(graph_debug)

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
                        if validation_checked is not None:
                            msg_entry["validation_checked"] = validation_checked
                        if validation_mismatch is not None:
                            msg_entry["validation_mismatch"] = validation_mismatch
                        if validation_reason:
                            msg_entry["validation_reason"] = validation_reason
                        if graph_debug is not None:
                            msg_entry["graph_debug"] = graph_debug
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
