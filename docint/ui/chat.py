"""Chat page: streaming Q&A with source rendering and NER."""

import json
from typing import Any, Mapping, Sequence

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
CHAT_SOURCES_CONTAINER_HEIGHT = 420
CHAT_DEBUG_CONTAINER_HEIGHT = 280
CHAT_SCOPE_OPTIONS: tuple[str, ...] = (
    "All content",
    "Images only",
    "Documents only",
    "Custom MIME",
)
DATE_FIELD_OPTIONS: tuple[str, ...] = (
    "None",
    "reference_metadata.timestamp",
    "created_at",
    "updated_at",
)
DOCUMENT_SOURCE_VALUES: tuple[str, ...] = ("document", "pdf_text", "vision_ocr")
CUSTOM_FIELD_OPTIONS: tuple[str, ...] = (
    "mimetype",
    "source",
    "filename",
    "reference_metadata.timestamp",
    "reference_metadata.author",
    "reference_metadata.author_id",
    "hate_speech.hate_speech",
    "Custom field",
)
CUSTOM_OPERATOR_OPTIONS: dict[str, str] = {
    "Equals": "eq",
    "Not equals": "neq",
    "Contains": "contains",
    "Greater than": "gt",
    "Greater than or equal": "gte",
    "Less than": "lt",
    "Less than or equal": "lte",
    "MIME match": "mime_match",
    "Date on or after": "date_on_or_after",
    "Date on or before": "date_on_or_before",
}


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


def _build_custom_metadata_rules(count: int) -> list[dict[str, Any]]:
    """Build request payloads for user-defined custom metadata rules.

    Args:
        count: Number of custom rules to process, as specified by the user in the UI.

    Returns:
        A list of metadata filter rules derived from the UI state.
    """
    rules: list[dict[str, Any]] = []
    for index in range(count):
        selected_field = st.session_state.get(
            f"chat_filter_custom_field_{index}",
            "mimetype",
        )
        field = (
            str(
                st.session_state.get(f"chat_filter_custom_field_name_{index}", "")
            ).strip()
            if selected_field == "Custom field"
            else str(selected_field).strip()
        )
        operator_label = str(
            st.session_state.get(
                f"chat_filter_custom_operator_{index}",
                "Equals",
            )
        )
        value = str(
            st.session_state.get(f"chat_filter_custom_value_{index}", "")
        ).strip()
        operator = CUSTOM_OPERATOR_OPTIONS.get(operator_label)
        if field and operator and value:
            rules.append({"field": field, "operator": operator, "value": value})
    return rules


def build_chat_metadata_filters(
    filter_state: Mapping[str, Any],
    *,
    custom_rules: Sequence[Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Translate chat filter state into the backend request payload.

    Args:
        filter_state: Mapping of chat filter UI state keys to their current values.
        custom_rules: Optional sequence of additional custom filter rules to include.

    Returns:
        A list of metadata filter rules to apply to the next chat retrieval request.
    """
    if not bool(filter_state.get("enabled")):
        return []

    filters: list[dict[str, Any]] = []
    scope = str(filter_state.get("scope") or "All content")
    if scope == "Images only":
        filters.append(
            {"field": "mimetype", "operator": "mime_match", "value": "image/*"}
        )
    elif scope == "Documents only":
        filters.append(
            {
                "field": "source",
                "operator": "in",
                "values": list(DOCUMENT_SOURCE_VALUES),
            }
        )
    elif scope == "Custom MIME":
        mime_pattern = str(filter_state.get("mime_pattern") or "").strip()
        if mime_pattern:
            filters.append(
                {
                    "field": "mimetype",
                    "operator": "mime_match",
                    "value": mime_pattern,
                }
            )

    date_field = str(filter_state.get("date_field") or "None").strip()
    if date_field and date_field != "None":
        start_date = str(filter_state.get("start_date") or "").strip()
        end_date = str(filter_state.get("end_date") or "").strip()
        if start_date:
            filters.append(
                {
                    "field": date_field,
                    "operator": "date_on_or_after",
                    "value": start_date,
                }
            )
        if end_date:
            filters.append(
                {
                    "field": date_field,
                    "operator": "date_on_or_before",
                    "value": end_date,
                }
            )

    if bool(filter_state.get("hate_speech_only")):
        filters.append(
            {
                "field": "hate_speech.hate_speech",
                "operator": "eq",
                "value": True,
            }
        )

    for rule in custom_rules or []:
        field = str(rule.get("field") or "").strip()
        operator = str(rule.get("operator") or "").strip()
        value = rule.get("value")
        values = rule.get("values")
        if field and operator:
            payload: dict[str, Any] = {"field": field, "operator": operator}
            if values:
                payload["values"] = list(values)
            elif value not in (None, ""):
                payload["value"] = value
            else:
                continue
            filters.append(payload)

    return filters


def _render_sources_panel(
    sources: Sequence[Mapping[str, Any]],
    collection: str,
) -> None:
    """Render chat source details in a stable-height popover.

    Args:
        sources: Source rows associated with a chat answer.
        collection: Active collection name for source downloads.
    """
    if not sources:
        return

    with st.popover("Sources"):
        with st.container(height=CHAT_SOURCES_CONTAINER_HEIGHT):
            for src in sources:
                render_source_item(dict(src), collection)
            st.markdown("**Information Extraction Overview**")
            render_ner_overview([dict(src) for src in sources])


def _render_graph_debug_panel(graph_debug: Mapping[str, Any]) -> None:
    """Render GraphRAG debug details in a popover.

    Args:
        graph_debug: GraphRAG debug payload for the current answer.
    """
    with st.popover("GraphRAG Debug"):
        with st.container(height=CHAT_DEBUG_CONTAINER_HEIGHT):
            st.json(dict(graph_debug))


def _render_retrieval_debug_panel(
    *,
    retrieval_query: str | None,
    retrieval_mode: str | None,
    coverage_unit: str | None,
) -> None:
    """Render retrieval diagnostics in a popover."""
    payload = {
        "retrieval_query": retrieval_query,
        "retrieval_mode": retrieval_mode,
        "coverage_unit": coverage_unit,
    }
    with st.popover("Retrieval Debug"):
        with st.container(height=CHAT_DEBUG_CONTAINER_HEIGHT):
            st.json(payload)


def _render_answer_tool_panels(
    *,
    graph_debug: Mapping[str, Any] | None,
    sources: Sequence[Mapping[str, Any]] | None,
    collection: str,
    retrieval_query: str | None = None,
    retrieval_mode: str | None = None,
    coverage_unit: str | None = None,
) -> None:
    """Render optional chat detail controls in one horizontal row.

    Args:
        graph_debug: Optional GraphRAG debug payload.
        sources: Optional source rows associated with a chat answer.
        collection: Active collection name for source downloads.
    """
    tool_count = (
        int(bool(sources))
        + int(isinstance(graph_debug, Mapping))
        + int(bool(retrieval_query or retrieval_mode or coverage_unit))
    )
    if tool_count == 0:
        return

    columns = st.columns(tool_count)
    column_index = 0

    if sources:
        with columns[column_index]:
            _render_sources_panel(sources, collection)
        column_index += 1

    if isinstance(graph_debug, Mapping):
        with columns[column_index]:
            _render_graph_debug_panel(graph_debug)
        column_index += 1

    if retrieval_query or retrieval_mode or coverage_unit:
        with columns[column_index]:
            _render_retrieval_debug_panel(
                retrieval_query=retrieval_query,
                retrieval_mode=retrieval_mode,
                coverage_unit=coverage_unit,
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
            if (
                msg.get("retrieval_query")
                or msg.get("retrieval_mode")
                or msg.get("coverage_unit")
            ):
                chat_text += (
                    "RETRIEVAL: "
                    f"query={msg.get('retrieval_query')}, "
                    f"mode={msg.get('retrieval_mode')}, "
                    f"coverage_unit={msg.get('coverage_unit')}\n\n"
                )
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
            label="Download",
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
            _render_answer_tool_panels(
                graph_debug=msg.get("graph_debug"),
                sources=msg.get("sources"),
                collection=collection,
                retrieval_query=msg.get("retrieval_query"),
                retrieval_mode=msg.get("retrieval_mode"),
                coverage_unit=msg.get("coverage_unit"),
            )

    with st.expander("Retrieval filters", expanded=False):
        st.checkbox(
            "Enable metadata filtering",
            key="chat_filter_enabled",
            help="Apply metadata constraints to the next retrieval request.",
        )
        st.selectbox(
            "Scope",
            options=CHAT_SCOPE_OPTIONS,
            key="chat_filter_scope",
        )
        if st.session_state.chat_filter_scope == "Custom MIME":
            st.text_input(
                "MIME pattern",
                key="chat_filter_mime_pattern",
                placeholder="image/*, application/pdf, text/*",
            )
        st.selectbox(
            "Date field",
            options=DATE_FIELD_OPTIONS,
            key="chat_filter_date_field",
        )
        if st.session_state.chat_filter_date_field != "None":
            date_cols = st.columns(2)
            with date_cols[0]:
                st.text_input(
                    "Start date",
                    key="chat_filter_start_date",
                    placeholder="YYYY-MM-DD",
                )
            with date_cols[1]:
                st.text_input(
                    "End date",
                    key="chat_filter_end_date",
                    placeholder="YYYY-MM-DD",
                )
        st.checkbox(
            "Only hate-speech positive chunks",
            key="chat_filter_hate_speech_only",
        )
        st.selectbox(
            "Additional custom filters",
            options=[0, 1, 2, 3],
            key="chat_filter_custom_count",
        )
        for index in range(int(st.session_state.chat_filter_custom_count or 0)):
            st.markdown(f"Custom filter {index + 1}")
            cols = st.columns([1.4, 1.1, 1.5])
            with cols[0]:
                st.selectbox(
                    "Field",
                    options=CUSTOM_FIELD_OPTIONS,
                    key=f"chat_filter_custom_field_{index}",
                    label_visibility="collapsed",
                )
                if (
                    st.session_state.get(f"chat_filter_custom_field_{index}")
                    == "Custom field"
                ):
                    st.text_input(
                        "Metadata field",
                        key=f"chat_filter_custom_field_name_{index}",
                        placeholder="reference_metadata.author_id",
                        label_visibility="collapsed",
                    )
            with cols[1]:
                st.selectbox(
                    "Operator",
                    options=list(CUSTOM_OPERATOR_OPTIONS.keys()),
                    key=f"chat_filter_custom_operator_{index}",
                    label_visibility="collapsed",
                )
            with cols[2]:
                st.text_input(
                    "Value",
                    key=f"chat_filter_custom_value_{index}",
                    placeholder="value",
                    label_visibility="collapsed",
                )

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
            retrieval_query: str | None = None
            coverage_unit: str | None = None
            retrieval_mode: str | None = None
            metadata_filters = build_chat_metadata_filters(
                {
                    "enabled": st.session_state.chat_filter_enabled,
                    "scope": st.session_state.chat_filter_scope,
                    "mime_pattern": st.session_state.chat_filter_mime_pattern,
                    "date_field": st.session_state.chat_filter_date_field,
                    "start_date": st.session_state.chat_filter_start_date,
                    "end_date": st.session_state.chat_filter_end_date,
                    "hate_speech_only": st.session_state.chat_filter_hate_speech_only,
                },
                custom_rules=_build_custom_metadata_rules(
                    int(st.session_state.chat_filter_custom_count or 0)
                ),
            )

            payload = {
                "question": prompt,
                "session_id": st.session_state.session_id,
                "metadata_filters": metadata_filters,
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
                            nonlocal retrieval_query
                            nonlocal coverage_unit
                            nonlocal retrieval_mode
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
                                if data.get("retrieval_query") is not None:
                                    retrieval_query = str(
                                        data.get("retrieval_query") or ""
                                    )
                                if data.get("coverage_unit") is not None:
                                    coverage_unit = str(data.get("coverage_unit") or "")
                                if data.get("retrieval_mode") is not None:
                                    retrieval_mode = str(
                                        data.get("retrieval_mode") or ""
                                    )
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
                        _render_answer_tool_panels(
                            graph_debug=graph_debug,
                            sources=sources,
                            collection=collection,
                            retrieval_query=retrieval_query,
                            retrieval_mode=retrieval_mode,
                            coverage_unit=coverage_unit,
                        )

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
                        if retrieval_query is not None:
                            msg_entry["retrieval_query"] = retrieval_query
                        if retrieval_mode is not None:
                            msg_entry["retrieval_mode"] = retrieval_mode
                        if coverage_unit is not None:
                            msg_entry["coverage_unit"] = coverage_unit
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
