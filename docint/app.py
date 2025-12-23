import json
import sys
from pathlib import Path
from typing import Any, Iterable, Tuple

import requests
import streamlit as st
from loguru import logger
from streamlit.runtime import exists
from streamlit.web import cli as st_cli

from docint.utils.env_cfg import load_host_env, set_offline_env
from docint.utils.logging_cfg import setup_logging

host_cfg = load_host_env()
BACKEND_HOST = host_cfg.backend_host
BACKEND_PUBLIC_HOST = host_cfg.backend_public_host or BACKEND_HOST


def _format_score(score: Any) -> str:
    """Format score values for display."""
    try:
        return f"{float(score):.2f}"
    except (TypeError, ValueError):
        return "—"


def _normalize_entities(entities: Iterable[Any] | None) -> list[dict[str, Any]]:
    """
    Return sanitized entity payloads.

    Args:
        entities: Iterable of entity dicts or None.

    Returns:
        List of normalized entity dicts.
    """
    normalized: list[dict[str, Any]] = []
    for ent in entities or []:
        if not isinstance(ent, dict):
            continue
        text_val = str(ent.get("text") or "").strip()
        if not text_val:
            continue
        normalized.append(
            {
                "text": text_val,
                "type": ent.get("type") or ent.get("label"),
                "score": ent.get("score"),
            }
        )
    return normalized


def _normalize_relations(relations: Iterable[Any] | None) -> list[dict[str, Any]]:
    """
    Return sanitized relation payloads.

    Args:
        relations: Iterable of relation dicts or None.

    Returns:
        List of normalized relation dicts.
    """
    normalized: list[dict[str, Any]] = []
    for rel in relations or []:
        if not isinstance(rel, dict):
            continue
        head = str(rel.get("head") or rel.get("subject") or "").strip()
        tail = str(rel.get("tail") or rel.get("object") or "").strip()
        if not head or not tail:
            continue
        normalized.append(
            {
                "head": head,
                "tail": tail,
                "label": rel.get("label") or rel.get("type"),
                "score": rel.get("score"),
            }
        )
    return normalized


def _source_label(src: dict) -> str:
    """
    Build a compact label for a source row.

    Args:
        src: Source dictionary with possible keys 'filename', 'file_path', 'page', 'row'.

    Returns:
        A string label representing the source.
    """
    filename_val = src.get("filename") or src.get("file_path") or "Unknown"
    filename = str(filename_val).strip() or "Unknown"
    parts: list[str] = []
    if src.get("page") is not None:
        parts.append(f"p{src['page']}")
    if src.get("row") is not None:
        parts.append(f"row {src['row']}")
    return f"{filename} ({', '.join(parts)})" if parts else filename


def _aggregate_ie(
    sources: Iterable[dict] | None,
) -> Tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Aggregate entities and relations across source payloads.

    Args:
        sources: Iterable of source dictionaries containing 'entities' and 'relations'.

    Returns:
        A tuple containing two lists:
        - List of aggregated entity dictionaries.
        - List of aggregated relation dictionaries.
    """

    entity_index: dict[tuple[str, str], dict[str, Any]] = {}
    relation_index: dict[tuple[str, str, str], dict[str, Any]] = {}

    for src in sources or []:
        if not isinstance(src, dict):
            continue
        label = _source_label(src)
        entities = _normalize_entities(src.get("entities"))
        relations = _normalize_relations(src.get("relations"))

        for ent in entities:
            text_val = str(ent.get("text") or "")
            type_val = str(ent.get("type") or "")
            ent_key: tuple[str, str] = (text_val.lower(), type_val.lower())
            if ent_key not in entity_index:
                entity_index[ent_key] = {
                    "text": text_val,
                    "type": ent.get("type"),
                    "best_score": ent.get("score"),
                    "count": 0,
                    "files": set(),
                    "occurrences": [],
                }
            entry = entity_index[ent_key]
            entry["count"] += 1
            entry["files"].add(label)
            if ent.get("score") is not None:
                prev = entry.get("best_score")
                entry["best_score"] = (
                    max(prev, ent["score"]) if prev is not None else ent["score"]
                )
            entry["occurrences"].append({"source": label, "score": ent.get("score")})

        for rel in relations:
            head_val = str(rel.get("head") or "")
            label_val = str(rel.get("label") or "")
            tail_val = str(rel.get("tail") or "")
            rel_key: tuple[str, str, str] = (
                head_val.lower(),
                label_val.lower(),
                tail_val.lower(),
            )
            if rel_key not in relation_index:
                relation_index[rel_key] = {
                    "head": head_val,
                    "tail": tail_val,
                    "label": rel.get("label"),
                    "best_score": rel.get("score"),
                    "count": 0,
                    "files": set(),
                    "occurrences": [],
                }
            entry = relation_index[rel_key]
            entry["count"] += 1
            entry["files"].add(label)
            if rel.get("score") is not None:
                prev = entry.get("best_score")
                entry["best_score"] = (
                    max(prev, rel["score"]) if prev is not None else rel["score"]
                )
            entry["occurrences"].append({"source": label, "score": rel.get("score")})

    entities_sorted: list[dict[str, Any]] = sorted(
        [{**v, "files": sorted(v["files"])} for v in entity_index.values()],
        key=lambda item: (
            -int(item.get("count", 0) or 0),
            str(item.get("text") or "").lower(),
        ),
    )
    relations_sorted: list[dict[str, Any]] = sorted(
        [{**v, "files": sorted(v["files"])} for v in relation_index.values()],
        key=lambda item: (
            -int(item.get("count", 0) or 0),
            str(item.get("head") or "").lower(),
            str(item.get("label") or ""),
        ),
    )
    return entities_sorted, relations_sorted


def _render_entities_relations(src: dict[str, Any]) -> None:
    """
    Render entities and relations for a single source.

    Args:
        src: Source dictionary containing 'entities' and 'relations'.
    """
    entities = _normalize_entities(src.get("entities"))
    relations = _normalize_relations(src.get("relations"))

    if not entities and not relations:
        return

    col_entities, col_relations = st.columns(2)
    if entities:
        with col_entities:
            st.caption("Entities")
            for ent in entities:
                score = _format_score(ent.get("score"))
                label = ent.get("type") or "Unlabeled"
                st.markdown(f"- **{ent['text']}** ({label}) — score {score}")

    if relations:
        with col_relations:
            st.caption("Relations")
            for rel in relations:
                score = _format_score(rel.get("score"))
                label = rel.get("label") or "rel"
                st.markdown(
                    f"- **{rel['head']}** — _{label}_ → **{rel['tail']}** (score {score})"
                )


def _render_ie_overview(sources: list[dict[str, Any]]) -> None:
    """
    Show aggregated IE results for a set of sources.

    Args:
        sources: List of source dictionaries containing 'entities' and 'relations'.
    """
    entities, relations = _aggregate_ie(sources)

    metrics = st.columns(2)
    metrics[0].metric("Unique entities", len(entities))
    metrics[1].metric("Unique relations", len(relations))

    if entities:
        st.markdown("#### Entities")
        st.dataframe(
            {
                "Entity": [e["text"] for e in entities],
                "Type": [e.get("type") or "Unlabeled" for e in entities],
                "Mentions": [e["count"] for e in entities],
                "Best score": [_format_score(e.get("best_score")) for e in entities],
                "Sources": [", ".join(e["files"]) for e in entities],
            },
            width="stretch",
            hide_index=True,
        )
        for ent in entities:
            with st.expander(
                f"{ent['text']} ({ent.get('type') or 'Unlabeled'}) — {ent['count']} mention(s)"
            ):
                for occ in ent["occurrences"]:
                    st.markdown(
                        f"- {occ['source']} (score {_format_score(occ.get('score'))})"
                    )
    else:
        st.caption("No entities detected.")

    if relations:
        st.markdown("#### Relations")
        st.dataframe(
            {
                "Head": [r["head"] for r in relations],
                "Label": [r.get("label") or "rel" for r in relations],
                "Tail": [r["tail"] for r in relations],
                "Mentions": [r["count"] for r in relations],
                "Best score": [_format_score(r.get("best_score")) for r in relations],
                "Sources": [", ".join(r["files"]) for r in relations],
            },
            width="stretch",
            hide_index=True,
        )
        for rel in relations:
            with st.expander(
                f"{rel['head']} — {rel.get('label') or 'rel'} → {rel['tail']} ({rel['count']} mention(s))"
            ):
                for occ in rel["occurrences"]:
                    st.markdown(
                        f"- {occ['source']} (score {_format_score(occ.get('score'))})"
                    )
    else:
        st.caption("No relations detected.")


def setup_app() -> None:
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
    if "chat_running" not in st.session_state:
        st.session_state.chat_running = False


def render_sidebar() -> None:
    """
    Render the sidebar controls.
    """
    with st.sidebar:
        # Add top padding so sidebar content starts lower
        st.markdown("<div style='height:59px'></div>", unsafe_allow_html=True)

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
            width="stretch",
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
                        width="stretch",
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
                        width="stretch",
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


def render_ingestion() -> None:
    """
    Render the ingestion interface.

    Raises:
        JSONDecodeError: If the response from the backend cannot be decoded.
    """
    st.header("Ingestion")

    if "ingest_summary" not in st.session_state:
        st.session_state.ingest_summary = None

    # New collection input
    new_col = st.text_input("New Collection Name")
    target_col = new_col if new_col else st.session_state.selected_collection

    uploaded_files = st.file_uploader("Upload Documents", accept_multiple_files=True)

    def _render_ingest_summary(summary: dict[str, Any] | None) -> None:
        """
        Render ingestion summary.

        Args:
            summary (dict[str, Any] | None): Ingestion summary data.
        """
        if not summary:
            return
        with st.container():
            st.subheader("Last ingestion summary")
            cols = st.columns(3)
            cols[0].metric("Total", summary.get("total", 0))
            cols[1].metric("Succeeded", summary.get("done", 0))
            cols[2].metric("Errors", summary.get("errors", 0))

            pills = [
                f"{name} — {status}"
                for name, status in sorted(summary.get("file_status", {}).items())
            ]
            if pills:
                st.markdown("\n".join(pills))

            events = summary.get("events") or []
            if events:
                st.caption("Recent events")
                st.markdown("\n".join(events[-8:]))

    _render_ingest_summary(st.session_state.ingest_summary)

    if uploaded_files and st.button("Upload & Ingest"):
        if not target_col:
            logger.error("No target collection specified.")
            st.error("Please select or enter a collection name.")
        else:
            files = [("files", (f.name, f, f.type)) for f in uploaded_files]
            data = {
                "collection": target_col,
                "hybrid": "True",
            }

            file_status: dict[str, str] = {f.name: "Queued" for f in uploaded_files}
            events: list[str] = []
            summary: dict[str, int] = {"processed": 0, "errors": 0}

            header_ph = st.empty()
            progress = st.progress(0.0, text="Starting upload...")
            board_ph = st.empty()
            feed_ph = st.empty()
            summary_ph = st.empty()

            def render_board(current_stage: str) -> None:
                """
                Render per-file pills, counts, and progress.

                Args:
                    current_stage: Current overall stage label.
                """
                total = len(file_status) or 1
                done = sum(
                    1 for s in file_status.values() if s in {"Processed", "Done"}
                )
                errs = sum(1 for s in file_status.values() if s == "Error")
                progress.progress(
                    done / total,
                    text=f"{current_stage} • {done}/{total} done • {errs} errors",
                )

                with header_ph.container():
                    cols = st.columns(3)
                    cols[0].metric("Total files", total)
                    cols[1].metric("Processed", done)
                    cols[2].metric("Errors", errs)

                pill_map = {
                    "Queued": "⏳",
                    "Uploading": "📤",
                    "Processing": "⚙️",
                    "IE": "🔍",
                    "Processed": "✅",
                    "Done": "✅",
                    "Error": "❌",
                }
                pill_rows = [
                    f"{pill_map.get(state, '•')} **{name}** — {state}"
                    for name, state in sorted(file_status.items())
                ]
                board_ph.markdown("\n".join(pill_rows))

            def render_feed() -> None:
                """
                Render recent event feed.
                """
                if not events:
                    return
                feed_ph.markdown("\n".join(events[-8:]))

            render_board("Starting upload")

            with st.status("Ingestion in progress", expanded=True) as status:
                try:
                    response = requests.post(
                        f"{BACKEND_HOST}/ingest/upload",
                        data=data,
                        files=files,
                        stream=True,
                    )

                    if response.status_code == 200:
                        for line in response.iter_lines():
                            if not line:
                                continue
                            decoded_line = line.decode("utf-8")
                            if not decoded_line.startswith("data: "):
                                continue
                            try:
                                event_data = json.loads(decoded_line[6:])
                            except json.JSONDecodeError:
                                continue

                            filename = event_data.get("filename") or event_data.get(
                                "file"
                            )
                            stage = event_data.get("stage") or event_data.get("status")
                            message = event_data.get("message")

                            if stage == "ingestion_progress" and message:
                                # Special case for granular progress - replace last if it was also progress
                                if events and events[-1].startswith("• Extracting"):
                                    events[-1] = f"• {message}"
                                else:
                                    events.append(f"• {message}")
                                render_board("Processing")
                                render_feed()
                                continue

                            if filename:
                                file_status[filename] = stage or "Processing"
                            if message:
                                events.append(
                                    f"• {filename + ': ' if filename else ''}{message}"
                                )
                            elif stage and filename:
                                events.append(f"• {filename}: {stage}")
                            render_board(stage or "Processing")
                            render_feed()

                            if filename and message and "error" in message.lower():
                                file_status[filename] = "Error"
                                summary["errors"] += 1

                            if stage and stage.lower() in {"processed", "done"}:
                                summary["processed"] += 1

                        # Mark remaining as done
                        for name in file_status:
                            if file_status[name] not in {"Processed", "Done", "Error"}:
                                file_status[name] = "Done"
                        render_board("Complete")

                        total = len(file_status)
                        errs = sum(1 for s in file_status.values() if s == "Error")
                        done = sum(
                            1
                            for s in file_status.values()
                            if s in {"Processed", "Done"}
                        )
                        st.session_state.ingest_summary = {
                            "total": total,
                            "done": done,
                            "errors": errs,
                            "file_status": dict(file_status),
                            "events": events[-20:],
                        }
                        with summary_ph.container():
                            _render_ingest_summary(st.session_state.ingest_summary)

                        status.update(
                            label="Ingestion complete!",
                            state="complete",
                            expanded=False,
                        )
                        if target_col:
                            st.session_state.selected_collection = target_col
                            st.rerun()
                    else:
                        status.update(label="Ingestion failed", state="error")
                        logger.error(f"Ingestion failed: {response.text}")
                        st.error(f"Ingestion failed: {response.text}")
                except Exception as e:
                    status.update(label="Error", state="error")
                    logger.error(f"Error during ingestion: {e}")
                    st.error(f"Error during ingestion: {e}")


def render_analysis() -> None:
    """
    Render the analysis interface.
    """
    st.header("Analysis")
    st.info("Generate summaries and insights from your collection.")

    if st.button("Run analysis"):
        if not st.session_state.selected_collection:
            logger.error("No collection selected.")
            st.error("No collection selected.")
        else:
            st.markdown(f"### Summary of '{st.session_state.selected_collection}'")
            summary_placeholder = st.empty()
            full_summary = ""

            with st.spinner("Thinking..."):
                try:
                    resp = requests.post(
                        f"{BACKEND_HOST}/summarize/stream", stream=True
                    )
                    if resp.status_code == 200:
                        sources = []
                        for line in resp.iter_lines():
                            if line:
                                decoded_line = line.decode("utf-8")
                                if decoded_line.startswith("data: "):
                                    data = json.loads(decoded_line[6:])
                                    if "token" in data:
                                        full_summary += data["token"]
                                        summary_placeholder.markdown(full_summary + "▌")
                                    elif "sources" in data:
                                        sources = data["sources"]
                                    elif "error" in data:
                                        st.error(f"Error: {data['error']}")

                        summary_placeholder.markdown(full_summary)

                        # Download button for analysis
                        analysis_text = (
                            f"COLLECTION: {st.session_state.selected_collection}\n\n"
                        )
                        analysis_text += f"SUMMARY:\n{full_summary}\n\n"

                        if sources:
                            with st.expander("Sources used for summary"):
                                for j, src in enumerate(sources):
                                    loc = ""
                                    if src.get("page") is not None:
                                        loc += f" (Page {src['page']})"
                                    if src.get("row") is not None:
                                        loc += f" (Row {src['row']})"

                                    score = ""
                                    if src.get("score"):
                                        score = f" - Score: {src['score']:.2f}"

                                    st.markdown(
                                        f"**{src.get('filename')}{loc}**{score}"
                                    )
                                    st.caption(src.get("preview_text", ""))
                                    _render_entities_relations(src)
                                    if src.get("file_hash"):
                                        link = f"{BACKEND_PUBLIC_HOST}/sources/preview?collection={st.session_state.selected_collection}&file_hash={src['file_hash']}"
                                        st.markdown(
                                            f'<a href="{link}" target="_blank">Download/View Original</a>',
                                            unsafe_allow_html=True,
                                        )
                                    st.divider()

                        # Fetch and aggregate IE for the whole collection
                        st.markdown("---")
                        st.markdown("### Collection-wide Information Extraction")
                        with st.spinner(
                            "Aggregating entities and relations from entire collection..."
                        ):
                            ie_resp = requests.get(f"{BACKEND_HOST}/collections/ie")
                            if ie_resp.status_code == 200:
                                all_ie_sources = ie_resp.json().get("sources", [])
                                if all_ie_sources:
                                    _render_ie_overview(all_ie_sources)

                                    # Add IE to download text
                                    entities, relations = _aggregate_ie(all_ie_sources)
                                    analysis_text += "ENTITIES:\n"
                                    for ent in entities:
                                        analysis_text += f"- {ent['text']} ({ent.get('type', 'Unlabeled')}): {ent['count']} mentions in {', '.join(ent['files'])}\n"
                                    analysis_text += "\nRELATIONS:\n"
                                    for rel in relations:
                                        analysis_text += f"- {rel['head']} --[{rel.get('label', 'rel')}]--> {rel['tail']}: {rel['count']} mentions in {', '.join(rel['files'])}\n"
                                else:
                                    st.info(
                                        "No entities or relations found in this collection."
                                    )
                            else:
                                st.error("Failed to fetch collection-wide IE data.")

                        st.download_button(
                            label="📥 Download Analysis (.txt)",
                            data=analysis_text,
                            file_name=f"analysis_{st.session_state.selected_collection}.txt",
                            mime="text/plain",
                        )
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

    # Download button for chat
    if st.session_state.messages:
        chat_text = ""
        for msg in st.session_state.messages:
            chat_text += f"{msg['role'].upper()}: {msg['content']}\n\n"
            if "reasoning" in msg and msg["reasoning"]:
                chat_text += f"REASONING: {msg['reasoning']}\n\n"

        st.download_button(
            label="📥 Download Chat (.txt)",
            data=chat_text,
            file_name=f"chat_{st.session_state.session_id or 'session'}.txt",
            mime="text/plain",
        )

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
                        if src.get("page") is not None:
                            loc += f" (Page {src['page']})"
                        if src.get("row") is not None:
                            loc += f" (Row {src['row']})"

                        score = ""
                        if src.get("score"):
                            score = f" - Score: {src['score']:.2f}"

                        st.markdown(f"**{src.get('filename')}{loc}**{score}")
                        st.caption(src.get("preview_text", ""))

                        if src.get("file_hash"):
                            link = f"{BACKEND_PUBLIC_HOST}/sources/preview?collection={st.session_state.selected_collection}&file_hash={src['file_hash']}"
                            st.markdown(
                                f'<a href="{link}" target="_blank">Download/View Original</a>',
                                unsafe_allow_html=True,
                            )
                        _render_entities_relations(src)
                        st.divider()

                    st.markdown("**Information Extraction Overview**")
                    _render_ie_overview(msg["sources"])

    def stop_generation():
        st.session_state.chat_running = False
        if st.session_state.get("current_answer"):
            st.session_state.messages.append(
                {"role": "assistant", "content": st.session_state.current_answer}
            )
            del st.session_state.current_answer

    def _start_chat():
        st.session_state.chat_running = True
        st.session_state.current_answer = ""

    # Chat Input
    if prompt := st.chat_input("Ask a question...", on_submit=_start_chat):
        # 1. User message
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        # 2. Bot response
        with st.chat_message("assistant"):
            # Layout for Stop button
            c1, c2 = st.columns([0.85, 0.15])
            with c2:
                if st.session_state.get("chat_running"):
                    st.button(
                        "⏹️ Stop", on_click=stop_generation, help="Stop Generation"
                    )

            with c1:
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
                        with c1:
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
                                            st.session_state.current_answer = (
                                                full_answer
                                            )
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
                            if not st.session_state.get("chat_running"):
                                break
                            process_line(line)

                        answer_placeholder.markdown(full_answer)

                        if reasoning:
                            with st.expander("Reasoning"):
                                st.markdown(reasoning)

                        if sources:
                            with st.expander("View Sources"):
                                for j, src in enumerate(sources):
                                    loc = ""
                                    if src.get("page") is not None:
                                        loc += f" (Page {src['page']})"
                                    if src.get("row") is not None:
                                        loc += f" (Row {src['row']})"

                                    score = ""
                                    if src.get("score"):
                                        score = f" - Score: {src['score']:.2f}"

                                    st.markdown(
                                        f"**{src.get('filename')}{loc}**{score}"
                                    )
                                    st.caption(src.get("preview_text", ""))
                                    if src.get("file_hash"):
                                        link = f"{BACKEND_PUBLIC_HOST}/sources/preview?collection={st.session_state.selected_collection}&file_hash={src['file_hash']}"
                                        st.markdown(
                                            f'<a href="{link}" target="_blank">Download/View Original</a>',
                                            unsafe_allow_html=True,
                                        )
                                    _render_entities_relations(src)
                                    st.divider()

                            st.markdown("**Information Extraction Overview**")
                            _render_ie_overview(sources)

                        # 3. Save bot message
                        msg_entry = {
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
                        logger.error(f"Query failed: {resp.text}")
                        st.error(f"Query failed: {resp.text}")
            except Exception as e:
                st.session_state.chat_running = False
                logger.error(f"Error: {e}")
                st.error(f"Error: {e}")


def main() -> None:
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
