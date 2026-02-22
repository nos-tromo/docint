"""Ingest page: file upload, streaming ingestion progress, and summary."""

import json
from typing import Any

import requests
import streamlit as st
from loguru import logger

from docint.ui.state import BACKEND_HOST


def render_ingestion() -> None:
    """
    Render the ingestion interface.
    """
    st.header("📥 Ingest")

    collection = st.session_state.selected_collection

    if "ingest_summary" not in st.session_state:
        st.session_state.ingest_summary = None

    # ── Target collection ─────────────────────────────────────
    with st.container(border=True):
        st.markdown("##### Target Collection")
        new_col = st.text_input(
            "New Collection Name",
            placeholder="Leave blank to use the current collection",
        )
        target_col = new_col if new_col else collection
        if target_col:
            st.caption(f"Documents will be ingested into **{target_col}**.")
        else:
            st.warning("Select an existing collection or enter a new name.")

    st.markdown("")  # spacer

    # ── File upload ───────────────────────────────────────────
    with st.container(border=True):
        st.markdown("##### Upload Documents")
        uploaded_files = st.file_uploader(
            "Choose files",
            accept_multiple_files=True,
            label_visibility="collapsed",
        )

    # ── Previous ingestion summary ────────────────────────────
    _render_ingest_summary(st.session_state.ingest_summary)

    # ── Ingest button & progress ──────────────────────────────
    if uploaded_files and st.button(
        "🚀 Upload & Ingest", type="primary", use_container_width=True
    ):
        if not target_col:
            logger.error("No target collection specified.")
            st.error("Please select or enter a collection name.")
        else:
            _run_ingestion(target_col, uploaded_files)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _render_ingest_summary(summary: dict[str, Any] | None) -> None:
    """
    Render the last-ingestion summary card.

    Args:
        summary: Ingestion summary dict (may be ``None``).
    """
    if not summary:
        return

    with st.container(border=True):
        st.markdown("##### Last Ingestion Summary")
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


def _run_ingestion(target_col: str, uploaded_files: list) -> None:
    """
    Execute the upload-and-ingest workflow with streaming progress.

    Args:
        target_col: Name of the collection to ingest into.
        uploaded_files: List of uploaded Streamlit file objects.
    """
    files = [("files", (f.name, f, f.type)) for f in uploaded_files]
    data = {"collection": target_col, "hybrid": "True"}

    file_status: dict[str, str] = {f.name: "Queued" for f in uploaded_files}
    events: list[str] = []
    summary: dict[str, int] = {"processed": 0, "errors": 0}

    header_ph = st.empty()
    progress = st.progress(0.0, text="Starting upload…")
    board_ph = st.empty()
    feed_ph = st.empty()
    summary_ph = st.empty()

    def _render_board(current_stage: str) -> None:
        """
        Render the main progress board with file statuses and overall progress.

        Args:
            current_stage (str): Description of the current stage of the ingestion process.
        """
        total = len(file_status) or 1
        done = sum(1 for s in file_status.values() if s in {"Processed", "Done"})
        errs = sum(1 for s in file_status.values() if s == "Error")
        progress.progress(
            done / total,
            text=f"{current_stage} · {done}/{total} done · {errs} errors",
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
            "NER": "🔍",
            "Processed": "✅",
            "Done": "✅",
            "Error": "❌",
        }
        pill_rows = [
            f"{pill_map.get(state, '•')} **{name}** — {state}"
            for name, state in sorted(file_status.items())
        ]
        board_ph.markdown("\n".join(pill_rows))

    def _render_feed() -> None:
        """
        Render the event feed sidebar with recent ingestion events.
        """
        if events:
            feed_ph.markdown("\n".join(events[-8:]))

    _render_board("Starting upload")

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

                    filename = event_data.get("filename") or event_data.get("file")
                    stage = event_data.get("stage") or event_data.get("status")
                    message = event_data.get("message")

                    if stage == "ingestion_progress" and message:
                        if events and events[-1].startswith("• Extracting"):
                            events[-1] = f"• {message}"
                        else:
                            events.append(f"• {message}")
                        _render_board("Processing")
                        _render_feed()
                        continue

                    if filename:
                        file_status[filename] = stage or "Processing"
                    if message:
                        events.append(
                            f"• {filename + ': ' if filename else ''}{message}"
                        )
                    elif stage and filename:
                        events.append(f"• {filename}: {stage}")
                    _render_board(stage or "Processing")
                    _render_feed()

                    if filename and message and "error" in message.lower():
                        file_status[filename] = "Error"
                        summary["errors"] += 1

                    if stage and stage.lower() in {"processed", "done"}:
                        summary["processed"] += 1

                # Mark remaining as done
                for name in file_status:
                    if file_status[name] not in {"Processed", "Done", "Error"}:
                        file_status[name] = "Done"
                _render_board("Complete")

                total = len(file_status)
                errs = sum(1 for s in file_status.values() if s == "Error")
                done = sum(
                    1 for s in file_status.values() if s in {"Processed", "Done"}
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
                logger.error("Ingestion failed: {}", response.text)
                st.error(f"Ingestion failed: {response.text}")
        except Exception as e:
            status.update(label="Error", state="error")
            logger.error("Error during ingestion: {}", e)
            st.error(f"Error during ingestion: {e}")
