"""Ingest page: file upload, streaming ingestion progress, and summary."""

import json
import re
from typing import Any

import requests
import streamlit as st
from loguru import logger

from docint.ui.state import BACKEND_HOST


def _normalize_file_status(raw: str | None) -> str:
    """Normalize raw stage/status text to user-facing labels.

    Args:
        raw: Raw stage or status text.

    Returns:
        Normalized status label.
    """
    txt = str(raw or "").strip().lower()
    if not txt:
        return "Processing"
    if txt in {"processed", "done", "ingestion_complete"}:
        return "Done"
    if "error" in txt:
        return "Error"
    if txt in {"upload_progress", "uploading", "file_saved"}:
        return "Uploading"
    if txt in {"ingestion_started", "processing"}:
        return "Processing"
    return str(raw).strip().title() if raw else "Processing"


def _prettify_progress_message(message: str) -> str:
    """Convert backend progress strings into concise, readable text.

    Args:
        message: Raw progress message from backend ingestion callback.

    Returns:
        Human-friendly progress message.
    """
    text = str(message or "").strip()
    if not text:
        return text

    ner_match = re.search(
        r"Extracting entities:\s*(\d+)/(\d+)\s*chunks processed",
        text,
        flags=re.IGNORECASE,
    )
    if ner_match:
        return f"NER extraction {ner_match.group(1)}/{ner_match.group(2)} chunks"

    pdf_match = re.search(
        r"Core pipeline processing PDF\s*\((\d+)/(\d+)\):\s*(.+)$",
        text,
        flags=re.IGNORECASE,
    )
    if pdf_match:
        return (
            f"Processing PDF {pdf_match.group(1)}/{pdf_match.group(2)}"
            f" ({pdf_match.group(3)})"
        )

    indexed_match = re.search(
        r"Core pipeline indexed\s*(\d+)\s*chunks:\s*(.+)$",
        text,
        flags=re.IGNORECASE,
    )
    if indexed_match:
        return f"Indexed {indexed_match.group(1)} chunks ({indexed_match.group(2)})"

    return text


def _coerce_event_entries(events_raw: list[Any] | None) -> list[dict[str, str]]:
    """Normalize event payloads to a common dict shape.

    Args:
        events_raw: List of raw events (dicts or legacy strings).

    Returns:
        Normalized list of event dictionaries.
    """
    entries: list[dict[str, str]] = []
    for raw in events_raw or []:
        if isinstance(raw, dict):
            entries.append(
                {
                    "stage": str(raw.get("stage") or ""),
                    "filename": str(raw.get("filename") or ""),
                    "message": str(raw.get("message") or ""),
                }
            )
            continue
        if isinstance(raw, str):
            text = raw.strip()
            if text.startswith("•"):
                text = text[1:].strip()
            entries.append({"stage": "", "filename": "", "message": text})
    return entries


def _render_event_list(events_raw: list[Any] | None, *, limit: int = 8) -> str:
    """Build markdown for a compact ingestion event timeline.

    Args:
        events_raw: Raw events list.
        limit: Number of last events to render.

    Returns:
        Markdown bullet list.
    """
    entries = _coerce_event_entries(events_raw)
    if not entries:
        return ""

    icons = {
        "start": "🚀",
        "upload_progress": "📤",
        "file_saved": "💾",
        "ingestion_started": "⚙️",
        "ingestion_progress": "🔄",
        "ingestion_complete": "✅",
        "error": "❌",
    }
    lines: list[str] = []
    for entry in entries[-max(1, int(limit)) :]:
        stage = entry.get("stage") or ""
        filename = entry.get("filename") or ""
        message = _prettify_progress_message(entry.get("message") or "")
        icon = icons.get(stage, "•")
        scope = f"`{filename}` · " if filename else ""
        if not message and stage:
            message = stage.replace("_", " ").title()
        if not message:
            continue
        lines.append(f"- {icon} {scope}{message}")
    return "\n".join(lines)


def _append_event(
    events: list[dict[str, str]],
    *,
    stage: str,
    filename: str = "",
    message: str = "",
) -> None:
    """Append an event with de-noising for repetitive progress updates.

    Args:
        events: Mutable event list.
        stage: Event stage/event-name.
        filename: Optional file scope.
        message: Optional event detail text.
    """
    entry = {
        "stage": stage,
        "filename": filename,
        "message": _prettify_progress_message(message),
    }
    if not entry["message"] and entry["stage"] == "upload_progress":
        return

    if events:
        last = events[-1]
        if stage == "ingestion_progress" and last.get("stage") == "ingestion_progress":
            events[-1] = entry
            return
        if last == entry:
            return
    events.append(entry)


def _infer_done_filename_from_message(message: str) -> str | None:
    """Infer filename from a pipeline indexed message.

    Args:
        message: Progress message.

    Returns:
        Filename if present in message, otherwise ``None``.
    """
    match = re.search(
        r"Core pipeline indexed\s*\d+\s*chunks:\s*(.+)$",
        str(message or ""),
        flags=re.IGNORECASE,
    )
    return match.group(1).strip() if match else None


def render_ingestion() -> None:
    """Render the ingestion interface."""
    st.header("📥 Ingest")

    collection = st.session_state.selected_collection

    if "ingest_summary" not in st.session_state:
        st.session_state.ingest_summary = None

    # -- Target collection --
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

    # -- File upload --
    with st.container(border=True):
        st.markdown("##### Upload Documents")
        uploaded_files = st.file_uploader(
            "Choose files",
            accept_multiple_files=True,
            label_visibility="collapsed",
        )

    # -- Previous ingestion summary --
    _render_ingest_summary(st.session_state.ingest_summary)

    # -- Ingest button & progress --
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
    """Render the last-ingestion summary card.

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

        file_status = summary.get("file_status", {}) or {}
        if file_status:
            st.caption("File status")
            status_rows = [
                {"File": name, "Status": _normalize_file_status(status)}
                for name, status in sorted(file_status.items())
            ]
            st.dataframe(status_rows, use_container_width=True, hide_index=True)

        events = summary.get("events") or []
        if events:
            st.caption("Recent events")
            timeline = _render_event_list(events, limit=8)
            if timeline:
                st.markdown(timeline)


def _run_ingestion(target_col: str, uploaded_files: list) -> None:
    """Execute the upload-and-ingest workflow with streaming progress.

    Args:
        target_col: Name of the collection to ingest into.
        uploaded_files: List of uploaded Streamlit file objects.
    """
    files = [("files", (f.name, f, f.type)) for f in uploaded_files]
    data = {"collection": target_col, "hybrid": "True"}

    file_status: dict[str, str] = {f.name: "Queued" for f in uploaded_files}
    events: list[dict[str, str]] = []

    header_ph = st.empty()
    progress = st.progress(0.0, text="Starting upload...")
    board_ph = st.empty()
    feed_ph = st.empty()
    summary_ph = st.empty()

    def _render_board(current_stage: str) -> None:
        """Render the main progress board with file statuses and overall progress.

        Args:
            current_stage: Description of the current stage.
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
            "Processed": "✅",
            "Done": "✅",
            "Error": "❌",
        }
        table_lines = ["| File | Status |", "|---|---|"]
        table_lines.extend(
            [
                f"| `{name}` | {pill_map.get(state, '•')} {state} |"
                for name, state in sorted(file_status.items())
            ]
        )
        board_ph.markdown("\n".join(table_lines))

    def _render_feed() -> None:
        """Render recent ingestion events."""
        timeline = _render_event_list(events, limit=8)
        if timeline:
            feed_ph.markdown(timeline)
        else:
            feed_ph.caption("Waiting for events...")

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
                current_event = ""
                for line in response.iter_lines():
                    if not line:
                        continue

                    decoded_line = line.decode("utf-8")
                    if decoded_line.startswith("event: "):
                        current_event = decoded_line[7:].strip().lower()
                        continue
                    if not decoded_line.startswith("data: "):
                        continue

                    try:
                        event_data = json.loads(decoded_line[6:])
                    except json.JSONDecodeError:
                        continue

                    filename = str(
                        event_data.get("filename") or event_data.get("file") or ""
                    ).strip()
                    stage = (
                        str(
                            event_data.get("stage")
                            or event_data.get("status")
                            or current_event
                            or ""
                        )
                        .strip()
                        .lower()
                    )
                    message = str(event_data.get("message") or "").strip()
                    current_event = ""

                    if stage == "start":
                        _append_event(
                            events,
                            stage=stage,
                            message=f"Upload started for collection `{target_col}`",
                        )
                        _render_feed()
                        continue

                    if stage == "upload_progress":
                        if filename:
                            file_status[filename] = "Uploading"
                        _render_board("Uploading")
                        continue

                    if stage == "file_saved":
                        if filename:
                            file_status[filename] = "Uploading"
                        _append_event(
                            events,
                            stage=stage,
                            filename=filename,
                            message="Saved upload",
                        )
                        _render_board("Uploading")
                        _render_feed()
                        continue

                    if stage == "ingestion_started":
                        for name, current in file_status.items():
                            if current not in {"Done", "Error"}:
                                file_status[name] = "Processing"
                        _append_event(
                            events,
                            stage=stage,
                            message=f"Ingestion started for `{target_col}`",
                        )
                        _render_board("Processing")
                        _render_feed()
                        continue

                    if stage == "ingestion_progress":
                        inferred_done = _infer_done_filename_from_message(message)
                        if inferred_done and inferred_done in file_status:
                            file_status[inferred_done] = "Done"
                        _append_event(
                            events,
                            stage=stage,
                            filename=filename,
                            message=message,
                        )
                        _render_board("Processing")
                        _render_feed()
                        continue

                    if stage == "ingestion_complete":
                        _append_event(
                            events,
                            stage=stage,
                            message=f"Ingestion completed for `{target_col}`",
                        )
                        _render_board("Complete")
                        _render_feed()
                        continue

                    if stage == "error":
                        if filename:
                            file_status[filename] = "Error"
                        _append_event(
                            events,
                            stage=stage,
                            filename=filename,
                            message=message or "Ingestion failed",
                        )
                        _render_board("Error")
                        _render_feed()
                        continue

                    # Fallback path for unknown/legacy payloads.
                    if filename and stage:
                        file_status[filename] = _normalize_file_status(stage)
                    if message:
                        _append_event(
                            events,
                            stage=stage or "info",
                            filename=filename,
                            message=message,
                        )
                    _render_board(stage.title() if stage else "Processing")
                    _render_feed()

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
