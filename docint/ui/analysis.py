"""Analysis page: collection summarisation and NER overview."""

import html
import json
import re
import time
from typing import Any

import requests
import streamlit as st

from docint.ui.components import (
    aggregate_ner,
    reference_metadata_inline,
    reference_metadata_text_block,
    render_response_validation,
    render_source_item,
    render_summary_diagnostics,
)
from docint.utils.env_cfg import load_host_env

BACKEND_HOST = load_host_env().backend_host


def render_analysis() -> None:
    """Render the analysis interface."""
    st.header("📈 Analysis")

    collection = st.session_state.selected_collection
    if not collection:
        st.info("Select a collection from the sidebar to view analysis.")
        return

    st.caption(f"📁 {collection}")

    ner_cache = _get_ner_cache()
    summary_cache = _get_summary_cache()

    if collection not in ner_cache:
        ner_result = _load_ner_analysis(collection, refresh=True)
    else:
        ner_result = ner_cache.get(collection, _empty_ner_result(collection))

    summary_state = summary_cache.setdefault(
        collection, _empty_summary_result(collection)
    )

    if st.button("🔄 Refresh NER analysis", width="stretch"):
        ner_result = _load_ner_analysis(collection, refresh=True)

    if ner_result.get("error"):
        st.warning(str(ner_result["error"]))

    summary_state = _render_summary_section(collection)
    _render_analysis_result(ner_result, collection, summary_state=summary_state)


def _update_summary_metadata(result: dict[str, Any], data: dict[str, Any]) -> None:
    """Update summary-related metadata fields in analysis state.

    Args:
        result: Mutable analysis-result state.
        data: Parsed SSE event payload.
    """
    if "validation_checked" in data:
        result["validation_checked"] = data.get("validation_checked")
    if "validation_mismatch" in data:
        result["validation_mismatch"] = data.get("validation_mismatch")
    if "validation_reason" in data:
        result["validation_reason"] = data.get("validation_reason")

    diagnostics = data.get("summary_diagnostics")
    if isinstance(diagnostics, dict):
        result["summary_diagnostics"] = diagnostics


def _empty_ner_result(collection: str) -> dict[str, Any]:
    """Build default NER analysis state for a collection.

    Args:
        collection: Currently selected collection name.

    Returns:
        Default state payload for NER analysis data and metadata.
    """
    return {
        "collection": collection,
        "ner": None,
        "hate_speech": [],
        "loaded": False,
        "error": None,
        "errors": {},
        "last_loaded_at": None,
    }


def _empty_summary_result(collection: str) -> dict[str, Any]:
    """Build default summary state for a collection.

    Args:
        collection: Currently selected collection name.

    Returns:
        Default state payload for optional summary output.
    """
    return {
        "collection": collection,
        "generated": False,
        "summary": "",
        "sources": [],
        "validation_checked": None,
        "validation_mismatch": None,
        "validation_reason": None,
        "summary_diagnostics": None,
        "error": None,
    }


def _get_ner_cache() -> dict[str, dict[str, Any]]:
    """Return mutable NER cache map keyed by collection.

    Returns:
        Mutable dictionary representing the NER cache.
    """
    cache = st.session_state.get("analysis_ner_by_collection")
    if not isinstance(cache, dict):
        cache = {}
        st.session_state.analysis_ner_by_collection = cache
    return cache


def _get_summary_cache() -> dict[str, dict[str, Any]]:
    """Return mutable summary cache map keyed by collection.

    Returns:
        Mutable dictionary representing the summary cache.
    """
    cache = st.session_state.get("analysis_summary_by_collection")
    if not isinstance(cache, dict):
        cache = {}
        st.session_state.analysis_summary_by_collection = cache
    return cache


def _fetch_ner_analysis(collection: str, *, refresh: bool = False) -> dict[str, Any]:
    """Fetch NER + hate-speech payloads for a collection.

    Args:
        collection: Currently selected collection name.
        refresh: Whether to bypass backend NER cache.

    Returns:
        NER payload with load/error metadata.
    """
    _ = collection
    result = _empty_ner_result(collection)
    result["loaded"] = True
    errors: dict[str, str] = {}

    try:
        params = {"refresh": "true"} if refresh else None
        ner_resp = requests.get(
            f"{BACKEND_HOST}/collections/ner",
            params=params,
            timeout=120,
        )
        if ner_resp.status_code == 200:
            result["ner"] = ner_resp.json().get("sources", [])
        else:
            errors["ner"] = (
                f"NER request failed ({ner_resp.status_code}): {ner_resp.text.strip()}"
            )
            result["ner"] = []
    except Exception as exc:
        errors["ner"] = f"NER request failed: {exc}"
        result["ner"] = []

    try:
        hate_resp = requests.get(
            f"{BACKEND_HOST}/collections/hate-speech",
            timeout=120,
        )
        if hate_resp.status_code == 200:
            result["hate_speech"] = hate_resp.json().get("results", [])
        else:
            errors["hate_speech"] = (
                f"Hate-speech request failed ({hate_resp.status_code}): "
                f"{hate_resp.text.strip()}"
            )
            result["hate_speech"] = []
    except Exception as exc:
        errors["hate_speech"] = f"Hate-speech request failed: {exc}"
        result["hate_speech"] = []

    result["errors"] = errors
    if errors:
        result["error"] = " ; ".join(errors.values())
    result["last_loaded_at"] = time.time()
    return result


def _load_ner_analysis(collection: str, *, refresh: bool) -> dict[str, Any]:
    """Load NER analysis data and persist it in session cache.

    Args:
        collection: Currently selected collection name.
        refresh: Whether to bypass backend NER cache.

    Returns:
        Loaded NER analysis payload.
    """
    with st.spinner("Loading entity and hate-speech analysis…"):
        result = _fetch_ner_analysis(collection, refresh=refresh)
    _get_ner_cache()[collection] = result
    return result


def _generate_summary(collection: str) -> None:
    """Generate summary for the selected collection and store it in cache.

    Args:
        collection: Currently selected collection name.
    """
    summary_placeholder = st.empty()
    full_summary = ""
    current_sources: list[dict[str, Any]] = []
    summary_state = _empty_summary_result(collection)

    with st.spinner("Generating summary…"):
        try:
            resp = requests.post(
                f"{BACKEND_HOST}/summarize/stream", stream=True, timeout=600
            )
            if resp.status_code == 200:
                for line in resp.iter_lines():
                    if not line:
                        continue
                    decoded = line.decode("utf-8")
                    if decoded.startswith("data: "):
                        data = json.loads(decoded[6:])
                        if "token" in data:
                            full_summary += data["token"]
                            summary_placeholder.markdown(full_summary + "▌")
                        elif "sources" in data:
                            current_sources = data["sources"]
                        _update_summary_metadata(summary_state, data)
                        if "error" in data:
                            summary_state["error"] = str(data["error"])
                summary_placeholder.markdown(full_summary)
                summary_state["summary"] = full_summary
                summary_state["sources"] = current_sources
                summary_state["generated"] = True
            else:
                summary_state["error"] = (
                    f"Summarisation failed ({resp.status_code}): {resp.text}"
                )
        except Exception as e:
            summary_state["error"] = f"Error: {e}"

    _get_summary_cache()[collection] = summary_state
    st.rerun()


def _render_summary_section(collection: str) -> dict[str, Any]:
    """Render manual summary controls and current summary output.

    Args:
        collection: Currently selected collection name.

    Returns:
        Summary state for the selected collection.
    """
    summary_cache = _get_summary_cache()
    summary_state = summary_cache.setdefault(
        collection, _empty_summary_result(collection)
    )

    st.markdown(f"### Summary of '{collection}'")
    st.caption("Summary generation uses LLM tokens.")
    button_label = (
        "Generate summary"
        if not summary_state.get("generated")
        else "Regenerate summary"
    )
    if st.button(button_label, type="primary", width="stretch"):
        _generate_summary(collection)

    if summary_state.get("error"):
        st.error(str(summary_state["error"]))

    if not summary_state.get("generated"):
        st.info("Summary has not been generated yet.")
        return summary_state

    st.markdown(str(summary_state.get("summary") or ""))
    render_response_validation(
        validation_checked=summary_state.get("validation_checked"),
        validation_mismatch=summary_state.get("validation_mismatch"),
        validation_reason=summary_state.get("validation_reason"),
    )
    render_summary_diagnostics(summary_state.get("summary_diagnostics"))

    sources = list(summary_state.get("sources") or [])
    if sources:
        with st.expander("Sources used for summary"):
            for src in sources:
                render_source_item(src, collection)

    return summary_state


def _entity_option_label(entity: dict[str, Any]) -> str:
    """Build a compact selectbox label for an aggregated entity row.

    Args:
        entity: The aggregated entity dictionary containing text, type, and mention count.

    Returns:
        A formatted string label for the entity selectbox option.
    """
    text = str(entity.get("text") or "Unknown")
    entity_type = str(entity.get("type") or "Unlabeled")
    mentions = int(entity.get("count", entity.get("mentions", 0)) or 0)
    return f"{text} [{entity_type}] · {mentions}"


def _entity_highlight_terms(entity: str | dict[str, Any]) -> list[str]:
    """Return unique entity terms that should be highlighted in chunk text.

    Args:
        entity: The selected entity, either as a string or an aggregated entity dictionary.

    Returns:
        A list of unique entity terms to highlight, sorted by length and deduplicated case-insensitively.
    """
    terms: list[str] = []
    if isinstance(entity, dict):
        for variant in list(entity.get("variants") or []):
            text = str(variant.get("text") or "").strip()
            if text:
                terms.append(text)
        selected_text = str(entity.get("text") or "").strip()
        if selected_text:
            terms.append(selected_text)
    else:
        selected_text = str(entity or "").strip()
        if selected_text:
            terms.append(selected_text)

    seen: set[str] = set()
    deduped: list[str] = []
    for term in sorted(terms, key=lambda item: (-len(item), item.lower())):
        lowered = term.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        deduped.append(term)
    return deduped


def _highlight_entity_text(chunk_text: str, entity: str | dict[str, Any]) -> str:
    """Return HTML with matching entity variants highlighted.

    Args:
        chunk_text: The original text of the chunk to process.
        entity: The selected entity, either as a string or an aggregated entity dictionary.

    Returns:
        An HTML string with entity terms highlighted and non-matching text escaped.
    """
    text = str(chunk_text or "")
    terms = _entity_highlight_terms(entity)
    if not text or not terms:
        return (
            "<div style='white-space: pre-wrap; line-height: 1.6;'>"
            f"{html.escape(text)}"
            "</div>"
        )

    pattern = re.compile(
        "|".join(re.escape(term) for term in terms),
        flags=re.IGNORECASE,
    )
    parts: list[str] = []
    cursor = 0
    for match in pattern.finditer(text):
        start, end = match.span()
        if start > cursor:
            parts.append(html.escape(text[cursor:start]))
        parts.append(
            "<mark style='background-color: rgba(255, 255, 0, 1.0); "
            "color: black; padding: 0 0.16rem; border-radius: 0.25rem;'>"
            f"{html.escape(match.group(0))}"
            "</mark>"
        )
        cursor = end
    if cursor < len(text):
        parts.append(html.escape(text[cursor:]))

    return (
        f"<div style='white-space: pre-wrap; line-height: 1.6;'>{''.join(parts)}</div>"
    )


def _entity_related_chunks(
    entity: str | dict[str, Any],
    sources: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return source chunks that mention a selected entity.

    Args:
        entity: The selected entity text or aggregated entity row to match.
        sources: List of source dictionaries containing chunk and entity information.

    Returns:
        A list of source dictionaries that contain the selected entity.
    """
    selected_text = str(entity or "").strip()
    selected_key = ""
    variant_lookups: set[str] = set()
    variant_compacts: set[str] = set()
    if isinstance(entity, dict):
        selected_text = str(entity.get("text") or "").strip()
        selected_key = str(entity.get("key") or "").strip().lower()
        for variant in list(entity.get("variants") or []):
            variant_text = str(variant.get("text") or "").strip().lower()
            if not variant_text:
                continue
            variant_lookups.add(variant_text)
            variant_compacts.add("".join(ch for ch in variant_text if ch.isalnum()))

    needle = selected_text.lower()
    if needle:
        variant_lookups.add(needle)
        variant_compacts.add("".join(ch for ch in needle if ch.isalnum()))
    if not variant_lookups and not selected_key:
        return []

    matches: list[dict[str, Any]] = []
    for src in sources:
        entities = src.get("entities")
        if not isinstance(entities, list):
            continue
        if not any(
            isinstance(ent, dict)
            and (
                (
                    bool(selected_key)
                    and str(ent.get("key") or "").strip().lower() == selected_key
                )
                or str(ent.get("text") or ent.get("name") or "").strip().lower()
                in variant_lookups
                or "".join(
                    ch
                    for ch in str(ent.get("text") or ent.get("name") or "")
                    .strip()
                    .lower()
                    if ch.isalnum()
                )
                in variant_compacts
                or str(ent.get("key") or "").split("::", maxsplit=1)[0].strip().lower()
                in variant_lookups
            )
            for ent in entities
        ):
            continue

        chunk_text = str(
            src.get("chunk_text") or src.get("text") or src.get("preview_text") or ""
        ).strip()
        matches.append(
            {
                "chunk_id": str(src.get("chunk_id") or "").strip(),
                "chunk_text": chunk_text,
                "filename": str(src.get("filename") or "Unknown"),
                "page": src.get("page"),
                "row": src.get("row"),
                "reference_metadata": src.get("reference_metadata"),
            }
        )
    return matches


def _entity_chunks_to_txt(entity: str, chunks: list[dict[str, Any]]) -> str:
    """Build downloadable text payload for entity-linked chunks.

    Args:
        entity: The selected entity text.
        chunks: List of source dictionaries containing chunk information.

    Returns:
        A formatted string containing metadata and text of chunks related to the entity.
    """
    lines = [f"Entity: {entity}", ""]
    for idx, chunk in enumerate(chunks, start=1):
        lines.append(
            f"[{idx}] {chunk.get('filename')} page={chunk.get('page')} row={chunk.get('row')} "
            f"chunk_id={chunk.get('chunk_id')}"
        )
        metadata_block = reference_metadata_text_block(
            chunk,
            include_text=False,
        )
        if metadata_block:
            lines.append(metadata_block)
        lines.append(str(chunk.get("chunk_text") or ""))
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def _hate_speech_chunks_to_txt(chunks: list[dict[str, Any]]) -> str:
    """Build downloadable text payload for hate-speech findings.

    Args:
        chunks: List of source dictionaries containing hate-speech finding information.

    Returns:
        A formatted string containing metadata and text of chunks flagged for hate-speech.
    """
    lines = ["Flagged hate-speech chunks", ""]
    for idx, chunk in enumerate(chunks, start=1):
        location_label = "page" if chunk.get("page") is not None else "row"
        location_value = (
            chunk.get("page")
            if chunk.get("page") is not None
            else chunk.get("row", "n/a")
        )
        lines.append(
            f"[{idx}]\n"
            f"- source: {chunk.get('source_ref')}\n"
            f"- {location_label}: {location_value}\n"
            f"- chunk_id: {chunk.get('chunk_id')}\n"
            f"- category: {chunk.get('category')}\n"
            f"- confidence: {chunk.get('confidence')}"
        )
        lines.append(f"reason: {chunk.get('reason')}")
        metadata_block = reference_metadata_text_block(chunk)
        if metadata_block:
            lines.append(metadata_block)
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def _render_entities_tab(result: dict[str, Any], collection: str) -> None:
    """Render entity drill-down UI.

    Args:
        result: The analysis result dictionary containing NER data.
        collection: The name of the currently selected collection.
    """
    ner_sources = result.get("ner") or []
    if not ner_sources:
        st.info("No entities or relations found in this collection.")
        return

    entities, _ = aggregate_ner(ner_sources)
    entity_rows = [
        entity for entity in entities if str(entity.get("text") or "").strip()
    ]
    if not entity_rows:
        st.caption("No entities available for chunk drill-down.")
        return

    entity_types = sorted(
        {
            str(entity.get("type") or "Unlabeled").strip()
            for entity in entity_rows
            if str(entity.get("type") or "Unlabeled").strip()
        }
    )
    filter_col, entity_col = st.columns([1, 2], gap="medium")
    with filter_col:
        selected_type = st.selectbox(
            "Entity category",
            options=["All", *entity_types],
            index=0,
            key=f"analysis_entity_type_{collection}",
        )

    filtered_entities = [
        entity
        for entity in entity_rows
        if selected_type == "All"
        or str(entity.get("type") or "Unlabeled") == selected_type
    ]
    if not filtered_entities:
        st.caption("No entities available for the selected category.")
        return

    option_map = {
        str(entity.get("key") or entity.get("text") or ""): entity
        for entity in filtered_entities
    }
    with entity_col:
        selected_entity_key = st.selectbox(
            "Entity",
            options=list(option_map),
            index=0,
            format_func=lambda option: _entity_option_label(option_map[option]),
            key=f"analysis_entity_selected_{collection}",
        )

    selected_entity = option_map.get(selected_entity_key)

    if selected_entity:
        selected_entity_text = str(selected_entity.get("text") or "").strip()
        chunks = _entity_related_chunks(selected_entity, ner_sources)
        st.markdown(f"#### Entity findings related to **{selected_entity_text}**")
        if chunks:
            for idx, chunk in enumerate(chunks, start=1):
                label = (
                    f"{chunk.get('filename')} "
                    f"(page={chunk.get('page')}, row={chunk.get('row')}, id={chunk.get('chunk_id') or 'n/a'})"
                )
                with st.expander(f"Chunk {idx}: {label}"):
                    metadata_line = reference_metadata_inline(chunk)
                    if metadata_line:
                        st.caption(metadata_line)
                    chunk_text = str(chunk.get("chunk_text") or "").strip()
                    if chunk_text:
                        st.markdown(
                            _highlight_entity_text(chunk_text, selected_entity),
                            unsafe_allow_html=True,
                        )
                    else:
                        st.caption("Chunk text unavailable for this record.")
            st.download_button(
                label="Download",
                data=_entity_chunks_to_txt(selected_entity_text, chunks),
                file_name=f"entity_{selected_entity_text}_{collection}.txt".replace(
                    " ", "_"
                ),
                mime="text/plain",
            )
        else:
            st.caption("No chunks were matched for the selected entity.")


def _render_hate_speech_tab(result: dict[str, Any], collection: str) -> None:
    """Render hate-speech findings with expandable chunk details.

    Displays a numbered summary table followed by expanders whose labels
    reference the corresponding table row so users can cross-reference
    findings easily.

    Args:
        result: The analysis result dictionary containing hate-speech findings.
        collection: The name of the currently selected collection.
    """
    findings = list(result.get("hate_speech") or [])
    st.markdown("#### Hate speech findings")
    if not findings:
        st.info("No hate speech flags detected for this collection.")
        return

    st.dataframe(
        {
            "#": list(range(1, len(findings) + 1)),
            "Category": [row.get("category") for row in findings],
            "Reason": [row.get("reason") for row in findings],
            "Source": [row.get("source_ref") for row in findings],
            "Chunk ID": [row.get("chunk_id") for row in findings],
        },
        width="stretch",
        hide_index=True,
    )
    st.caption("Expand a row below to view the flagged chunk text.")
    for idx, row in enumerate(findings, start=1):
        category = row.get("category") or "unknown"
        reason = row.get("reason") or ""
        with st.expander(f"{idx} — {category.capitalize()}: {reason}"):
            source = row.get("source_ref") or "Unknown source"
            page = row.get("page")
            row_number = row.get("row")
            reason = row.get("reason") or ""
            chunk_text = (row.get("chunk_text") or "").strip()
            metadata_line = reference_metadata_inline(row)
            location_label = "Page" if page is not None else "Row"
            location_value = page if page is not None else row_number
            detail_lines = [
                f"**Source:** {source}",
                f"**{location_label}:** {location_value if location_value is not None else 'n/a'}",
            ]
            if metadata_line:
                detail_lines.append(metadata_line)
            st.caption("  \n".join(detail_lines))
            if chunk_text:
                st.write(chunk_text)
            else:
                st.caption("Chunk text unavailable for this record.")

    st.download_button(
        label="Download",
        data=_hate_speech_chunks_to_txt(findings),
        file_name=f"hate_speech_{collection}.txt",
        mime="text/plain",
    )


def _render_analysis_result(
    result: dict[str, Any],
    collection: str,
    *,
    summary_state: dict[str, Any] | None = None,
) -> None:
    """Render NER analysis output for a collection.

    Args:
        result: NER analysis result dictionary.
        collection: The name of the currently selected collection.
        summary_state: Optional summary state used when composing downloadable text.
    """
    st.markdown("---")
    tabs = st.tabs(["Entities", "Hate Speech"])
    with tabs[0]:
        _render_entities_tab(result, collection)

    with tabs[1]:
        _render_hate_speech_tab(result, collection)
