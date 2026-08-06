"""Dependency-injected map-reduce (tree) summarization pipeline.

Pure: no Qdrant, no RAG, no model runtimes. The orchestration layer in
:mod:`docint.core.rag` supplies the LLM completer, the chunk fetcher, the
prompt templates, and the map cache.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any, Protocol

from loguru import logger

from docint.core.summary.units import MapUnit

_EVIDENCE_RE = re.compile(r"^EVIDENCE_INDICES:\s*([0-9,\s]+)\s*$", re.MULTILINE)
_MAX_EVIDENCE_PER_UNIT = 2


@dataclass(frozen=True)
class UnitChunk:
    """One retrievable piece of text belonging to a :class:`MapUnit`.

    Attributes:
        chunk_id: Stable identity of the chunk (typically a Qdrant point id).
        text: The chunk's raw text content.
    """

    chunk_id: str
    text: str


@dataclass
class UnitMapResult:
    """The map stage's output for a single unit.

    Attributes:
        unit_key: The unit's stable identity (see :class:`MapUnit`).
        kind: ``"document"`` or ``"social_bucket"``.
        label: Human-readable name shown in reduce briefs and diagnostics.
        summary: The unit's summary text (evidence line stripped).
        evidence_ids: Chunk ids picked as representative, at most
            :data:`_MAX_EVIDENCE_PER_UNIT`.
        from_cache: ``True`` when this result was served from the cache
            without any LLM calls.
    """

    unit_key: str
    kind: str
    label: str
    summary: str
    evidence_ids: list[str] = field(default_factory=list)
    from_cache: bool = False


@dataclass
class TreeSummaryResult:
    """The pipeline's overall output.

    Attributes:
        response: The final synthesized summary text.
        unit_results: Per-unit map results for covered units, in input order.
        covered_units: Count of units that produced a summary (cache hit or
            successful map).
        total_units: Count of units passed to :meth:`TreeSummarizer.build`.
        partial: ``True`` when the LLM-call cap truncated mapping before all
            units were covered.
        llm_calls: Total number of ``complete()`` invocations made (map +
            intra-unit fold + reduce fold + synthesis).
    """

    response: str
    unit_results: list[UnitMapResult]
    covered_units: int
    total_units: int
    partial: bool
    llm_calls: int


class MapCache(Protocol):
    """Storage seam for per-unit map results, keyed by content fingerprint."""

    def get(self, unit_key: str, validator: str) -> dict[str, Any] | None:
        """Return a cached entry for ``unit_key`` iff its validator matches.

        Args:
            unit_key: The unit's stable identity.
            validator: Content fingerprint the caller expects to match.

        Returns:
            dict[str, Any] | None: ``{"summary": str, "evidence_ids": list[str]}``
            on a hit, else ``None``.
        """
        ...

    def put(self, unit_key: str, validator: str, entry: dict[str, Any]) -> None:
        """Store ``entry`` for ``unit_key`` under ``validator``.

        Args:
            unit_key: The unit's stable identity.
            validator: Content fingerprint that validates this entry.
            entry: ``{"summary": str, "evidence_ids": list[str]}``.
        """
        ...


def _parse_evidence(text: str, window_chunk_ids: list[str]) -> tuple[str, list[str]]:
    """Strip the ``EVIDENCE_INDICES`` line and resolve indices to chunk ids.

    Args:
        text: Raw map/fold response text, possibly ending in an
            ``EVIDENCE_INDICES: 1,2`` line.
        window_chunk_ids: Chunk ids for the window, in the same order used
            to number the prompt's ``chunk_block`` (1-based).

    Returns:
        tuple[str, list[str]]: The summary text with the evidence line
        removed, and the resolved chunk ids (falls back to the window's
        first chunk id when the line is missing or unparseable).
    """
    matches = list(_EVIDENCE_RE.finditer(text))
    if not matches:
        return text.strip(), window_chunk_ids[:1]
    last = matches[-1]
    summary = (text[: last.start()] + text[last.end() :]).strip()
    ids: list[str] = []
    for token in last.group(1).split(","):
        token = token.strip()
        if not token.isdigit():
            continue
        index = int(token) - 1
        if 0 <= index < len(window_chunk_ids):
            ids.append(window_chunk_ids[index])
    return summary, (ids[:_MAX_EVIDENCE_PER_UNIT] or window_chunk_ids[:1])


def _windows(chunks: list[UnitChunk], window_chars: int) -> list[list[UnitChunk]]:
    """Greedily pack chunks into windows of at most ``window_chars`` chars.

    Empty/whitespace-only chunks are dropped. A single chunk longer than
    ``window_chars`` is truncated to fit rather than dropped or split
    further.

    Args:
        chunks: The unit's chunks, in reading order.
        window_chars: Maximum total text length per window.

    Returns:
        list[list[UnitChunk]]: Non-empty windows, each within budget.
    """
    usable: list[UnitChunk] = []
    for chunk in chunks:
        text = chunk.text.strip()
        if not text:
            continue
        if len(text) > window_chars:
            text = text[:window_chars]
        usable.append(UnitChunk(chunk_id=chunk.chunk_id, text=text))

    windows: list[list[UnitChunk]] = []
    current: list[UnitChunk] = []
    current_len = 0
    for chunk in usable:
        chunk_len = len(chunk.text)
        if current and current_len + chunk_len > window_chars:
            windows.append(current)
            current = []
            current_len = 0
        current.append(chunk)
        current_len += chunk_len
    if current:
        windows.append(current)
    return windows


def _chunked(items: list[str], size: int) -> Iterator[list[str]]:
    """Yield successive ``size``-sized slices of ``items``.

    Args:
        items: The list to slice.
        size: Maximum length of each slice (the last slice may be shorter).

    Yields:
        list[str]: Successive slices covering all of ``items``.
    """
    for start in range(0, len(items), size):
        yield items[start : start + size]


class TreeSummarizer:
    """Pure map-reduce summarization over pre-partitioned :class:`MapUnit` s.

    All LLM access, chunk fetching, prompt rendering, and caching are
    injected as callables/protocols so this class has no knowledge of
    Qdrant, RAG, or any model runtime.
    """

    def __init__(
        self,
        *,
        complete: Callable[[str], str],
        fetch_chunks: Callable[[MapUnit], list[UnitChunk]],
        map_prompt: str,
        fold_prompt: str,
        build_synthesis_prompt: Callable[[list[str], dict[str, Any]], str],
        cache: MapCache | None = None,
        window_chars: int = 12000,
        reduce_fanin: int = 10,
        max_llm_calls: int = 500,
        progress: Callable[[int, int], None] | None = None,
    ) -> None:
        """Configure the pipeline.

        Args:
            complete: One LLM call: prompt text in, response text out.
            fetch_chunks: Returns a unit's member chunks in reading order.
            map_prompt: Format string with ``{label}``/``{chunk_block}``
                placeholders for one window's map call.
            fold_prompt: Format string with a ``{summaries_block}``
                placeholder for folding multiple summaries into one.
            build_synthesis_prompt: Builds the final synthesis prompt from
                per-unit briefs and a diagnostics dict.
            cache: Optional per-unit map-result cache, validated by content
                fingerprint.
            window_chars: Maximum characters packed into one map window
                (``map_window_tokens * 4``, a char-ratio approximation).
            reduce_fanin: Maximum briefs folded together per reduce round.
            max_llm_calls: Cap on LLM calls during the map stage; the
                reduce stage is never blocked by it.
            progress: Optional callback invoked ``(processed, total)``
                after each unit resolves (hit, mapped, or failed).
        """
        self._complete_fn = complete
        self._fetch_chunks = fetch_chunks
        self._map_prompt = map_prompt
        self._fold_prompt = fold_prompt
        self._build_synthesis_prompt = build_synthesis_prompt
        self._cache = cache
        self.window_chars = window_chars
        self.reduce_fanin = reduce_fanin
        self.max_llm_calls = max_llm_calls
        self._progress = progress
        self._calls = 0

    def _complete(self, prompt: str) -> str:
        """Invoke the injected completer and count the call.

        Args:
            prompt: Fully rendered prompt text.

        Returns:
            str: The completer's response text.
        """
        self._calls += 1
        return self._complete_fn(prompt)

    def _map_unit(self, unit: MapUnit) -> UnitMapResult | None:
        """Map one unit to a summary via windowed LLM calls.

        Args:
            unit: The unit to summarize.

        Returns:
            UnitMapResult | None: The map result, or ``None`` when the unit
            has no usable chunks or the completer raised.
        """
        try:
            chunks = self._fetch_chunks(unit)
            windows = _windows(chunks, self.window_chars)
            if not windows:
                return None

            window_summaries: list[str] = []
            evidence_ids: list[str] = []
            for window in windows:
                window_ids = [chunk.chunk_id for chunk in window]
                chunk_block = "\n\n".join(f"[{i}] {chunk.text}" for i, chunk in enumerate(window, start=1))
                prompt = self._map_prompt.format(label=unit.label, chunk_block=chunk_block)
                response = self._complete(prompt)
                summary, ids = _parse_evidence(response, window_ids)
                window_summaries.append(summary)
                for evidence_id in ids:
                    if len(evidence_ids) >= _MAX_EVIDENCE_PER_UNIT:
                        break
                    if evidence_id not in evidence_ids:
                        evidence_ids.append(evidence_id)

            if len(window_summaries) > 1:
                summaries_block = "\n".join(f"- {s}" for s in window_summaries)
                summary = self._complete(self._fold_prompt.format(summaries_block=summaries_block))
            else:
                summary = window_summaries[0]

            return UnitMapResult(
                unit_key=unit.unit_key,
                kind=unit.kind,
                label=unit.label,
                summary=summary,
                evidence_ids=evidence_ids[:_MAX_EVIDENCE_PER_UNIT],
                from_cache=False,
            )
        except Exception as exc:  # noqa: BLE001 - map failures are per-unit, not fatal
            logger.warning("Tree summary map failed for unit '{}': {}", unit.unit_key, exc)
            return None

    def build(self, units: list[MapUnit]) -> TreeSummaryResult:
        """Run the full map-reduce pipeline over ``units``.

        Args:
            units: Pre-partitioned units to summarize (see
                :func:`docint.core.summary.units.partition_units`).

        Returns:
            TreeSummaryResult: The synthesized summary plus per-unit
            results and coverage/partial diagnostics.
        """
        self._calls = 0
        total_units = len(units)
        if total_units == 0:
            return TreeSummaryResult(
                response="", unit_results=[], covered_units=0, total_units=0, partial=False, llm_calls=0
            )

        results_by_key: dict[str, UnitMapResult] = {}
        misses: list[MapUnit] = []

        # First pass: resolve every cache hit — they cost no LLM calls and
        # must never be blocked by the call cap.
        for unit in units:
            entry = self._cache.get(unit.unit_key, unit.fingerprint) if self._cache is not None else None
            if entry is not None:
                results_by_key[unit.unit_key] = UnitMapResult(
                    unit_key=unit.unit_key,
                    kind=unit.kind,
                    label=unit.label,
                    summary=entry["summary"],
                    evidence_ids=list(entry.get("evidence_ids") or []),
                    from_cache=True,
                )
            else:
                misses.append(unit)

        # Second pass: map the cache misses, respecting the call cap. Progress
        # is reported for cache hits first (input order), then for each
        # mapped/failed miss, since hits already resolved in the first pass.
        partial = False
        processed = 0
        for unit in units:
            if unit.unit_key in results_by_key:
                processed += 1
                if self._progress is not None:
                    self._progress(processed, total_units)

        for unit in misses:
            if self._calls >= self.max_llm_calls:
                partial = True
                break
            result = self._map_unit(unit)
            if result is not None:
                results_by_key[unit.unit_key] = result
                if self._cache is not None:
                    self._cache.put(
                        unit.unit_key,
                        unit.fingerprint,
                        {"summary": result.summary, "evidence_ids": result.evidence_ids},
                    )
            processed += 1
            if self._progress is not None:
                self._progress(processed, total_units)

        unit_results = [results_by_key[unit.unit_key] for unit in units if unit.unit_key in results_by_key]
        covered_units = len(unit_results)

        briefs = [
            f"- {'Document' if r.kind == 'document' else 'Posts'}: {r.label}\n  {r.summary}" for r in unit_results
        ]
        while len(briefs) > self.reduce_fanin:
            briefs = [
                self._complete(self._fold_prompt.format(summaries_block="\n\n".join(group)))
                for group in _chunked(briefs, self.reduce_fanin)
            ]

        diagnostics = {"covered_units": covered_units, "total_units": total_units, "partial": partial}
        response = self._complete(self._build_synthesis_prompt(briefs, diagnostics))

        return TreeSummaryResult(
            response=response,
            unit_results=unit_results,
            covered_units=covered_units,
            total_units=total_units,
            partial=partial,
            llm_calls=self._calls,
        )
