"""Pure map-reduce pipeline tests: fake LLM, fake fetcher, dict cache."""

from collections.abc import Callable
from typing import Any

from docint.core.summary.tree import MapCache, TreeSummarizer, UnitChunk
from docint.core.summary.units import MapUnit

MAP_PROMPT = "MAP {label}\n{chunk_block}"
FOLD_PROMPT = "FOLD\n{summaries_block}"


def _unit(key: str, n_members: int = 1, kind: str = "document", fingerprint: str = "fp") -> MapUnit:
    return MapUnit(
        unit_key=key,
        kind=kind,
        label=f"label-{key}",
        fingerprint=fingerprint,
        member_ids=tuple(f"{key}-m{i}" for i in range(n_members)),
    )


class DictCache:
    """In-memory :class:`MapCache` stub keyed by ``(unit_key, validator)``."""

    def __init__(self) -> None:
        self.store: dict[str, dict[str, Any]] = {}

    def get(self, unit_key: str, validator: str) -> dict[str, Any] | None:
        entry = self.store.get(unit_key)
        if entry and entry["validator"] == validator:
            return entry["entry"]
        return None

    def put(self, unit_key: str, validator: str, entry: dict[str, Any]) -> None:
        self.store[unit_key] = {"validator": validator, "entry": entry}


class FakeLLM:
    """Echoes a marker per call; map calls end with an evidence line."""

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if prompt.startswith("MAP"):
            return f"map-summary-{len(self.prompts)}\nEVIDENCE_INDICES: 1"
        if prompt.startswith("FOLD"):
            return f"fold-summary-{len(self.prompts)}"
        return f"final-summary-{len(self.prompts)}"


def _summarizer(
    llm: Callable[[str], str],
    fetch: Callable[[MapUnit], list[UnitChunk]],
    cache: MapCache | None = None,
    **kw: Any,
) -> TreeSummarizer:
    defaults: dict[str, Any] = dict(
        complete=llm,
        fetch_chunks=fetch,
        map_prompt=MAP_PROMPT,
        fold_prompt=FOLD_PROMPT,
        build_synthesis_prompt=lambda briefs, diag: "SYNTH\n" + "\n".join(briefs),
        cache=cache,
        window_chars=100,
        reduce_fanin=10,
        max_llm_calls=500,
    )
    defaults.update(kw)
    return TreeSummarizer(**defaults)


def _small_fetch(unit: MapUnit) -> list[UnitChunk]:
    return [UnitChunk(chunk_id=m, text=f"text of {m}") for m in unit.member_ids]


def test_single_unit_one_window() -> None:
    llm = FakeLLM()
    result = _summarizer(llm, _small_fetch).build([_unit("a")])
    assert result.total_units == 1
    assert result.covered_units == 1
    assert result.partial is False
    assert result.llm_calls == 2  # one map + one synthesis
    assert result.response.startswith("final-summary")
    assert result.unit_results[0].summary == "map-summary-1"
    assert result.unit_results[0].evidence_ids == ["a-m0"]
    assert "EVIDENCE_INDICES" not in result.unit_results[0].summary


def test_windowing_splits_long_units_and_folds() -> None:
    llm = FakeLLM()

    def fetch(unit: MapUnit) -> list[UnitChunk]:
        return [UnitChunk(chunk_id=f"{unit.unit_key}-m{i}", text="x" * 60) for i in range(4)]

    result = _summarizer(llm, fetch, window_chars=100).build([_unit("a", n_members=4)])
    # 4 chunks of 60 chars into 100-char windows -> 4 windows -> 4 map calls
    # + 1 intra-unit fold + 1 synthesis.
    assert result.llm_calls == 6
    assert result.unit_results[0].summary.startswith("fold-summary")


def test_oversize_single_chunk_is_truncated_not_dropped() -> None:
    llm = FakeLLM()

    def fetch(unit: MapUnit) -> list[UnitChunk]:
        return [UnitChunk(chunk_id="big", text="y" * 500)]

    result = _summarizer(llm, fetch, window_chars=100).build([_unit("a")])
    assert result.covered_units == 1
    map_prompt = next(p for p in llm.prompts if p.startswith("MAP"))
    assert len(map_prompt) < 300


def test_cache_hit_skips_llm_and_delta_only_remaps() -> None:
    cache = DictCache()
    llm1 = FakeLLM()
    units = [_unit("a", fingerprint="fa"), _unit("b", fingerprint="fb")]
    _summarizer(llm1, _small_fetch, cache=cache).build(units)

    llm2 = FakeLLM()
    changed = [_unit("a", fingerprint="fa"), _unit("b", fingerprint="fb-CHANGED")]
    result = _summarizer(llm2, _small_fetch, cache=cache).build(changed)
    map_calls = [p for p in llm2.prompts if p.startswith("MAP")]
    assert len(map_calls) == 1  # only unit b re-mapped
    by_key = {u.unit_key: u for u in result.unit_results}
    assert by_key["a"].from_cache is True
    assert by_key["b"].from_cache is False


def test_call_cap_marks_partial_but_still_synthesizes() -> None:
    llm = FakeLLM()
    units = [_unit(f"u{i}", fingerprint=f"f{i}") for i in range(5)]
    result = _summarizer(llm, _small_fetch, max_llm_calls=2).build(units)
    assert result.partial is True
    assert result.covered_units == 2
    assert result.response.startswith("final-summary")


def test_empty_units_list_returns_empty_without_llm() -> None:
    llm = FakeLLM()
    result = _summarizer(llm, _small_fetch).build([])
    assert result.response == ""
    assert result.llm_calls == 0
    assert llm.prompts == []


def test_unit_with_no_chunks_is_uncovered() -> None:
    llm = FakeLLM()
    result = _summarizer(llm, lambda unit: []).build([_unit("a")])
    assert result.covered_units == 0
    assert result.total_units == 1


def test_map_failure_skips_unit_and_continues() -> None:
    calls = {"n": 0}

    def flaky(prompt: str) -> str:
        calls["n"] += 1
        if prompt.startswith("MAP label-a"):
            raise RuntimeError("llm down")
        if prompt.startswith("MAP"):
            return "ok\nEVIDENCE_INDICES: 1"
        return "final"

    result = _summarizer(flaky, _small_fetch).build([_unit("a"), _unit("b", fingerprint="fb")])
    assert result.covered_units == 1
    assert result.total_units == 2
    assert result.response == "final"


def test_reduce_folds_when_over_fanin() -> None:
    llm = FakeLLM()
    units = [_unit(f"u{i}", fingerprint=f"f{i}") for i in range(5)]
    result = _summarizer(llm, _small_fetch, reduce_fanin=2).build(units)
    fold_calls = [p for p in llm.prompts if p.startswith("FOLD")]
    assert len(fold_calls) >= 2
    assert result.response.startswith("final-summary")


def test_progress_callback_reports_per_unit() -> None:
    seen: list[tuple[int, int]] = []
    llm = FakeLLM()
    _summarizer(llm, _small_fetch, progress=lambda done, total: seen.append((done, total))).build(
        [_unit("a"), _unit("b", fingerprint="fb")]
    )
    assert seen == [(1, 2), (2, 2)]
