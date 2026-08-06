"""Pure map-reduce pipeline tests: fake LLM, fake fetcher, dict cache."""

from collections.abc import Callable
from typing import Any

import pytest

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
        """Initialize an empty cache store."""
        self.store: dict[str, dict[str, Any]] = {}

    def get(self, unit_key: str, validator: str) -> dict[str, Any] | None:
        """Return the stored entry for ``unit_key`` iff its validator matches.

        Args:
            unit_key: The unit's stable identity.
            validator: Content fingerprint the caller expects to match.

        Returns:
            dict[str, Any] | None: The cached entry on a hit, else ``None``.
        """
        entry = self.store.get(unit_key)
        if entry and entry["validator"] == validator:
            return entry["entry"]
        return None

    def put(self, unit_key: str, validator: str, entry: dict[str, Any]) -> None:
        """Store ``entry`` for ``unit_key`` under ``validator``.

        Args:
            unit_key: The unit's stable identity.
            validator: Content fingerprint that validates this entry.
            entry: The value to store.
        """
        self.store[unit_key] = {"validator": validator, "entry": entry}


class FakeLLM:
    """Echoes a marker per call; map calls end with an evidence line."""

    def __init__(self) -> None:
        """Initialize the call log."""
        self.prompts: list[str] = []

    def __call__(self, prompt: str) -> str:
        """Record ``prompt`` and return a deterministic marker response.

        Args:
            prompt: The rendered prompt text.

        Returns:
            str: A marker response whose kind (map/fold/synthesis) is
            inferred from ``prompt``'s prefix.
        """
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
    """A single small unit maps in one window and synthesizes in one call."""
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
    """Chunks that overflow ``window_chars`` split into multiple windows and fold."""
    llm = FakeLLM()

    def fetch(unit: MapUnit) -> list[UnitChunk]:
        return [UnitChunk(chunk_id=f"{unit.unit_key}-m{i}", text="x" * 60) for i in range(4)]

    result = _summarizer(llm, fetch, window_chars=100).build([_unit("a", n_members=4)])
    # 4 chunks of 60 chars into 100-char windows -> 4 windows -> 4 map calls
    # + 1 intra-unit fold + 1 synthesis.
    assert result.llm_calls == 6
    assert result.unit_results[0].summary.startswith("fold-summary")


def test_oversize_single_chunk_is_truncated_not_dropped() -> None:
    """A single chunk longer than ``window_chars`` is truncated, not dropped."""
    llm = FakeLLM()

    def fetch(unit: MapUnit) -> list[UnitChunk]:
        return [UnitChunk(chunk_id="big", text="y" * 500)]

    result = _summarizer(llm, fetch, window_chars=100).build([_unit("a")])
    assert result.covered_units == 1
    map_prompt = next(p for p in llm.prompts if p.startswith("MAP"))
    assert len(map_prompt) < 300


def test_cache_hit_skips_llm_and_delta_only_remaps() -> None:
    """An unchanged unit's fingerprint hits the cache; only the changed unit re-maps."""
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
    """Tripping ``max_llm_calls`` during mapping stops mapping but still synthesizes."""
    llm = FakeLLM()
    units = [_unit(f"u{i}", fingerprint=f"f{i}") for i in range(5)]
    result = _summarizer(llm, _small_fetch, max_llm_calls=2).build(units)
    assert result.partial is True
    assert result.covered_units == 2
    assert result.response.startswith("final-summary")


def test_cap_truncates_windows_inside_a_single_unit() -> None:
    """The cap bites *inside* one unit's window loop, not only between units.

    A between-units-only check let one huge unit (a 50 MB transcript is
    thousands of windows) issue thousands of map calls under a cap of 500.
    """
    llm = FakeLLM()

    def fetch(unit: MapUnit) -> list[UnitChunk]:
        return [UnitChunk(chunk_id=f"{unit.unit_key}-m{i}", text="x" * 60) for i in range(20)]

    result = _summarizer(llm, fetch, window_chars=100, max_llm_calls=3).build([_unit("a", n_members=20)])

    map_calls = [p for p in llm.prompts if p.startswith("MAP")]
    assert len(map_calls) == 3  # 20 windows requested, 3 allowed
    assert result.partial is True
    assert result.covered_units == 1  # what was summarized is kept
    # The intra-unit fold is capped too, so the windows are joined locally.
    assert not any(p.startswith("FOLD") for p in llm.prompts)
    # The final synthesis call is exempt from the cap and still ran.
    assert result.response.startswith("final-summary")


def test_cap_truncated_unit_is_not_written_to_the_map_cache() -> None:
    """A cap-truncated unit summary must not be cached under its full fingerprint.

    Caching it would store a partial summary against the unit's *complete*
    content fingerprint, so every later build would serve it as complete.
    """
    cache = DictCache()
    llm = FakeLLM()

    def fetch(unit: MapUnit) -> list[UnitChunk]:
        return [UnitChunk(chunk_id=f"{unit.unit_key}-m{i}", text="x" * 60) for i in range(20)]

    _summarizer(llm, fetch, cache=cache, window_chars=100, max_llm_calls=3).build([_unit("a", n_members=20)])

    assert cache.store == {}


def test_cap_bounds_the_reduce_fold_tier() -> None:
    """Fold tiers respect the cap; 10k units cannot issue ~1,111 uncapped folds."""
    cache = DictCache()
    units = [_unit(f"u{i}", fingerprint=f"f{i}") for i in range(20)]
    # Warm every unit so the map stage costs zero calls and the whole budget
    # is available to (and must be enforced on) the fold tier.
    _summarizer(FakeLLM(), _small_fetch, cache=cache).build(units)

    llm = FakeLLM()
    result = _summarizer(llm, _small_fetch, cache=cache, reduce_fanin=2, max_llm_calls=3).build(units)

    fold_calls = [p for p in llm.prompts if p.startswith("FOLD")]
    assert len(fold_calls) == 3  # uncapped this tier alone would issue 10
    assert result.partial is True
    assert result.covered_units == 20  # every unit still resolved, from cache
    assert result.response.startswith("final-summary")
    assert result.llm_calls == 4  # 3 folds + the exempt synthesis call


def test_final_synthesis_runs_even_when_the_cap_is_zero_budget() -> None:
    """A build with no map budget at all still produces a synthesized answer."""
    llm = FakeLLM()
    units = [_unit(f"u{i}", fingerprint=f"f{i}") for i in range(3)]

    result = _summarizer(llm, _small_fetch, max_llm_calls=1).build(units)

    assert result.partial is True
    assert result.response.startswith("final-summary")
    assert llm.prompts[-1].startswith("SYNTH")


def test_empty_units_list_returns_empty_without_llm() -> None:
    """``build([])`` short-circuits to an empty result without any LLM call."""
    llm = FakeLLM()
    result = _summarizer(llm, _small_fetch).build([])
    assert result.response == ""
    assert result.llm_calls == 0
    assert llm.prompts == []


def test_unit_with_no_chunks_is_uncovered() -> None:
    """A unit whose fetcher returns no chunks is counted uncovered, not errored."""
    llm = FakeLLM()
    result = _summarizer(llm, lambda unit: []).build([_unit("a")])
    assert result.covered_units == 0
    assert result.total_units == 1


def test_map_failure_skips_unit_and_continues() -> None:
    """A ``complete()`` exception during mapping marks only that unit uncovered."""
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
    """More briefs than ``reduce_fanin`` fold in groups before synthesis."""
    llm = FakeLLM()
    units = [_unit(f"u{i}", fingerprint=f"f{i}") for i in range(5)]
    result = _summarizer(llm, _small_fetch, reduce_fanin=2).build(units)
    fold_calls = [p for p in llm.prompts if p.startswith("FOLD")]
    assert len(fold_calls) >= 2
    assert result.response.startswith("final-summary")


def test_progress_callback_reports_per_unit() -> None:
    """``progress`` fires once per unit as ``(processed, total)``."""
    seen: list[tuple[int, int]] = []
    llm = FakeLLM()
    _summarizer(llm, _small_fetch, progress=lambda done, total: seen.append((done, total))).build(
        [_unit("a"), _unit("b", fingerprint="fb")]
    )
    assert seen == [(1, 2), (2, 2)]


def test_cache_hit_survives_cap_trip_during_mapping() -> None:
    """A cache hit is retained even when the call cap trips while mapping misses.

    Regression guard for semantics item 5: cache hits must resolve in a
    pass that is never gated by ``max_llm_calls`` — collapsing the
    two-pass loop into one cap-gated pass would drop the cached unit
    whenever the cap trips before its turn.
    """
    cache = DictCache()
    warm_llm = FakeLLM()
    _summarizer(warm_llm, _small_fetch, cache=cache).build([_unit("a", fingerprint="fa")])

    llm = FakeLLM()
    units = [
        _unit("a", fingerprint="fa"),
        _unit("b", fingerprint="fb"),
        _unit("c", fingerprint="fc"),
    ]
    result = _summarizer(llm, _small_fetch, cache=cache, max_llm_calls=1).build(units)

    assert result.partial is True
    by_key = {u.unit_key: u for u in result.unit_results}
    assert by_key["a"].from_cache is True
    assert by_key["b"].from_cache is False
    assert "c" not in by_key  # capped before mapping ever starts on it
    assert result.covered_units == 2  # cached "a" + mapped "b"
    assert result.response.startswith("final-summary")


def test_reduce_exception_propagates() -> None:
    """An exception raised during reduce/synthesis escapes ``build()``.

    Unlike a map-stage failure (which only marks its unit uncovered), a
    reduce-stage failure must propagate so the job layer can fail the job.
    """

    def flaky(prompt: str) -> str:
        if prompt.startswith("MAP"):
            return "ok\nEVIDENCE_INDICES: 1"
        raise RuntimeError("synthesis down")

    with pytest.raises(RuntimeError, match="synthesis down"):
        _summarizer(flaky, _small_fetch).build([_unit("a")])
