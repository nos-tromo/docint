"""Pure entity-resolution pipeline (no I/O, no model runtime).

This mirrors chorus's ``chorus/ingestion/resolution.py`` decision logic but
keeps every side effect behind injected callables so the module is trivially
unit-testable:

- the entity store (lookups / vector clustering / mint / attach) is an
  :class:`EntityStoreProtocol`,
- embedding is an ``embed_fn`` callable,
- the chat LLM tie-break is a ``chat_fn`` callable.

Pipeline per surface form::

    normalize_surface
      -> store.lookup_alias        (exact, idempotent skip)
      -> store.cluster_candidates  (type-blocked vector search >= threshold)
      -> llm_tiebreaker            (only when > 1 candidate)
      -> store.mint_entity         (when no confident match)

The batch runner :func:`resolve_collection` processes surfaces
most-mentioned-first (so the most common surface becomes the canonical name)
and keeps an in-run ``(normalized_surface, type)`` cache so case/whitespace
variants cluster deterministically without depending on vector-index lag.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any, Protocol

# A vector-search hit: ``{"id", "canonical_name", "type", "score"}``.
Candidate = dict[str, Any]

EmbedFn = Callable[[list[str]], list[list[float]]]
ChatFn = Callable[[str], str]

_ID_TOKEN_RE = re.compile(r"[\w-]+")


class EntityStoreProtocol(Protocol):
    """Persistence contract the resolution pipeline depends on."""

    def lookup_alias(self, surface: str, entity_type: str) -> str | None:
        """Return the entity id for an exact ``(surface, type)`` alias, if any."""
        ...

    def cluster_candidates(self, embedding: list[float], threshold: float, k: int, entity_type: str) -> list[Candidate]:
        """Return up to ``k`` same-type entities scoring ``>= threshold``."""
        ...

    def mint_entity(self, surface: str, embedding: list[float], entity_type: str) -> str:
        """Create a new canonical entity seeded with ``surface`` and return its id."""
        ...

    def attach_alias(self, entity_id: str, surface: str) -> None:
        """Attach ``surface`` as an alias of an existing entity."""
        ...


@dataclass(frozen=True)
class SurfaceMention:
    """A distinct surface form and its mention count for one entity type.

    Attributes:
        surface (str): Raw surface text as it appeared in documents.
        entity_type (str): Normalized entity type/label (a type block).
        mentions (int): Number of mentions across the collection (used to
            order processing so the most common surface becomes canonical).
    """

    surface: str
    entity_type: str
    mentions: int


@dataclass(frozen=True)
class ResolutionSummary:
    """Outcome counts for one :func:`resolve_collection` run.

    Attributes:
        processed (int): Surface forms processed.
        minted (int): New canonical entities created.
        attached (int): Surfaces attached to an existing entity (via vector
            match, LLM tie-break, top-k fallback, or the in-run cache).
        skipped (int): Surfaces already resolved on a prior run (idempotent).
        entities_touched (int): Distinct entity ids minted or attached to.
    """

    processed: int
    minted: int
    attached: int
    skipped: int
    entities_touched: int


def normalize_surface(surface: str, *, case_normalize: bool) -> str:
    """Normalize a surface form for in-run cache clustering.

    Args:
        surface (str): Raw surface text.
        case_normalize (bool): When ``True``, casefold for case-insensitive
            clustering; otherwise only strip surrounding whitespace.

    Returns:
        str: Normalized surface used as part of the in-run cache key.
    """
    out = surface.strip()
    if case_normalize:
        out = out.casefold()
    return out


def llm_tiebreaker(
    surface: str,
    candidates: list[Candidate],
    *,
    chat_fn: ChatFn,
    prompt_header: str,
) -> str | None:
    r"""Ask the chat LLM to pick the candidate matching ``surface``.

    The reply is parsed strictly: it must contain exactly one known candidate
    id (tokenized on ``[\\w-]+`` so UUID/hyphenated ids survive and ``e-1`` is
    never matched by ``e-12``). Anything else — ``NONE``, an unknown token, or
    several ids — abstains by returning ``None`` so the caller can mint rather
    than force a wrong merge.

    Args:
        surface (str): The surface form being resolved.
        candidates (list[Candidate]): Above-threshold candidate entities.
        chat_fn (ChatFn): Callable that sends a prompt and returns the reply.
        prompt_header (str): Locale-aware instructional preamble (the
            translatable prose); the candidate scaffold is appended in English.

    Returns:
        str | None: The chosen entity id, or ``None`` to abstain.
    """
    if not candidates:
        return None
    lines = [f"{i + 1}. id={c['id']} name={c['canonical_name']!r} type={c['type']}" for i, c in enumerate(candidates)]
    prompt = f"{prompt_header.strip()}\n\nSurface form: {surface!r}\nCandidates:\n" + "\n".join(lines) + "\n"
    reply = (chat_fn(prompt) or "").strip()
    tokens = set(_ID_TOKEN_RE.findall(reply))
    matched = [str(c["id"]) for c in candidates if str(c["id"]) in tokens]
    return matched[0] if len(matched) == 1 else None


def resolve_surface(
    store: EntityStoreProtocol,
    surface: str,
    embedding: list[float],
    cfg: Any,
    *,
    entity_type: str,
    chat_fn: ChatFn,
    prompt_header: str,
) -> tuple[str, str]:
    """Resolve one surface form to an entity id, minting if needed.

    Args:
        store (EntityStoreProtocol): The entity persistence layer.
        surface (str): Raw surface text to resolve.
        embedding (list[float]): Embedding of ``surface``.
        cfg (ResolutionConfig): Resolution thresholds/toggles.
        entity_type (str): Type block for candidate filtering.
        chat_fn (ChatFn): Chat callable for the tie-break.
        prompt_header (str): Locale-aware tie-break preamble.

    Returns:
        tuple[str, str]: ``(entity_id, method)`` where method is one of
        ``"skipped"``, ``"minted"``, ``"vector_single"``, ``"vector_llm"``,
        ``"vector_topk"``.
    """
    existing = store.lookup_alias(surface, entity_type)
    if existing is not None:
        return existing, "skipped"

    candidates = store.cluster_candidates(embedding, cfg.embed_cluster_threshold, cfg.vector_k, entity_type)
    if not candidates:
        return store.mint_entity(surface, embedding, entity_type), "minted"

    if len(candidates) == 1:
        entity_id, method = str(candidates[0]["id"]), "vector_single"
    elif cfg.llm_tiebreak_enabled:
        chosen = llm_tiebreaker(surface, candidates, chat_fn=chat_fn, prompt_header=prompt_header)
        if chosen is None:
            return store.mint_entity(surface, embedding, entity_type), "minted"
        entity_id, method = chosen, "vector_llm"
    else:
        entity_id, method = str(candidates[0]["id"]), "vector_topk"

    store.attach_alias(entity_id, surface)
    return entity_id, method


def resolve_collection(
    store: EntityStoreProtocol,
    surfaces: Iterable[SurfaceMention],
    *,
    embed_fn: EmbedFn,
    chat_fn: ChatFn,
    prompt_header: str,
    cfg: Any,
) -> ResolutionSummary:
    """Resolve every surface form in a collection into canonical entities.

    Surfaces are processed most-mentioned-first so the most common surface
    mints first and becomes the canonical name. An in-run
    ``(normalized_surface, type)`` cache clusters case/whitespace variants in
    a single pass regardless of vector-index lag.

    Args:
        store (EntityStoreProtocol): The entity persistence layer.
        surfaces (Iterable[SurfaceMention]): Distinct ``(surface, type,
            mentions)`` triples for the collection.
        embed_fn (EmbedFn): Batch embed callable (handles its own chunking).
        chat_fn (ChatFn): Chat callable for tie-breaks.
        prompt_header (str): Locale-aware tie-break preamble.
        cfg (ResolutionConfig): Resolution thresholds/toggles.

    Returns:
        ResolutionSummary: Outcome counts for the run.
    """
    ordered = sorted(surfaces, key=lambda m: (-m.mentions, m.surface))
    if not ordered:
        return ResolutionSummary(processed=0, minted=0, attached=0, skipped=0, entities_touched=0)

    vectors = embed_fn([m.surface for m in ordered])

    run_cache: dict[tuple[str, str], str] = {}
    touched: set[str] = set()
    minted = attached = skipped = 0

    for mention, vector in zip(ordered, vectors, strict=True):
        cache_key = (
            normalize_surface(mention.surface, case_normalize=cfg.case_normalize),
            mention.entity_type.lower(),
        )
        if cache_key in run_cache:
            entity_id = run_cache[cache_key]
            store.attach_alias(entity_id, mention.surface)
            attached += 1
            touched.add(entity_id)
            continue

        entity_id, method = resolve_surface(
            store,
            mention.surface,
            list(vector),
            cfg,
            entity_type=mention.entity_type,
            chat_fn=chat_fn,
            prompt_header=prompt_header,
        )
        run_cache[cache_key] = entity_id
        touched.add(entity_id)
        if method == "skipped":
            skipped += 1
        elif method == "minted":
            minted += 1
        else:
            attached += 1

    return ResolutionSummary(
        processed=len(ordered),
        minted=minted,
        attached=attached,
        skipped=skipped,
        entities_touched=len(touched),
    )
