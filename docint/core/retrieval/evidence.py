"""What an assessment asks of retrieval beyond what a lookup asks.

An assessment searches through the turn's own engine, so filters, rerank and
parent context apply as on a lookup. Its searches carry :class:`EveryChildQuery`
as their bundle; the engine hands that same bundle to every postprocessor, which
is how parent-context expansion knows to keep every retrieved child of a section
rather than the first alone. A lookup's plain bundle changes nothing.
"""

from __future__ import annotations

from dataclasses import dataclass

from llama_index.core.schema import QueryBundle


@dataclass
class EveryChildQuery(QueryBundle):
    """A query whose parent-context expansion keeps every child the reranker returned.

    It adds no field: the type is the request. Parent-context expansion folds every
    later child of a section into that section's one emission instead of dropping
    it, so the node list stays exactly a lookup's. The flag rides on the bundle
    rather than on the engine, which a cached engine would share across requests.
    """
