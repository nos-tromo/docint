"""Retrieval shapes that sit beside the main text lane.

The modules here own retrieval behaviour that is not the ordinary
vector/hybrid pass over the main collection: today the visual target, which
answers from the ``{collection}_images`` companion alone. They import
llama-index, Qdrant models and the search helpers, never
:mod:`docint.core.rag` — everything RAG-bound arrives as an injected
callable, the way :class:`~docint.core.rag.MultimodalRetriever` already takes
its image lane.
"""
