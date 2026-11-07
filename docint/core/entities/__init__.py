"""Entity resolution: durable canonical entities over Qdrant.

The :mod:`resolution` module holds the pure, dependency-injected pipeline
(normalize -> exact alias -> vector cluster -> conservative LLM tie-break ->
mint). The :mod:`store` module is the
Qdrant-backed companion collection that persists canonical entities and their
aliases.
"""
