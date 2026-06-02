"""Entity resolution: durable canonical entities over Qdrant.

The :mod:`resolution` module holds the pure, dependency-injected pipeline
(normalize -> exact alias -> vector cluster -> conservative LLM tie-break ->
mint), mirroring the chorus resolution stage. The :mod:`store` module is the
Qdrant-backed companion collection that persists canonical entities and their
aliases.
"""
