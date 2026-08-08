"""Full-text keyword search over collection chunk text.

The package owns a ``search_text`` payload field on each Qdrant point plus a
prefix + lowercase text index over it, and compiles keyword queries into native
Qdrant filters. It deliberately imports nothing from :mod:`docint.core.rag` —
the payload-to-text extractor is injected — so it stays unit-testable without a
RAG instance and cannot create a circular import.
"""
