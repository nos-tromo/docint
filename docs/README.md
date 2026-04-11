# Docint Documentation

This directory contains the in-repo reference manual for **Docint**, the
Document Intelligence RAG stack. It complements the top-level
[`README.md`](../README.md) (which focuses on install and quick starts) with
topic-by-topic deep dives.

## Table of contents

| Document | What it covers |
|---|---|
| [getting-started.md](getting-started.md) | Install, Docker quick start, local dev quick start, first ingest and first query |
| [architecture.md](architecture.md) | Runtime architecture, component map, end-to-end request flow |
| [configuration.md](configuration.md) | Every environment variable grouped by the dataclass that reads it, with defaults |
| [api-reference.md](api-reference.md) | Every FastAPI route, method, tag, request/response shape, and streaming semantics |
| [cli-reference.md](cli-reference.md) | `docint`, `ingest`, `query`, `query-eval`, `load-models` — flags and examples |
| [ingestion.md](ingestion.md) | Document ingestion pipeline: triage, layout, OCR, extraction, chunking, embedding |
| [retrieval-and-agents.md](retrieval-and-agents.md) | Agent orchestration, hybrid retrieval, reranking, parent-context expansion, Graph-RAG |
| [ui-guide.md](ui-guide.md) | Streamlit UI pages — Dashboard, Chat, Ingest, Analysis, Inspector |
| [deployment.md](deployment.md) | Docker profiles, shared volumes, networks, vLLM co-deployment, proxies |
| [development.md](development.md) | Dev workflow, `uv`, pre-commit, pytest layout, CI, extension points |

## Who this is for

- **Operators** deploying Docint behind their own inference stack — start
  with [getting-started.md](getting-started.md), then
  [configuration.md](configuration.md) and [deployment.md](deployment.md).
- **Backend developers** extending the RAG engine, agents, or ingestion
  pipeline — start with [architecture.md](architecture.md), then the
  module-level docs in
  [ingestion.md](ingestion.md) and [retrieval-and-agents.md](retrieval-and-agents.md).
- **API consumers** wiring a client against the FastAPI surface — go
  straight to [api-reference.md](api-reference.md).
- **UI contributors** — see [ui-guide.md](ui-guide.md) and the Streamlit
  entry point `docint/app.py`.

## Conventions used in these docs

- **Source references** use the `path:line` format (for example
  `docint/core/api.py:427`) so editors can jump directly to the symbol.
- **Environment-variable tables** always show the variable name, the
  dataclass it lives in, and the default value baked into the
  `load_*_env()` factory.
- **Endpoint tables** use the tag groups from
  `docint/core/api.py` (`Collections`, `Query`, `Sessions`, `Agent`,
  `Ingestion`, `Sources`).
- Documentation is plain Markdown (GitHub Flavored). No MkDocs/Sphinx
  build step is required.
