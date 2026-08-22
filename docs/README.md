# Docint Documentation

This directory contains the in-repo reference manual for **Docint**, the
Document Intelligence RAG stack. It complements the top-level
[`README.md`](../README.md) (which orients you: what docint is, how to start
it, and what it can do) with topic-by-topic deep dives.

## Table of contents

| Document | What it covers |
|---|---|
| [getting-started.md](getting-started.md) | Install, Docker quick start, local dev quick start, first ingest and first query |
| [architecture.md](architecture.md) | Runtime architecture, component map, end-to-end request flow |
| [configuration.md](configuration.md) | Every environment variable grouped by the dataclass that reads it, with defaults |
| [api-reference.md](api-reference.md) | Every FastAPI route, method, tag, request/response shape, and streaming semantics |
| [cli-reference.md](cli-reference.md) | `docint`, `ingest`, `resolve`, `search-index`, `search-index-all`, `query`, `query-eval`, `verify`, `load-models` — flags and examples |
| [ingestion.md](ingestion.md) | Document ingestion pipeline: triage, layout, OCR, extraction, chunking, embedding |
| [retrieval-and-agents.md](retrieval-and-agents.md) | Agent orchestration, hybrid retrieval, reranking, parent-context expansion, Graph-RAG |
| [ui-guide.md](ui-guide.md) | React SPA pages — Dashboard, Chat, Ingest, Analysis, Inspector, Report — plus localization |
| [reports.md](reports.md) | Report Builder: curating findings into a case file, frozen image evidence, the five export formats |
| [deployment.md](deployment.md) | Docker services, shared volumes, networks, vLLM co-deployment, proxies |
| [migrations.md](migrations.md) | Changes existing collections do not pick up on their own, and the one action that fixes each |
| [development.md](development.md) | Dev workflow, `uv`, pre-commit, pytest layout, CI, extension points |

## Who this is for

- **Operators** deploying Docint behind their own inference stack — start
  with [getting-started.md](getting-started.md), then
  [configuration.md](configuration.md) and [deployment.md](deployment.md).
- **Backend developers** extending the RAG engine, agents, or ingestion
  pipeline — start with [architecture.md](architecture.md), then the
  module-level docs in
  [ingestion.md](ingestion.md) and [retrieval-and-agents.md](retrieval-and-agents.md).
- **Investigators** working a case in the app — [ui-guide.md](ui-guide.md)
  for the screens, then [reports.md](reports.md) for turning findings into an
  exportable case file.
- **API consumers** wiring a client against the FastAPI surface — go
  straight to [api-reference.md](api-reference.md).
- **UI contributors** — see [ui-guide.md](ui-guide.md) and the React SPA
  under `frontend/`.

## Conventions used in these docs

- **Source references** use the `path:line` format (for example
  `docint/core/api.py:427`) so editors can jump directly to the symbol.
- **Environment-variable tables** always show the variable name, the
  dataclass it lives in, and the default value baked into the
  `load_*_env()` factory.
- **Endpoint tables** use the tag groups from
  `docint/core/api.py` (`Meta`, `Collections`, `Query`, `Sessions`,
  `Reports`, `Agent`, `Ingestion`, `Sources`).
- Documentation is plain Markdown (GitHub Flavored). No MkDocs/Sphinx
  build step is required.
- Dated `YYYY-MM-DD-*.md` files alongside these pages are design and plan
  history, not reference material; they are not listed above.
