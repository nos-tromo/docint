# Document Intelligence

Document Intelligence (docint) is a document RAG stack for ingestion,
retrieval, and chat. It exists for analysis and investigative work: point it at a
pile of documents and social-media exports, and it will read them — text, tables,
scanned pages, images, audio and video — so you can ask questions of the whole
corpus, follow the entities through it, and export what you found as a case
file that stands on its own.

It ships a FastAPI backend, a React SPA served by an nginx sidecar that
reverse-proxies API requests to the backend on the internal Docker network, and
pluggable inference via any OpenAI-compatible API or an external routed vLLM
service. docint runs no models itself — every ML call (chat, embeddings,
rerank, NER, CLIP, OCR, transcription) leaves over HTTP.

## What you need

- **Docker** (Engine + Compose v2) for the containerized setup.
- **Python 3.11** and [`uv`](https://github.com/astral-sh/uv) for local
  development; **Node 20 + pnpm** to run the React dev server.
- **An OpenAI-compatible inference endpoint**, configured via `.env`
  (`INFERENCE_PROVIDER`, `OPENAI_API_BASE`, `OPENAI_API_KEY`). vLLM is deployed
  separately and consumed through one routed base URL. On a non-CUDA dev host,
  `vllm-service` ships standalone CPU-only profiles (`gliner-only`,
  `rerank-only`, `clip-only`, `embed-only`) you can point the per-service base
  URLs at — see
  [configuration.md](docs/configuration.md#dense-embedding-client--embedclientconfig)
  for the per-service defaults and the `EMBED_API_BASE`-vs-`SPARSE_API_BASE`
  gotcha.
- **Qdrant**, provided by the sibling `data-plane` project — not by this stack.
  Start it with `cd ../data-plane && make up`.

## Quick start

```bash
cp .env.example .env      # then point it at your inference endpoint
make network              # external inference-net + data-net + edge-net
make volumes              # external cache + state volumes
make build
make up-dev               # http://localhost:8080 (DOCINT_HOST_PORT to override)
```

`make up-dev` layers the dev overlay so the frontend port is published;
`make up` runs the production shape with no host ports. The backend is reached
only through the nginx sidecar. Full walkthrough, including the local (non-Docker)
path and a first ingest and query:
[getting-started.md](docs/getting-started.md).

## What it does

**Multimodal RAG over one collection.** Text, tables, PDFs (page-level layout,
OCR for scanned pages), images and audio/video all land in the same collection
and compete for the same answer slots. An image — a standalone file, a figure
inside a PDF, a video keyframe — is retrieved by CLIP, reranked against the text
chunks on the same scale, and cited by number like any other source; where an
OCR model is configured, the text printed *inside* it is indexed too. Audio and
video are transcribed by an external Nextext service and keyframe-extracted.
Social-media exports that pair `postings.csv` with a `media.csv` manifest are
linked, so a post and all of its media cite as one thing. See
[ingestion.md](docs/ingestion.md) and
[retrieval-and-agents.md](docs/retrieval-and-agents.md#image-retrieval-lane).

**The Report Builder turns findings into a case file.** Collection-wide CSV
exports are all-or-nothing; a case usually needs a curated document instead. An
"+ Report" control on every chat answer, entity finding and hate-speech finding
snapshots that one artifact — image evidence frozen in as a data URI — into an
owner-scoped report you can reorder, annotate, and export as Markdown, HTML,
JSON, a CSV bundle, or a paginated PDF that references nothing outside itself.
See [reports.md](docs/reports.md).

**Multi-user by default.** Collections, chat sessions and reports are
owner-scoped: two users can each hold a `my_collection` without collision,
cross-owner access is a 404, and the active collection is resolved per request
so concurrent users never interfere. The principal comes from a trusted header
injected by the gateway (`X-Auth-User`); production leaves
`DOCINT_DEFAULT_IDENTITY` unset so an unauthenticated request is rejected. See
[architecture.md](docs/architecture.md#multi-tenancy-and-data-isolation) and
[configuration.md](docs/configuration.md#identity-and-authentication--principalconfig).

## Operating

`make help` lists every target. The ones you will actually reach for:

| Command | What it does |
|---|---|
| `make network` · `make volumes` | Create the external Docker networks and volumes. Idempotent; once per host. |
| `make build` · `make up-dev` · `make stop` | Build the images, start with the frontend port published, stop. `make up` is the production shape. |
| `make health` | Ask the running backend whether Qdrant is reachable. Chain it: `make up health`. |
| `make resolve COLLECTION=<name>` | Merge duplicate and semantically-similar entities (`USA`/`United States`) into durable canonical records. Re-runnable and idempotent. |
| `make search-index COLLECTION=<name>` · `make search-index-all` | Build the full-text search index for one collection, or backport every collection on the host. Payload-only — no re-embedding, no inference. |
| `make verify` · `make test` | Pre-push gate (ruff + pyrefly + frontend lint/build), then pytest + vitest. |

The nine console scripts — `docint`, `ingest`, `query`, `query-eval`,
`resolve`, `search-index`, `search-index-all`, `verify`, `load-models` — are
documented with their flags in [cli-reference.md](docs/cli-reference.md). Every environment variable, grouped
by the dataclass that reads it, is in
[configuration.md](docs/configuration.md).

## Documentation

The [`docs/`](docs/README.md) directory is the in-repo reference manual:

- [Getting started](docs/getting-started.md) — install, first ingest, first query
- [Architecture](docs/architecture.md) — runtime components, request flow, multi-tenancy
- [Configuration](docs/configuration.md) — every env var grouped by dataclass, with defaults
- [API reference](docs/api-reference.md) — every FastAPI route
- [CLI reference](docs/cli-reference.md) — the nine console scripts
- [Ingestion pipeline](docs/ingestion.md) — readers, chunking, NER, media, social exports
- [Retrieval and agents](docs/retrieval-and-agents.md) — orchestrator, hybrid retrieval, graph-RAG, citations
- [UI guide](docs/ui-guide.md) — React SPA screens and localization
- [Reports](docs/reports.md) — the Report Builder and its export formats
- [Deployment](docs/deployment.md) — Docker services, volumes, co-deployment, offline bundles
- [Migrations](docs/migrations.md) — what existing collections do not pick up on their own
- [Development](docs/development.md) — dev workflow, pre-commit, pytest layout, CI

## Pointers

- [vllm-service](https://github.com/nos-tromo/vllm-service) — the inference
  router docint calls for chat, embeddings, rerank, NER, CLIP and OCR.
- [nextext](https://github.com/nos-tromo/nextext) — the transcription service
  behind audio and video ingestion.
- `../data-plane` — owns Qdrant and its volumes; bring it up first.
- Issues: <https://github.com/nos-tromo/docint/issues>
