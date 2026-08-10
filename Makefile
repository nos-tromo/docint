# Build-host helpers for docint.
#
# docint is a CPU-only Python app. All ML inference (chat, embeddings,
# rerank, NER, CLIP) is delegated over HTTP to the external vllm-service
# stack — see CLAUDE.md. There is no PROFILE / CUDA toggle here.
#
# The compose lifecycle (network/volumes/build/bundle/up/up-dev/dev/stop/down/
# logs/pre-commit/test) + the versioned image tag come from make/common.mk,
# vendored from nos-tromo/.github. Only docint-specific config, the help
# text, and the `resolve` target live here.

.DEFAULT_GOAL := help

REPO     := docint
NETWORKS := inference-net data-net edge-net
VOLUMES  := docling-cache huggingface-cache ollama-cache sessions-storage source-preview-cache pipeline-storage
include make/common.mk

.PHONY: help resolve health search-index search-index-all

help:
	@echo "docint — build-host helpers."
	@echo
	@echo "  make network    create the external inference-net + data-net + edge-net"
	@echo "  make volumes    create the external Docker volumes"
	@echo "  make build      build images"
	@echo "  make bundle     ship the built images as a versioned .tar.gz (latest annotated release tag)"
	@echo "  make bundle-dev like 'bundle', but from the current working tree (dev/soak)"
	@echo "  make up         run docint detached, no build (production shape, no host ports)"
	@echo "  make up-dev     like 'up', but publishes the frontend port on the host (no build)"
	@echo "  make dev        build, then up-dev"
	@echo "  make stop       stop docint containers"
	@echo "  make down       stop + remove containers (never touches data-plane state)"
	@echo "  make health     check backend dependency status (Qdrant reachability); chain as 'make up health'"
	@echo "  make resolve    merge duplicate/similar entities (COLLECTION=<name> optional)"
	@echo "  make search-index  build the full-text search index (COLLECTION=<name> optional)"
	@echo "  make search-index-all  build it for every collection on this host (one-time backport)"
	@echo "  make pre-commit run pre-commit hooks (ruff + pyrefly)"
	@echo "  make verify     pre-push gate: pre-commit + frontend lint/build; mirrors CI's lint gate"
	@echo "  make test       run the test suite"

# Dependency status report, printed on the build host's terminal. `up`/`up-dev`
# are detached, so the backend's own startup probe log never reaches stdout;
# this asks the running backend via GET /health (which re-runs the Qdrant
# probe on demand). Runs inside the backend container because the production
# shape publishes no host ports. Exits 0 either way — a Qdrant outage is a
# warning, not a failed `up` (the backend deliberately serves without it).
# Chain it: `make up health` / `make up-dev health`.
health:
	@$(COMPOSE) exec -T backend python3 -c "\
	import json, urllib.request; \
	d = json.load(urllib.request.urlopen('http://localhost:8000/health', timeout=15)); \
	print('docint health: ' + d['status']); \
	d['qdrant'] or print('WARNING: Qdrant is unreachable — ingest and query will fail until the data-plane stack is up on data-net.')" \
	|| echo "docint health: UNKNOWN — backend not answering /health (backend down, or image predates the endpoint)."

# Resolve duplicate / semantically-similar entities for a collection into
# durable canonicals (see CLAUDE.md). Runs the `resolve` CLI in a one-off
# backend container so it reaches the qdrant / vllm-router network aliases —
# production is Docker-only, no host `uv`. Interactive by default; pass
# COLLECTION=<name> to run non-interactively. Requires the backend image
# built (make build/up) and data-plane + vllm-service already up.
resolve:
	@if [ -n "$(COLLECTION)" ]; then \
		printf '%s\n' "$(COLLECTION)" | $(COMPOSE) run --rm -T backend resolve; \
	else \
		$(COMPOSE) run --rm backend resolve; \
	fi

# Build the full-text search index for a collection: creates the `search_text`
# payload index and backfills the field across existing points. Payload-only —
# no re-embedding, no inference — so it is safe on an airgapped host, and
# re-running it skips populated points. Runs in a one-off backend container so
# it reaches the qdrant network alias; production is Docker-only, no host `uv`.
# Interactive by default; pass COLLECTION=<name> to run non-interactively.
search-index:
	@if [ -n "$(COLLECTION)" ]; then \
		printf '%s\n' "$(COLLECTION)" | $(COMPOSE) run --rm -T backend search-index; \
	else \
		$(COMPOSE) run --rm backend search-index; \
	fi

# Backport the full-text search index across every collection on this host.
# Works on physical collection names, so two users owning the same logical name
# are simply two entries — no ambiguity to resolve. Idempotent: collections
# already carrying `search_text` are scanned and skipped cheaply, so re-running
# after a partial failure is safe. One failing collection does not strand the
# rest; the run exits non-zero and names every failure at the end.
search-index-all:
	$(COMPOSE) run --rm -T backend search-index-all
