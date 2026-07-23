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
NETWORKS := inference-net data-net
VOLUMES  := docling-cache huggingface-cache ollama-cache sessions-storage source-preview-cache
include make/common.mk

.PHONY: help resolve

help:
	@echo "docint — build-host helpers."
	@echo
	@echo "  make network    create the external inference-net + data-net"
	@echo "  make volumes    create the external Docker volumes"
	@echo "  make build      build images"
	@echo "  make bundle     ship the built images as a versioned .tar.gz (latest annotated release tag)"
	@echo "  make bundle-dev like 'bundle', but from the current working tree (dev/soak)"
	@echo "  make up         run docint detached, no build (production shape, no host ports)"
	@echo "  make up-dev     like 'up', but publishes the frontend port on the host (no build)"
	@echo "  make dev        build, then up-dev"
	@echo "  make stop       stop docint containers"
	@echo "  make down       stop + remove containers (never touches data-plane state)"
	@echo "  make resolve    merge duplicate/similar entities (COLLECTION=<name> optional)"
	@echo "  make pre-commit run pre-commit hooks (ruff + pyrefly)"
	@echo "  make verify     pre-push gate: pre-commit + frontend lint/build; mirrors CI's lint gate"
	@echo "  make test       run the test suite"

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
