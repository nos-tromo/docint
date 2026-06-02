# Build-host helpers for docint.
#
# docint is a CPU-only Python app. All ML inference (chat, embeddings,
# rerank, NER, CLIP) is delegated over HTTP to the external vllm-service
# stack — see CLAUDE.md. There is no PROFILE / CUDA toggle here anymore.

.DEFAULT_GOAL := help

.PHONY: help network volumes build bundle up up-dev stop down resolve pre-commit test

# Versioned image tag.
# On production: read from .docint-version written by bundle_images.sh.
# On dev: compute YYYY-MM-DD[-<short-sha>] on the fly.
# Override entirely by exporting DOCINT_VERSION before invoking make.
DOCINT_VERSION ?= $(shell \
    cat .docint-version 2>/dev/null || \
    { _s=$$(git rev-parse --short HEAD 2>/dev/null); \
      echo "$$(date +%Y-%m-%d)$${_s:+-$$_s}"; } )
export DOCINT_VERSION

COMPOSE     := docker compose --env-file .env -f docker/compose.yaml
COMPOSE_DEV := docker compose --env-file .env -f docker/compose.yaml -f docker/compose.override.yaml

help:
	@echo "docint — build-host helpers."
	@echo
	@echo "  make network    create the external inference-net + data-net"
	@echo "  make volumes    create the external Docker volumes"
	@echo "  make build      build images"
	@echo "  make bundle     ship images as a versioned .tar.gz pair"
	@echo "  make up         build + run docint (production shape, no host ports)"
	@echo "  make up-dev     like 'up', but publishes the frontend port on the host"
	@echo "  make stop       stop docint containers"
	@echo "  make down       stop + remove containers (never touches data-plane state)"
	@echo "  make resolve    merge duplicate/similar entities (COLLECTION=<name> optional)"
	@echo "  make pre-commit run pre-commit hooks (ruff + mypy)"
	@echo "  make test       run the test suite"

# Create the external Docker networks (one-time per host; idempotent).
network:
	docker network create inference-net >/dev/null 2>&1 || true
	docker network create data-net >/dev/null 2>&1 || true

# Create the external Docker volumes (one-time per host; idempotent).
volumes:
	./scripts/create_docker_volumes.sh

# Build images.
build:
	DOCKER_BUILDKIT=1 $(COMPOSE) build

# Build images and ship as a versioned .tar.gz pair (built + pulled).
bundle:
	./scripts/bundle_images.sh

# Build and run docint in production shape (no host ports).
up:
	DOCKER_BUILDKIT=1 $(COMPOSE) up

# Like 'up' but layers compose.override.yaml on top to publish the
# frontend (React SPA) port on the host.
up-dev:
	DOCKER_BUILDKIT=1 $(COMPOSE_DEV) up

# Stop docint containers.
stop:
	$(COMPOSE) stop

# Stop + remove containers. All docint volumes (sessions-storage, caches)
# are declared external, and Qdrant lives in data-plane, so this is safe —
# `down` never destroys session state or vector data.
down:
	$(COMPOSE) down

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

# Run pre-commit hooks (ruff + mypy).
pre-commit:
	uv run pre-commit run --all-files

# Run the test suite.
test:
	uv run pytest -q
