# Build-host helpers for docint.
#
# docint is a CPU-only Python app. All ML inference (chat, embeddings,
# rerank, NER, CLIP) is delegated over HTTP to the external vllm-service
# stack — see CLAUDE.md. There is no PROFILE / CUDA toggle here anymore.

.DEFAULT_GOAL := help

.PHONY: help network volumes build bundle up stop pre-commit test

# Versioned image tag.
# On production: read from .docint-version written by bundle_images.sh.
# On dev: compute YYYY-MM-DD[-<short-sha>] on the fly.
# Override entirely by exporting DOCINT_VERSION before invoking make.
DOCINT_VERSION ?= $(shell \
    cat .docint-version 2>/dev/null || \
    { _s=$$(git rev-parse --short HEAD 2>/dev/null); \
      echo "$$(date +%Y-%m-%d)$${_s:+-$$_s}"; } )
export DOCINT_VERSION

COMPOSE := docker compose --env-file .env -f docker/compose.yaml -f docker/compose.override.yaml

help:
	@echo "docint — build-host helpers."
	@echo
	@echo "  make network    create the external inference-net + data-net"
	@echo "  make volumes    create the external Docker volumes"
	@echo "  make build      build images"
	@echo "  make bundle     ship images as a versioned .tar.gz pair"
	@echo "  make up         build + run docint"
	@echo "  make stop       stop docint containers"
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

# Build and run docint.
up:
	DOCKER_BUILDKIT=1 $(COMPOSE) up

# Stop docint containers.
stop:
	$(COMPOSE) stop

# Run pre-commit hooks (ruff + mypy).
pre-commit:
	uv run pre-commit run --all-files

# Run the test suite.
test:
	uv run pytest -q
