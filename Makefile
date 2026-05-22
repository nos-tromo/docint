# Build-host helpers for docint.

.PHONY: network volumes bundle bundle-cuda build build-cuda up up-cuda stop pre-commit test

# Versioned image tag.
# On production: read from .docint-version written by bundle_images.sh.
# On dev: compute YYYY-MM-DD[-<short-sha>] on the fly.
# Override entirely by exporting DOCINT_VERSION before invoking make.
DOCINT_VERSION ?= $(shell \
    cat .docint-version 2>/dev/null || \
    { _s=$$(git rev-parse --short HEAD 2>/dev/null); \
      echo "$$(date +%Y-%m-%d)$${_s:+-$$_s}"; } )
export DOCINT_VERSION

# Create the external Docker network (one-time per host; idempotent)
network:
	DOCKER_BUILDKIT=1 docker network create inference-net

# Create the external Docker volumes (one-time per host; idempotent).
volumes:
	./scripts/create_docker_volumes.sh

# Build CPU stack and ship as versioned .tar.gz pair (built + pulled).
bundle:
	./scripts/bundle_images.sh cpu

# Build CUDA stack and ship as versioned .tar.gz pair (built + pulled).
bundle-cuda:
	./scripts/bundle_images.sh cuda

# Build the CPU profile (backend-cpu, frontend-cpu).
build:
	DOCKER_BUILDKIT=1 docker compose --profile cpu build

# Build the CUDA profile (backend-cuda, frontend-cuda).
build-cuda:
	DOCKER_BUILDKIT=1 docker compose --profile cuda build

# Build and run the CPU profile (backend-cpu, frontend-cpu, qdrant-cpu).
up:
	DOCKER_BUILDKIT=1 docker compose --profile cpu up

# Build and run the CUDA profile (backend-cuda, frontend-cuda, qdrant-cuda).
up-cuda:
	DOCKER_BUILDKIT=1 docker compose --profile cuda up

# Stop the CPU profile containers.
stop:
	docker compose --profile cpu stop

# Stop the CUDA profile containers.
stop-cuda:
	docker compose --profile cuda stop

# Pre-commit checks: lint, typecheck, test.
pre-commit:
	uv run pre-commit run --all-files

# Run the test suite (requires build).
test:
	uv run pytest
