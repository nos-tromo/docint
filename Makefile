# Build-host helpers for docint.

.PHONY: volumes bundle bundle-cuda build build-cuda up up-cuda stop

# Versioned image tag: YYYY-MM-DD-<short-sha>. Override by exporting
# DOCINT_VERSION before invoking make. Mirrors scripts/bundle_images.sh.
DOCINT_VERSION ?= $(shell date +%Y-%m-%d)-$(shell git rev-parse --short HEAD)
export DOCINT_VERSION

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