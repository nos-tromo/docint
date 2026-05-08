# Build-host helpers for docint.

.PHONY: volumes build-cpu build-cuda bundle-cpu bundle-cuda up-cpu up-cuda

# Versioned image tag: YYYY-MM-DD-<short-sha>. Override by exporting
# DOCINT_VERSION before invoking make. Mirrors scripts/bundle_images.sh.
DOCINT_VERSION ?= $(shell date +%Y-%m-%d)-$(shell git rev-parse --short HEAD)
export DOCINT_VERSION

# Create the external Docker volumes (one-time per host; idempotent).
volumes:
	./scripts/create_docker_volumes.sh

# Build the CPU profile (backend-cpu, frontend-cpu).
build-cpu:
	@echo "DOCINT_VERSION=$(DOCINT_VERSION)"
	DOCKER_BUILDKIT=1 docker compose --profile cpu build

# Build the CUDA profile (backend-cuda, frontend-cuda).
build-cuda:
	@echo "DOCINT_VERSION=$(DOCINT_VERSION)"
	DOCKER_BUILDKIT=1 docker compose --profile cuda build

# Build CPU stack and ship as versioned .tar.gz pair (built + pulled).
bundle-cpu:
	./scripts/bundle_images.sh cpu

# Build CUDA stack and ship as versioned .tar.gz pair (built + pulled).
bundle-cuda:
	./scripts/bundle_images.sh cuda

# Build and run the CPU profile (backend-cpu, frontend-cpu, qdrant-cpu).
up-cpu:
      @echo "DOCINT_VERSION=$(DOCINT_VERSION)"
	DOCKER_BUILDKIT=1 docker compose --profile cpu up

# Build and run the CUDA profile (backend-cuda, frontend-cuda, qdrant-cuda).
up-cuda:
      @echo "DOCINT_VERSION=$(DOCINT_VERSION)"
	DOCKER_BUILDKIT=1 docker compose --profile cuda up
