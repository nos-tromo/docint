# Build-host helpers for docint.
#
# The Docker profile (cpu/cuda) is read from PROFILE in .env, so plain
# `make up` follows the host's hardware. Override per-invocation with
# `make up PROFILE=cuda`.

.DEFAULT_GOAL := help

.PHONY: help network volumes bundle build up stop pre-commit test

# Docker profile (cpu/cuda). Read from .env, default cpu. Override on the
# command line: make up PROFILE=cuda
PROFILE ?= $(or $(strip $(shell test -f .env && grep -E '^PROFILE=' .env | cut -d= -f2)),cpu)

# Versioned image tag.
# On production: read from .docint-version written by bundle_images.sh.
# On dev: compute YYYY-MM-DD[-<short-sha>] on the fly.
# Override entirely by exporting DOCINT_VERSION before invoking make.
DOCINT_VERSION ?= $(shell \
    cat .docint-version 2>/dev/null || \
    { _s=$$(git rev-parse --short HEAD 2>/dev/null); \
      echo "$$(date +%Y-%m-%d)$${_s:+-$$_s}"; } )
export DOCINT_VERSION

COMPOSE      := docker compose --env-file .env -f docker/compose.yaml -f docker/compose.override.yaml
PROFILE_FLAG := --profile $(PROFILE)

help:
	@echo "docint — build-host helpers. Active profile: $(PROFILE)"
	@echo
	@echo "  make network   	create the external inference-net + data-net"
	@echo "  make volumes   	create the external Docker volumes"
	@echo "  make build     	build images for the $(PROFILE) profile"
	@echo "  make up        	build + run the $(PROFILE) profile"
	@echo "  make stop      	stop the $(PROFILE) profile containers"
	@echo "  make bundle    	ship images as a versioned .tar.gz pair ($(PROFILE))"
	@echo "  make pre-commit 	run pre-commit checks"
	@echo "  make test      	run tests"
	@echo
	@echo "Set PROFILE=cpu|cuda in .env, or override: make up PROFILE=cuda"

# Create the external Docker networks (one-time per host; idempotent).
network:
	docker network create inference-net >/dev/null 2>&1 || true
	docker network create data-net >/dev/null 2>&1 || true

# Create the external Docker volumes (one-time per host; idempotent).
volumes:
	./scripts/create_docker_volumes.sh

# Build the stack and ship as a versioned .tar.gz pair (built + pulled).
bundle:
	./scripts/bundle_images.sh $(PROFILE)

# Build images for the active profile.
build:
	DOCKER_BUILDKIT=1 $(COMPOSE) $(PROFILE_FLAG) build

# Build and run the active profile.
up:
	DOCKER_BUILDKIT=1 $(COMPOSE) $(PROFILE_FLAG) up

# Stop the active profile's containers.
stop:
	$(COMPOSE) $(PROFILE_FLAG) stop

pre-commit:
	uv run pre-commit run --all-files

test:
	uv run pytest -q
