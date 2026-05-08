#!/usr/bin/env bash
# Download Python wheels for the frontend dependency group for offline/airgapped
# Docker builds.
#
# Run this script on an internet-connected machine that has Docker available.
# It launches the exact Python base image used in Dockerfile.frontend so the
# downloaded wheels are guaranteed to match the build target's platform and
# Python version.
#
# Usage:
#   bash scripts/download-frontend-wheels.sh
#
# After the script finishes:
#   1.  tar czf wheels.tar.gz wheels/
#   2.  Transfer wheels.tar.gz alongside the git bundle to the airgapped server.
#   3.  On the airgapped server: tar xzf wheels.tar.gz
#   4.  OFFLINE_BUILD=1 docker compose up --build --pull never

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
WHEELS_DIR="$REPO_DIR/wheels"

# Must match ARG PYTHON_SLIM_BOOKWORM_IMAGE in Dockerfile.frontend
BASE_IMAGE="${BASE_IMAGE:-python:3.11.12-slim-bookworm}"

mkdir -p "$WHEELS_DIR"

echo "Exporting frontend group requirements from uv.lock..."
REQUIREMENTS_FILE="$(mktemp /tmp/requirements-frontend-XXXXXX.txt)"
trap 'rm -f "$REQUIREMENTS_FILE"' EXIT

cd "$REPO_DIR"
uv export --only-group frontend --no-hashes --no-emit-project -o "$REQUIREMENTS_FILE"

echo "Downloading wheels inside $BASE_IMAGE..."
docker run --rm \
    -v "$WHEELS_DIR:/wheels" \
    -v "$REQUIREMENTS_FILE:/requirements.txt:ro" \
    "$BASE_IMAGE" \
    bash -c "
        set -e
        python3 -m ensurepip --upgrade 2>/dev/null || true
        python3 -m pip install --quiet --upgrade pip
        python3 -m pip download \
            --no-deps \
            -r /requirements.txt \
            -d /wheels
    "

COUNT=$(find "$WHEELS_DIR" -name "*.whl" | wc -l)
echo ""
echo "Downloaded $COUNT wheel(s) to $WHEELS_DIR/"
echo ""
echo "Next steps:"
echo "  1.  tar czf wheels.tar.gz wheels/"
echo "  2.  Transfer wheels.tar.gz alongside the git bundle to the airgapped server."
echo "  3.  On the airgapped server: tar xzf wheels.tar.gz"
echo "  4.  OFFLINE_BUILD=1 docker compose up --build --pull never"
