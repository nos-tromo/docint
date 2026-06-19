#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
. scripts/bundle-lib.sh

COMPOSE=(docker compose --env-file .env -f docker/compose.yaml)
bundle_version docint; VER="$BUNDLE_VERSION"

"${COMPOSE[@]}" build
bundle_partition_images < <("${COMPOSE[@]}" config --images)

echo "Built images: ${BUNDLE_BUILT[*]:-<none>}"
if (( ${#BUNDLE_BUILT[@]} > 0 )); then
  docker save "${BUNDLE_BUILT[@]}" | gzip > "docint-built-${VER}.tar.gz"
fi
echo "Wrote: docint-built-${VER}.tar.gz"
