#!/usr/bin/env bash
set -euo pipefail

PROFILE="${1:-cpu}"
COMPOSE="docker compose --env-file .env -f docker/compose.yaml"

# Always compute a fresh version from git so repeated bundle runs produce
# distinct tags. Uses the commit date (not the build date) for reproducibility.
# Falls back to today's date when not in a git repo.
# .docint-version (if present) is never used as input here — it is only
# written as output for production hosts.
# To pin a specific tag, set DOCINT_VERSION_OVERRIDE in your shell before
# invoking make.
if [[ -n "${DOCINT_VERSION_OVERRIDE:-}" ]]; then
  export DOCINT_VERSION="$DOCINT_VERSION_OVERRIDE"
else
  _git_sha=$(git rev-parse --short HEAD 2>/dev/null || true)
  _git_date=$(git log -1 --format=%cs 2>/dev/null || true)
  _date="${_git_date:-$(date +%Y-%m-%d)}"
  export DOCINT_VERSION="${_date}${_git_sha:+-${_git_sha}}"
fi
echo "DOCINT_VERSION=$DOCINT_VERSION"

# Persist the version so production hosts can run 'make no-build-*' without
# git or the original build date. Copy this file alongside docker/compose.yaml.
echo "$DOCINT_VERSION" > .docint-version

# Build locally-defined services (frontend + backend for the chosen profile)
$COMPOSE --profile "$PROFILE" build

# Pull externally hosted services (any image:-only services)
$COMPOSE --profile "$PROFILE" pull --ignore-buildable

# Partition compose's image list and ensure local tag bindings exist:
#   built  = local-only names like "docint-backend-cpu" (already tagged by build)
#   pulled = registry refs like "docker.io/qdrant/qdrant:v1.17.0@sha256:..."
#
# Docker Desktop sometimes drops the name:tag binding when you pull
# `name:tag@digest`, leaving only the digest. We re-tag explicitly so
# `docker save` produces a tarball that loads back with both tag and digest
# bindings — which compose needs for its `image: name:tag@digest` references.
built=()
pulled=()
while IFS= read -r img; do
  [[ -z "$img" ]] && continue
  if [[ "$img" == */* ]]; then
    if [[ "$img" =~ ^(.+):([^@]+)@(sha256:[a-f0-9]+)$ ]]; then
      name="${BASH_REMATCH[1]}"
      tag="${BASH_REMATCH[2]}"
      digest="${BASH_REMATCH[3]}"
      docker tag "${name}@${digest}" "${name}:${tag}"
      pulled+=("${name}:${tag}")
    else
      pulled+=("$img")
    fi
  else
    built+=("$img")
  fi
done < <($COMPOSE --profile "$PROFILE" config --images)

echo "Built images:  ${built[*]:-<none>}"
echo "Pulled images: ${pulled[*]:-<none>}"

if (( ${#built[@]} > 0 )); then
  docker save "${built[@]}" | gzip > "docint-built-${PROFILE}-${DOCINT_VERSION}.tar.gz"
fi

if (( ${#pulled[@]} > 0 )); then
  docker save "${pulled[@]}" | gzip > "docint-pulled-${PROFILE}-${DOCINT_VERSION}.tar.gz"
fi

echo "Wrote: docint-built-${PROFILE}-${DOCINT_VERSION}.tar.gz, docint-pulled-${PROFILE}-${DOCINT_VERSION}.tar.gz"
