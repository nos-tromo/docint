#!/usr/bin/env bash
set -euo pipefail

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

# Build locally-defined services (backend + frontend).
$COMPOSE build

# Collect the built image names so docker save can bundle them. Every
# service in this compose file is locally built (backend + frontend);
# stateful/remote images (Qdrant) live in the data-plane project, not here,
# so there is nothing to pull.
built=()
while IFS= read -r img; do
  [[ -z "$img" ]] && continue
  built+=("$img")
done < <($COMPOSE config --images)

echo "Built images: ${built[*]:-<none>}"

if (( ${#built[@]} > 0 )); then
  docker save "${built[@]}" | gzip > "docint-built-${DOCINT_VERSION}.tar.gz"
fi

echo "Wrote: docint-built-${DOCINT_VERSION}.tar.gz"
