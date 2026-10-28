#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
EXTRAS="$(uv run --locked python -m scripts.run "$@" --print-extras)"
UV_ARGS=(run --locked)
if [[ "$EXTRAS" == "hkg" ]]; then
    UV_ARGS+=(--extra hkg)
fi
exec uv "${UV_ARGS[@]}" earth-bench-pro-max "$@"
