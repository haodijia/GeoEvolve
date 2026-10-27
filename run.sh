#!/usr/bin/env bash
# Run one GeoEvolve experiment from the repository root.
#
# Usage:
#   bash run.sh
#   bash run.sh configs/qwen3_8b_baseline.yaml
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

CONFIG_PATH="${1:-configs/deepseek_v32_rsmem.yaml}"

exec uv run --locked --extra hkg geoevolve run --config "$CONFIG_PATH"
