#!/usr/bin/env bash
set -euo pipefail
cd /home/zhangjai/whu/7_jia/04-agent/GeoEvolve
source ./scripts/env.a800.sh
exec bash scripts/run.sh "$@"
