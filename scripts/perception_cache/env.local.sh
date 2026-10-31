#!/usr/bin/env bash
export UV_PROJECT_ENVIRONMENT=/home/dijkstra/model/.venv-earthbench
export TMPDIR=/home/dijkstra/data/GeoEvolve/cache-repair/tmp
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=4
export PYTHONUNBUFFERED=1
export SAM2_BUILD_CUDA=0
export NO_PROXY=127.0.0.1,localhost
export no_proxy=127.0.0.1,localhost
mkdir -p "$TMPDIR"
