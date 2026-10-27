# HKG embedding model

Baseline and FAR require no local model weights. Perception uses the fixed
observation cache and masks included with EarthBench-Fixed.

HKG uses `sentence-transformers/all-mpnet-base-v2`. Install its Python dependencies
and download the pinned checkpoint:

```bash
uv sync --locked --extra hkg
uv run --locked --extra hkg hf download sentence-transformers/all-mpnet-base-v2 \
  --revision e8c3b32edf5434bc2275fc9bab85f82640a19130 \
  --local-dir models/all-mpnet-base-v2
```

Run HKG/RSMeM with `uv run --locked --extra hkg geoevolve run --config <CONFIG>`.
The default configuration reads `models/all-mpnet-base-v2`. Weights are not stored
in Git. Use the same checkpoint for every HKG comparison.
