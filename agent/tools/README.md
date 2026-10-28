# Earth-Bench pro max MCP tools

The benchmark exposes 110 local MCP tools across Analysis, Index, Inversion,
Perception, and Statistics. CPU raster operations use Rasterio. Perception reads
the fixed observation CSV and masks; benchmark runs require no GPU model service.

```bash
uv sync --locked
uv run --locked python -m benchmark.prepare
uv run --locked python -m scripts.validate_mcp_tools
uv run --locked python -m scripts.validate_mcp_tools --replay-ground-truth
```

The first validator command checks all 1,378 reference-call schemas and invokes
all 110 tools on generated fixtures, without requiring benchmark imagery. The
full replay additionally checks source-file flow, executes all 248 reference
trajectories, and invokes all 461 real cached observations. A missing asset,
invalid schema, or failed call makes validation fail.

Each invocation writes its outputs and `validation.json` into a new directory
under `tmp/mcp-validation/`. `--work-dir /path/to/new-directory` selects a new
location. It must not already exist. Replay input paths are relocated to this
isolated output directory; shared `tmp/out` is neither reused nor deleted.

MCP servers are configured in `agent/mcp.json`. The runner supplies each
question's `result/out/question<id>/` directory and starts Python servers with its own interpreter.
`concurrency` (or `--concurrency`) bounds simultaneous questions and defaults to 1.
Each question gets an independent agent and MCP configuration. After each attempt,
including failures, the runner saves its answer and trajectory under `result/`
and removes only that question's output directory. Later attempts recreate
required intermediate files.
Evaluation and reflection read the recorded tool results, not those temporary files.

## Observation cache

`benchmark/earthbench/perception/observations.csv` contains 461 observations: MSCN 172, RemoteCLIP 172,
SM3Det 57, InstructSAM 24, RemoteSAM 10, SAM2 9, and ChangeOS 17. The pinned dataset
contains all 26 referenced masks. `EARTH_BENCH_PERCEPTION_CACHE` explicitly selects a
different CSV; normal benchmark runs use the pinned cache.

Lookups require a recorded image, prompt, and bounding boxes. Unknown requests
raise an error. Strip-R-CNN has no observations in this version and cannot serve
real benchmark requests; its interface is exercised only with a validator
fixture. No replacement detector or runtime inference is attempted.

The released snapshot contains the fixed observation cache, masks, and their
provenance under its `provenance/` directory. Runtime setup uses these verified
assets directly.

## Reference-tool inventory

Official dialogs call six interfaces absent from the released 104-tool inventory:
`calculate_ndvi`, `calculate_ndwi`, `calculate_nbr`, `calculate_ndti`, `argmax`, and
`index_to_date_range`. This environment exposes all six for a total of 110 tools.
All compared methods use this same repaired inventory and reference trajectories.
