# Earth-Bench pro max MCP tools

The benchmark exposes the official 104 local MCP tools across Analysis, Index, Inversion,
Perception, and Statistics. CPU raster operations use Rasterio. Perception reads
the fixed observation CSV and masks; benchmark runs require no GPU model service.

```bash
uv sync --locked
uv run --locked python -m benchmark.prepare
uv run --locked python -m scripts.validate_mcp_tools
uv run --locked python -m scripts.validate_mcp_tools --replay-ground-truth
```

The first validator command checks all 1,331 reference-call schemas and invokes
all 104 tools on generated fixtures, without requiring benchmark imagery. The
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

The exposed tool names match the official Earth-Agent inventory at commit
`ca1dfaef750654331308a434966d307f1728fd90`: Analysis 10, Index 12,
Inversion 17, Perception 15, and Statistics 50.

Source dialogs contain six names outside that inventory. Published reference
trajectories use `calculate_batch_ndvi`, `calculate_batch_ndwi`,
`calculate_batch_nbr`, and `calculate_batch_ndti` with list inputs in place of
the single-image calls. Consecutive independent calls are combined into a batch.
`argmax` is represented by `max_value_and_index`, and date-range selection uses
`get_list_object_via_indexes`. The source dialogs remain archived under dataset
provenance; no additional tool interfaces are exposed.
