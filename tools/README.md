# GeoEvolve MCP Tools

All remote-sensing tools run locally during a benchmark run.

```
tools/
├── Analysis.py, Index.py, Inversion.py, Statistics.py   # CPU tools
├── Perception.py                                        # Offline cache tools
├── build_perception_cache.py
├── generate_perception_artifacts.py
└── requirements-tools.txt
```

## Install and validate

```bash
cd GeoEvolve
uv venv --python 3.10
source .venv/bin/activate
uv pip install -r requirements.txt -r tools/requirements-tools.txt
uv run --no-sync python tools/validate_mcp_tools.py
```

The validator starts each MCP server once, validates every official EarthBench
ground-truth call against the published tool schemas, and then invokes every
registered tool. Its runtime fixtures exercise both batch raster inputs and
null results from rasters with no valid pixels.

Raster tools whose official trajectories pass aligned path lists process those
lists as one bounded-parallel MCP call and return results in input order. The
ground truth is not expanded into separate calls.

To execute all 248 trajectories and every intermediate raster operation, use:

```bash
uv run --no-sync python tools/validate_mcp_tools.py --replay-ground-truth
```

The replay starts the five MCP servers once, runs the 1,378 ground-truth calls
in question order, and fails on missing sources, unresolved outputs, schema
errors, or tool runtime errors.

MCP tool servers are configured in [`configs/agent_mcp.json`](../configs/agent_mcp.json); YAML configs point to it via `agent_config_path`.

## Tool inventory repair

The original Earth-Agent and RSMeM MCP servers register 104 tools, but official dialogs call six additional names 45 times across 23 questions. GeoEvolve exposes those names, for 110 registered tools: `calculate_ndvi`, `calculate_ndwi`, `calculate_nbr`, `calculate_ndti`, `argmax`, and `index_to_date_range`. The four index calculations preserve the original internal implementations; the other two reproduce the official example outputs. These interfaces repair the mismatch between the released dialogs and MCP tool inventory. Keep this same repaired inventory for all methods; see [REPRODUCTION_REVIEW.md](../REPRODUCTION_REVIEW.md).

## Offline perception cache

This evaluation follows official EarthBench's precomputed-observation approach. Upstream RSMeM's GPU HTTP services are not used or required. `Perception.py` performs strict lookups in `benchmark/model_results.csv`. No model inference or alternative tool is attempted on a cache miss. This is a fixed-observation environment, not a reproduction of all possible online perception requests.

The merged cache contains 461 observations: MSCN 172, RemoteCLIP 172, SM3Det 57, InstructSAM 24, RemoteSAM 10, SAM2 9, and ChangeOS 17. All 461 keys were invoked successfully through MCP during the local audit. Strip-R-CNN has no observations or local checkpoint; it remains unavailable in practice. Unseen prompts, boxes, or images also require observations before they can run.

Source priority is explicit: valid official Earth-Agent observations, then GeoEvolve observations for missing entries, then locally rebuilt SAM2 observations for missing entries. Six InstructSAM numerical conflicts now use the official values; all conflicts and exclusions are recorded in provenance. The merge reads observation CSVs, never GT answers or GT-derived fabricated tool results.

Local source assets and provenance are in `/home/dijkstra/data/earth_bench_perception/`. Reproduce the merge with:

```bash
uv run --no-sync python tools/build_perception_cache.py \
  --sources /home/dijkstra/data/earth_bench_perception/official_model_results.csv \
    /home/dijkstra/data/earth_bench_perception/geoevolve_observations.csv \
    /home/dijkstra/data/earth_bench_perception/sam2_observations.csv \
  --output benchmark/model_results.csv \
  --provenance /home/dijkstra/data/earth_bench_perception/merge_provenance.json
```

The 172 RemoteCLIP observations come directly from the official CSV. Upstream specifies ViT-L/14; the local RemoteCLIP checkpoint is ViT-B/32, so it was not substituted. Three missing SAM2 masks for question214 A/B/C were rebuilt using local SAM2.1 Hiera Small weights and the official image/box inputs. Their absolute mask paths point into `/home/dijkstra/data/earth_bench_perception/sam2/`; copy the masks and update the paths when moving machines. Other SAM2 and ChangeOS masks are under `benchmark/model_out/`.

`rebuild_sam2_cache.py --help` documents the explicit observation, current-cache, input/output-root, checkpoint, CSV, and provenance arguments. The model environment used locally is `/home/dijkstra/model/.venv-earthbench`. `generate_perception_artifacts.py` supplies its SAM2 inference routine and the earlier mask generation utilities. Normal benchmark runs only read the generated masks.

Set `RSMEM_PERCEPTION_CACHE` to explicitly select a different observation CSV. For fair method comparisons, use exactly the same cache and mask assets for every method, and disclose synthesized/rebuilt observations.
