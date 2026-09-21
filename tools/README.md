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
uv run python tools/validate_mcp_tools.py
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
uv run python tools/validate_mcp_tools.py --replay-ground-truth
```

The replay starts the five MCP servers once, runs the 1,378 ground-truth calls
in question order, and fails on missing sources, unresolved outputs, schema
errors, or tool runtime errors.

MCP tool servers are configured in [`configs/agent_mcp.json`](../configs/agent_mcp.json); YAML configs point to it via `agent_config_path`.

## Offline perception cache

`Perception.py` performs strict lookups against
[`benchmark/model_results.csv`](../benchmark/model_results.csv). SAM2 and
ChangeOS rows reference checked-in masks under `benchmark/model_out/`. Set
`RSMEM_PERCEPTION_CACHE` only when intentionally validating another cache.

The cache contains all 219 perception calls referenced by EarthBench ground
truth. `build_perception_cache.py` rebuilds the CSV from that ground truth.
`generate_perception_artifacts.py` is the one-time GPU-host utility used to
regenerate the six SAM2 and seventeen ChangeOS masks; it is not part of normal
benchmark execution.
