# Earth-Bench pro max MCP tools

The benchmark exposes the official 104 local MCP tools across Analysis, Index, Inversion,
Perception, and Statistics. CPU raster operations use Rasterio. Perception reads
the released, consolidated CSV and masks. Evaluation uses only offline
observations and requires no GPU model service. See [cache repair](../../docs/perception-cache-repair.md).

```bash
uv sync --locked
uv run --locked python -m benchmark.prepare
uv run --locked python -m scripts.validate_mcp_tools
uv run --locked python -m scripts.validate_mcp_tools --replay-ground-truth
uv run --locked python -m scripts.validate_mcp_tools --replay-ground-truth --question-ids 69 70 184
```

The first validator command checks all published reference-call schemas and invokes
all 104 tools on generated fixtures, without requiring benchmark imagery. The
full replay additionally checks source-file flow, executes all 248 reference
trajectories, and invokes all 569 real cached observations. A missing asset,
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

The eight model tools have distinct contracts:

| Tool | Task and required input |
| --- | --- |
| MSCN | Whole-image scene classification using AID's 30 categories; no object counts or locations. |
| RemoteCLIP | Whole-image scene classification using a separate 19-category taxonomy. Its labels are not interchangeable with MSCN's. |
| Strip-R-CNN | Ships and vessel types only. Other object categories use SM3Det. |
| SM3Det | All matching object boxes in original-image pixels, retaining the prompt's conditions. |
| RemoteSAM | One region described by the complete referring expression. Preserve all target conditions; multiple unresolved candidates are an error. |
| InstructSAM | Integer count of all objects satisfying the complete requested conditions. |
| SAM2 | Union foreground mask from explicit pixel-coordinate boxes; supplied boxes determine the region. |
| ChangeOS | Damaged-building mask from an aligned, ordered image pair; identical inputs produce building footprints. |

Image paths, prompts and output paths must be nonblank. SAM2 requires at least
one box, with exactly four numeric coordinates per box. Tool schemas describe
these parameters without exposing per-question cache contents or vocabulary.
No cache miss is interpreted as zero objects. RemoteSAM preserves referring
conditions and never selects an arbitrary box from ambiguous observations.

`benchmark/earthbench/perception/observations.csv` contains 569 frozen observations, including reviewed supplements. Referenced masks ship in the same dataset. `EARTH_BENCH_PERCEPTION_CACHE` selects another CSV; normal evaluation uses the pinned release. External patch loading and service calls have been removed. Unknown requests remain errors. See the repair document for model provenance and review boundaries.

The released snapshot contains the fixed observation cache, masks, and their
provenance under its `provenance/` directory. Runtime setup uses these verified
assets directly. On cache load, inputs are compared byte for byte (without file
hashes); existing observations are expanded to identical images under other
recorded paths. The pinned snapshot gains seven exact query combinations this
way. Model, prompt, bounding boxes, and input order are preserved; recorded keys
are never overwritten. Tool descriptions explain the task, required conditions,
parameters and return values without exposing per-question cached vocabulary.

Numeric and array parameters have explicit JSON schema types. Output paths may
be relative to the question workspace or absolute within it; paths outside the
workspace are rejected. `calc_threshold_value_mean` pairs lists by acquisition
date (`YYYY-MM-DD` or `YYYY_MM_DD`, optionally with `HHMM`), requires a unique
target for every condition date, and reports invalid pairs or no valid pixels
as errors. Its pixel-weighted mean excludes nodata/nonfinite pixels and accepts
targets outside 0..1, including LST values.

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
