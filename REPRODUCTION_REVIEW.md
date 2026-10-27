# EarthBench-Fixed protocol

## Dataset and environment

GeoEvolve uses `haodijia/EarthBench-Fixed` at revision
`05525260e1e0f99d8e8ddd5e204cc2b975eb7c4a`. `benchmark/dataset_manifest.json` pins
13,699 release files by path, byte size, and SHA-256. The runtime uses 13,688 of
those files: 13,659 question data files, 26 masks, the question manifest, the
observation cache, and the reference trajectories.

Python dependencies are declared in `pyproject.toml` and pinned in `uv.lock`.
The default environment contains the agent, CPU tools, and evaluator. All raster
I/O uses Rasterio; its wheel bundles GDAL. There is no direct `osgeo` dependency.
HKG's PyTorch/Transformers environment and embedding checkpoint are installed
explicitly with the `hkg` extra.

## Data sources

The snapshot preserves the official questions and answers and supplies 461
precomputed perception observations. Six InstructSAM conflicts use the official
values. Cache assets are repository-relative and available in the published
snapshot. Its `provenance/` directory records merging and mask reconstruction.

Question148/149 use four annual night-light rasters synthesized from fixed
summary statistics. These are fixed-version benchmark inputs, not original
remote-sensing observations. Three question214 SAM2 masks were rebuilt with
SAM2.1 Hiera Small. Reports retain these source annotations and separately report
answer accuracy on the 246 released-imagery and two synthetic-raster questions.
All 248 questions remain in the primary denominator.

Perception is a fixed-observation environment. Unrecorded images, prompts, and
boxes fail lookup. Strip-R-CNN has no real observations in the snapshot. Runtime
requests never invoke replacement models. Every compared method uses the same
cache and masks.

## Reference protocol and scores

The environment exposes 110 tools: the original 104 plus six names already used
by official dialogs but absent from the public inventory. Repaired reference
trajectories contain 248 questions and 1,378 calls. Rebuilding them from the
question manifest reproduces the published fixed snapshot.

Accuracy uses the final canonical Answer tag and divides by all 248 questions;
missing, malformed, and execution-failed answers receive zero. Tool_Any_Order,
Tool_In_Order, Tool_Exact_Match, and Parameter preserve the documented reference
formulas. Storage-root normalization does not change filenames or numeric
parameters. Latest and cumulative call ratios are reported separately. See
[evaluation/README.md](evaluation/README.md) for formulas and token-cost boundaries.

Evaluation reports record the dataset revision and cache/reference checksums.
The pipeline recomputes by default. Explicit resume requires matching input,
evaluator, environment-lock, and output checksums. All earlier attempt files
needed for a requested round must be present.

## Validation

```bash
uv sync --locked
uv run --locked python benchmark/prepare.py
uv run --locked python benchmark/prepare.py --check
uv run --locked python tools/validate_mcp_tools.py --replay-ground-truth
```

Data verification checks the SHA-256 of every runtime asset. Full tool validation
checks all reference-call schemas, replays every question in its own output
workspace, invokes all 461 real cached observations, and exercises all 110 tool
interfaces on fixtures. JSON reports record individual outcomes.

The fixed snapshot has passed all 248 reference trajectories / 1,378 calls,
461 cache observations, and 110 tool checks after the Rasterio migration.
The same complete replay also passes in a newly created uv environment.
The deterministic scoring pipeline has also been checked with reference-format
inputs, explicit failures, missing questions, multiround retries, changed inputs,
and damaged cached reports. These checks verify execution and evaluator
behavior; they are not model experiment scores or independent scientific
validation of each reference output.
