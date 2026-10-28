# EarthBench-Fixed protocol

## Dataset and environment

Earth-Bench pro max uses `haodijia/EarthBench-Fixed`, release `2026-10-28`.
`benchmark/earthbench.lock.json` pins the release manifest's SHA-256;
`benchmark/earthbench/manifest.json` pins every other release file by relative
path, byte size, and SHA-256. Inputs include 13,658 question data files, 26 masks,
248 questions, 461 fixed observations, and 1,378 reference tool calls.
The data-only release lives in `benchmark/earthbench/`; Python loaders,
downloading, and reference reconstruction live directly in `benchmark/`.

Python dependencies are declared in `pyproject.toml` and pinned in `uv.lock`.
The default environment contains the agent, CPU tools, and evaluator. All raster
I/O uses Rasterio; its wheel bundles GDAL. There is no direct `osgeo` dependency.
HKG's PyTorch/Transformers environment and embedding checkpoint are installed
explicitly with the `hkg` extra.

## Data sources

The release preserves the source question content and answers and supplies 461
precomputed perception observations. Six InstructSAM conflicts use the official
values. Cache assets use dataset-relative paths under `perception/`. Its `provenance/` directory records merging and mask reconstruction.

Question148/149 use four annual night-light rasters synthesized from fixed
summary statistics. These are fixed-version benchmark inputs, not original
remote-sensing observations. Three question214 SAM2 masks were rebuilt with
SAM2.1 Hiera Small. Reports retain these source annotations and separately report
answer accuracy on the 246 released-imagery and two synthetic-raster questions.
Evaluation uses the distinct questions executed in the selected run. The complete
dataset contains 248 questions; a sampled run uses only its executed subset.

Perception is a fixed-observation environment. Unrecorded images, prompts, and
boxes fail lookup. Strip-R-CNN has no real observations in the snapshot. Runtime
requests never invoke replacement models. Every compared method uses the same
cache and masks.

## Reference protocol and scores

The environment exposes 110 tools: the original 104 plus six names already used
by official dialogs but absent from the public inventory. Repaired reference
trajectories contain 248 questions and 1,378 calls. Rebuilding them from the
archived source question manifest reproduces the normalized reference file.

Accuracy uses the final canonical Answer tag and divides by the number of distinct
questions in the run's final result package. Reference answers and trajectories are
restricted to this same question set, which is checked against the saved dataset
snapshot. Missing, malformed, and execution-failed answers within that set receive
zero. Methods with internal iterations submit one final answer per question, using
this same denominator. Tool_Any_Order,
Tool_In_Order, Tool_Exact_Match, and Parameter preserve the documented reference
formulas. Storage-root normalization does not change filenames or numeric
parameters. Latest and cumulative call ratios are reported separately. See
[README.md](README.md) for formulas and token-cost boundaries.

Evaluation reports record the dataset release version and manifest/cache/reference checksums.
The pipeline recomputes by default. Explicit resume requires matching input,
evaluator, environment-lock, and output checksums. Every method submits the same final-result contract. Evaluation does not read
internal attempt files or select rounds. Raw execution records live in each run's `result/` directory. Reports are written
to the sibling `evaluation/` directory.

## Validation

```bash
uv sync --locked
uv run --locked python -m benchmark.prepare
uv run --locked python -m benchmark.prepare --check
uv run --locked python -m scripts.validate_mcp_tools --replay-ground-truth
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
