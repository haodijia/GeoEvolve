# EarthBench-Fixed protocol

## Dataset and environment

The code-side `benchmark/earthbench.lock.json` pins a ModelScope commit and release version.
The current release is `2026-10-31-contracts-v3`: 248 questions, 104 public tools and 569 frozen perception observations.
The data-only release contains questions, actual reference inputs/outputs, rasters, cache masks and provenance.
Manifest schema 2 records relative paths and byte sizes; download and offline verification use that inventory.
Python dependencies use the project's `pyproject.toml`, `uv.lock` and `uv` environment.

## Repair scope and sources

Question `auto`/`instruct` strings and public tool arguments, annotations and descriptions remain unchanged.
Repairs address tool implementation, fixed observations, reference dependency/path wiring and answers inconsistent with actual tool outputs.
They do not repair evaluated agents' planning, arguments, reasoning or historical scores.

Q148/149 retain the earlier synthetic-input declaration. Reviewed supplemental perceptions retain actual backend and review provenance.
Frozen observations are not a claim that every image was manually annotated. Unknown requests fail explicitly; runtime makes no service calls.
All methods must use the same pinned release.

## Reference and scoring

Published reference calls contain actual tool outputs. `provenance/reference-plan.json` records their dependencies;
`provenance/answer-review.json` records numeric conversions, ranks and other answer evidence. These assets do not enter agent prompts.
Previous questions, references and observations are archived in the release for audit.

Accuracy divides correct final answers by the distinct question set in the run. Missing, malformed and execution-failed answers score zero.
A sampled run uses only its selected questions, not a fixed denominator of 248.
Tool_Any_Order, Tool_In_Order, Tool_Exact_Match, Parameter and token-cost boundaries remain documented in README.md.
Reference repairs do not retroactively rewrite model results. New-release comparisons require using the new release consistently.

Existing runtime/evaluation provenance digests are retained. No new file-hash manifest is added.

## Validation

```bash
uv sync --locked
uv run --locked python -m benchmark.prepare
uv run --locked python -m benchmark.prepare --check
uv run --locked python scripts/validate_mcp_tools.py --replay-ground-truth
```

Full validation checks schema and file flow, executes all 248 reference chains through MCP,
compares each return value with the published reference, validates answer evidence,
invokes every frozen observation and exercises all 104 tool interfaces using the existing fixtures.
This verifies benchmark execution consistency, not model performance or independent scientific validity of every source observation.
