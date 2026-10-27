# EarthBench-Fixed data

The benchmark uses [haodijia/EarthBench-Fixed](https://modelscope.cn/datasets/haodijia/EarthBench-Fixed)
at revision `05525260e1e0f99d8e8ddd5e204cc2b975eb7c4a`. The version and SHA-256 of all
13,699 published files are pinned in `dataset_manifest.json`.

```bash
uv sync --locked
uv run --locked python benchmark/prepare.py
uv run --locked python benchmark/prepare.py --check
```

The download is about 16.6 GB. Its default destination is
`benchmark/.earthbench/`; use `--snapshot-dir /path/to/EarthBench-Fixed` to choose
another disk. The script links `benchmark/data` to the verified snapshot and
installs the question manifest and masks when absent. Existing tracked assets
must match the pinned snapshot. No API key is required.

To reuse downloaded data without modifying the source:

```bash
uv run --locked python benchmark/prepare.py --reuse-data /path/to/existing/question-folders
```

Matching files are copied after checksum verification. Missing or changed files
are downloaded from the pinned revision. Partial downloads use `.part` files and
are activated only after checksum verification. Rerunning the command reuses
already verified snapshot files.

`--check` performs no downloads and verifies all 13,688 runtime assets: 13,659 data
files, 26 masks, the question manifest, the observation CSV, and the reference
trajectories. Its JSON report defaults to `tmp/data-validation.json`. The full
snapshot also preserves the dataset license, source notices, and provenance.

The fixed snapshot covers all 248 questions and 461 perception observations.
Question148/149 use four synthetic annual night-light rasters with fixed summary
statistics. Those files are part of this version's inputs; their origin is
recorded in the manifest and evaluation reports. Three question214 SAM2 masks
were rebuilt with SAM2.1 Hiera Small. Runtime mask paths are relative to the
repository, so no paths from the release author's machine are needed.

All methods must use this same snapshot. Input data are shared read-only;
each experiment writes its own `<experiment_dir>/out`. The validator also uses
an isolated output directory for each invocation.
