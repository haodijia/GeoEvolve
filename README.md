# GeoEvolve

GeoEvolve is an evolving framework for building reliable, tool-using remote-sensing agents. It combines domain knowledge, execution traces, and failure-driven refinement to improve multi-step geospatial workflows.

This repository is independently maintained and under active development. Its architecture, interfaces, and documentation will continue to change as the project evolves.

## Features

- Knowledge-guided planning for remote-sensing tasks
- Local MCP tools for perception, analysis, indexing, inversion, and statistics
- Failure-aware refinement across repeated execution attempts
- Deterministic EarthBench evaluation and token accounting
- YAML-based experiment configuration with resumable runs

## Installation

Use Python 3.10–3.12 and uv. The repository selects Python 3.10 by default.

```bash
git clone https://github.com/haodijia/GeoEvolve.git
cd GeoEvolve
uv sync --locked
uv run --locked python benchmark/prepare.py
uv run --locked python tools/validate_mcp_tools.py --replay-ground-truth
```

The default environment includes the agent runtime, Rasterio, all CPU tools, and
the evaluator. Rasterio uses its bundled GDAL library; no separate GDAL Python
package, system installation, or GPU perception service is needed. Required
Rasterio, NumPy, and OpenCV wheels must be available for the selected platform.

The downloader installs the pinned [EarthBench-Fixed](https://modelscope.cn/datasets/haodijia/EarthBench-Fixed)
snapshot (about 16.6 GB), verifies every file against its SHA-256 manifest, and
activates the dataset. Existing matching files can be reused; see
[benchmark/README.md](benchmark/README.md).

HKG additionally needs `uv sync --locked --extra hkg` and the embedding weights
specified in [models/README.md](models/README.md). Baseline and FAR use the default environment.

## Configuration

Copy a configuration from `configs/` to a local `*_local.yaml` file, then add the OpenAI-compatible endpoint and API key. Local configuration files are ignored by Git.

```yaml
base_url: ""
api_key: ""
```

MCP tool servers are configured in `configs/agent_mcp.json`. Each experiment writes tool outputs to its own absolute `<experiment_dir>/out` directory. The perception server strictly looks up recorded observations; its external masks must be available. See [tools/README.md](tools/README.md) for cache provenance and coverage limits.

All Python dependencies are declared in `pyproject.toml` and pinned in `uv.lock`. Use `uv sync --locked` to reproduce the environment. See [REPRODUCTION_REVIEW.md](REPRODUCTION_REVIEW.md) for the benchmark protocol and validation scope.

## Running an experiment

```bash
uv run --locked --extra hkg geoevolve run --config configs/deepseek_v32_rsmem.yaml
```

Run the deterministic evaluation pipeline after an experiment completes:

```bash
uv run --locked python evaluation/run_pipeline.py \
  --base-dir <RESULTS_DIR> \
  --rounds 1,2,3 \
  --result-dir-name geoevolve_results
```

The internal Python package is currently named `rsmem`; it will be migrated as GeoEvolve's public API is redesigned.

## Data and model weights

Benchmark data and embedding weights are not stored in this repository. See the focused documentation in [`benchmark/`](benchmark/README.md), [`models/`](models/README.md), [`tools/`](tools/README.md), and [`evaluation/`](evaluation/README.md).

## Acknowledgements

GeoEvolve began as a repaired and extended derivative of [RSMeM](https://github.com/AI9Stars/RSMeM), which itself incorporates code derived from [Earth-Agent](https://github.com/opendatalab/Earth-Agent). Thanks to their authors for releasing their work under the MIT License.

The project also uses ideas, datasets, or tooling from [EarthBench](https://github.com/opendatalab/Earth-Agent), [ThinkGeo](https://github.com/mbzuai-oryx/ThinkGeo), and [LangChain](https://github.com/langchain-ai/langchain).

See [`NOTICE`](NOTICE) for detailed attribution and third-party license information.

## License

GeoEvolve is distributed under the MIT License. See [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).
