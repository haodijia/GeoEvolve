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

GeoEvolve requires Python 3.10 or later.

```bash
git clone https://github.com/haodijia/GeoEvolve.git
cd GeoEvolve

uv venv --python 3.10
source .venv/bin/activate
uv pip install -r requirements.txt
uv pip install -e .
uv pip install -r tools/requirements-tools.txt
```

## Configuration

Copy a configuration from `configs/` to a local `*_local.yaml` file, then add the OpenAI-compatible endpoint and API key. Local configuration files are ignored by Git.

```yaml
base_url: ""
api_key: ""
```

MCP tool servers are configured in `configs/agent_mcp.json`. The perception server uses the checked-in offline cache, so the standard EarthBench workflow does not require a GPU or a separate inference service.

## Running an experiment

```bash
uv run geoevolve run --config configs/deepseek_v32_rsmem.yaml
```

Run the deterministic evaluation pipeline after an experiment completes:

```bash
uv run python evaluation/run_pipeline.py \
  --base-dir <RESULTS_DIR> \
  --model-name deepseek-v3.2 \
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
