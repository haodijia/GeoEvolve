# EarthBench data

Data are stored outside the repository, under `/home/dijkstra/data`:

```bash
uv tool run --from huggingface_hub hf download Sssunset/Earth-Bench \
  --repo-type dataset --local-dir /home/dijkstra/data/earth_bench
ln -s /home/dijkstra/data/earth_bench benchmark/data
curl -L -o /home/dijkstra/data/earth_bench/question.json \
  https://raw.githubusercontent.com/opendatalab/Earth-Agent/main/benchmark/question.json
ln -s /home/dijkstra/data/earth_bench/question.json benchmark/question.json
```

The official question manifest expects `benchmark/data/questionN`. Share these read-only data across experiments; the runner gives each experiment its own `<experiment_dir>/out` directory. There is no need to copy the whole dataset for parallel runs.

## Missing q148/q149 rasters

`reconstruct_missing_data.py` constructs synthetic annual Los Angeles night-light rasters with hardcoded means and hotspot counts, borrowing the grid of a released building raster. It does not recover the true imagery or spatial pattern. Never describe those files as authentic reconstructed observations.

The local audit preserves `/home/dijkstra/data/earth_bench` unchanged. `benchmark/data` currently points to `/home/dijkstra/data/earth_bench_geoevolve_review`, an overlay of official file links plus four synthetic rasters used by q148/q149. All 248 questions can execute with this overlay, but formal results must disclose it and report these two questions separately if assessing sensitivity to missing data.

See [REPRODUCTION_REVIEW.md](../REPRODUCTION_REVIEW.md) for validation evidence and perception-cache provenance. Downloaded data, local manifests, and generated outputs are ignored by Git.
