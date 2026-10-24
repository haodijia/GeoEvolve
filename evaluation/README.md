# EarthBench evaluation

The deterministic pipeline uses no judge model:

```bash
uv run --no-sync python evaluation/run_pipeline.py \
  --base-dir <EXPERIMENT_DIR> \
  --rounds 1,2,3 \
  --result-dir-name <NAME> --strict-rounds --force
```

Inputs are new runner outputs `results_attempt_N.jsonl`, including `question_id`, `attempt`, `answer`, `execution_error`, `tool_calls`, `token_usage`, and absolute `tool_output_dir`. Older logs are not parsed or guessed. Requesting only round 3 still reads attempts 1–3. A question not retried retains its latest answer and latest trajectory. Failed and pending tool calls remain in the trajectory.

The pipeline writes `results_summary_ROUNDN.jsonl` and `extracted_tool_calls_roundN.json` beside the run, then metric reports under `evaluation/results/<NAME>/`. Trajectory exports contain both latest-attempt `tool_calls` and all-attempt `cumulative_tool_calls`. Use a distinct result name for each experiment. `--force` recomputes reports after an input or evaluator change.

## Main metrics

All step scores are averaged over all 248 GT questions, including zeros for missing predictions. Official score formulas were checked against the published Earth-Agent evaluator on all 248 published Kimi trajectories: zero differences when using the same GT and normalized inputs.

| Paper/display name | JSON key | Definition |
|---|---|---|
| Tool_Any_Order | `contains_all_tool_calls_any_order` | Fraction of distinct GT tool names present anywhere in the prediction. Repeats do not count separately. |
| Tool_In_Order | `contains_all_tool_calls_in_order` | Matched GT tool-name subsequence prefix / GT length. Other predicted calls may occur between matches. Matching stops at the first GT tool not found later. |
| Tool_Exact_Match | `trajectory_step_wise_score` | Matching tool-name prefix / GT length. This is the official naming; it is a fractional prefix score. |
| Parameter | `parameter_accuracy` | Matching prefix requiring both tool name and complete input equality / GT length. Storage roots are normalized as described below. |
| Extra diagnostic | `contains_all_tool_calls_in_order_exact_match` | Matching tool-name prefix / max(predicted length, GT length). This is not the official Tool_Exact_Match. |

The end-to-end report contains:

| Key | Definition |
|---|---|
| `accuracy` | Correct answers / all 248 questions. Missing, malformed, and failed answers receive zero. |
| `accuracy_on_evaluated` | Correct / present predictions; diagnostic only. |
| `coverage` | Present predictions / all GT questions. |
| `fail_rate` | Explicit execution failures or empty answers / all GT questions. |
| `average_efficiency` | Mean latest-attempt tool-call count / GT count, over present traces. |
| `cumulative_tool_call_ratio` | Mean all-attempt tool-call count / GT count, over present traces. |

Call ratios measure cost, not solution quality. An empty trace has ratio zero; always report accuracy and coverage alongside cost. No present traces yields `null` cost. Batch operations remain one tool call; lists are never expanded into separate per-file calls.

Answers must use the final `<Answer>value<Answer>` pair (`</Answer>` is also accepted). No free-text letter guessing or model polishing is performed. The GT contains 246 multiple-choice answers and two numeric answers. The runner's `heuristic_success` indicates answer-format validity, not correctness, and must not be reported as benchmark accuracy.

## Paths and GT protocol

`earthbench_paths.py` normalizes declared per-experiment output roots, canonical `tmp/out`, and the repository's benchmark-data root aliases. Relative output parameters are also mapped to the same artifact root. Filenames, numeric values, list order, and other parameters are preserved. Relocating all 248 GT trajectories to an isolated experiment directory preserves parameter score 1.

`build_ground_truth.py` takes answers from the official `evaluation[0].gt_answer.whitelist`, and executable tool calls from dialogs. It repairs stale paths, parameter errors, batch alignment, and broken reference trajectories; the current repaired GT has 248 questions and 1,378 calls. Rebuild with:

```bash
uv run --no-sync python evaluation/build_ground_truth.py
```

The repaired environment exposes 110 tools: the original 104 plus six interfaces already called by official dialogs but absent from the published MCP inventory. The repaired GT, path normalization, all-question denominator, and strict answer format define our comparison protocol. They differ from some published scripts and polished result archives. Formula agreement alone does not make our numbers directly comparable to the papers. All compared methods must use this same GT, dataset variant, observation cache, and evaluator. Disclose question148/149's synthetic replacement data and the rebuilt perception masks; see [REPRODUCTION_REVIEW.md](../REPRODUCTION_REVIEW.md).

## Token and reflection cost

`experiment_summary.json` records actor and reflection costs by attempt. Reflection and compressor responses are logged before content parsing in `feedback/attempt_N/model_usage.jsonl`, so format retries count even if their output is rejected. Resume imports earlier actor costs and feedback ledgers. Reflections no longer use fixed fallback text on failure.

To compare round 1 and round 3:

```bash
uv run --no-sync python evaluation/steps/metric_token_efficiency.py \
  --summary-file <EXPERIMENT_DIR>/experiment_summary.json \
  --round-before 1 --round-after 3 \
  --acc-before <ROUND1_ACCURACY> --acc-after <ROUND3_ACCURACY>
```

Round N cost includes actor responses from attempts 1–N, plus reflection/compressor responses generated after attempts 1–N−1. Feedback generated after N serves a later round and is excluded. Count every model role when imposing a common budget; a common actor model alone does not equalize teacher/compressor cost. These are provider-reported response tokens, not a monetary invoice; a request without reported usage makes cost unknown and raises an error.

The token script computes cumulative totals, `eta = accuracy / log10(tokens)`, and `RCBR = accuracy_gain / (reflection_tokens / total_tokens)`. RCBR is `null` when reflection cost is zero. The old before/after aggregated-answer token subtraction has been removed. `metric_experience_density.py` computes `NEI = accuracy * (1 - reflection_tokens / total_tokens)` from these measured totals; accuracy must be in [0, 1]. These are optional method diagnostics, not the primary EarthBench metrics. `run_pipeline_thinkgeo.py` is a separate evaluator with a separate input contract.
