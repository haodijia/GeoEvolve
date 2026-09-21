# Evaluation

The EarthBench evaluation path is deterministic and uses no judge model:

```text
results_attempt_{N}.jsonl
        │
        ├─ complete ──> results_summary_ROUND{N}.jsonl
        ├─ extract  ──> extracted_tool_calls_round{N}.json
        ├─ step2step -> evaluation/results/<name>/step_by_step_evaluation_results_round{N}.json
        └─ end2end  ──> evaluation/results/<name>/end_to_end_evaluation_results_round{N}.json
```

Run it with:

```bash
uv run python evaluation/run_pipeline.py \
  --base-dir <RESULTS_DIR> \
  --model-name <MODEL_NAME> \
  --rounds 1,2,3 \
  --result-dir-name <NAME>
```

## Main EarthBench metrics

The step-by-step report contains five per-question tool-trajectory scores. Missing predictions receive zero and are included in the dataset-level mean.

| Output key | Definition |
|---|---|
| `contains_all_tool_calls_any_order` | Recall of distinct GT tool names found anywhere in the predicted trajectory. Order, parameters, and repeated occurrences are ignored. |
| `contains_all_tool_calls_in_order` | Fraction of the GT tool-name sequence matched as an ordered subsequence. Matching stops at the first GT tool that cannot be found later in the prediction. |
| `contains_all_tool_calls_in_order_exact_match` | Correct prefix length divided by the longer of predicted and GT trajectory lengths. Tool names only. |
| `trajectory_step_wise_score` | Correct prefix length divided by GT trajectory length. Tool names only. |
| `parameter_accuracy` | Correct prefix length divided by GT trajectory length, requiring exact equality of both tool name and the complete `input` object. |

The end-to-end report contains:

| Output key | Definition |
|---|---|
| `accuracy` | Correct answers divided by all 248 GT questions. Missing, malformed, and failed answers count as incorrect. |
| `accuracy_on_evaluated` | Correct answers divided only by present predictions; diagnostic only. |
| `coverage` | Present predictions divided by all GT questions. |
| `fail_rate` | Explicit failures divided by all GT questions. |
| `average_efficiency` | Mean of `predicted tool-call count / GT tool-call count` over present tool traces. This is a call-count ratio, not a quality score; 1 means the same number of calls. |

A batch request remains one tool-call object throughout extraction and scoring;
the evaluator does not unfold list-valued inputs into per-file calls.

Answers are extracted only from the final canonical tag pair, `<Answer>value<Answer>` (the conventional `</Answer>` closing tag is also accepted). No free-text letter guessing or model-based normalization is performed. EarthBench includes 246 multiple-choice answers and two numeric answers, so `value` is not restricted to A-D.

## Analysis-only metrics

These scripts are not part of the default pipeline:

- `metric_token_efficiency.py`: accuracy gain, token totals, `eta = accuracy / log10(tokens)`, and reflection cost-benefit ratio `RCBR = gain / reflection_token_ratio`.
- `metric_experience_density.py`: `NEI = accuracy * (1 - reflection_tokens / total_tokens)`.
- `run_pipeline_thinkgeo.py`: separate ThinkGeo evaluation with its own input contract.

## Ground truth

`build_ground_truth.py` reads final answers from `question.json`'s authoritative `evaluation[0].gt_answer.whitelist` and tool calls from the dialogs. It does not infer an answer from the last dialog message, because several EarthBench dialogs end with a tool result rather than an assistant answer.

Rebuild the executable ground truth after downloading the benchmark data:

```bash
uv run python evaluation/build_ground_truth.py
```

The builder preserves one call per official batch operation, resolves stale
upstream paths against the actual question inventory, and writes every derived
raster under the canonical `tmp/out` dataflow. It also repairs the published
trajectories whose arguments cannot execute against the released files.
