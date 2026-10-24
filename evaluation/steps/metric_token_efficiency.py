"""Compute optional cost diagnostics from measured cumulative round costs."""

import argparse
import json
import math
from pathlib import Path


def round_cost(statistics: list[dict], round_number: int) -> tuple[int, int]:
    if round_number < 1 or round_number > len(statistics):
        raise ValueError("Requested round must have completed cost statistics")
    actor = 0
    reflection = 0
    for index, stat in enumerate(statistics, start=1):
        if stat["attempt"] != index:
            raise ValueError("Attempt cost statistics must be contiguous and ordered")
        if index > round_number:
            continue
        usage = stat["token_usage"]["total_tokens"]
        feedback = stat["reflection_token_usage"]["total_tokens"]
        if any(type(value) is not int or value < 0 for value in (usage, feedback)):
            raise ValueError("Token counts must be nonnegative integers")
        actor += usage
        # Feedback generated after this round only serves a later round.
        if index < round_number:
            reflection += feedback
    return actor + reflection, reflection


def calculate_metrics(summary_file: Path, round_before: int, round_after: int,
                      acc_before: float, acc_after: float) -> dict:
    if not all(math.isfinite(value) and 0 <= value <= 1 for value in (acc_before, acc_after)):
        raise ValueError("Accuracy must be in [0, 1]")
    if round_before >= round_after:
        raise ValueError("round-before must precede round-after")
    summary = json.loads(summary_file.read_text(encoding="utf-8"))
    statistics = summary["statistics"]["per_attempt_stats"]
    before, _ = round_cost(statistics, round_before)
    after, reflection = round_cost(statistics, round_after)
    if before <= 1 or after <= 1:
        raise ValueError("Token efficiency requires measured totals greater than one")
    gain = acc_after - acc_before
    ratio = reflection / after
    eta_before = acc_before / math.log10(before)
    eta_after = acc_after / math.log10(after)
    return {
        "round_before": round_before,
        "round_after": round_after,
        "accuracy_before": acc_before,
        "accuracy_after": acc_after,
        "improvement_gain": gain,
        "total_tokens_before": before,
        "total_tokens_after": after,
        "reflection_tokens": reflection,
        "reflection_ratio": ratio,
        "token_efficiency_before": eta_before,
        "token_efficiency_after": eta_after,
        "token_efficiency_improvement": eta_after - eta_before,
        "rcbr": gain / ratio if reflection else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-file", type=Path, required=True, help="experiment_summary.json")
    parser.add_argument("--round-before", type=int, default=1)
    parser.add_argument("--round-after", type=int, required=True)
    parser.add_argument("--acc-before", type=float, required=True, help="Full-dataset accuracy in [0, 1]")
    parser.add_argument("--acc-after", type=float, required=True, help="Full-dataset accuracy in [0, 1]")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    metrics = calculate_metrics(args.summary_file, args.round_before, args.round_after,
                                args.acc_before, args.acc_after)
    rendered = json.dumps(metrics, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    print(rendered, end="")
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
