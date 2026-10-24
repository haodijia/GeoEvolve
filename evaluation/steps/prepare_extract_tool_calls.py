"""Export final and cumulative trajectories directly from structured run results."""

import argparse
import json
from pathlib import Path


def extract_rounds(base_dir: Path, rounds: list[int]) -> None:
    latest: dict[str, dict] = {}
    cumulative: dict[str, list[dict]] = {}
    roots: dict[str, set[str]] = {}
    for attempt in range(1, max(rounds) + 1):
        source = base_dir / f"results_attempt_{attempt}.jsonl"
        seen = set()
        for line in source.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            result = json.loads(line)
            qid = str(result["question_id"])
            if qid in seen or result["attempt"] != attempt:
                raise ValueError(f"Invalid or duplicate attempt record: {source}, question {qid}")
            seen.add(qid)
            calls = result["tool_calls"]
            if not isinstance(calls, list):
                raise TypeError(f"question {qid}: tool_calls must be a list")
            cumulative.setdefault(qid, []).extend(calls)
            roots.setdefault(qid, set()).add(result["tool_output_dir"])
            latest[qid] = {
                "question_index": qid,
                "latest_attempt": attempt,
                "tool_calls": calls,
            }
        if attempt in rounds:
            output = [
                {
                    **latest[qid],
                    "output_roots": sorted(roots[qid]),
                    "cumulative_tool_calls": list(cumulative[qid]),
                }
                for qid in sorted(latest, key=int)
            ]
            destination = base_dir / f"extracted_tool_calls_round{attempt}.json"
            destination.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(f"ROUND {attempt}: exported {len(output)} questions to {destination}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, required=True)
    parser.add_argument("--rounds", required=True, help="Comma-separated rounds, e.g. 1,2,3")
    args = parser.parse_args()
    rounds = sorted({int(value) for value in args.rounds.split(",")})
    if not rounds or rounds[0] < 1:
        raise ValueError("Rounds must be positive integers")
    extract_rounds(args.base_dir, rounds)


if __name__ == "__main__":
    main()
