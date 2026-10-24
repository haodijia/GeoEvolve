"""Write the most recent attempted answer per question for each requested round."""

import argparse
import json
from pathlib import Path


def complete_results(base_dir: Path, rounds: list[int]) -> None:
    latest: dict[str, dict] = {}
    for attempt in range(1, max(rounds) + 1):
        path = base_dir / f"results_attempt_{attempt}.jsonl"
        seen = set()
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            result = json.loads(line)
            qid = str(result["question_id"])
            if qid in seen or result["attempt"] != attempt:
                raise ValueError(f"Invalid or duplicate attempt record: {path}, question {qid}")
            seen.add(qid)
            latest[qid] = result
        if attempt in rounds:
            output = base_dir / f"results_summary_ROUND{attempt}.jsonl"
            output.write_text(
                "".join(json.dumps(latest[qid], ensure_ascii=False) + "\n" for qid in sorted(latest, key=int)),
                encoding="utf-8",
            )
            print(f"ROUND {attempt}: {len(latest)} answers -> {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, required=True)
    parser.add_argument("--rounds", required=True)
    args = parser.parse_args()
    rounds = sorted({int(value) for value in args.rounds.split(",")})
    if not rounds or rounds[0] < 1:
        raise ValueError("Rounds must be positive integers")
    complete_results(args.base_dir, rounds)


if __name__ == "__main__":
    main()
