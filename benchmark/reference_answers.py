"""Evaluate data-only arithmetic evidence for benchmark reference answers."""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np


def derive(expression, calls):
    if isinstance(expression, list):
        return [derive(item, calls) for item in expression]
    if not isinstance(expression, dict):
        return expression
    for field in ("output", "input"):
        if field in expression:
            index, *keys = expression[field]
            value = calls[index][field]
            for key in keys:
                value = value[key]
            return value
    operation = expression["op"]
    args = [derive(arg, calls) for arg in expression["args"]]
    if operation == "add":
        return args[0] + args[1]
    if operation == "subtract":
        return args[0] - args[1]
    if operation == "multiply":
        return args[0] * args[1]
    if operation == "divide":
        return args[0] / args[1]
    if operation == "abs":
        return abs(args[0])
    if operation == "mean":
        return float(np.mean(args[0]))
    if operation == "percentile":
        return float(np.percentile(args[0], args[1]))
    if operation == "get":
        return args[0][int(args[1])]
    if operation == "date":
        match = re.search(r"((?:19|20)[0-9]{2})[-_]?([0-9]{2})[-_]?([0-9]{2})", Path(args[0]).name)
        if match is None:
            raise ValueError(f"No acquisition date: {args[0]}")
        return "-".join(match.groups())
    if operation == "rank":
        values = args[0]
        groups = {}
        for label, value in zip("ABCDEFGHIJKLMNOPQRSTUVWXYZ", values):
            groups.setdefault(value, []).append(label)
        return " > ".join(" = ".join(groups[v]) for v in sorted(groups, reverse=True))
    if operation == "count_class":
        return sum(value["predicted_class"] == args[1] for value in args[0])
    if operation == "closest":
        distances = [abs(args[0] - value) for value in args[1]]
        index = min(range(len(distances)), key=distances.__getitem__)
        if distances.count(distances[index]) != 1:
            raise ValueError("Answer has no unique closest option")
        return chr(65 + index)
    raise ValueError(f"Unknown reference arithmetic operation: {operation}")


def verify_answer(question, reference, review):
    if reference["final_answer"] != review["answer"]:
        raise ValueError("Reference and reviewed answer differ")
    evidence = []
    for item in review["evidence"]:
        actual = derive(item["expression"], reference["tool_calls"])
        expected = item["expected"]
        if isinstance(expected, (int, float)):
            matches = isinstance(actual, (int, float)) and math.isclose(
                actual, expected, rel_tol=1e-7, abs_tol=item.get("tolerance", 1e-7)
            )
        else:
            matches = actual == expected
        if not matches:
            raise ValueError(f"Answer evidence differs: {actual!r} != {expected!r}")
        evidence.append(actual)
    if not evidence:
        raise ValueError("Answer requires evidence from actual tool outputs")
    choices = question["choices"]
    selected = choices[ord(review["answer"]) - 65] if choices else review["answer"]
    if selected != review["selected_text"]:
        raise ValueError("Selected option differs from reviewed evidence")
    return evidence


def verify_release(root: Path):
    questions = json.loads((root / "questions.json").read_text())
    references = json.loads((root / "ground_truth.json").read_text())
    reviews = json.loads((root / "provenance/answer-review.json").read_text())
    if len(questions) != len(references) or len(questions) != len(reviews):
        raise ValueError("Question, reference and answer-review inventories differ")
    for question, reference, review in zip(questions, references, reviews):
        expected_id = "question" + str(question["question_id"])
        if reference["question_index"] != expected_id or review["question_index"] != expected_id:
            raise ValueError("Reference answer IDs differ")
        verify_answer(question, reference, review)
    return {"questions": len(questions), "answer_evidence": "passed"}
