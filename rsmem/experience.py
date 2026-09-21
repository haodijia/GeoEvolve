"""Grounded full-trajectory summaries for Summary+FTRO reflection."""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel, Field

class GroundedClaim(BaseModel):
    text: str = Field(min_length=3, max_length=900)
    source_step_ids: list[int] = Field(min_length=1, max_length=512)


class TrajectorySummary(BaseModel):
    question_id: str
    task_summary: GroundedClaim
    attempted_strategy: list[GroundedClaim] = Field(default_factory=list, max_length=8)
    verified_state: list[GroundedClaim] = Field(default_factory=list, max_length=10)
    observed_failures: list[GroundedClaim] = Field(default_factory=list, max_length=10)
    unfinished_subgoals: list[GroundedClaim] = Field(default_factory=list, max_length=8)
    budget_signals: list[GroundedClaim] = Field(default_factory=list, max_length=6)


class ReflectionResponse(BaseModel):
    question_id: str
    reflective_text: str = Field(min_length=20, max_length=3000)


_ANSWER_LEAK = re.compile(
    r"<\s*/?\s*answer\s*>|"
    r"\b(?:choose|select|answer(?:\s+is)?)\s+(?:option\s+)?[A-D]\b|"
    r"\b(?:matches?|corresponds?\s+to|closest\s+to)\s+(?:the\s+)?option\s+[A-D]\b|"
    r'"answer"\s*:',
    flags=re.I,
)
_ABSOLUTE_PATH = re.compile(
    r"(?:[A-Za-z]:[\\/]|/(?:home|root|mnt|data|tmp|workspace|ultrafast_share)/)"
    r"[^\s\"']*",
    flags=re.I,
)


def extract_object(text: str, required_key: str) -> dict[str, Any]:
    stripped = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.I)
    decoder = json.JSONDecoder()
    for index, character in enumerate(stripped):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(stripped[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and required_key in value:
            return value
    raise ValueError(f"no JSON object containing {required_key!r}")


def parse_and_validate_summary(
    text: str, *, question_id: str, step_count: int
) -> TrajectorySummary:
    value = extract_object(text, "task_summary")
    summary = TrajectorySummary.model_validate(value)
    if summary.question_id != str(question_id):
        raise ValueError(
            f"question_id mismatch: expected {question_id}, got {summary.question_id}"
        )
    claims = [
        summary.task_summary,
        *summary.attempted_strategy,
        *summary.verified_state,
        *summary.observed_failures,
        *summary.unfinished_subgoals,
        *summary.budget_signals,
    ]
    for claim in claims:
        invalid = [step for step in claim.source_step_ids if step < 1 or step > step_count]
        if invalid:
            raise ValueError(f"q{question_id}: invalid source steps {invalid}; maximum={step_count}")
        if _ANSWER_LEAK.search(claim.text):
            raise ValueError(f"q{question_id}: answer leakage in summary")
        if _ABSOLUTE_PATH.search(claim.text):
            raise ValueError(f"q{question_id}: absolute path copied into summary")
    return summary


def parse_and_validate_reflection(text: str, *, question_id: str) -> ReflectionResponse:
    reflection = ReflectionResponse.model_validate(extract_object(text, "reflective_text"))
    if reflection.question_id != str(question_id):
        raise ValueError(
            f"question_id mismatch: expected {question_id}, got {reflection.question_id}"
        )
    if _ANSWER_LEAK.search(reflection.reflective_text):
        raise ValueError(f"q{question_id}: answer leakage in reflection")
    return reflection
