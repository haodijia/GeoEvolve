#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Adapted from the Earth-Agent evaluation framework
# (https://github.com/opendatalab/Earth-Agent), MIT License,
# Copyright (c) 2026 OpenDataLab. See the NOTICE file at the repo root.
import re
from typing import Dict, List, Any
from evaluation.context import MetricContext

REQUIRES = ("answers", "trajectories")


def extract_answer_from_text(text: str) -> str:
    """Extract the content of the final canonical Answer tag pair."""
    if not text:
        return "FAIL"

    matches = re.findall(
        r"<\s*Answer\s*>(.*?)<\s*/?\s*Answer\s*>",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if matches:
        answer = matches[-1].strip()
        return answer.upper() if len(answer) == 1 and answer.isalpha() else answer
    return "UNKNOWN"


def count_tool_calls(data: Dict) -> int:
    """Count tool calls"""
    tool_calls = data.get("tool_calls", [])
    return len(tool_calls)


def calculate_accuracy(ground_truth_data: List[Dict], predicted_data: Any) -> Dict:
    """Calculate accuracy"""
    # Create index mapping for ground truth
    gt_dict = {}
    for item in ground_truth_data:
        key = str(item["question_index"])
        if key.isdigit():
            key = f"question{key}"
        gt_dict[key] = item

    if not isinstance(predicted_data, list):
        raise TypeError("Predictions must be a list of structured result records")
    pred_dict = {}
    for item in predicted_data:
        key = str(item["question_id"])
        if key.isdigit():
            key = f"question{key}"
        if key in pred_dict or key not in gt_dict:
            raise ValueError(f"Duplicate or unknown question ID: {key}")
        pred_dict[key] = item

    results = {
        "total_questions": len(gt_dict),
        "evaluated_questions": 0,
        "correct_answers": 0,
        "fail_answers": 0,
        "unknown_answers": 0,
        "missing_predictions": [],
        "coverage": 0.0,
        "accuracy": 0.0,
        "accuracy_on_evaluated": 0.0,
        "detailed_results": [],
    }

    for question_index, gt_item in gt_dict.items():
        gt_answer = gt_item.get("final_answer", "")
        if gt_answer is None:
            gt_answer = ""
        gt_answer = str(gt_answer).strip()

        if question_index not in pred_dict:
            results["missing_predictions"].append(question_index)
            results["detailed_results"].append(
                {
                    "question_index": question_index,
                    "ground_truth": gt_answer,
                    "predicted": "MISSING",
                    "correct": False,
                    "status": "missing",
                }
            )
            continue

        results["evaluated_questions"] += 1
        pred_item = pred_dict[question_index]
        pred_answer_text = pred_item.get("answer", "")
        pred_answer = extract_answer_from_text(str(pred_answer_text))
        # Judge correctness
        is_correct = False
        status = "incorrect"

        if pred_item["execution_error"] or pred_answer == "FAIL":
            results["fail_answers"] += 1
            status = "fail"
        elif pred_answer == "UNKNOWN":
            results["unknown_answers"] += 1
            status = "unknown"

        elif pred_answer == gt_answer:
            results["correct_answers"] += 1
            is_correct = True
            status = "correct"

        results["detailed_results"].append(
            {
                "question_index": question_index,
                "ground_truth": gt_answer,
                "predicted": pred_answer,
                "correct": is_correct,
                "status": status,
            }
        )

    if results["total_questions"] > 0:
        results["coverage"] = (
            results["evaluated_questions"] / results["total_questions"]
        )
        results["accuracy"] = results["correct_answers"] / results["total_questions"]
    if results["evaluated_questions"] > 0:
        results["accuracy_on_evaluated"] = (
            results["correct_answers"] / results["evaluated_questions"]
        )

    return results


def index_tool_calls(data: List[Dict]) -> Dict:
    """Load model tool calls data"""
    tool_calls_dict = {}
    for item in data:
        key = item["question_index"]
        if key.isdigit():
            key = f"question{key}"
        if key in tool_calls_dict:
            raise ValueError(f"Duplicate trajectory for {key}")
        tool_calls_dict[key] = item
    return tool_calls_dict


def calculate_efficiency_with_tool_calls(
    ground_truth_data: List[Dict], model_tool_calls_data: Dict
) -> Dict:
    """
    Calculate efficiency: model tool calls count / ground truth tool calls count
    Value > 1 means model used more tools than needed
    Value < 1 means model used fewer tools than expected
    """
    gt_dict = {}
    for item in ground_truth_data:
        key = str(item["question_index"])
        if key.isdigit():
            key = f"question{key}"
        gt_dict[key] = item

    results = {
        "total_questions": len(gt_dict),
        "evaluated_questions": 0,
        "efficiency_scores": [],
        "cumulative_efficiency_scores": [],
        "average_efficiency": None,
        "cumulative_tool_call_ratio": None,
        "detailed_results": [],
    }

    for question_index, gt_item in gt_dict.items():
        gt_tool_count = count_tool_calls(gt_item)

        if question_index not in model_tool_calls_data:
            results["detailed_results"].append(
                {
                    "question_index": question_index,
                    "gt_tool_count": gt_tool_count,
                    "model_tool_count": 0,
                    "efficiency": 0.0,
                    "status": "missing",
                }
            )
            continue

        results["evaluated_questions"] += 1
        model_item = model_tool_calls_data[question_index]
        model_tool_count = count_tool_calls(model_item)
        cumulative_tool_count = len(model_item["cumulative_tool_calls"])

        if gt_tool_count == 0:
            raise ValueError(
                f"{question_index}: cumulative call ratio requires a nonempty reference trajectory"
            )
        efficiency = model_tool_count / gt_tool_count
        results["efficiency_scores"].append(efficiency)
        cumulative_efficiency = cumulative_tool_count / gt_tool_count
        results["cumulative_efficiency_scores"].append(cumulative_efficiency)
        results["detailed_results"].append(
            {
                "question_index": question_index,
                "gt_tool_count": gt_tool_count,
                "model_tool_count": model_tool_count,
                "cumulative_tool_count": cumulative_tool_count,
                "cumulative_efficiency": cumulative_efficiency,
                "efficiency": efficiency,
                "status": "evaluated",
            }
        )

    # Calculate average efficiency
    if results["efficiency_scores"]:
        results["average_efficiency"] = sum(results["efficiency_scores"]) / len(
            results["efficiency_scores"]
        )
        results["cumulative_tool_call_ratio"] = sum(
            results["cumulative_efficiency_scores"]
        ) / len(results["cumulative_efficiency_scores"])
    results["coverage"] = results["evaluated_questions"] / results["total_questions"]

    return results


def compute(context: MetricContext) -> Dict:
    ground_truth_data = context.run.ground_truth
    predicted_data = context.extracted["answers"]
    accuracy_results = calculate_accuracy(ground_truth_data, predicted_data)
    efficiency_results = calculate_efficiency_with_tool_calls(
        ground_truth_data, index_tool_calls(context.extracted["trajectories"])
    )
    protocol = context.run.snapshot
    synthetic = {f"question{number}" for number in protocol["synthetic_data_questions"]}
    by_data_source = {}
    groups = {
        "released_imagery": [
            q for q in ground_truth_data if q["question_index"] not in synthetic
        ],
        "synthetic_rasters": [
            q for q in ground_truth_data if q["question_index"] in synthetic
        ],
    }
    for label, subset in groups.items():
        ids = {q["question_index"] for q in subset}
        predictions = []
        for prediction in predicted_data:
            key = str(prediction["question_id"])
            if key.isdigit():
                key = f"question{key}"
            if key in ids:
                predictions.append(prediction)
        group = calculate_accuracy(subset, predictions)
        by_data_source[label] = {
            key: group[key]
            for key in (
                "total_questions",
                "evaluated_questions",
                "correct_answers",
                "accuracy",
                "coverage",
            )
        }
    return {
        "by_data_source": by_data_source,
        "accuracy": accuracy_results,
        "efficiency": efficiency_results,
        "summary": {
            "total_questions": accuracy_results["total_questions"],
            "accuracy_rate": accuracy_results["accuracy"],
            "coverage": accuracy_results["coverage"],
            "average_efficiency": efficiency_results.get("average_efficiency", 0.0),
            "cumulative_tool_call_ratio": efficiency_results.get(
                "cumulative_tool_call_ratio"
            ),
            "fail_rate": accuracy_results["fail_answers"]
            / accuracy_results["total_questions"]
            if accuracy_results["total_questions"] > 0
            else 0.0,
        },
    }
