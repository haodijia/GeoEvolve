"""Measured actor and plugin costs for producing the submitted results."""

import math

from evaluation.context import MetricContext
from evaluation.metrics.answers import calculate_accuracy

REQUIRES = ("answers",)


def compute(context: MetricContext) -> dict:
    costs = context.run.summary["token_usage"]
    actor, plugins, total = (costs[role] for role in ("actor", "plugins", "total"))
    accuracy = calculate_accuracy(
        context.run.ground_truth, context.extracted["answers"]
    )["accuracy"]
    return {
        "summary": {
            "actor": actor,
            "plugins": plugins,
            "total": total,
            "plugin_ratio": plugins["total_tokens"] / total["total_tokens"]
            if total["total_tokens"]
            else None,
            "accuracy": accuracy,
            "accuracy_per_log10_tokens": accuracy / math.log10(total["total_tokens"])
            if total["total_tokens"] > 1
            else None,
        }
    }
