"""Lifecycle used by all agent components."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agent.config import ExperimentConfig


@dataclass
class PluginContext:
    config: ExperimentConfig
    output_dir: Path
    actor: Any


@dataclass
class PromptContribution:
    text: str = ""
    metadata: dict = field(default_factory=dict)


class AgentPlugin:
    def __init__(self, options: dict) -> None:
        if options:
            raise ValueError(f"Unsupported plugin options: {sorted(options)}")

    async def setup(self, context: PluginContext) -> None:
        self.context = context

    async def before_question(self, question: dict, query: str) -> PromptContribution:
        return PromptContribution()

    async def after_attempt(
        self, attempt: int, failed_results: list[dict], chat_log_path: Path
    ) -> dict[str, int]:
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
