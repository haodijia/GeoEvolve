"""Failure-Aware Reflection lifecycle; state and model costs belong to this plugin."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent.agent import load_model
from agent.config import ModelConfig
from agent.tokens import TOKEN_FIELDS, sum_usage_ledger
from plugins.base import AgentPlugin, PluginContext, PromptContribution
from plugins.far.memory import FailureAwareMemory
from plugins.far.prompts import DEFAULT_REFLECTION_TEMPLATE
from plugins.far.strategies import (
    generate_reflections_for_failed_questions,
    get_reflection,
    get_trajectory,
    get_truncation,
)


@dataclass
class FARConfig:
    reflector: str | dict[str, Any] = "actor"
    compressor: str | dict[str, Any] = "actor"
    truncation_strategy: str = "default"
    trajectory_strategy: str = "full_tool_result_only"
    reflection_strategy: str = "default"
    reflection_template: str = DEFAULT_REFLECTION_TEMPLATE
    summary_max_step_chars: int = 6000
    summary_max_trace_chars: int = 120000

    def __post_init__(self) -> None:
        get_truncation(self.truncation_strategy)
        get_trajectory(self.trajectory_strategy)
        get_reflection(self.reflection_strategy)
        if (
            self.reflection_strategy == "summary_ftro"
            and self.trajectory_strategy != "summary_ftro_source"
        ):
            raise ValueError("summary_ftro requires summary_ftro_source trajectories")
        if self.summary_max_step_chars < 1 or self.summary_max_trace_chars < 1:
            raise ValueError("Summary limits must be positive")
        for model in (self.reflector, self.compressor):
            if not isinstance(model, dict) and model != "actor":
                raise ValueError(
                    "Plugin model must be 'actor' or an explicit model configuration"
                )


class FARPlugin(AgentPlugin):
    def __init__(self, options: dict) -> None:
        self.config = FARConfig(**options)

    def _model(self, specification: str | dict) -> Any:
        if specification == "actor":
            return self.context.actor
        return load_model(ModelConfig.from_dict(specification))

    async def setup(self, context: PluginContext) -> None:
        await super().setup(context)
        self.root = context.output_dir / "plugins" / "far"
        self.root.mkdir(parents=True, exist_ok=True)
        self.memory = FailureAwareMemory(self.root / "memory.json")
        self.reflector = self._model(self.config.reflector)
        self.compressor = (
            self._model(self.config.compressor)
            if self.config.reflection_strategy == "summary_ftro"
            else None
        )

    async def before_question(self, question: dict, query: str) -> PromptContribution:
        reflection = self.memory.get_latest_reflection(question["question_id"])
        return PromptContribution(
            f"### Previous Attempt Reflection\n{reflection}" if reflection else ""
        )

    async def after_attempt(
        self, attempt: int, failed_results: list[dict], chat_log_path: Path
    ) -> dict[str, int]:
        if not failed_results:
            return dict.fromkeys(TOKEN_FIELDS, 0)
        root = self.root / f"attempt_{attempt}"
        usage_path = root / "model_usage.jsonl"
        pending = [
            result
            for result in failed_results
            if not any(
                entry["attempt"] == attempt
                for entry in self.memory.by_qid.get(str(result["question_id"]), [])
            )
        ]
        kwargs: dict = {"reflection_template": self.config.reflection_template}
        trajectory_kwargs: dict = {}
        if self.config.reflection_strategy == "summary_ftro":
            kwargs.update(compressor_llm=self.compressor, artifact_dir=root)
            trajectory_kwargs = {
                "max_step_chars": self.config.summary_max_step_chars,
                "max_trace_chars": self.config.summary_max_trace_chars,
            }
        # Persist each reflection, so an interrupted batch resumes per question.
        for result in pending:
            reflections, _ = await generate_reflections_for_failed_questions(
                llm=self.reflector,
                chat_log_path=chat_log_path,
                failed_results=[result],
                attempt_num=attempt,
                truncation_strategy=self.config.truncation_strategy,
                trajectory_strategy=self.config.trajectory_strategy,
                reflection_strategy=self.config.reflection_strategy,
                reflection_kwargs=kwargs,
                trajectory_kwargs=trajectory_kwargs,
                usage_path=usage_path,
            )
            reflection = reflections[0]
            self.memory.add_reflection(
                reflection["question_id"], reflection["reflective_text"], attempt
            )
            self.memory.save()
        return sum_usage_ledger(usage_path)
