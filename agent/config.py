"""Run configuration shared by the base agent and experiment launcher."""

from __future__ import annotations

import os
import re
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

from agent.prompts import SYSTEM_PROMPT


@dataclass
class PluginConfig:
    name: str
    options: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not re.fullmatch(
            r"[a-z][a-z0-9_]*(?::[A-Za-z_][A-Za-z0-9_.]*)?", self.name.replace(".", "_")
        ):
            raise ValueError(f"Invalid plugin name: {self.name!r}")
        if not isinstance(self.options, dict):
            raise ValueError("Plugin options must be a mapping")


@dataclass
class ModelConfig:
    model_name: str = "deepseek-v4-flash"
    base_url: str = "https://api.deepseek.com"
    api_key_env: str = "OPENAI_API_KEY"
    request_timeout: float = 120.0
    temperature: float = 0.1
    generate_args: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.model_name, str) or not self.model_name.strip():
            raise ValueError("model_name must be a non-empty string")
        if not isinstance(self.api_key_env, str) or not self.api_key_env:
            raise ValueError(
                "api_key_env must be a non-empty environment variable name"
            )
        if self.request_timeout <= 0:
            raise ValueError("request_timeout must be positive")
        if not isinstance(self.generate_args, dict):
            raise ValueError("generate_args must be a mapping")

    def resolve_api_key(self) -> str:
        value = os.environ.get(self.api_key_env, "")
        if not value:
            raise ValueError(f"Set the API key environment variable {self.api_key_env}")
        return value

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelConfig:
        unknown = set(data) - {item.name for item in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown configuration fields: {sorted(unknown)}")
        return cls(**data)


@dataclass
class ExperimentConfig(ModelConfig):
    model_label: str = ""
    datasets: list[str] = field(default_factory=lambda: ["earthbench"])
    plugins: list[PluginConfig] = field(default_factory=list)
    autoplanning: bool = True
    sample: int | None = None
    concurrency: int = 1
    max_attempts: int = 1
    sys_prompt: str = SYSTEM_PROMPT
    agent_config_path: str = "agent/mcp.json"
    output_base_dir: str = "runs"
    question_timeout: float = 300.0
    recursion_limit: int = 50

    def __post_init__(self) -> None:
        super().__post_init__()
        if not isinstance(self.model_label, str):
            raise ValueError("model_label must be a string")
        if self.model_label and not re.fullmatch(r"[a-z][a-z0-9_]*", self.model_label):
            raise ValueError(
                "model_label must contain lowercase letters, digits and underscores"
            )
        if (
            not isinstance(self.datasets, list)
            or not self.datasets
            or any(not isinstance(name, str) for name in self.datasets)
        ):
            raise ValueError("datasets must be a non-empty list of dataset names")
        if len(set(self.datasets)) != len(self.datasets):
            raise ValueError("Duplicate datasets")
        for name in self.datasets:
            if not re.fullmatch(r"[a-z][a-z0-9_]*", name):
                raise ValueError(f"Invalid dataset name: {name!r}")
        if not isinstance(self.plugins, list):
            raise ValueError("plugins must be a list")
        self.plugins = [
            PluginConfig(**item) if isinstance(item, dict) else item
            for item in self.plugins
        ]
        if any(not isinstance(item, PluginConfig) for item in self.plugins):
            raise ValueError("Each plugin must contain name and options")
        if len({item.name for item in self.plugins}) != len(self.plugins):
            raise ValueError("Duplicate plugins")
        for name in ("concurrency", "max_attempts", "recursion_limit"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.sample is not None and (
            type(self.sample) is not int or self.sample < 1
        ):
            raise ValueError(
                "sample must be null (all questions) or a positive integer"
            )
        if self.request_timeout <= 0 or self.question_timeout <= 0:
            raise ValueError("Timeouts must be positive")

    @property
    def method_label(self) -> str:
        return (
            "+".join(item.name.rsplit(":", 1)[-1] for item in self.plugins)
            or "baseline"
        )

    def to_yaml(self, path: str | Path) -> None:
        Path(path).write_text(
            yaml.safe_dump(asdict(self), sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> ExperimentConfig:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("Run configuration must be a mapping")
        return cls.from_dict(raw)
