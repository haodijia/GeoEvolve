"""Explicit, per-run LangSmith tracing with concurrent-question context isolation."""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from langsmith import Client, tracing_context

from agent.config import ExperimentConfig


def langsmith_api_key() -> str:
    key = os.environ.get("LANGSMITH_API_KEY", "").strip()
    if not key:
        raise ValueError("Set LANGSMITH_API_KEY when LangSmith tracing is enabled")
    return key


@asynccontextmanager
async def trace_run(
    config: ExperimentConfig, dataset_name: str, run_dir: Path, *, perception_cache: dict
) -> AsyncIterator[None]:
    client = Client(api_key=langsmith_api_key()) if config.langsmith else None
    try:
        with tracing_context(
            enabled=config.langsmith,
            client=client,
            project_name=run_dir.name,
            parent=False,
            tags=["earth-bench-pro-max", dataset_name],
            metadata={
                "experiment": run_dir.name,
                "dataset": dataset_name,
                "model": config.model_name,
                "plugins": [plugin.name for plugin in config.plugins],
                "concurrency": config.concurrency,
                "perception_cache": perception_cache,
            },
        ):
            yield
    finally:
        if client is not None:
            await asyncio.to_thread(client.close)
