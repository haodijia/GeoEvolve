"""LangChain MCP agent bootstrap and single-question runner (extracted)."""

from __future__ import annotations

import asyncio
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent.answers import extract_answer_from_response, extract_tagged_answer
from agent.config import ExperimentConfig, ModelConfig
from agent.io import save_chat_message
from agent.tokens import ModelUsageRecorder
from agent.model import (
    ProviderChatOpenAI,
    capture_async_response,
    capture_sync_response,
    record_model_calls,
)
from benchmark.dataset import REPO_ROOT, Dataset
from plugins.base import AgentPlugin


def _mcp_and_react_factories() -> tuple[Any, Any]:
    from langchain_mcp_adapters.client import MultiServerMCPClient
    from langgraph.prebuilt import create_react_agent

    return MultiServerMCPClient, create_react_agent


def load_model(config: ModelConfig) -> Any:
    import httpx

    kwargs = {
        "model": config.model_name,
        "api_key": config.resolve_api_key(),
        "temperature": config.temperature,
        "request_timeout": config.request_timeout,
        "max_retries": 0,
        "preserve_reasoning": config.preserve_reasoning,
        "http_client": httpx.Client(event_hooks={"response": [capture_sync_response]}),
        "http_async_client": httpx.AsyncClient(
            event_hooks={"response": [capture_async_response]}
        ),
    }
    if config.base_url:
        kwargs["base_url"] = config.base_url
    if config.generate_args:
        kwargs["extra_body"] = config.generate_args
    return ProviderChatOpenAI(**kwargs)


def load_tool_servers(
    config: ExperimentConfig, tool_output_dir: Path, dataset: Dataset
) -> dict:
    """Bind every MCP tool to one question's isolated workspace."""
    import os

    tool_output_dir = tool_output_dir.resolve()
    configuration = json.loads(
        Path(config.agent_config_path).read_text(encoding="utf-8")
    )
    raw_servers = configuration["mcpServers"]
    if not raw_servers:
        raise ValueError("Earth-Agent requires at least one MCP tool server")
    servers = {}
    for name, server in raw_servers.items():
        if "--temp_dir" in server["args"]:
            raise ValueError("MCP output directories are assigned by the runner")
        environment = dict(os.environ)
        environment.update(server.get("env", {}))
        environment.update(dataset.runtime_environment())
        servers[name] = {
            "command": sys.executable
            if server["command"] == "python"
            else server["command"],
            "args": [*server["args"], "--temp_dir", str(tool_output_dir)],
            "transport": "stdio",
            "env": environment,
            "cwd": str(REPO_ROOT),
        }
    return servers


async def create_agent(llm, mcp_servers) -> tuple[Any, Any]:
    """Create ReAct agent with MCP tools. Returns ``(agent, client)``."""

    MCPClient, react_agent_factory = _mcp_and_react_factories()
    client = MCPClient(mcp_servers)

    try:
        tools = await client.get_tools()
        agent = react_agent_factory(llm, tools)
        return agent, client
    except BaseException:
        if hasattr(client, "close"):
            close_result = client.close()
            if hasattr(close_result, "__await__"):
                await close_result
        raise


def _log_info(logger: Any | None, *args: Any) -> None:
    if logger is None:
        return
    info_fn = getattr(logger, "info", None)
    if callable(info_fn):
        info_fn(*args)


def extract_tool_calls_from_response(response: dict) -> list[dict]:
    calls: list[dict] = []
    pending: dict[str, dict] = {}
    for message in response["messages"]:
        if message.type == "ai":
            for call in [*message.tool_calls, *message.invalid_tool_calls]:
                item = {
                    "id": call["id"],
                    "name": call["name"],
                    "input": call["args"],
                    "output": None,
                }
                calls.append(item)
                pending[call["id"]] = item
        elif message.type == "tool":
            item = pending.pop(message.tool_call_id)
            item["output"] = message.content
            item["status"] = message.status
    return calls


@dataclass
class QuestionOutcome:
    answer: str
    token_usage: dict
    tool_calls: list[dict]
    status: str
    finish_reason: str | None


async def handle_question(
    agent,
    question: dict,
    chat_log_path: Path,
    config: ExperimentConfig,
    plugins: list[AgentPlugin],
    *,
    plugin_lock: asyncio.Lock,
    tool_output_dir: Path,
    attempt: int,
    logger=None,
) -> QuestionOutcome:
    """Run one question and preserve all attempted calls, including failures."""

    qid_str = str(question["question_id"])
    response = {"messages": []}
    failure = None
    status = "completed"
    usage_recorder = ModelUsageRecorder()

    try:
        data_str = (
            question["data"]
            if isinstance(question["data"], str)
            else json.dumps(question["data"], ensure_ascii=False)
        )
        base_query = (
            question["auto"] + data_str
            if config.autoplanning
            else question["instruct"] + data_str
        )

        base_question_text = (
            question["auto"] if config.autoplanning else question["instruct"]
        )
        data_path = question["data"]
        question_body = (
            f"### Question\n{base_question_text}\n\n### Data Path\n{data_path}"
        )

        choices = question.get("choices")
        if choices and isinstance(choices, list):
            choices_text = "\n".join(
                f"{chr(ord('A') + i)}. {choice}" for i, choice in enumerate(choices)
            )
            question_body += f"\n\n### Options\n{choices_text}"

        full_query = question_body
        metadata = {}
        async with plugin_lock:
            for plugin in plugins:
                contribution = await plugin.before_question(question, base_query)
                if contribution.text:
                    full_query += f"\n\n{contribution.text}"
                for key, value in contribution.metadata.items():
                    if key in metadata:
                        raise ValueError(f"Duplicate plugin metadata key: {key}")
                    metadata[key] = value
        full_query += (
            "\n\n### Temporary Workspace\n"
            f"This question's tool output directory is {tool_output_dir.resolve()}. "
            "The output directory is empty at the start of this question attempt. "
            "Generated files are deleted when the attempt ends. "
            "Use the original inputs and this attempt's tool outputs; recreate any "
            "intermediate files needed from earlier attempts."
        )
        save_chat_message(chat_log_path, "user", full_query, qid_str, metadata=metadata)

        from langchain_core.messages import HumanMessage, SystemMessage

        async def invoke() -> None:
            nonlocal response
            async for state in agent.astream(
                {
                    "messages": [
                        SystemMessage(content=config.sys_prompt),
                        HumanMessage(content=full_query),
                    ]
                },
                config={
                    "recursion_limit": config.recursion_limit,
                    "run_name": f"question{qid_str}/attempt_{attempt}",
                    "callbacks": [usage_recorder],
                    "metadata": {
                        "question_id": qid_str,
                        "attempt": attempt,
                        "model_role": "actor",
                    },
                },
                stream_mode="values",
            ):
                response = state

        with record_model_calls(
            Path(chat_log_path).parent / "traces" / "model_calls", qid_str, attempt
        ):
            await asyncio.wait_for(invoke(), timeout=config.question_timeout)
        final_answer = extract_answer_from_response(response)
        if not final_answer.strip():
            status = "empty_response"
        elif final_answer == "Sorry, need more steps to process this request.":
            status = "step_limit"
        elif not extract_tagged_answer(final_answer):
            status = "format_error"
    except Exception as error:
        status = "timeout" if isinstance(error, asyncio.TimeoutError) else "execution_error"
        failure = (
            f"Error processing question {qid_str}: {type(error).__name__}: {error}"
        )
        final_answer = failure
        save_chat_message(
            chat_log_path,
            "system",
            [{"type": "text", "content": failure}],
            qid_str,
            metadata={"error": True},
        )

    calls = extract_tool_calls_from_response(response)
    conversation_log = []
    for message in response["messages"]:
        if message.type == "system":
            conversation_log.append({"role": "system", "content": message.content})
        elif message.type == "human":
            conversation_log.append({"role": "user", "content": message.content})
        elif message.type == "ai":
            content = []
            if message.content:
                content.append({"type": "text", "content": message.content})
            for call in [*message.tool_calls, *message.invalid_tool_calls]:
                content.append(
                    {"id": call["id"], "name": call["name"], "input": call["args"]}
                )
            conversation_log.append({
                "role": "assistant", "content": content,
                "reasoning_content": message.additional_kwargs.get("reasoning_content"),
                "response_metadata": message.response_metadata,
            })
            save_chat_message(
                chat_log_path,
                "assistant",
                [
                    {"type": "text", "text": message.content},
                    *[
                        {
                            "type": "tool_use",
                            "id": call["id"],
                            "name": call["name"],
                            "input": call["args"],
                        }
                        for call in [*message.tool_calls, *message.invalid_tool_calls]
                    ],
                ],
                qid_str,
                metadata={
                    "reasoning_content": message.additional_kwargs.get("reasoning_content"),
                    "response_metadata": message.response_metadata,
                },
            )
        elif message.type == "tool":
            conversation_log.append(
                {
                    "role": "tool",
                    "name": message.name,
                    "tool_call_id": message.tool_call_id,
                    "content": [{"output": [{"text": str(message.content)}]}],
                }
            )
            save_chat_message(
                chat_log_path,
                "system",
                [
                    {
                        "type": "tool_result",
                        "id": message.tool_call_id,
                        "output": [
                            {
                                "type": "text",
                                "text": str(message.content),
                                "annotations": None,
                                "meta": None,
                            }
                        ],
                        "name": message.name,
                    }
                ],
                qid_str,
                metadata=None,
            )

    _log_info(
        logger,
        "Error" if failure else "Chat Content",
        question["question_id"],
        conversation_log,
        final_answer,
    )
    finish_reason = (
        usage_recorder.messages[-1].response_metadata.get("finish_reason")
        if usage_recorder.messages else None
    )
    if status in {"empty_response", "format_error"} and finish_reason == "length":
        status = "token_limit"
    return QuestionOutcome(final_answer, usage_recorder.totals(), calls, status, finish_reason)
