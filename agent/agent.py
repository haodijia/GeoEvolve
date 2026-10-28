"""LangChain MCP agent bootstrap and single-question runner (extracted)."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from agent.answers import extract_answer_from_response
from agent.config import ExperimentConfig, ModelConfig
from agent.io import save_chat_message
from agent.tokens import message_token_usage
from benchmark.dataset import REPO_ROOT, Dataset
from plugins.base import AgentPlugin


def _chat_openai_class() -> Any:
    from langchain_openai import ChatOpenAI

    return ChatOpenAI


def _mcp_and_react_factories() -> tuple[Any, Any]:
    from langchain_mcp_adapters.client import MultiServerMCPClient
    from langgraph.prebuilt import create_react_agent

    return MultiServerMCPClient, create_react_agent


def load_model(config: ModelConfig) -> Any:
    kwargs = {
        "model": config.model_name,
        "api_key": config.resolve_api_key(),
        "temperature": config.temperature,
        "request_timeout": config.request_timeout,
        "max_retries": 0,
    }
    if config.base_url:
        kwargs["base_url"] = config.base_url
    if config.generate_args:
        kwargs["extra_body"] = config.generate_args
    return _chat_openai_class()(**kwargs)


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


def extract_token_usage_from_response(response: dict) -> dict[str, int]:
    totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for message in response["messages"]:
        if message.type != "ai":
            continue
        usage = message_token_usage(message)
        for key in totals:
            totals[key] += usage[key]
    return totals


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
) -> tuple[str, dict, list[dict]]:
    """Run one question and preserve all attempted calls, including failures."""

    qid_str = str(question["question_id"])
    response = {"messages": []}
    failure = None

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

        full_query = f"{config.sys_prompt}\n\n{question_body}"
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

        from langchain_core.messages import HumanMessage

        async def invoke() -> None:
            nonlocal response
            async for state in agent.astream(
                {"messages": [HumanMessage(content=full_query)]},
                config={
                    "recursion_limit": config.recursion_limit,
                    "run_name": f"question{qid_str}/attempt_{attempt}",
                    "metadata": {
                        "question_id": qid_str,
                        "attempt": attempt,
                        "model_role": "actor",
                    },
                },
                stream_mode="values",
            ):
                response = state

        await asyncio.wait_for(invoke(), timeout=config.question_timeout)
        final_answer = extract_answer_from_response(response)
    except Exception as error:
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
        if message.type == "human":
            conversation_log.append({"role": "user", "content": message.content})
        elif message.type == "ai":
            content = []
            if message.content:
                content.append({"type": "text", "content": message.content})
            for call in [*message.tool_calls, *message.invalid_tool_calls]:
                content.append(
                    {"id": call["id"], "name": call["name"], "input": call["args"]}
                )
            conversation_log.append({"role": "assistant", "content": content})
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
                metadata=None,
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
    return final_answer, extract_token_usage_from_response(response), calls
