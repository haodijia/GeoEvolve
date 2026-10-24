"""LangChain MCP agent bootstrap and single-question runner (extracted)."""

from __future__ import annotations

import asyncio
import json
import sys

from pathlib import Path
from typing import Any

from rsmem.config import ExperimentConfig
from rsmem.evaluation import extract_answer_from_response
from rsmem.io import save_chat_message
from rsmem.memory import FailureAwareMemory
from rsmem.retrieval import hybrid_retrieval
from rsmem.tokens import message_token_usage


_LAZY = object()

ChatOpenAI: Any = _LAZY
MultiServerMCPClient: Any = _LAZY
create_react_agent: Any = _LAZY


def _chat_openai_class() -> Any:
    cls = globals()["ChatOpenAI"]
    if cls is not _LAZY:
        return cls
    from langchain_openai import ChatOpenAI as Impl

    globals()["ChatOpenAI"] = Impl
    return Impl


def _mcp_and_react_factories() -> tuple[Any, Any]:
    mcc = globals()["MultiServerMCPClient"]
    if mcc is _LAZY:
        from langchain_mcp_adapters.client import MultiServerMCPClient as MCPClient

        globals()["MultiServerMCPClient"] = MCPClient
        mcc = MCPClient
    cra = globals()["create_react_agent"]
    if cra is _LAZY:
        from langgraph.prebuilt import create_react_agent as react_agent

        globals()["create_react_agent"] = react_agent
        cra = react_agent
    return mcc, cra


def load_config(config: ExperimentConfig, output_dir: Path | str) -> tuple[Any, dict]:
    """Build ``(llm, mcp_servers)`` from an :class:`ExperimentConfig`.

    The model is configured entirely from the YAML config: ``model_name``,
    ``base_url`` and ``generate_args`` (forwarded as ``extra_body``). The API key
    is resolved by :meth:`ExperimentConfig.resolve_api_key` (from the YAML
    ``api_key``, falling back to the ``"EMPTY"`` sentinel).

    The agent/MCP JSON at ``config.agent_config_path`` only contributes its
    ``mcpServers`` block (tool servers); it does not configure the model. When no
    such file is given, the agent runs tool-free, which keeps the framework
    importable and runnable without the full perception/analysis tool stack.
    """

    out_root = Path(output_dir).resolve()
    tool_output_dir = out_root / "out"
    tool_output_dir.mkdir(parents=True, exist_ok=True)

    model_name = config.model_name
    base_url = config.base_url
    generate_args = config.generate_args or None
    raw_mcp_servers: dict = {}

    agent_cfg_path = (config.agent_config_path or "").strip()
    if agent_cfg_path and Path(agent_cfg_path).exists():
        with open(agent_cfg_path, encoding="utf-8") as f:
            agent_cfg = json.load(f)
        raw_mcp_servers = agent_cfg.get("mcpServers", {}) or {}

    api_key = config.resolve_api_key()
    llm_kwargs: dict[str, Any] = {
        "model": model_name,
        "api_key": api_key,
        "temperature": 0.1,
        "request_timeout": config.request_timeout,
    }
    if base_url:
        llm_kwargs["base_url"] = base_url
    if generate_args:
        llm_kwargs["extra_body"] = generate_args

    co = _chat_openai_class()
    llm = co(**llm_kwargs)

    mcp_servers: dict = {}
    for server_name, server_config in raw_mcp_servers.items():
        server_args = server_config["args"]
        if "--temp_dir" in server_args:
            raise ValueError("MCP output directories are assigned by the experiment runner")
        updated_args = [*server_args, "--temp_dir", str(tool_output_dir)]

        command = server_config["command"]
        if command in ("python", "python3"):
            command = sys.executable

        import os
        env = {k: v for k, v in os.environ.items()}
        if server_config.get("env"):
            env.update(server_config["env"])

        mcp_servers[server_name] = {
            "command": command,
            "args": updated_args,
            "transport": "stdio",
            "env": env,
        }

    return llm, mcp_servers


async def create_agent(llm, mcp_servers) -> tuple[Any, Any]:
    """Create ReAct agent with MCP tools. Returns ``(agent, client)``."""

    MCPClient, react_agent_factory = _mcp_and_react_factories()
    client = MCPClient(mcp_servers)

    try:
        tools = await client.get_tools()
        agent = react_agent_factory(llm, tools)
        return agent, client
    except Exception:
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
    memory: FailureAwareMemory | None = None,
    kb_embeds: dict | None = None,
    emb_model=None,
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

        retrieved_context = ""
        if config.enable_hkg and kb_embeds and emb_model:
            retrieval_results = hybrid_retrieval(
                query=base_query,
                kb_embeds=kb_embeds,
                emb_model=emb_model,
                top_k=1,
            )
            if retrieval_results:
                result = retrieval_results[0]
                summary = result.get("summary", "")[:1000].strip()
                retrieved_context = (
                    f"**Domain**: {result.get('domain', 'Unknown')}\n"
                    f"**Sub-task**: {result.get('sub_title', 'General')}\n"
                    f"**Core Methodology**: {summary}\n"
                    f"**Execution Rule**: Prioritize the above logic while dynamically adjusting "
                    f"file paths and parameters according to the current environment."
                )

        reflection_block = ""
        if config.enable_far and memory is not None:
            reflection_text = memory.get_latest_reflection(question["question_id"])
            if reflection_text:
                reflection_block = f"""### Previous Attempt Reflection\n{reflection_text}\n"""

        base_question_text = (
            question["auto"] if config.autoplanning else question["instruct"]
        )
        data_path = question["data"]
        question_body = f"### Question\n{base_question_text}\n\n### Data Path\n{data_path}"

        choices = question.get("choices")
        if choices and isinstance(choices, list):
            choices_text = "\n".join(
                f"{chr(ord('A') + i)}. {choice}"
                for i, choice in enumerate(choices)
            )
            question_body += f"\n\n### Options\n{choices_text}"

        full_query = f"{config.resolve_sys_prompt()}\n\n{question_body}"
        if retrieved_context:
            full_query += f"\n\n### Reference Knowledge (Priority Execution)\n{retrieved_context}"
        if reflection_block:
            full_query += f"\n\n{reflection_block}"

        save_chat_message(
            chat_log_path,
            "user",
            full_query,
            qid_str,
            metadata={"retrieved_context": retrieved_context},
        )

        from langchain_core.messages import HumanMessage

        async def invoke() -> None:
            nonlocal response
            async for state in agent.astream(
                {"messages": [HumanMessage(content=full_query)]},
                config={"recursion_limit": 50},
                stream_mode="values",
            ):
                response = state

        await asyncio.wait_for(invoke(), timeout=300)
        final_answer = extract_answer_from_response(response)
    except Exception as error:
        failure = f"Error processing question {qid_str}: {type(error).__name__}: {error}"
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
                content.append({"id": call["id"], "name": call["name"], "input": call["args"]})
            conversation_log.append({"role": "assistant", "content": content})
            save_chat_message(
                chat_log_path,
                "assistant",
                [
                    {"type": "text", "text": message.content},
                    *[
                        {"type": "tool_use", "id": call["id"], "name": call["name"], "input": call["args"]}
                        for call in [*message.tool_calls, *message.invalid_tool_calls]
                    ],
                ],
                qid_str,
                metadata=None,
            )
        elif message.type == "tool":
            conversation_log.append({
                "role": "tool",
                "name": message.name,
                "tool_call_id": message.tool_call_id,
                "content": [{"output": [{"text": str(message.content)}]}],
            })
            save_chat_message(
                chat_log_path,
                "system",
                [{"type": "tool_result", "id": message.tool_call_id,
                  "output": [{"type": "text", "text": str(message.content), "annotations": None, "meta": None}],
                  "name": message.name}],
                qid_str,
                metadata=None,
            )

    _log_info(logger, "Error" if failure else "Chat Content", question["question_id"], conversation_log, final_answer)
    return final_answer, extract_token_usage_from_response(response), calls
