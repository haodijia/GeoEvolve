"""Provider response capture and reasoning-aware chat completion messages."""

from __future__ import annotations

import json
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path
from time import monotonic
from typing import Any
from uuid import uuid4

import httpx
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI
from pydantic import Field


_question: ContextVar[dict | None] = ContextVar("provider_question", default=None)
_call: ContextVar[dict | None] = ContextVar("provider_call", default=None)


@contextmanager
def record_model_calls(directory: Path, question_id: str, attempt: int):
    directory.mkdir(parents=True, exist_ok=True)
    token = _question.set(
        {"directory": directory, "question_id": question_id, "attempt": attempt}
    )
    try:
        yield
    finally:
        _question.reset(token)


def _capture_response(response: httpx.Response) -> None:
    call = _call.get()
    if call is not None:
        call.update(
            request=json.loads(response.request.content),
            http_status=response.status_code,
            response_body=response.text,
        )


async def capture_async_response(response: httpx.Response) -> None:
    await response.aread()
    _capture_response(response)


def capture_sync_response(response: httpx.Response) -> None:
    response.read()
    _capture_response(response)


@contextmanager
def _record_call():
    question = _question.get()
    call = {
        "id": uuid4().hex,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    token = _call.set(call)
    started = monotonic()
    try:
        yield
    except BaseException as error:
        call["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        call["elapsed_seconds"] = monotonic() - started
        _call.reset(token)
        if question is not None:
            call.update(question_id=question["question_id"], attempt=question["attempt"])
            path = question["directory"] / f"question{question['question_id']}.jsonl"
            with path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(call, ensure_ascii=False) + "\n")


class ProviderChatOpenAI(ChatOpenAI):
    """Preserve provider reasoning and explicitly opt into reasoning history."""

    preserve_reasoning: bool = Field(default=False, exclude=True)

    def _get_request_payload(self, input_: Any, *, stop=None, **kwargs: Any) -> dict:
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        if self.preserve_reasoning:
            messages = self._convert_input(input_).to_messages()
            for message, outgoing in zip(messages, payload["messages"]):
                if isinstance(message, AIMessage):
                    reasoning = message.additional_kwargs.get("reasoning_content")
                    if reasoning is not None:
                        outgoing["reasoning"] = reasoning
        return payload

    def _create_chat_result(self, response: Any, generation_info=None):
        raw = response if isinstance(response, dict) else response.model_dump()
        result = super()._create_chat_result(response, generation_info)
        for generation, choice in zip(result.generations, raw["choices"]):
            message = choice["message"]
            # vLLM uses `reasoning` on the wire and maps it to
            # `reasoning_content` for Qwen's chat template.
            for field in ("reasoning", "reasoning_content"):
                if message.get(field) is not None:
                    generation.message.additional_kwargs["reasoning_content"] = message[field]
            generation.generation_info["stop_reason"] = choice.get("stop_reason")
        return result

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        with _record_call():
            return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        with _record_call():
            return await super()._agenerate(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )
