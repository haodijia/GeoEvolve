"""Count provider-reported usage, including unsuccessful reflection formats."""

import json
from pathlib import Path
from typing import Any

from langchain_core.callbacks import AsyncCallbackHandler

TOKEN_FIELDS = ("prompt_tokens", "completion_tokens", "total_tokens")


def message_token_usage(message: Any) -> dict[str, int]:
    usage = message.usage_metadata
    if usage is None:
        raise ValueError("The model did not report token usage; experiment cost is unknown")
    result = dict(zip(TOKEN_FIELDS, (
        usage["input_tokens"], usage["output_tokens"], usage["total_tokens"],
    )))
    if any(type(value) is not int or value < 0 for value in result.values()):
        raise ValueError(f"Invalid token usage: {result}")
    if result["total_tokens"] != result["prompt_tokens"] + result["completion_tokens"]:
        raise ValueError(f"Inconsistent token usage: {result}")
    return result


class ModelUsageRecorder(AsyncCallbackHandler):
    """Count actual model responses before the agent replaces any messages."""

    def __init__(self) -> None:
        self.messages: list[Any] = []

    async def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        self.messages.extend(
            generation.message
            for generations in response.generations
            for generation in generations
        )

    def totals(self) -> dict[str, int]:
        total = dict.fromkeys(TOKEN_FIELDS, 0)
        for message in self.messages:
            usage = message_token_usage(message)
            for key in TOKEN_FIELDS:
                total[key] += usage[key]
        return total


def sum_usage_ledger(path: Path) -> dict[str, int]:
    total = dict.fromkeys(TOKEN_FIELDS, 0)
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if "error" in row:
            raise ValueError(f"Incomplete model cost in {path}: {row['error']}")
        for key in TOKEN_FIELDS:
            total[key] += row["token_usage"][key]
    return total


class MeteredModel:
    """Persist each returned response before a strategy parses its content."""

    def __init__(self, model: Any, path: Path, question_id: str, role: str):
        self.model = model
        self.path = path
        self.question_id = str(question_id)
        self.role = role

    async def ainvoke(self, messages: list) -> Any:
        row = {"question_id": self.question_id, "role": self.role,
               "error": "Model call did not complete"}
        try:
            response = await self.model.ainvoke(
                messages,
                config={
                    "run_name": f"{self.role}/question{self.question_id}",
                    "metadata": {
                        "question_id": self.question_id,
                        "model_role": self.role,
                    },
                },
            )
            row["token_usage"] = message_token_usage(response)
            del row["error"]
        except Exception as error:
            row["error"] = f"{type(error).__name__}: {error}"
            raise
        finally:
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        return response
