"""Keep Qwen reasoning turns open until thinking or a tool transition ends."""

from dataclasses import dataclass

import torch

from vllm.logger import init_logger
from vllm.tokenizers import cached_tokenizer_from_config
from vllm.v1.sample.logits_processor.interface import (
    BatchUpdate,
    LogitsProcessor,
    MoveDirectionality,
)


logger = init_logger(__name__)


@dataclass
class ReasoningState:
    thinking: bool
    output_ids: list[int]
    seen: int = 0


class QwenReasoningEOSGuard(LogitsProcessor):
    """Mask model EOS IDs while the current request remains inside thinking."""

    def __init__(self, vllm_config, device, is_pin_memory):
        if vllm_config.speculative_config is not None:
            raise ValueError("The reasoning EOS guard requires non-speculative decoding")
        tokenizer = cached_tokenizer_from_config(vllm_config.model_config)
        ids = []
        for text in ("<think>", "</think>", "<tool_call>"):
            encoded = tokenizer.encode(text, add_special_tokens=False)
            if len(encoded) != 1:
                raise ValueError(f"Expected a single Qwen delimiter token: {text}")
            ids.append(encoded[0])
        self.think_start, self.think_end, self.tool_start = ids
        eos = vllm_config.model_config.try_get_generation_config()["eos_token_id"]
        eos_ids = {eos} if isinstance(eos, int) else set(eos)
        eos_ids.add(tokenizer.eos_token_id)
        self.eos_ids = tuple(sorted(eos_ids))
        self.device = device
        self.pin_memory = is_pin_memory
        self.states: dict[int, ReasoningState] = {}
        self.rows: tuple[int, ...] = ()
        self.indices = (self._tensor([]), self._tensor([]))
        self.negative_infinity = torch.tensor(-float("inf"), device=device)
        logger.info("Reasoning EOS guard enabled: EOS=%s, delimiters=%s", self.eos_ids, ids)

    def _tensor(self, values):
        return torch.tensor(
            values, dtype=torch.int64, device="cpu", pin_memory=self.pin_memory
        ).to(self.device, non_blocking=True)

    def _prompt_is_thinking(self, prompt_ids):
        for token in reversed(prompt_ids):
            if token in (self.think_start, self.think_end, self.tool_start):
                return token == self.think_start
        return False

    def is_argmax_invariant(self):
        return False

    def update_state(self, batch_update: BatchUpdate | None):
        if batch_update is not None:
            for index in batch_update.removed:
                self.states.pop(index, None)
            for index, _, prompt_ids, output_ids in batch_update.added:
                if prompt_ids is None:
                    raise ValueError("The reasoning EOS guard requires prompt token IDs")
                self.states[index] = ReasoningState(
                    self._prompt_is_thinking(prompt_ids), output_ids
                )
            for first, second, direction in batch_update.moved:
                first_state = self.states.pop(first, None)
                second_state = self.states.pop(second, None)
                if first_state is not None:
                    self.states[second] = first_state
                if direction == MoveDirectionality.SWAP and second_state is not None:
                    self.states[first] = second_state

        for state in self.states.values():
            for token in state.output_ids[state.seen:]:
                if token == self.think_start:
                    state.thinking = True
                elif token in (self.think_end, self.tool_start):
                    state.thinking = False
            state.seen = len(state.output_ids)

        rows = tuple(sorted(index for index, state in self.states.items() if state.thinking))
        if rows != self.rows:
            self.rows = rows
            self.indices = (
                self._tensor([row for row in rows for _ in self.eos_ids]),
                self._tensor(list(self.eos_ids) * len(rows)),
            )

    def apply(self, logits):
        if self.rows:
            logits.index_put_(self.indices, self.negative_infinity)
        return logits
