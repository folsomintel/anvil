"""binding a real language model to the controller.

the controller only needs step(messages) -> str. this wraps a raw generate
callable into that, which is the seam where model/generation.py plugs in once
it exists. nothing here imports the model package, so the harness stays
runnable and testable without weights.

expected callable:

    generate(prompt: str, max_new_tokens: int, stop: list[str]) -> str

returning only the continuation, not the prompt. if your generate cannot stop
on a string, pass stop_supported=False and this will cut the output itself.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from .budget import Budget
from .model import Message
from .schema import CLOSE_TAG

Generate = Callable[..., str]


def flatten(messages: list[Message]) -> str:
    """the wire format the model is trained on.

    no role tags. the block markup already says who is speaking: a <tool> block
    is the model, a <tool_result> block is the harness. role labels would be
    pure overhead in a 512 token window.
    """
    return "\n".join(message["content"] for message in messages)


@dataclass
class GenerativeModel:
    """adapts a generate callable to the Model protocol."""

    generate: Generate
    budget: Budget = field(default_factory=Budget)
    stop_supported: bool = True
    # generation is capped well under the window, a tool call is short
    max_new_tokens: int = 96

    def step(self, messages: list[Message]) -> str:
        prompt = flatten(messages) + "\n"
        raw = self.generate(
            prompt,
            max_new_tokens=self.max_new_tokens,
            stop=[CLOSE_TAG] if self.stop_supported else [],
        )
        return self._close(raw)

    def _close(self, raw: str) -> str:
        """make sure the block is terminated.

        a stop sequence usually consumes the closing tag, and a small model
        often just runs out of tokens mid-block. either way the parser needs a
        closing tag, and reconstructing it here is better than burning a step
        on a parse error the model cannot see the cause of.
        """
        text = raw.strip()
        if CLOSE_TAG in text:
            return text[: text.index(CLOSE_TAG) + len(CLOSE_TAG)]
        if "<tool>" in text:
            return text + "\n" + CLOSE_TAG
        return text
