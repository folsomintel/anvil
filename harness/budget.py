"""token accounting and transcript trimming.

the 30m preset has a 512 token window. that is the binding constraint on this
whole design: the system prompt, the task, and every tool result have to share
it. so nothing is measured in lines here, only in tokens, and the transcript is
trimmed to fit before every model call.

the counter is pluggable because the real tokenizer does not exist yet. swap
`Budget(count=tokenizer.count)` in once it does. the default heuristic is
deliberately pessimistic: code tokenizes worse than prose, and a 4096-entry
vocab splits identifiers hard.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from .model import Message

# a 4096 vocab on code lands near 3 chars per token, so this over-counts
# slightly rather than silently overflowing the window
CHARS_PER_TOKEN = 3.0


def estimate_tokens(text: str) -> int:
    return int(len(text) / CHARS_PER_TOKEN) + 1


@dataclass
class Budget:
    """the context window, and the policy for fitting a transcript into it."""

    window: int = 512
    # room left for the model to actually emit a tool call
    reserve_for_output: int = 96
    count: Callable[[str], int] = field(default=estimate_tokens)

    @property
    def input_budget(self) -> int:
        return max(0, self.window - self.reserve_for_output)

    def tokens(self, messages: list[Message]) -> int:
        return sum(self.count(message["content"]) for message in messages)

    def fits(self, messages: list[Message]) -> bool:
        return self.tokens(messages) <= self.input_budget

    def fit(self, messages: list[Message]) -> tuple[list[Message], int]:
        """drop the oldest turns until the transcript fits.

        the system prompt and the task are pinned: without them the model has
        no idea what it is doing. everything else is a sliding window over the
        most recent turns, because the newest tool result is almost always the
        one that matters.
        """
        if not messages:
            return [], 0
        pinned = messages[:2]
        rest = list(messages[2:])
        dropped = 0
        while rest and self.tokens(pinned + rest) > self.input_budget:
            rest.pop(0)
            dropped += 1
        if dropped and rest:
            # tell the model its history was cut, otherwise a truncated
            # transcript looks like the episode just started
            note = {"role": "tool", "content": f"<note>{dropped} earlier turns dropped</note>"}
            rest = [note, *rest]
            while rest and self.tokens(pinned + rest) > self.input_budget:
                rest.pop(1) if len(rest) > 1 else rest.pop(0)
                dropped += 1
        return pinned + rest, dropped

    def clip(self, text: str, max_tokens: int) -> str:
        """hard-clip a single string to a token ceiling."""
        if self.count(text) <= max_tokens:
            return text
        # binary search the character cut that lands under the ceiling
        low, high = 0, len(text)
        while low < high:
            mid = (low + high + 1) // 2
            if self.count(text[:mid]) <= max_tokens:
                low = mid
            else:
                high = mid - 1
        return text[:low]
