"""the model interface the controller drives.

anything with a step(messages) -> str method works. the real model plugs in
here later. two stand-ins ship now so the loop is testable without weights.
"""

from __future__ import annotations

from typing import Protocol

Message = dict[str, str]


class Model(Protocol):
    def step(self, messages: list[Message]) -> str: ...


class ScriptedModel:
    """replays a fixed list of outputs. used by the harness tests."""

    def __init__(self, outputs: list[str]) -> None:
        self.outputs = list(outputs)
        self.calls: list[list[Message]] = []

    def step(self, messages: list[Message]) -> str:
        self.calls.append(list(messages))
        if not self.outputs:
            return "<tool>\nname: finish\nsummary: out of scripted outputs\n</tool>"
        return self.outputs.pop(0)


class StdinModel:
    """a human at the keyboard, for exercising tools by hand."""

    def step(self, messages: list[Message]) -> str:
        print(messages[-1]["content"])
        print("\nemit a tool call, end with a blank line:")
        lines: list[str] = []
        while True:
            try:
                line = input()
            except EOFError:
                break
            if not line.strip() and lines:
                break
            lines.append(line)
        return "\n".join(lines)


def render_transcript(messages: list[Message]) -> str:
    """flatten the transcript for models that want a single string."""
    return "\n\n".join(f"[{m['role']}]\n{m['content']}" for m in messages)
