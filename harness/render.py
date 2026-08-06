"""normalization layer: everything the model sees is built here.

a small model drowns in raw pytest or lsp output, so tools never return their
own text. they return a ToolResult with a status, a one-line summary, and a
small tree of fields, and this module renders it into the single block format
the model is trained on.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

INDENT = "  "

# hard ceilings so one bad tool call cannot eat the whole context window
MAX_BLOCK_LINES = 200
MAX_LINE_CHARS = 400


class Status:
    OK = "ok"
    FAILED = "failed"
    ERROR = "error"
    REJECTED = "rejected"


@dataclass
class ToolResult:
    """the only thing a tool is allowed to hand back."""

    tool: str
    status: str
    summary: str
    fields: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status == Status.OK

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool": self.tool,
            "status": self.status,
            "summary": self.summary,
            "fields": self.fields,
        }


def render_result(result: ToolResult, max_lines: int = MAX_BLOCK_LINES) -> str:
    lines = [
        "<tool_result>",
        f"tool: {result.tool}",
        f"status: {result.status}",
        f"summary: {result.summary}",
    ]
    body = _render_mapping(result.fields, level=0)
    lines.extend(_truncate(body, max_lines))
    lines.append("</tool_result>")
    return "\n".join(lines)


def _truncate(lines: list[str], max_lines: int) -> list[str]:
    if len(lines) <= max_lines:
        return lines
    dropped = len(lines) - max_lines
    return [*lines[:max_lines], f"... {dropped} more lines truncated"]


def _clip(text: str) -> str:
    if len(text) <= MAX_LINE_CHARS:
        return text
    return text[:MAX_LINE_CHARS] + " ..."


def _render_mapping(mapping: dict[str, Any], level: int) -> list[str]:
    out: list[str] = []
    for key, value in mapping.items():
        out.extend(_render_pair(key, value, level))
    return out


def _render_pair(key: str, value: Any, level: int) -> list[str]:
    pad = INDENT * level
    if value is None:
        return []
    if isinstance(value, dict):
        if not value:
            return []
        return [f"{pad}{key}:", *_render_mapping(value, level + 1)]
    if isinstance(value, (list, tuple)):
        if not value:
            return []
        out = [f"{pad}{key}:"]
        for item in value:
            out.extend(_render_item(item, level + 1))
        return out
    text = "true" if value is True else "false" if value is False else str(value)
    if "\n" in text:
        out = [f"{pad}{key}:"]
        for line in text.split("\n"):
            out.append(f"{pad}{INDENT}{_clip(line)}")
        return out
    return [f"{pad}{key}: {_clip(text)}"]


def _render_item(item: Any, level: int) -> list[str]:
    pad = INDENT * level
    if isinstance(item, dict):
        rendered = _render_mapping(item, level + 1)
        if not rendered:
            return []
        # first field goes on the dash line so entries stay compact
        head = rendered[0].lstrip()
        return [f"{pad}- {head}", *rendered[1:]]
    return [f"{pad}- {_clip(str(item))}"]
