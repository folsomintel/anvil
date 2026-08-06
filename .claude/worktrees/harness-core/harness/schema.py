"""tool specs, the call format, and the parser.

the call format is deliberately not json. a small model matching braces and
quotes fails in ways that are hard to recover from, so calls are flat key/value
lines inside one tagged block:

    <tool>
    name: read_file
    path: harness/workspace.py
    </tool>

multi-line values use a pipe and two-space indentation, which is what
write_file and apply_patch will need:

    <tool>
    name: write_file
    path: a.py
    content: |
      def f():
          return 1
    </tool>
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .errors import ParseError
from .render import ToolResult

OPEN_TAG = "<tool>"
CLOSE_TAG = "</tool>"
BLOCK_RE = re.compile(r"<tool>(.*?)</tool>", re.DOTALL)
KEY_RE = re.compile(r"^([a-z_][a-z0-9_]*):\s?(.*)$")

TRUE_WORDS = frozenset({"true", "yes", "1", "on"})
FALSE_WORDS = frozenset({"false", "no", "0", "off"})


@dataclass(frozen=True)
class Arg:
    name: str
    type: str = "str"  # str | int | bool
    required: bool = False
    default: Any = None
    help: str = ""


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    args: tuple[Arg, ...]
    fn: Callable[..., ToolResult]
    # terminal tools end the episode instead of feeding a result back in
    terminal: bool = False
    # mutating tools invalidate cached state such as tests_passing
    mutates: bool = False

    def signature(self) -> str:
        parts = []
        for arg in self.args:
            parts.append(arg.name if arg.required else f"{arg.name}?")
        return f"{self.name}({', '.join(parts)})"

    def docs(self) -> str:
        lines = [f"{self.signature()} - {self.description}"]
        for arg in self.args:
            flag = "required" if arg.required else "optional"
            detail = f"  {arg.name} ({arg.type}, {flag})"
            if arg.help:
                detail += f": {arg.help}"
            if not arg.required and arg.default is not None:
                detail += f" [default {arg.default}]"
            lines.append(detail)
        return "\n".join(lines)


@dataclass
class ToolCall:
    name: str
    args: dict[str, Any]
    raw: str = ""

    def key(self) -> tuple[str, tuple[tuple[str, Any], ...]]:
        """identity used to detect repeated identical calls."""
        return (self.name, tuple(sorted(self.args.items(), key=lambda kv: kv[0])))


@dataclass
class Registry:
    tools: dict[str, Tool] = field(default_factory=dict)

    def add(self, tool: Tool) -> Tool:
        if tool.name in self.tools:
            raise ValueError(f"duplicate tool: {tool.name}")
        self.tools[tool.name] = tool
        return tool

    def get(self, name: str) -> Tool:
        try:
            return self.tools[name]
        except KeyError:
            known = ", ".join(sorted(self.tools))
            raise ParseError(f"unknown tool: {name}. available: {known}") from None

    def docs(self) -> str:
        return "\n\n".join(self.tools[name].docs() for name in sorted(self.tools))

    def parse(self, text: str) -> ToolCall:
        """parse the model output into a validated call against this registry."""
        call = parse_call(text)
        tool = self.get(call.name)
        return ToolCall(name=tool.name, args=bind_args(tool, call.args), raw=call.raw)


def parse_call(text: str) -> ToolCall:
    """extract the last tool block and its raw string args."""
    blocks = BLOCK_RE.findall(text)
    if not blocks:
        if OPEN_TAG in text:
            raise ParseError(f"tool block was opened but never closed with {CLOSE_TAG}")
        raise ParseError(
            f"no tool call found. emit exactly one {OPEN_TAG} ... {CLOSE_TAG} block"
        )
    raw = blocks[-1]
    fields = _parse_fields(raw)
    name = fields.pop("name", "").strip()
    if not name:
        raise ParseError("tool block is missing a name field")
    return ToolCall(name=name, args=fields, raw=raw.strip())


def _parse_fields(raw: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    lines = raw.split("\n")
    index = 0
    while index < len(lines):
        line = lines[index]
        index += 1
        if not line.strip():
            continue
        match = KEY_RE.match(line.strip())
        if not match:
            raise ParseError(f"cannot parse tool argument line: {line.strip()!r}")
        key, value = match.group(1), match.group(2)
        if value.strip() == "|":
            block, index = _read_block(lines, index)
            fields[key] = block
        else:
            fields[key] = value.strip()
    return fields


def _read_block(lines: list[str], index: int) -> tuple[str, int]:
    """consume indented lines following a pipe marker, stripping two spaces."""
    body: list[str] = []
    while index < len(lines):
        line = lines[index]
        if line.strip() and not line.startswith("  "):
            break
        body.append(line[2:] if line.startswith("  ") else "")
        index += 1
    while body and not body[-1].strip():
        body.pop()
    return "\n".join(body), index


def bind_args(tool: Tool, given: dict[str, str]) -> dict[str, Any]:
    """validate and coerce raw string args against the tool spec."""
    spec = {arg.name: arg for arg in tool.args}
    unknown = sorted(set(given) - set(spec))
    if unknown:
        allowed = ", ".join(sorted(spec)) or "none"
        raise ParseError(
            f"{tool.name} got unknown argument(s): {', '.join(unknown)}. allowed: {allowed}"
        )
    bound: dict[str, Any] = {}
    for arg in tool.args:
        if arg.name not in given or given[arg.name] == "":
            if arg.required:
                raise ParseError(f"{tool.name} is missing required argument: {arg.name}")
            bound[arg.name] = arg.default
            continue
        bound[arg.name] = _coerce(tool.name, arg, given[arg.name])
    return bound


def _coerce(tool_name: str, arg: Arg, value: str) -> Any:
    if arg.type == "str":
        return value
    if arg.type == "int":
        try:
            return int(value.strip())
        except ValueError:
            raise ParseError(
                f"{tool_name}.{arg.name} expects an integer, got {value!r}"
            ) from None
    if arg.type == "bool":
        lowered = value.strip().lower()
        if lowered in TRUE_WORDS:
            return True
        if lowered in FALSE_WORDS:
            return False
        raise ParseError(f"{tool_name}.{arg.name} expects true or false, got {value!r}")
    raise ParseError(f"unsupported argument type: {arg.type}")
