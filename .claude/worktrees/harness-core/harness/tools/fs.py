"""filesystem tools. slice one is read-only: list_files and read_file."""

from __future__ import annotations

from ..context import Context
from ..render import Status, ToolResult
from ..schema import Arg, Tool


def list_files(ctx: Context, path: str | None = None, depth: int | None = None) -> ToolResult:
    depth = depth if depth is not None else 2
    if depth < 1:
        depth = 1
    start = ctx.workspace.resolve(path)
    entries = ctx.workspace.walk(start, depth)
    limit = ctx.limits.max_files
    shown = entries[:limit]
    listing = [
        ctx.workspace.rel(entry) + ("/" if entry.is_dir() else "") for entry in shown
    ]
    fields: dict[str, object] = {
        "path": ctx.workspace.rel(start),
        "depth": depth,
        "entries": listing,
    }
    summary = f"{len(shown)} entries"
    if len(entries) > limit:
        fields["truncated"] = f"{len(entries) - limit} entries omitted, narrow the path"
        summary += f" of {len(entries)}"
    return ToolResult("list_files", Status.OK, summary, fields)


def read_file(ctx: Context, path: str) -> ToolResult:
    target = ctx.workspace.resolve(path)
    text = ctx.workspace.read_text(target)
    lines = text.split("\n")
    # a trailing newline produces an empty final element that is not a real line
    if lines and lines[-1] == "":
        lines.pop()
    limit = ctx.limits.max_file_lines
    shown = lines[:limit]
    width = len(str(len(shown))) if shown else 1
    numbered = "\n".join(
        f"{str(i).rjust(width)}| {line}" for i, line in enumerate(shown, start=1)
    )
    fields: dict[str, object] = {
        "path": ctx.workspace.rel(target),
        "lines": len(lines),
        "content": numbered,
    }
    summary = f"{len(lines)} lines"
    if len(lines) > limit:
        fields["truncated"] = (
            f"showing lines 1-{limit} of {len(lines)}, use read_range for the rest"
        )
        summary = f"{limit} of {len(lines)} lines"
    return ToolResult("read_file", Status.OK, summary, fields)


LIST_FILES = Tool(
    name="list_files",
    description="list files and directories under a workspace path",
    args=(
        Arg("path", "str", help="workspace-relative directory, defaults to the root"),
        Arg("depth", "int", default=2, help="how many levels to descend"),
    ),
    fn=list_files,
)

READ_FILE = Tool(
    name="read_file",
    description="read a text file with line numbers",
    args=(Arg("path", "str", required=True, help="workspace-relative file path"),),
    fn=read_file,
)

TOOLS = (LIST_FILES, READ_FILE)
