"""read-side filesystem tools.

everything here clips against the token ceiling, not a line count. in a 512
token window a whole-file read almost never fits, so read_range and search_text
are the primary navigation tools and read_file is the convenience case for
small files.
"""

from __future__ import annotations

from ..context import Context
from ..render import Status, ToolResult
from ..schema import Arg, Tool

# a search hit costs context, so only a handful are ever worth returning
MAX_HITS = 8


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
        "entries": listing,
    }
    summary = f"{len(shown)} entries"
    if len(entries) > limit:
        fields["truncated"] = f"{len(entries) - limit} omitted, narrow the path"
        summary += f" of {len(entries)}"
    return ToolResult("list_files", Status.OK, summary, fields)


def _numbered(lines: list[str], start: int) -> str:
    width = len(str(start + len(lines) - 1)) if lines else 1
    return "\n".join(
        f"{str(i).rjust(width)}| {line}" for i, line in enumerate(lines, start=start)
    )


def _lines_of(ctx: Context, path: str) -> tuple[list[str], str]:
    target = ctx.workspace.resolve(path)
    text = ctx.workspace.read_text(target)
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    return lines, ctx.workspace.rel(target)


def read_file(ctx: Context, path: str) -> ToolResult:
    lines, rel = _lines_of(ctx, path)
    content, clipped = ctx.clip(_numbered(lines, 1))
    fields: dict[str, object] = {"path": rel, "lines": len(lines), "content": content}
    summary = f"{len(lines)} lines"
    if clipped:
        shown = len(content.split("\n"))
        fields["truncated"] = f"showing ~{shown} of {len(lines)} lines, use read_range"
        summary = f"{shown} of {len(lines)} lines"
    return ToolResult("read_file", Status.OK, summary, fields)


def read_range(ctx: Context, path: str, start_line: int, end_line: int) -> ToolResult:
    lines, rel = _lines_of(ctx, path)
    if start_line < 1:
        start_line = 1
    if end_line < start_line:
        end_line = start_line
    window = lines[start_line - 1 : end_line]
    if not window:
        return ToolResult(
            "read_range",
            Status.ERROR,
            f"no such lines, file has {len(lines)}",
            {"path": rel},
        )
    content, clipped = ctx.clip(_numbered(window, start_line))
    fields: dict[str, object] = {"path": rel, "content": content}
    if clipped:
        fields["truncated"] = "range clipped, ask for fewer lines"
    return ToolResult(
        "read_range", Status.OK, f"lines {start_line}-{start_line + len(window) - 1}", fields
    )


def search_text(ctx: Context, query: str, path: str | None = None) -> ToolResult:
    """literal substring search. cheaper for the model to aim than a regex."""
    start = ctx.workspace.resolve(path)
    targets = (
        [start]
        if start.is_file()
        else [p for p in ctx.workspace.walk(start, 8) if p.is_file() and p.suffix == ".py"]
    )
    hits: list[dict[str, object]] = []
    for target in targets:
        try:
            text = ctx.workspace.read_text(target)
        except Exception:
            continue
        for number, line in enumerate(text.split("\n"), start=1):
            if query in line:
                hits.append(
                    {
                        "path": ctx.workspace.rel(target),
                        "line": number,
                        "text": line.strip()[:100],
                    }
                )
                if len(hits) >= MAX_HITS:
                    break
        if len(hits) >= MAX_HITS:
            break
    if not hits:
        return ToolResult("search_text", Status.OK, "no matches", {"query": query})
    return ToolResult("search_text", Status.OK, f"{len(hits)} matches", {"hits": hits})


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

READ_RANGE = Tool(
    name="read_range",
    description="read a line range, use when a file is too big to read whole",
    args=(
        Arg("path", "str", required=True, help="workspace-relative file path"),
        Arg("start_line", "int", required=True, help="first line, 1-indexed"),
        Arg("end_line", "int", required=True, help="last line, inclusive"),
    ),
    fn=read_range,
)

SEARCH_TEXT = Tool(
    name="search_text",
    description="find a literal string in python files",
    args=(
        Arg("query", "str", required=True, help="exact text to find"),
        Arg("path", "str", help="file or directory to search, defaults to the root"),
    ),
    fn=search_text,
)

TOOLS = (LIST_FILES, READ_FILE, READ_RANGE, SEARCH_TEXT)
