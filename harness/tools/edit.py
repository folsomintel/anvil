"""mutation tools: apply_patch, write_file, diff, revert_changes.

apply_patch takes an exact find/replace pair rather than a unified diff. hunk
headers and line offsets are a lot of bookkeeping for a small model to get
right, and every off-by-one is a wasted step. an exact string match either
lands or fails loudly, and the failure is easy to explain back.

the match must be unique. an ambiguous find is rejected instead of guessing,
because silently patching the first of three occurrences is the kind of bug
that only shows up as a confusing test failure two steps later.
"""

from __future__ import annotations

from ..context import Context
from ..errors import ToolError
from ..render import Status, ToolResult
from ..schema import Arg, Tool

# a diff bigger than this is a rewrite, not a fix, and is worth flagging
LARGE_DIFF_LINES = 60


def _write(ctx: Context, path: str, text: str) -> str:
    target = ctx.workspace.resolve(path)
    if target.is_dir():
        raise ToolError(f"path is a directory: {ctx.workspace.rel(target)}")
    target.parent.mkdir(parents=True, exist_ok=True)
    if text and not text.endswith("\n"):
        text += "\n"
    target.write_text(text, encoding="utf-8")
    return ctx.workspace.rel(target)


def write_file(ctx: Context, path: str, content: str) -> ToolResult:
    existed = ctx.workspace.resolve(path).exists()
    rel = _write(ctx, path, content)
    lines = len(content.split("\n"))
    return ToolResult(
        "write_file",
        Status.OK,
        f"{'overwrote' if existed else 'created'} {rel}, {lines} lines",
        {"path": rel},
    )


def apply_patch(ctx: Context, path: str, find: str, replace: str | None = None) -> ToolResult:
    target = ctx.workspace.resolve(path)
    text = ctx.workspace.read_text(target)
    replace = replace or ""

    occurrences = text.count(find)
    if occurrences == 0:
        return ToolResult(
            "apply_patch",
            Status.ERROR,
            "find text not present",
            {
                "path": ctx.workspace.rel(target),
                "hint": "read the file again, find must match exactly including indentation",
            },
        )
    if occurrences > 1:
        return ToolResult(
            "apply_patch",
            Status.ERROR,
            f"find text appears {occurrences} times",
            {
                "path": ctx.workspace.rel(target),
                "hint": "include more surrounding lines so the match is unique",
            },
        )
    if find == replace:
        return ToolResult(
            "apply_patch",
            Status.REJECTED,
            "patch would change nothing",
            {"hint": "find and replace are identical"},
        )

    rel = _write(ctx, path, text.replace(find, replace, 1))
    removed = len(find.split("\n"))
    added = len(replace.split("\n")) if replace else 0
    return ToolResult(
        "apply_patch",
        Status.OK,
        f"patched {rel}, -{removed} +{added}",
        {"path": rel},
    )


def diff(ctx: Context) -> ToolResult:
    """what has changed since the episode started."""
    if ctx.baseline is None:
        return ToolResult("diff", Status.ERROR, "no baseline snapshot", {})
    changes = ctx.baseline.diff(ctx.workspace)
    if not changes:
        return ToolResult("diff", Status.OK, "no changes", {})
    total = sum(int(change["added"]) + int(change["removed"]) for change in changes)
    fields: dict[str, object] = {}
    body = "\n".join(str(change["diff"]) for change in changes)
    clipped, was_clipped = ctx.clip(body)
    fields["changes"] = clipped
    if was_clipped:
        fields["truncated"] = "diff clipped"
    if total > LARGE_DIFF_LINES:
        # a suspiciously large diff usually means the model rewrote a file it
        # should have patched, which is worth surfacing to the controller
        ctx.state["large_diff"] = True
        fields["warning"] = "this is a large change for a targeted fix"
    return ToolResult(
        "diff", Status.OK, f"{len(changes)} file(s) changed, {total} lines", fields
    )


def revert_changes(ctx: Context) -> ToolResult:
    if ctx.baseline is None:
        return ToolResult("revert_changes", Status.ERROR, "no baseline snapshot", {})
    changed = ctx.baseline.restore(ctx.workspace)
    ctx.state.pop("tests_passing", None)
    ctx.state.pop("large_diff", None)
    return ToolResult(
        "revert_changes",
        Status.OK,
        f"restored {changed} file(s) to the starting state",
        {},
    )


WRITE_FILE = Tool(
    name="write_file",
    description="create a file or replace it whole, prefer apply_patch for edits",
    args=(
        Arg("path", "str", required=True, help="workspace-relative file path"),
        Arg("content", "str", required=True, help="full file contents, use a pipe block"),
    ),
    fn=write_file,
    mutates=True,
)

APPLY_PATCH = Tool(
    name="apply_patch",
    description="replace an exact unique snippet in a file, the default way to edit",
    args=(
        Arg("path", "str", required=True, help="workspace-relative file path"),
        Arg("find", "str", required=True, help="exact text to replace, must be unique"),
        Arg("replace", "str", help="replacement text, empty to delete"),
    ),
    fn=apply_patch,
    mutates=True,
)

DIFF = Tool(
    name="diff",
    description="show what you have changed so far",
    args=(),
    fn=diff,
)

REVERT_CHANGES = Tool(
    name="revert_changes",
    description="undo all your edits and start over",
    args=(),
    fn=revert_changes,
    mutates=True,
)

TOOLS = (APPLY_PATCH, WRITE_FILE, DIFF, REVERT_CHANGES)
