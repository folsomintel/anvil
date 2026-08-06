"""control tools. slice one is finish only."""

from __future__ import annotations

from ..context import Context
from ..render import Status, ToolResult
from ..schema import Arg, Tool


def finish(ctx: Context, summary: str) -> ToolResult:
    return ToolResult(
        "finish",
        Status.OK,
        "episode finished",
        # not named summary, the block already has a summary line
        {"task_summary": summary, "tests_passing": ctx.state.get("tests_passing")},
    )


FINISH = Tool(
    name="finish",
    description="end the episode with a one-line summary of what you did",
    args=(Arg("summary", "str", required=True, help="what you changed and why"),),
    fn=finish,
    terminal=True,
)

TOOLS = (FINISH,)
