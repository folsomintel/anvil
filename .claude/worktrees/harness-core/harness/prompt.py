"""system prompt construction.

kept in one place because the prompt is training surface, not decoration. every
change here changes the distribution the model is trained on.
"""

from __future__ import annotations

from .schema import Registry

HEADER = """you are working inside a python workspace.

each turn you emit exactly one tool call and nothing else. the harness runs it
and returns a <tool_result> block. repeat until the task is done, then call
finish.

call format:

<tool>
name: <tool name>
<arg>: <value>
</tool>

rules:
- one tool call per turn, no prose outside the block
- paths are workspace-relative, never absolute
- multi-line values use a pipe and two-space indentation
- call finish when the task is done or you cannot make progress

tools:
"""


def build_system_prompt(registry: Registry) -> str:
    return HEADER + "\n" + registry.docs()


def build_task_prompt(task: str) -> str:
    return f"task: {task.strip()}"
