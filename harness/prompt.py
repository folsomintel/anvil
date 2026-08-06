"""system prompt construction.

kept in one place because the prompt is training surface, not decoration.

there are two modes. verbose is for a large model that has never seen this
format and has to infer it from the docs. compact is for the trained model:
signatures only, no prose, because at a 512 token window the verbose prompt
alone consumes the entire context and leaves nothing for the task. a model
trained on this format does not need the format explained back to it.
"""

from __future__ import annotations

from .schema import Registry

VERBOSE_HEADER = """you are working inside a python workspace.

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

COMPACT_HEADER = "tools:"


def build_system_prompt(registry: Registry, compact: bool = False) -> str:
    if compact:
        signatures = "\n".join(
            registry.tools[name].signature() for name in sorted(registry.tools)
        )
        return f"{COMPACT_HEADER}\n{signatures}"
    return VERBOSE_HEADER + "\n" + registry.docs()


def build_task_prompt(task: str) -> str:
    return f"task: {task.strip()}"
