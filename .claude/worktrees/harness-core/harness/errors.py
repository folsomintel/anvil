"""harness error types.

all tool failures reach the model as normalized tool results, never as python
tracebacks. these exceptions let a tool bail out with a clean message that the
controller renders.
"""


class HarnessError(Exception):
    """base for anything the controller knows how to render."""


class WorkspaceError(HarnessError):
    """path escaped the workspace, or does not exist, or is the wrong type."""


class ToolError(HarnessError):
    """a tool ran and failed in an expected way."""


class ParseError(HarnessError):
    """the model emitted something that is not a valid tool call."""
