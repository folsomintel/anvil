"""the deterministic agent loop.

the loop itself has no cleverness in it: one model turn produces one tool call,
the tool runs, the normalized result goes back in. all the intelligence the
controller is allowed to have lives in the guards, which watch for the failure
modes small models fall into (looping on an identical call, burning steps on
malformed output, grinding after the tests already pass).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .context import Context, Limits
from .errors import HarnessError, ParseError
from .model import Message, Model
from .prompt import build_system_prompt, build_task_prompt
from .render import Status, ToolResult, render_result
from .schema import Registry, ToolCall
from .tools import build_registry
from .workspace import Workspace


class Stop:
    FINISHED = "finished"
    MAX_STEPS = "max_steps"
    TOO_MANY_ERRORS = "too_many_errors"
    REPEATED_CALLS = "repeated_calls"


@dataclass
class ControllerConfig:
    max_steps: int = 20
    # consecutive parse or tool errors before the episode is abandoned
    max_consecutive_errors: int = 3
    # how many times an identical call is allowed before it is rejected
    max_repeats: int = 2
    # rejections in a row before the episode is abandoned as a loop
    max_rejections: int = 2


@dataclass
class Step:
    index: int
    output: str
    call: dict[str, Any] | None
    result: dict[str, Any]

    @property
    def status(self) -> str:
        return str(self.result["status"])


@dataclass
class Episode:
    task: str
    steps: list[Step] = field(default_factory=list)
    stop_reason: str = Stop.MAX_STEPS
    summary: str = ""
    transcript: list[Message] = field(default_factory=list)
    tests_passing: bool | None = None

    @property
    def finished(self) -> bool:
        return self.stop_reason == Stop.FINISHED

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "stop_reason": self.stop_reason,
            "summary": self.summary,
            "tests_passing": self.tests_passing,
            "steps": [asdict(step) for step in self.steps],
            "transcript": self.transcript,
        }


class Controller:
    def __init__(
        self,
        workspace: Workspace | str | Path,
        registry: Registry | None = None,
        config: ControllerConfig | None = None,
        limits: Limits | None = None,
    ) -> None:
        if not isinstance(workspace, Workspace):
            workspace = Workspace(Path(workspace))
        self.registry = registry or build_registry()
        self.config = config or ControllerConfig()
        self.ctx = Context(workspace=workspace, limits=limits or Limits())

    def run(self, model: Model, task: str) -> Episode:
        messages: list[Message] = [
            {"role": "system", "content": build_system_prompt(self.registry)},
            {"role": "user", "content": build_task_prompt(task)},
        ]
        episode = Episode(task=task, transcript=messages)
        seen: Counter[tuple] = Counter()
        consecutive_errors = 0
        consecutive_rejections = 0

        for index in range(self.config.max_steps):
            output = model.step(messages)
            messages.append({"role": "assistant", "content": output})

            call, result = self._resolve(output, seen)
            episode.steps.append(
                Step(
                    index=index,
                    output=output,
                    call={"name": call.name, "args": call.args} if call else None,
                    result=result.to_dict(),
                )
            )
            messages.append({"role": "tool", "content": render_result(result)})

            if call and result.ok and self.registry.get(call.name).terminal:
                episode.stop_reason = Stop.FINISHED
                episode.summary = str(result.fields.get("task_summary", ""))
                break

            if result.status == Status.REJECTED:
                consecutive_rejections += 1
                if consecutive_rejections >= self.config.max_rejections:
                    episode.stop_reason = Stop.REPEATED_CALLS
                    break
            else:
                consecutive_rejections = 0

            if result.status == Status.ERROR:
                consecutive_errors += 1
                if consecutive_errors >= self.config.max_consecutive_errors:
                    episode.stop_reason = Stop.TOO_MANY_ERRORS
                    break
            else:
                consecutive_errors = 0

        episode.tests_passing = self.ctx.state.get("tests_passing")
        return episode

    def _resolve(
        self, output: str, seen: Counter[tuple]
    ) -> tuple[ToolCall | None, ToolResult]:
        try:
            call = self.registry.parse(output)
        except ParseError as exc:
            return None, ToolResult(
                "parse", Status.ERROR, "invalid tool call", {"error": str(exc)}
            )

        seen[call.key()] += 1
        if seen[call.key()] > self.config.max_repeats:
            return call, ToolResult(
                call.name,
                Status.REJECTED,
                "identical call already made, result will not change",
                {"hint": "use a different tool or different arguments, or call finish"},
            )

        if call.name == "run_tests" and self.ctx.state.get("tests_passing"):
            return call, ToolResult(
                call.name,
                Status.REJECTED,
                "tests already passing",
                {"hint": "call finish with a summary of what you changed"},
            )

        return call, self._execute(call)

    def _execute(self, call: ToolCall) -> ToolResult:
        tool = self.registry.get(call.name)
        if tool.mutates:
            # the workspace is about to change, so any test verdict is stale
            self.ctx.state.pop("tests_passing", None)
        try:
            return tool.fn(self.ctx, **call.args)
        except HarnessError as exc:
            return ToolResult(call.name, Status.ERROR, "tool failed", {"error": str(exc)})
        except Exception as exc:  # a tool bug must not kill the episode
            return ToolResult(
                call.name,
                Status.ERROR,
                "internal tool error",
                {"error": f"{type(exc).__name__}: {exc}"},
            )
