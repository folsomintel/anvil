"""how much of the window a real episode actually costs.

run this after changing the prompt, a tool's output shape, or the limits. if
peak exceeds the window the model never sees the start of its own episode.

    uv run python scripts/budget_report.py
"""

from __future__ import annotations

import pathlib
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from harness import (  # noqa: E402
    Budget,
    Controller,
    ControllerConfig,
    ScriptedModel,
    build_registry,
)
from harness.prompt import build_system_prompt  # noqa: E402

WINDOW = 512


def build_workspace() -> pathlib.Path:
    root = pathlib.Path(tempfile.mkdtemp())
    (root / "pkg").mkdir()
    (root / "tests").mkdir()
    (root / "pkg" / "m.py").write_text(
        "def clamp(v, lo, hi):\n    if v < lo:\n        return lo\n    return v\n",
        encoding="utf-8",
    )
    (root / "tests" / "test_m.py").write_text(
        "import sys, pathlib\n"
        "sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))\n"
        "from pkg.m import clamp\n\n"
        "def test_upper():\n    assert clamp(15, 0, 10) == 10\n",
        encoding="utf-8",
    )
    return root


def call(name: str, **args: object) -> str:
    lines = [f"name: {name}"] + [f"{k}: {v}" for k, v in args.items()]
    return "<tool>\n" + "\n".join(lines) + "\n</tool>"


def main() -> int:
    budget = Budget(window=WINDOW)
    registry = build_registry()

    print(f"window: {WINDOW} tokens, input budget: {budget.input_budget}")
    for label, compact in [("compact", True), ("verbose", False)]:
        cost = budget.count(build_system_prompt(registry, compact=compact))
        share = 100 * cost / WINDOW
        print(f"  {label} system prompt: {cost} tokens ({share:.0f}% of window)")

    patch = (
        "<tool>\nname: apply_patch\npath: pkg/m.py\nfind: |\n      return v\n"
        "replace: |\n      if v > hi:\n          return hi\n      return v\n</tool>"
    )
    model = ScriptedModel(
        [
            call("run_tests"),
            call("read_file", path="pkg/m.py"),
            patch,
            call("run_tests"),
            call("finish", summary="clamped upper bound"),
        ]
    )
    controller = Controller(
        build_workspace(),
        config=ControllerConfig(compact_prompt=True),
        budget=budget,
    )
    episode = controller.run(model, "make the tests pass")

    print("\nper tool result:")
    for step in episode.steps:
        name = step.call["name"] if step.call else "parse"
        result = episode.transcript[3 + step.index * 2]["content"]
        print(f"  {name:<14} {budget.count(result):>4} tokens  ({step.status})")

    peak = 0
    running: list[dict[str, str]] = []
    for message in episode.transcript:
        running.append(message)
        peak = max(peak, budget.tokens(running))
    print(f"\nfull transcript: {budget.tokens(episode.transcript)} tokens")
    print(f"peak if never trimmed: {peak} tokens")
    print(f"turns dropped by the budget: {episode.dropped_turns}")
    print(f"result: {episode.stop_reason}, tests_passing={episode.tests_passing}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
