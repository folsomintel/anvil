"""runtime tools. slice one is run_tests only.

pytest output is parsed into structured failures. the model never sees a raw
traceback dump, it sees at most `limits.max_failures` normalized entries.
"""

from __future__ import annotations

import re
import subprocess
import sys

from ..context import Context
from ..render import Status, ToolResult
from ..schema import Arg, Tool

# "FAILED tests/test_math.py::test_upper - AssertionError: assert 15 == 10"
FAILED_RE = re.compile(r"^(?:FAILED|ERROR)\s+(\S+?)(?:::(\S+))?(?:\s+-\s+(.*))?$")
# "--tb=line" location: "/abs/tests/test_math.py:14: AssertionError: assert 15 == 10"
TB_LINE_RE = re.compile(r"^(?P<path>[^:\s][^:]*\.py):(?P<line>\d+):\s*(?P<msg>.*)$")
COUNT_RE = re.compile(
    r"(\d+)\s+(passed|failed|errors?|skipped|xfailed|xpassed|deselected)"
)
ASSERT_RE = re.compile(r"assert\s+(.+?)\s*==\s*(.+?)\s*$")

NO_TESTS_EXIT = 5


def run_tests(
    ctx: Context, path: str | None = None, test_filter: str | None = None
) -> ToolResult:
    target = ctx.workspace.resolve(path)
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "--tb=line",
        "-rf",
        "--color=no",
        "-p",
        "no:cacheprovider",
        "-p",
        "no:randomly",
        ctx.workspace.rel(target),
    ]
    if test_filter:
        cmd.extend(["-k", test_filter])

    try:
        proc = subprocess.run(
            cmd,
            cwd=ctx.workspace.root,
            env=ctx.workspace.env(),
            capture_output=True,
            text=True,
            timeout=ctx.limits.test_timeout,
        )
    except subprocess.TimeoutExpired:
        ctx.state["tests_passing"] = False
        return ToolResult(
            "run_tests",
            Status.ERROR,
            f"timed out after {ctx.limits.test_timeout:.0f}s",
            {"hint": "a test is hanging, narrow the run with path or test_filter"},
        )
    except FileNotFoundError:
        return ToolResult(
            "run_tests", Status.ERROR, "python interpreter not available", {}
        )

    output = (proc.stdout or "") + (proc.stderr or "")
    return _summarize(ctx, output, proc.returncode)


def _summarize(ctx: Context, output: str, code: int) -> ToolResult:
    lines = [line.rstrip() for line in output.split("\n")]
    counts = _counts(lines)
    failures = _failures(lines)[: ctx.limits.max_failures]

    if code == NO_TESTS_EXIT:
        ctx.state["tests_passing"] = False
        return ToolResult(
            "run_tests",
            Status.ERROR,
            "no tests collected",
            {"hint": "check the path, or drop test_filter"},
        )

    summary = _summary_text(counts) or ("passed" if code == 0 else "failed")

    if code == 0:
        ctx.state["tests_passing"] = True
        return ToolResult("run_tests", Status.OK, summary, {})

    ctx.state["tests_passing"] = False
    fields: dict[str, object] = {}
    if failures:
        fields["failures" if len(failures) > 1 else "failure"] = (
            failures if len(failures) > 1 else failures[0]
        )
    else:
        # collection error or an internal pytest problem, keep a small tail
        tail = [line for line in lines if line.strip()][-8:]
        fields["output"] = "\n".join(tail)
    total_failed = counts.get("failed", 0) + counts.get("error", 0) + counts.get("errors", 0)
    if total_failed > len(failures):
        fields["omitted"] = f"{total_failed - len(failures)} more failures not shown"
    return ToolResult("run_tests", Status.FAILED, summary, fields)


def _counts(lines: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for line in reversed(lines):
        found = COUNT_RE.findall(line)
        if found:
            for number, label in found:
                counts[label] = int(number)
            break
    return counts


def _summary_text(counts: dict[str, int]) -> str:
    order = ["failed", "errors", "error", "passed", "skipped", "xfailed", "xpassed"]
    parts = [f"{counts[key]} {key}" for key in order if counts.get(key)]
    return ", ".join(parts)


def _failures(lines: list[str]) -> list[dict[str, object]]:
    """pair short-summary FAILED lines with their --tb=line locations."""
    locations = [
        match.groupdict()
        for line in lines
        if (match := TB_LINE_RE.match(line.strip())) and "::" not in line
    ]
    entries: list[dict[str, object]] = []
    for line in lines:
        stripped = line.strip()
        if not (stripped.startswith("FAILED") or stripped.startswith("ERROR ")):
            continue
        match = FAILED_RE.match(stripped)
        if not match:
            continue
        path, test, message = match.group(1), match.group(2), match.group(3) or ""
        entry: dict[str, object] = {"path": path}
        if test:
            entry["test"] = test
        location = _match_location(locations, path)
        if location:
            entry["line"] = int(location["line"])
            message = message or str(location["msg"])
        if message:
            entry["error"] = message
            expected, actual = _expected_actual(message)
            if expected is not None:
                entry["expected"] = expected
                entry["actual"] = actual
        entries.append(entry)
    return entries


def _match_location(locations: list[dict[str, str]], path: str) -> dict[str, str] | None:
    name = path.split("/")[-1]
    for index, location in enumerate(locations):
        if location["path"].split("/")[-1] == name:
            return locations.pop(index)
    return None


def _expected_actual(message: str) -> tuple[str | None, str | None]:
    """pull the two sides out of a simple equality assertion, if present."""
    match = ASSERT_RE.search(message)
    if not match:
        return None, None
    left, right = match.group(1).strip(), match.group(2).strip()
    # pytest rewrites to `assert <actual> == <expected>`
    if len(left) > 80 or len(right) > 80:
        return None, None
    return right, left


RUN_TESTS = Tool(
    name="run_tests",
    description="run pytest and report normalized failures",
    args=(
        Arg("path", "str", help="file or directory to test, defaults to the root"),
        Arg("test_filter", "str", help="pytest -k expression"),
    ),
    fn=run_tests,
)

TOOLS = (RUN_TESTS,)
