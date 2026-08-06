from __future__ import annotations

from pathlib import Path

import pytest

from harness import Context, Status
from harness.errors import WorkspaceError
from harness.render import render_result
from harness.tools.control import finish
from harness.tools.fs import list_files, read_file
from harness.tools.runtime import run_tests


def test_list_files_is_sorted_and_prunes_noise(ctx: Context):
    result = list_files(ctx, depth=2)
    entries = result.fields["entries"]
    assert "pkg/" in entries
    assert "pkg/math_utils.py" in entries
    assert not any("__pycache__" in entry for entry in entries)
    # the listing must be stable across calls, the loop depends on determinism
    assert entries == list_files(ctx, depth=2).fields["entries"]


def test_list_files_depth_one_stays_shallow(ctx: Context):
    entries = list_files(ctx, depth=1).fields["entries"]
    assert "pkg/" in entries
    assert "pkg/math_utils.py" not in entries


def test_list_files_truncates(ctx: Context):
    ctx.limits.max_files = 2
    result = list_files(ctx)
    assert len(result.fields["entries"]) == 2
    assert "truncated" in result.fields


def test_read_file_numbers_lines(ctx: Context):
    result = read_file(ctx, "pkg/math_utils.py")
    assert result.fields["lines"] == 4
    assert result.fields["content"].splitlines()[0] == "1| def clamp(value, lower, upper):"


def test_read_file_clips_against_the_token_ceiling(ctx: Context):
    ctx.limits.max_result_tokens = 12
    result = read_file(ctx, "pkg/math_utils.py")
    assert ctx.budget.count(result.fields["content"]) <= 12
    assert "read_range" in result.fields["truncated"]


def test_read_file_rejects_a_directory(ctx: Context):
    with pytest.raises(WorkspaceError, match="directory"):
        read_file(ctx, "pkg")


def test_read_file_reports_missing_paths(ctx: Context):
    with pytest.raises(WorkspaceError, match="no such file"):
        read_file(ctx, "pkg/nope.py")


@pytest.mark.parametrize("escape", ["../outside.py", "pkg/../../outside.py", "~/secrets"])
def test_paths_cannot_escape_the_workspace(ctx: Context, escape: str):
    with pytest.raises(WorkspaceError):
        read_file(ctx, escape)


def test_symlink_out_of_the_workspace_is_blocked(ctx: Context, tmp_path: Path):
    outside = tmp_path.parent / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    link = ctx.workspace.root / "link.txt"
    link.symlink_to(outside)
    with pytest.raises(WorkspaceError, match="outside the workspace"):
        read_file(ctx, "link.txt")


def test_run_tests_reports_a_normalized_failure(ctx: Context):
    result = run_tests(ctx)
    assert result.status == Status.FAILED
    assert result.summary == "1 failed, 1 passed"
    failure = result.fields["failure"]
    assert failure["path"].endswith("tests/test_math.py")
    assert failure["test"] == "test_clamp_upper_bound"
    assert failure["line"] == 10
    assert failure["expected"] == "10"
    assert failure["actual"] == "15"
    assert ctx.state["tests_passing"] is False


def test_run_tests_passes_after_the_fix(ctx: Context):
    target = ctx.workspace.root / "pkg" / "math_utils.py"
    target.write_text(
        "def clamp(value, lower, upper):\n"
        "    if value < lower:\n"
        "        return lower\n"
        "    if value > upper:\n"
        "        return upper\n"
        "    return value\n",
        encoding="utf-8",
    )
    result = run_tests(ctx)
    assert result.status == Status.OK
    assert result.summary == "2 passed"
    assert result.fields == {}
    assert ctx.state["tests_passing"] is True


def test_run_tests_filter_narrows_the_run(ctx: Context):
    result = run_tests(ctx, test_filter="lower_bound")
    assert result.status == Status.OK
    assert result.summary.startswith("1 passed")


def test_run_tests_reports_empty_collection(ctx: Context):
    (ctx.workspace.root / "empty").mkdir()
    result = run_tests(ctx, path="empty")
    assert result.status == Status.ERROR
    assert result.summary == "no tests collected"


def test_run_tests_survives_a_collection_error(ctx: Context):
    (ctx.workspace.root / "tests" / "test_broken.py").write_text(
        "import nonexistent_module_xyz\n", encoding="utf-8"
    )
    result = run_tests(ctx)
    assert result.status == Status.FAILED
    rendered = render_result(result)
    assert "nonexistent_module_xyz" in rendered or "output" in rendered


def test_finish_carries_the_summary(ctx: Context):
    ctx.state["tests_passing"] = True
    result = finish(ctx, summary="fixed the upper bound")
    assert result.fields["task_summary"] == "fixed the upper bound"
    assert result.fields["tests_passing"] is True
