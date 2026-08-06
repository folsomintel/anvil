from __future__ import annotations

from harness import Context, Snapshot, Status
from harness.tools.edit import apply_patch, diff, revert_changes, write_file

FIXED = (
    "def clamp(value, lower, upper):\n"
    "    if value < lower:\n"
    "        return lower\n"
    "    if value > upper:\n"
    "        return upper\n"
    "    return value\n"
)


def baseline(ctx: Context) -> Context:
    ctx.baseline = Snapshot.take(ctx.workspace)
    return ctx


def test_apply_patch_replaces_a_unique_snippet(ctx: Context):
    result = apply_patch(
        baseline(ctx),
        path="pkg/math_utils.py",
        find="    return value",
        replace="    if value > upper:\n        return upper\n    return value",
    )
    assert result.status == Status.OK
    assert (ctx.workspace.root / "pkg" / "math_utils.py").read_text() == FIXED


def test_apply_patch_rejects_a_missing_match(ctx: Context):
    result = apply_patch(baseline(ctx), path="pkg/math_utils.py", find="not here", replace="x")
    assert result.status == Status.ERROR
    assert "not present" in result.summary
    assert "exactly" in result.fields["hint"]


def test_apply_patch_refuses_an_ambiguous_match(ctx: Context):
    # "return" appears more than once, patching the first would be a guess
    result = apply_patch(baseline(ctx), path="pkg/math_utils.py", find="return", replace="pass")
    assert result.status == Status.ERROR
    assert "appears" in result.summary
    assert "unique" in result.fields["hint"]


def test_apply_patch_rejects_a_no_op(ctx: Context):
    # a unique find, so it reaches the no-op check rather than the ambiguity one
    unchanged = "def clamp(value, lower, upper):"
    result = apply_patch(
        baseline(ctx), path="pkg/math_utils.py", find=unchanged, replace=unchanged
    )
    assert result.status == Status.REJECTED
    assert "nothing" in result.summary


def test_apply_patch_can_delete(ctx: Context):
    result = apply_patch(baseline(ctx), path="pkg/math_utils.py", find="    return value\n")
    assert result.status == Status.OK
    assert "return value" not in (ctx.workspace.root / "pkg" / "math_utils.py").read_text()


def test_write_file_creates_and_overwrites(ctx: Context):
    created = write_file(baseline(ctx), path="pkg/new.py", content="x = 1")
    assert "created" in created.summary
    assert (ctx.workspace.root / "pkg" / "new.py").read_text() == "x = 1\n"
    overwritten = write_file(ctx, path="pkg/new.py", content="x = 2")
    assert "overwrote" in overwritten.summary


def test_write_file_cannot_escape_the_workspace(ctx: Context):
    from harness.errors import WorkspaceError

    try:
        write_file(baseline(ctx), path="../evil.py", content="boom")
    except WorkspaceError as exc:
        assert "outside the workspace" in str(exc)
    else:
        raise AssertionError("escape was not blocked")


def test_diff_reports_nothing_before_an_edit(ctx: Context):
    assert diff(baseline(ctx)).summary == "no changes"


def test_diff_shows_the_edit(ctx: Context):
    baseline(ctx)
    apply_patch(ctx, path="pkg/math_utils.py", find="    return value", replace="    return 0")
    result = diff(ctx)
    assert result.status == Status.OK
    assert "1 file(s) changed" in result.summary
    assert "math_utils.py" in result.fields["changes"]
    assert "+    return 0" in result.fields["changes"]


def test_diff_flags_a_suspiciously_large_change(ctx: Context):
    baseline(ctx)
    write_file(ctx, path="pkg/math_utils.py", content="\n".join(f"x{i} = {i}" for i in range(80)))
    result = diff(ctx)
    assert "warning" in result.fields
    assert ctx.state["large_diff"] is True


def test_revert_undoes_everything(ctx: Context):
    baseline(ctx)
    original = (ctx.workspace.root / "pkg" / "math_utils.py").read_text()
    apply_patch(ctx, path="pkg/math_utils.py", find="    return value", replace="    return 0")
    write_file(ctx, path="pkg/junk.py", content="garbage")
    ctx.state["tests_passing"] = False

    result = revert_changes(ctx)
    assert result.status == Status.OK
    assert (ctx.workspace.root / "pkg" / "math_utils.py").read_text() == original
    assert not (ctx.workspace.root / "pkg" / "junk.py").exists()
    assert diff(ctx).summary == "no changes"


def test_revert_without_a_baseline_is_an_error(ctx: Context):
    assert revert_changes(ctx).status == Status.ERROR
