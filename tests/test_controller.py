from __future__ import annotations

from pathlib import Path

from harness import Controller, ControllerConfig, ScriptedModel, Status, Stop


def call(name: str, **args: object) -> str:
    lines = [f"name: {name}"] + [f"{k}: {v}" for k, v in args.items()]
    return "<tool>\n" + "\n".join(lines) + "\n</tool>"


def controller(sandbox: Path, **overrides: object) -> Controller:
    return Controller(sandbox, config=ControllerConfig(**overrides))


def test_happy_path_reads_then_finishes(sandbox: Path):
    model = ScriptedModel(
        [
            call("list_files", depth=1),
            call("read_file", path="pkg/math_utils.py"),
            call("finish", summary="looked at clamp"),
        ]
    )
    episode = controller(sandbox).run(model, "inspect clamp")
    assert episode.stop_reason == Stop.FINISHED
    assert episode.summary == "looked at clamp"
    assert len(episode.steps) == 3
    assert all(step.status == Status.OK for step in episode.steps)


def test_transcript_alternates_assistant_and_tool(sandbox: Path):
    model = ScriptedModel([call("list_files"), call("finish", summary="done")])
    episode = controller(sandbox).run(model, "look around")
    roles = [message["role"] for message in episode.transcript]
    assert roles == ["system", "user", "assistant", "tool", "assistant", "tool"]
    assert episode.transcript[3]["content"].startswith("<tool_result>")


def test_tool_docs_reach_the_system_prompt(sandbox: Path):
    model = ScriptedModel([call("finish", summary="nothing to do")])
    episode = controller(sandbox).run(model, "noop")
    system = episode.transcript[0]["content"]
    for name in ["list_files", "read_file", "run_tests", "finish"]:
        assert name in system


def test_parse_errors_are_returned_not_raised(sandbox: Path):
    model = ScriptedModel(["i think i should look at the files", call("finish", summary="ok")])
    episode = controller(sandbox).run(model, "explore")
    assert episode.steps[0].status == Status.ERROR
    assert "no tool call found" in episode.steps[0].result["fields"]["error"]
    assert episode.stop_reason == Stop.FINISHED


def test_repeated_parse_errors_abandon_the_episode(sandbox: Path):
    model = ScriptedModel(["nope"] * 5)
    episode = controller(sandbox, max_consecutive_errors=3).run(model, "explore")
    assert episode.stop_reason == Stop.TOO_MANY_ERRORS
    assert len(episode.steps) == 3


def test_a_good_call_resets_the_error_streak(sandbox: Path):
    model = ScriptedModel(
        ["nope", "nope", call("list_files"), "nope", "nope", call("finish", summary="ok")]
    )
    episode = controller(sandbox, max_consecutive_errors=3).run(model, "explore")
    assert episode.stop_reason == Stop.FINISHED


def test_identical_calls_are_rejected_then_the_loop_is_cut(sandbox: Path):
    model = ScriptedModel([call("list_files", depth=1)] * 6)
    episode = controller(sandbox, max_repeats=2, max_rejections=2).run(model, "explore")
    statuses = [step.status for step in episode.steps]
    assert statuses == [Status.OK, Status.OK, Status.REJECTED, Status.REJECTED]
    assert episode.stop_reason == Stop.REPEATED_CALLS


def test_a_different_call_clears_the_rejection_streak(sandbox: Path):
    model = ScriptedModel(
        [
            call("list_files", depth=1),
            call("list_files", depth=1),
            call("list_files", depth=1),
            call("read_file", path="pkg/math_utils.py"),
            call("finish", summary="ok"),
        ]
    )
    episode = controller(sandbox, max_repeats=2).run(model, "explore")
    assert episode.stop_reason == Stop.FINISHED


def test_rerunning_passing_tests_is_rejected(sandbox: Path):
    (sandbox / "pkg" / "math_utils.py").write_text(
        "def clamp(value, lower, upper):\n"
        "    if value < lower:\n"
        "        return lower\n"
        "    if value > upper:\n"
        "        return upper\n"
        "    return value\n",
        encoding="utf-8",
    )
    model = ScriptedModel(
        [
            call("run_tests"),
            call("run_tests", test_filter="upper"),
            call("finish", summary="already green"),
        ]
    )
    episode = controller(sandbox).run(model, "fix clamp")
    assert episode.steps[0].status == Status.OK
    assert episode.steps[1].status == Status.REJECTED
    assert "call finish" in episode.steps[1].result["fields"]["hint"]
    assert episode.stop_reason == Stop.FINISHED
    assert episode.tests_passing is True


def test_failing_tests_can_be_rerun(sandbox: Path):
    model = ScriptedModel([call("run_tests"), call("run_tests", test_filter="upper")])
    episode = controller(sandbox, max_steps=2).run(model, "fix clamp")
    assert [step.status for step in episode.steps] == [Status.FAILED, Status.FAILED]
    assert episode.tests_passing is False


def test_step_budget_is_enforced(sandbox: Path):
    model = ScriptedModel([call("read_file", path="pkg/math_utils.py")] * 10)
    episode = controller(sandbox, max_steps=3, max_repeats=99).run(model, "read forever")
    assert len(episode.steps) == 3
    assert episode.stop_reason == Stop.MAX_STEPS
    assert not episode.finished


def test_escaping_the_workspace_is_an_error_not_a_crash(sandbox: Path):
    escape = call("read_file", path="../../../etc/passwd")
    model = ScriptedModel([escape, call("finish", summary="x")])
    episode = controller(sandbox).run(model, "exfiltrate")
    assert episode.steps[0].status == Status.ERROR
    assert "outside the workspace" in episode.steps[0].result["fields"]["error"]


def test_episode_serializes_for_training_data(sandbox: Path):
    model = ScriptedModel([call("list_files"), call("finish", summary="done")])
    episode = controller(sandbox).run(model, "look")
    data = episode.to_dict()
    assert data["stop_reason"] == Stop.FINISHED
    assert data["steps"][0]["call"] == {"name": "list_files", "args": {"path": None, "depth": 2}}
    assert data["transcript"][0]["role"] == "system"


def test_a_full_repair_episode(sandbox: Path):
    """the mvp claim: read, patch, verify green, finish."""
    model = ScriptedModel(
        [
            call("run_tests"),
            call("read_file", path="pkg/math_utils.py"),
            "<tool>\nname: apply_patch\npath: pkg/math_utils.py\nfind: |\n"
            "      return value\nreplace: |\n"
            "      if value > upper:\n          return upper\n      return value\n</tool>",
            call("run_tests"),
            call("finish", summary="clamped the upper bound"),
        ]
    )
    episode = controller(sandbox).run(model, "make the clamp tests pass")
    assert [step.status for step in episode.steps] == [
        Status.FAILED,
        Status.OK,
        Status.OK,
        Status.OK,
        Status.OK,
    ]
    assert episode.stop_reason == Stop.FINISHED
    assert episode.tests_passing is True
    assert episode.changed_files == ["pkg/math_utils.py"]


def test_a_bad_patch_can_be_reverted(sandbox: Path):
    model = ScriptedModel(
        [
            "<tool>\nname: write_file\npath: pkg/math_utils.py\ncontent: |\n  garbage\n</tool>",
            call("run_tests"),
            call("revert_changes"),
            call("run_tests"),
        ]
    )
    episode = controller(sandbox, max_steps=4).run(model, "fix clamp")
    assert episode.steps[1].status == Status.FAILED
    assert episode.steps[2].status == Status.OK
    # back to the original failure, not the garbage one
    assert episode.changed_files == []


def test_the_transcript_is_trimmed_to_the_window(sandbox: Path):
    from harness import Budget

    model = ScriptedModel([call("list_files")] * 4 + [call("finish", summary="ok")])
    ctl = Controller(sandbox, config=ControllerConfig(max_repeats=99), budget=Budget(window=256))
    episode = ctl.run(model, "look around")
    assert episode.dropped_turns > 0
    # whatever was dropped, the model always saw the task
    assert all(len(seen) >= 2 and seen[1]["content"].startswith("task:") for seen in model.calls)


def test_the_loop_is_deterministic(sandbox: Path):
    def episode():
        model = ScriptedModel(
            [call("list_files"), call("read_file", path="pkg/math_utils.py"), call("run_tests")]
        )
        return controller(sandbox, max_steps=3).run(model, "inspect").to_dict()

    assert episode() == episode()
