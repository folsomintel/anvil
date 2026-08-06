from __future__ import annotations

from harness.render import MAX_LINE_CHARS, Status, ToolResult, render_result


def test_renders_the_documented_shape():
    result = ToolResult(
        tool="run_tests",
        status=Status.FAILED,
        summary="1 failed, 7 passed",
        fields={
            "failure": {
                "path": "tests/test_math.py",
                "line": 14,
                "test": "test_clamp_upper_bound",
                "expected": 10,
                "actual": 15,
            }
        },
    )
    assert render_result(result) == (
        "<tool_result>\n"
        "tool: run_tests\n"
        "status: failed\n"
        "summary: 1 failed, 7 passed\n"
        "failure:\n"
        "  path: tests/test_math.py\n"
        "  line: 14\n"
        "  test: test_clamp_upper_bound\n"
        "  expected: 10\n"
        "  actual: 15\n"
        "</tool_result>"
    )


def test_lists_render_as_dashes():
    result = ToolResult("list_files", Status.OK, "2 entries", {"entries": ["a.py", "b/"]})
    assert "entries:\n  - a.py\n  - b/" in render_result(result)


def test_list_of_dicts_stays_compact():
    result = ToolResult(
        "run_tests",
        Status.FAILED,
        "2 failed",
        {"failures": [{"path": "a.py", "line": 1}, {"path": "b.py", "line": 2}]},
    )
    body = render_result(result)
    assert "  - path: a.py\n    line: 1\n  - path: b.py\n    line: 2" in body


def test_empty_and_none_fields_are_dropped():
    result = ToolResult("finish", Status.OK, "done", {"a": None, "b": [], "c": {}, "d": "kept"})
    body = render_result(result).split("\n")
    assert "d: kept" in body
    assert not any(line.startswith(("a:", "b:", "c:")) for line in body)


def test_bools_render_lowercase():
    result = ToolResult("finish", Status.OK, "done", {"tests_passing": False})
    assert "tests_passing: false" in render_result(result)


def test_multiline_values_are_indented():
    result = ToolResult("read_file", Status.OK, "2 lines", {"content": "1| a\n2| b"})
    assert "content:\n  1| a\n  2| b" in render_result(result)


def test_block_line_budget_is_enforced():
    result = ToolResult("read_file", Status.OK, "many", {"entries": [str(i) for i in range(500)]})
    rendered = render_result(result, max_lines=10)
    assert "more lines truncated" in rendered
    # 4 header lines + 10 body lines + the truncation note + the closing tag
    assert len(rendered.split("\n")) == 16


def test_long_lines_are_clipped():
    result = ToolResult("read_file", Status.OK, "wide", {"content": "x" * 5000})
    body = render_result(result).split("\n")
    line = [entry for entry in body if entry.startswith("content:")][0]
    assert len(line) <= MAX_LINE_CHARS + 20
    assert line.endswith("...")
