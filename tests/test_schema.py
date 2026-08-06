from __future__ import annotations

import pytest

from harness import build_registry
from harness.errors import ParseError
from harness.schema import parse_call

REGISTRY = build_registry()


def test_parses_simple_call():
    call = REGISTRY.parse("<tool>\nname: read_file\npath: pkg/a.py\n</tool>")
    assert call.name == "read_file"
    assert call.args == {"path": "pkg/a.py"}


def test_ignores_prose_around_the_block():
    call = REGISTRY.parse("thinking about it\n<tool>\nname: list_files\n</tool>\ndone")
    assert call.name == "list_files"
    assert call.args == {"path": None, "depth": 2}


def test_uses_the_last_block_when_several_are_emitted():
    text = "<tool>\nname: list_files\n</tool>\n<tool>\nname: read_file\npath: a.py\n</tool>"
    assert REGISTRY.parse(text).name == "read_file"


def test_coerces_int_args():
    call = REGISTRY.parse("<tool>\nname: list_files\ndepth: 3\n</tool>")
    assert call.args["depth"] == 3


def test_multiline_block_values():
    call = parse_call("<tool>\nname: write_file\ncontent: |\n  def f():\n      return 1\n</tool>")
    assert call.args["content"] == "def f():\n    return 1"


@pytest.mark.parametrize(
    "text",
    [
        "no block at all",
        "<tool>\nname: read_file\npath: a.py",
        "<tool>\npath: a.py\n</tool>",
    ],
)
def test_malformed_blocks_raise(text: str):
    with pytest.raises(ParseError):
        REGISTRY.parse(text)


def test_unknown_tool_lists_the_alternatives():
    with pytest.raises(ParseError, match="available:"):
        REGISTRY.parse("<tool>\nname: rm_rf\n</tool>")


def test_unknown_argument_is_rejected():
    with pytest.raises(ParseError, match="unknown argument"):
        REGISTRY.parse("<tool>\nname: read_file\npath: a.py\nmode: fast\n</tool>")


def test_missing_required_argument_is_rejected():
    with pytest.raises(ParseError, match="missing required"):
        REGISTRY.parse("<tool>\nname: read_file\n</tool>")


def test_bad_int_is_rejected():
    with pytest.raises(ParseError, match="integer"):
        REGISTRY.parse("<tool>\nname: list_files\ndepth: deep\n</tool>")


def test_call_key_is_order_independent():
    a = REGISTRY.parse("<tool>\nname: run_tests\npath: t\ntest_filter: k\n</tool>")
    b = REGISTRY.parse("<tool>\nname: run_tests\ntest_filter: k\npath: t\n</tool>")
    assert a.key() == b.key()
