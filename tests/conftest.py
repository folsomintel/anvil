from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harness import Context, Limits, Workspace  # noqa: E402


@pytest.fixture
def sandbox(tmp_path: Path) -> Path:
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "math_utils.py").write_text(
        "def clamp(value, lower, upper):\n"
        "    if value < lower:\n"
        "        return lower\n"
        "    return value\n",
        encoding="utf-8",
    )
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_math.py").write_text(
        "import sys\n"
        "from pathlib import Path\n"
        "sys.path.insert(0, str(Path(__file__).resolve().parents[1]))\n"
        "from pkg.math_utils import clamp\n"
        "\n"
        "def test_clamp_lower_bound():\n"
        "    assert clamp(-1, 0, 10) == 0\n"
        "\n"
        "def test_clamp_upper_bound():\n"
        "    assert clamp(15, 0, 10) == 10\n",
        encoding="utf-8",
    )
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "junk.pyc").write_text("x", encoding="utf-8")
    return tmp_path


@pytest.fixture
def ctx(sandbox: Path) -> Context:
    return Context(workspace=Workspace(sandbox), limits=Limits())
