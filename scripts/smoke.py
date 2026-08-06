"""end-to-end smoke run against a throwaway workspace.

prints the exact transcript the model would see. not part of the test suite,
it exists so the prompt and result formats can be eyeballed after a change.

    uv run python scripts/smoke.py
"""

from __future__ import annotations

import pathlib
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from harness import Controller, ScriptedModel  # noqa: E402


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
    model = ScriptedModel(
        [
            call("list_files", depth=2),
            call("read_file", path="pkg/m.py"),
            call("run_tests"),
            call("finish", summary="diagnosed the missing upper bound"),
        ]
    )
    episode = Controller(build_workspace()).run(model, "find out why the clamp test fails")
    print(episode.transcript[0]["content"])
    for message in episode.transcript[1:]:
        print(f"\n--- {message['role']} ---")
        print(message["content"])
    print(
        f"\nstop_reason={episode.stop_reason} "
        f"steps={len(episode.steps)} tests_passing={episode.tests_passing}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
