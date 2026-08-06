"""drive the harness by hand: uv run python -m harness <workspace> "<task>"

useful for checking what a tool actually returns before any model exists.
"""

from __future__ import annotations

import argparse
import json
import sys

from .controller import Controller
from .model import StdinModel


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="harness")
    parser.add_argument("workspace")
    parser.add_argument("task")
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--json", action="store_true", help="dump the episode as json")
    args = parser.parse_args(argv)

    controller = Controller(args.workspace)
    controller.config.max_steps = args.max_steps
    episode = controller.run(StdinModel(), args.task)

    if args.json:
        print(json.dumps(episode.to_dict(), indent=2))
    else:
        print(f"\nstop_reason: {episode.stop_reason}")
        print(f"steps: {len(episode.steps)}")
        if episode.summary:
            print(f"summary: {episode.summary}")
    return 0 if episode.finished else 1


if __name__ == "__main__":
    sys.exit(main())
