"""the object every tool receives as its first argument."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .workspace import Workspace


@dataclass
class Limits:
    """context-budget knobs. tools clamp their own output against these."""

    max_files: int = 200
    max_file_lines: int = 400
    max_failures: int = 5
    test_timeout: float = 120.0


@dataclass
class Context:
    workspace: Workspace
    limits: Limits = field(default_factory=Limits)
    # scratch space shared across tools and the controller, e.g. tests_passing
    state: dict[str, Any] = field(default_factory=dict)
