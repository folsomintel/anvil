"""the object every tool receives as its first argument."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .budget import Budget
from .snapshot import Snapshot
from .workspace import Workspace


@dataclass
class Limits:
    """output ceilings. measured in tokens, not lines, because the window is.

    max_result_tokens is the important one: it is the most a single tool result
    may contribute to the context. at a 512 token window a result costing 160
    tokens already claims a third of everything the model can see.
    """

    max_files: int = 60
    max_result_tokens: int = 160
    max_failures: int = 2
    test_timeout: float = 120.0


@dataclass
class Context:
    workspace: Workspace
    limits: Limits = field(default_factory=Limits)
    budget: Budget = field(default_factory=Budget)
    # taken at episode start, used by diff and revert_changes
    baseline: Snapshot | None = None
    # scratch space shared across tools and the controller, e.g. tests_passing
    state: dict[str, Any] = field(default_factory=dict)

    def clip(self, text: str) -> tuple[str, bool]:
        """clip a tool payload to the per-result token ceiling."""
        clipped = self.budget.clip(text, self.limits.max_result_tokens)
        return clipped, clipped != text
