"""workspace snapshots: undo, and the diff the model is shown.

a training episode has to be recoverable. if the model writes garbage into a
file there must be a way back, or the sandbox is burnt and the episode is
wasted. snapshots are plain file copies rather than git, because a generated
task workspace is not a repo and requiring one would be a tax on every task.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass, field
from pathlib import Path

from .workspace import Workspace

# a file bigger than this is not something a 30m model is editing, and copying
# it into every snapshot is pure waste
MAX_SNAPSHOT_BYTES = 512 * 1024


@dataclass
class Snapshot:
    """the text of every tracked file at a point in time."""

    files: dict[str, str] = field(default_factory=dict)

    @classmethod
    def take(cls, workspace: Workspace) -> Snapshot:
        files: dict[str, str] = {}
        for path in _tracked(workspace):
            try:
                if path.stat().st_size > MAX_SNAPSHOT_BYTES:
                    continue
                files[workspace.rel(path)] = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                # binary or unreadable, not something the model edits
                continue
        return cls(files=files)

    def restore(self, workspace: Workspace) -> int:
        """put the workspace back. returns how many files changed."""
        changed = 0
        current = Snapshot.take(workspace)
        for rel, text in self.files.items():
            if current.files.get(rel) != text:
                target = workspace.root / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(text, encoding="utf-8")
                changed += 1
        # files the model created after the snapshot are removed
        for rel in current.files:
            if rel not in self.files:
                (workspace.root / rel).unlink(missing_ok=True)
                changed += 1
        return changed

    def diff(self, workspace: Workspace) -> list[dict[str, object]]:
        """per-file unified diff against the current workspace state."""
        current = Snapshot.take(workspace)
        out: list[dict[str, object]] = []
        for rel in sorted(set(self.files) | set(current.files)):
            before = self.files.get(rel, "")
            after = current.files.get(rel, "")
            if before == after:
                continue
            body = "".join(
                difflib.unified_diff(
                    before.splitlines(keepends=True),
                    after.splitlines(keepends=True),
                    fromfile=rel,
                    tofile=rel,
                    n=1,
                )
            )
            added = sum(1 for line in body.split("\n") if line.startswith("+") and line[1:2] != "+")
            removed = sum(
                1 for line in body.split("\n") if line.startswith("-") and line[1:2] != "-"
            )
            out.append({"path": rel, "added": added, "removed": removed, "diff": body.strip()})
        return out


def _tracked(workspace: Workspace) -> list[Path]:
    found: list[Path] = []
    stack = [workspace.root]
    while stack:
        current = stack.pop()
        try:
            entries = sorted(current.iterdir())
        except (PermissionError, FileNotFoundError):
            continue
        for entry in entries:
            if workspace.is_ignored(entry):
                continue
            if entry.is_symlink():
                continue
            if entry.is_dir():
                stack.append(entry)
            elif entry.is_file():
                found.append(entry)
    return sorted(found)
