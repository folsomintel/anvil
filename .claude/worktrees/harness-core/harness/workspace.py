"""the sandbox every tool goes through.

nothing in the harness touches a path directly. tools ask the workspace to
resolve a model-supplied path, and the workspace guarantees the result is inside
the root, symlinks included.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from .errors import WorkspaceError

# directories that are never interesting to the model and would blow up context
IGNORED_DIRS = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        "__pycache__",
        ".venv",
        "venv",
        "node_modules",
        ".mypy_cache",
        ".ruff_cache",
        ".pytest_cache",
        ".tox",
        ".idea",
        ".vscode",
        "dist",
        "build",
        ".eggs",
    }
)

IGNORED_SUFFIXES = frozenset({".pyc", ".pyo", ".so", ".egg-info"})


@dataclass
class Workspace:
    """a rooted, sandboxed view of the directory the model is allowed to touch."""

    root: Path
    ignored_dirs: frozenset[str] = field(default=IGNORED_DIRS)

    def __post_init__(self) -> None:
        self.root = Path(self.root).resolve()
        if not self.root.is_dir():
            raise WorkspaceError(f"workspace root is not a directory: {self.root}")

    def resolve(self, path: str | None) -> Path:
        """turn a model-supplied path into an absolute path inside the root.

        rejects absolute paths, parent traversal, and symlinks that point out.
        """
        raw = (path or ".").strip()
        if raw.startswith("~"):
            raise WorkspaceError("home-relative paths are not allowed")
        candidate = Path(raw)
        if candidate.is_absolute():
            # allow an absolute path only if it is already inside the root
            resolved = candidate.resolve()
        else:
            resolved = (self.root / candidate).resolve()
        if not self._contains(resolved):
            raise WorkspaceError(f"path is outside the workspace: {raw}")
        return resolved

    def _contains(self, resolved: Path) -> bool:
        return resolved == self.root or self.root in resolved.parents

    def rel(self, path: Path) -> str:
        """workspace-relative display path, always posix-style."""
        try:
            return path.resolve().relative_to(self.root).as_posix() or "."
        except ValueError:
            return str(path)

    def is_ignored(self, path: Path) -> bool:
        name = path.name
        if name in self.ignored_dirs:
            return True
        return any(name.endswith(suffix) for suffix in IGNORED_SUFFIXES)

    def read_text(self, path: Path) -> str:
        if not path.exists():
            raise WorkspaceError(f"no such file: {self.rel(path)}")
        if path.is_dir():
            raise WorkspaceError(f"path is a directory, not a file: {self.rel(path)}")
        try:
            return path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            raise WorkspaceError(f"file is not utf-8 text: {self.rel(path)}") from None

    def walk(self, start: Path, depth: int) -> list[Path]:
        """breadth-limited walk, deterministic order, ignored dirs pruned.

        depth 1 lists the immediate children of start.
        """
        if not start.is_dir():
            raise WorkspaceError(f"not a directory: {self.rel(start)}")
        found: list[Path] = []
        frontier = [(start, 0)]
        while frontier:
            current, level = frontier.pop(0)
            if level >= depth:
                continue
            try:
                entries = sorted(current.iterdir(), key=lambda p: (p.is_file(), p.name))
            except PermissionError:
                continue
            for entry in entries:
                if self.is_ignored(entry):
                    continue
                found.append(entry)
                if entry.is_dir() and not entry.is_symlink():
                    frontier.append((entry, level + 1))
        return found

    def env(self) -> dict[str, str]:
        """deterministic environment for subprocesses."""
        base = dict(os.environ)
        base["PYTHONHASHSEED"] = "0"
        base["PYTHONDONTWRITEBYTECODE"] = "1"
        base["NO_COLOR"] = "1"
        base["PY_COLORS"] = "0"
        base["COLUMNS"] = "100"
        return base
