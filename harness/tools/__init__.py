"""tool registry assembly.

the surface stays deliberately small. at a 512 token window every signature in
the prompt costs context the model could have spent on the task, so the
structural and lsp groups are a later curriculum stage, not a default.
"""

from __future__ import annotations

from ..schema import Registry
from . import control, edit, fs, runtime

MODULES = (fs, edit, runtime, control)


def build_registry() -> Registry:
    registry = Registry()
    for module in MODULES:
        for tool in module.TOOLS:
            registry.add(tool)
    return registry
