"""tool registry assembly.

the surface stays deliberately small. new tools are added by appending a module
to MODULES, which is where the fs mutation, structural, and lsp groups will go.
"""

from __future__ import annotations

from ..schema import Registry
from . import control, fs, runtime

MODULES = (fs, runtime, control)


def build_registry() -> Registry:
    registry = Registry()
    for module in MODULES:
        for tool in module.TOOLS:
            registry.add(tool)
    return registry
