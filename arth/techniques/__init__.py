"""Technique auto-discovery registry.

Scans subdirectories for modules that export a class inheriting
:class:`~arth.techniques.base.BaseTechnique`.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
from pathlib import Path

from arth.techniques.base import BaseTechnique

_REGISTRY: dict[str, BaseTechnique] | None = None


def _discover_techniques() -> dict[str, BaseTechnique]:
    """Walk sub-packages and instantiate every BaseTechnique subclass."""
    registry: dict[str, BaseTechnique] = {}
    package_dir = Path(__file__).resolve().parent

    for finder, module_name, is_pkg in pkgutil.iter_modules([str(package_dir)]):
        if not is_pkg:
            continue
        full_name = f"arth.techniques.{module_name}"
        try:
            mod = importlib.import_module(full_name)
        except Exception:
            continue

        for _attr_name, obj in inspect.getmembers(mod, inspect.isclass):
            if issubclass(obj, BaseTechnique) and obj is not BaseTechnique:
                instance = obj()
                registry[instance.name] = instance
                break  # one class per sub-package

    return registry


def _ensure_registry() -> dict[str, BaseTechnique]:
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = _discover_techniques()
    return _REGISTRY


def get_technique(name: str) -> BaseTechnique:
    """Return a technique instance by name.

    Args:
        name: Technique identifier (e.g. ``'refusal_direction'``).

    Raises:
        KeyError: If the technique is not found.
    """
    reg = _ensure_registry()
    if name not in reg:
        available = ", ".join(sorted(reg.keys()))
        raise KeyError(f"Unknown technique {name!r}. Available: {available}")
    return reg[name]


def list_techniques() -> dict[str, BaseTechnique]:
    """Return a dict of all discovered technique instances keyed by name."""
    return dict(_ensure_registry())
