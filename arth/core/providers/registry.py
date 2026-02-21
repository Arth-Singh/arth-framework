"""Provider auto-discovery registry.

Mirrors the technique registry pattern: scans provider modules in this
package and registers every concrete :class:`BaseProvider` subclass.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
from pathlib import Path
from typing import Type

from arth.core.providers.base import BaseProvider

_REGISTRY: dict[str, Type[BaseProvider]] | None = None


def _discover_providers() -> dict[str, Type[BaseProvider]]:
    """Walk sibling modules and collect every concrete BaseProvider subclass."""
    registry: dict[str, Type[BaseProvider]] = {}
    package_dir = Path(__file__).resolve().parent

    for _finder, module_name, _is_pkg in pkgutil.iter_modules([str(package_dir)]):
        if module_name in ("base", "registry"):
            continue
        full_name = f"arth.core.providers.{module_name}"
        try:
            mod = importlib.import_module(full_name)
        except Exception:
            # Provider module may fail to import if optional deps are missing;
            # that is fine -- we skip it and report on demand via get_provider.
            continue

        for _attr_name, obj in inspect.getmembers(mod, inspect.isclass):
            if issubclass(obj, BaseProvider) and obj is not BaseProvider:
                # Instantiate temporarily to read the name property
                try:
                    instance = obj()
                    registry[instance.name] = obj
                except Exception:
                    pass
                break  # one provider class per module

    return registry


def _ensure_registry() -> dict[str, Type[BaseProvider]]:
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = _discover_providers()
    return _REGISTRY


def register_provider(name: str, cls: Type[BaseProvider]) -> None:
    """Manually register a provider class.

    Useful for third-party plugins that add custom providers.

    Args:
        name: Identifier that users specify via ``--provider``.
        cls: A concrete :class:`BaseProvider` subclass.
    """
    reg = _ensure_registry()
    reg[name] = cls


def get_provider(name: str) -> BaseProvider:
    """Instantiate and return a provider by name.

    Args:
        name: Provider identifier (e.g. ``'transformer_lens'``).

    Raises:
        KeyError: If the provider is not found.
    """
    reg = _ensure_registry()
    if name not in reg:
        available = ", ".join(sorted(reg.keys()))
        raise KeyError(
            f"Unknown provider {name!r}. Available: {available or '(none discovered)'}. "
            "Make sure the required dependencies are installed."
        )
    return reg[name]()


def list_providers() -> dict[str, BaseProvider]:
    """Return a dict of all discovered providers (name -> instance)."""
    reg = _ensure_registry()
    result: dict[str, BaseProvider] = {}
    for name, cls in reg.items():
        try:
            result[name] = cls()
        except Exception:
            pass
    return result
