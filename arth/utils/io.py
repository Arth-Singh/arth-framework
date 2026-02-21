"""File I/O utilities for saving and loading tensors and results."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import Tensor


def save_vector(
    tensor: Tensor, path: Path, metadata: dict | None = None
) -> Path:
    """Save a tensor and optional metadata as a ``.pt`` file.

    The file stores a dict with keys ``"tensor"`` and ``"metadata"``.

    Args:
        tensor: The tensor to save.
        path: Destination path (should end in ``.pt``).
        metadata: Optional metadata dict to store alongside the tensor.

    Returns:
        The path the file was written to.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"tensor": tensor, "metadata": metadata or {}}
    torch.save(payload, path)
    return path


def load_vector(path: Path) -> tuple[Tensor, dict]:
    """Load a tensor and metadata from a ``.pt`` file saved by :func:`save_vector`.

    Uses ``weights_only=True`` for security.  Falls back to
    ``weights_only=False`` only for legacy files that contain
    non-standard Python objects, with a warning.

    Args:
        path: Path to the ``.pt`` file.

    Returns:
        Tuple of (tensor, metadata).
    """
    import warnings

    path = Path(path)
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        warnings.warn(
            f"Could not load {path} with weights_only=True (may contain "
            f"non-standard objects). Falling back to weights_only=False. "
            f"Only load files you trust.",
            stacklevel=2,
        )
        payload = torch.load(path, map_location="cpu", weights_only=False)

    return payload["tensor"], payload.get("metadata", {})


def save_results(data: dict, path: Path) -> Path:
    """Save a results dict as a JSON file.

    Handles common non-serializable types (Path, Tensor) by converting them
    to strings or lists.

    Args:
        data: Dictionary to save.
        path: Destination JSON path.

    Returns:
        The path the file was written to.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def _default(obj: object) -> object:
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, Tensor):
            return obj.tolist()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=_default)
    return path


def load_results(path: Path) -> dict:
    """Load a results dict from a JSON file.

    Args:
        path: Path to the JSON file.

    Returns:
        The loaded dictionary.
    """
    path = Path(path)
    with open(path) as f:
        return json.load(f)
