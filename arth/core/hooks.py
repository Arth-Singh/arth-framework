"""TransformerLens hook functions for activation manipulation."""

from __future__ import annotations

from typing import Callable

import torch
from torch import Tensor


def ablation_hook(direction: Tensor) -> Callable:
    """Return a hook that removes the component along *direction* from the residual stream.

    This is the core operation behind refusal ablation: by projecting out the
    refusal direction, the model loses its ability to refuse.

    The returned function has the TransformerLens hook signature:
    ``(activation, hook) -> modified_activation``.

    Args:
        direction: Tensor of shape ``(d_model,)``.  Normalized internally.

    Returns:
        A hook function suitable for use with ``model.run_with_hooks``
        or ``model.generate(fwd_hooks=...)``.

    Raises:
        ValueError: If *direction* has near-zero norm.
    """
    if direction.norm().item() < 1e-8:
        raise ValueError("Cannot create ablation hook: direction vector has near-zero norm.")
    d_f32 = torch.nn.functional.normalize(direction.float(), dim=-1)

    def hook_fn(activation: Tensor, hook: object) -> Tensor:
        # activation shape: (batch, seq, d_model)
        d = d_f32.to(activation.dtype)
        proj = (activation @ d).unsqueeze(-1) * d
        return activation - proj

    return hook_fn


def steering_hook(
    vector: Tensor, scale: float = 1.0, position: str = "all"
) -> Callable:
    """Return a hook that adds a scaled steering vector to the residual stream.

    This implements activation addition / steering: injecting a direction
    into the residual stream to shift model behavior.

    Args:
        vector: Tensor of shape ``(d_model,)``.
        scale: Multiplier for the vector before addition.
        position: ``"all"`` adds to every token position, ``"last"`` adds
            only to the last token.

    Returns:
        A hook function.
    """
    if vector.norm().item() < 1e-8:
        raise ValueError("Cannot create steering hook: vector has near-zero norm.")
    scaled_f32 = (vector.float() * scale)

    def hook_fn(activation: Tensor, hook: object) -> Tensor:
        scaled = scaled_f32.to(activation.dtype)
        if position == "last":
            activation = activation.clone()
            activation[:, -1, :] = activation[:, -1, :] + scaled
            return activation
        # "all" - add to every position
        return activation + scaled

    return hook_fn


def patching_hook(clean_activation: Tensor, layer_idx: int = 0) -> Callable:
    """Return a hook that replaces the activation with a clean (patched) activation.

    Used in activation patching / causal tracing to measure the causal
    effect of individual activations.

    Args:
        clean_activation: The activation tensor to patch in. Shape should
            match the hooked activation (typically ``(batch, seq, d_model)``).
        layer_idx: Reserved for identification; not used in the hook logic
            itself but can help with debugging.

    Returns:
        A hook function that replaces the activation with *clean_activation*.
    """
    def hook_fn(activation: Tensor, hook: object) -> Tensor:
        return clean_activation

    return hook_fn
