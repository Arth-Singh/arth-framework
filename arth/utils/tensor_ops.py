"""Tensor operation utilities for mechanistic interpretability."""

from __future__ import annotations

import torch
from torch import Tensor


def difference_in_means(activations_a: Tensor, activations_b: Tensor) -> Tensor:
    """Compute mean difference between two sets of activations.

    Used for refusal direction extraction: the mean difference between
    harmful and harmless activations reveals the "refusal direction."

    Args:
        activations_a: Tensor of shape (n_a, d_model).
        activations_b: Tensor of shape (n_b, d_model).

    Returns:
        Tensor of shape (d_model,) representing the mean difference.
    """
    return activations_a.mean(dim=0) - activations_b.mean(dim=0)


def pca(activations: Tensor, n_components: int = 1) -> tuple[Tensor, Tensor]:
    """PCA via SVD on centered activations.

    Args:
        activations: Tensor of shape (n_samples, d_model).
        n_components: Number of principal components to return.

    Returns:
        Tuple of (components, explained_variance) where components has shape
        (n_components, d_model) and explained_variance has shape (n_components,).

    Raises:
        ValueError: If activations has fewer than 2 samples or *n_components*
            exceeds the effective rank.
    """
    if activations.shape[0] < 2:
        raise ValueError("PCA requires at least 2 samples.")

    centered = activations - activations.mean(dim=0, keepdim=True)
    U, S, Vh = torch.linalg.svd(centered, full_matrices=False)

    available = min(S.shape[0], Vh.shape[0])
    if n_components > available:
        raise ValueError(
            f"Requested {n_components} components but data has effective "
            f"rank {available}."
        )

    components = Vh[:n_components]
    dof = max(activations.shape[0] - 1, 1)
    explained_variance = (S[:n_components] ** 2) / dof
    return components, explained_variance


def project_out(vectors: Tensor, direction: Tensor) -> Tensor:
    """Remove the component along *direction* from *vectors*.

    Used for ablation: projecting out the refusal direction from residual
    stream activations removes the model's ability to refuse.

    Args:
        vectors: Tensor of shape (..., d_model).
        direction: Tensor of shape (d_model,).  Will be L2-normalized internally.

    Returns:
        Tensor with the same shape as *vectors*, with the *direction* component removed.

    Raises:
        ValueError: If *direction* has near-zero norm.
    """
    if direction.norm().item() < 1e-8:
        raise ValueError("Cannot project out a near-zero direction vector.")
    d = normalize(direction)
    # dot product along last dim, keep dims for broadcasting
    proj = (vectors @ d).unsqueeze(-1) * d
    return vectors - proj


def cosine_sim(a: Tensor, b: Tensor) -> Tensor:
    """Cosine similarity between tensors.

    Supports both vector-vector and batch comparisons.  Returns 0.0 if
    either input has near-zero norm (instead of NaN).

    Args:
        a: Tensor of shape (..., d).
        b: Tensor of shape (..., d).

    Returns:
        Cosine similarity scalar or tensor.
    """
    eps = 1e-8
    a_norm = a.norm(dim=-1, keepdim=True).clamp(min=eps)
    b_norm = b.norm(dim=-1, keepdim=True).clamp(min=eps)
    return ((a / a_norm) * (b / b_norm)).sum(dim=-1)


def normalize(v: Tensor, eps: float = 1e-8) -> Tensor:
    """L2 normalize a vector along the last dimension.

    Args:
        v: Tensor of arbitrary shape.
        eps: Minimum norm to avoid division by zero.

    Returns:
        L2-normalized tensor with the same shape.
    """
    return torch.nn.functional.normalize(v, dim=-1, eps=eps)
