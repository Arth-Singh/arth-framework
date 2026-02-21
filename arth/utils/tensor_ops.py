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
    """
    centered = activations - activations.mean(dim=0, keepdim=True)
    U, S, Vh = torch.linalg.svd(centered, full_matrices=False)
    components = Vh[:n_components]
    # Explained variance proportional to squared singular values
    explained_variance = (S[:n_components] ** 2) / (activations.shape[0] - 1)
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
    """
    d = normalize(direction)
    # dot product along last dim, keep dims for broadcasting
    proj = (vectors @ d).unsqueeze(-1) * d
    return vectors - proj


def cosine_sim(a: Tensor, b: Tensor) -> Tensor:
    """Cosine similarity between tensors.

    Supports both vector-vector and batch comparisons.

    Args:
        a: Tensor of shape (..., d).
        b: Tensor of shape (..., d).

    Returns:
        Cosine similarity scalar or tensor.
    """
    a_norm = torch.nn.functional.normalize(a, dim=-1)
    b_norm = torch.nn.functional.normalize(b, dim=-1)
    return (a_norm * b_norm).sum(dim=-1)


def normalize(v: Tensor) -> Tensor:
    """L2 normalize a vector along the last dimension.

    Args:
        v: Tensor of arbitrary shape.

    Returns:
        L2-normalized tensor with the same shape.
    """
    return torch.nn.functional.normalize(v, dim=-1)
