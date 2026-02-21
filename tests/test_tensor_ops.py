"""Tests for tensor operation utilities in arth.utils.tensor_ops."""

from __future__ import annotations

import pytest
import torch
from torch import Tensor

from arth.utils.tensor_ops import (
    cosine_sim,
    difference_in_means,
    normalize,
    pca,
    project_out,
)


# ---------------------------------------------------------------------------
# difference_in_means
# ---------------------------------------------------------------------------

class TestDifferenceInMeans:
    def test_basic_computation(self) -> None:
        a = torch.tensor([[2.0, 4.0], [4.0, 6.0]])  # mean = [3, 5]
        b = torch.tensor([[1.0, 1.0], [3.0, 3.0]])  # mean = [2, 2]
        result = difference_in_means(a, b)
        expected = torch.tensor([1.0, 3.0])
        assert torch.allclose(result, expected)

    def test_different_batch_sizes(self) -> None:
        a = torch.tensor([[6.0, 8.0]])  # mean = [6, 8]
        b = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])  # mean = [3, 4]
        result = difference_in_means(a, b)
        expected = torch.tensor([3.0, 4.0])
        assert torch.allclose(result, expected)

    def test_single_sample_each(self) -> None:
        a = torch.tensor([[5.0, 10.0]])
        b = torch.tensor([[1.0, 2.0]])
        result = difference_in_means(a, b)
        expected = torch.tensor([4.0, 8.0])
        assert torch.allclose(result, expected)

    def test_output_shape(self) -> None:
        a = torch.randn(10, 64)
        b = torch.randn(8, 64)
        result = difference_in_means(a, b)
        assert result.shape == (64,)

    def test_identical_inputs_give_zero(self) -> None:
        a = torch.randn(5, 32)
        result = difference_in_means(a, a)
        assert torch.allclose(result, torch.zeros(32), atol=1e-6)


# ---------------------------------------------------------------------------
# pca
# ---------------------------------------------------------------------------

class TestPCA:
    def test_returns_correct_shapes(self) -> None:
        data = torch.randn(100, 64)
        components, variance = pca(data, n_components=3)
        assert components.shape == (3, 64)
        assert variance.shape == (3,)

    def test_components_are_orthogonal(self) -> None:
        data = torch.randn(100, 32)
        components, _ = pca(data, n_components=3)
        # Inner products between different components should be near zero
        for i in range(3):
            for j in range(i + 1, 3):
                dot = (components[i] @ components[j]).abs().item()
                assert dot < 1e-4, f"Components {i} and {j} are not orthogonal: dot={dot}"

    def test_variance_sorted_descending(self) -> None:
        data = torch.randn(100, 32)
        _, variance = pca(data, n_components=5)
        for i in range(len(variance) - 1):
            assert variance[i].item() >= variance[i + 1].item() - 1e-6

    def test_n_components_parameter(self) -> None:
        data = torch.randn(50, 16)
        components_1, var_1 = pca(data, n_components=1)
        assert components_1.shape == (1, 16)
        assert var_1.shape == (1,)

        components_5, var_5 = pca(data, n_components=5)
        assert components_5.shape == (5, 16)
        assert var_5.shape == (5,)

    def test_single_component_has_unit_norm_approx(self) -> None:
        """The first component from SVD should have approx unit norm (Vh rows are unit)."""
        data = torch.randn(50, 32)
        components, _ = pca(data, n_components=1)
        norm = components[0].norm().item()
        assert abs(norm - 1.0) < 1e-5

    def test_single_sample_raises(self) -> None:
        data = torch.randn(1, 16)
        with pytest.raises(ValueError, match="at least 2"):
            pca(data, n_components=1)

    def test_too_many_components_raises(self) -> None:
        data = torch.randn(5, 16)
        with pytest.raises(ValueError, match="effective rank"):
            pca(data, n_components=100)


# ---------------------------------------------------------------------------
# project_out
# ---------------------------------------------------------------------------

class TestProjectOut:
    def test_removes_direction_component(self) -> None:
        direction = torch.tensor([1.0, 0.0, 0.0])
        vectors = torch.tensor([[3.0, 4.0, 5.0]])
        result = project_out(vectors, direction)
        # x-component should be zero
        assert abs(result[0, 0].item()) < 1e-6
        # y and z should be preserved
        assert abs(result[0, 1].item() - 4.0) < 1e-6
        assert abs(result[0, 2].item() - 5.0) < 1e-6

    def test_result_orthogonal_to_direction(self) -> None:
        direction = torch.randn(64)
        vectors = torch.randn(10, 64)
        result = project_out(vectors, direction)
        d_norm = direction / direction.norm()
        dots = result @ d_norm
        assert dots.abs().max().item() < 1e-5

    def test_idempotent(self) -> None:
        direction = torch.randn(64)
        vectors = torch.randn(5, 64)
        once = project_out(vectors, direction)
        twice = project_out(once, direction)
        assert torch.allclose(once, twice, atol=1e-5)

    def test_batched_input(self) -> None:
        """Works with (..., d_model) shaped inputs like (batch, seq, d_model)."""
        direction = torch.randn(64)
        vectors = torch.randn(2, 5, 64)
        result = project_out(vectors, direction)
        assert result.shape == (2, 5, 64)
        d_norm = direction / direction.norm()
        dots = result @ d_norm
        assert dots.abs().max().item() < 1e-5

    def test_zero_direction_raises(self) -> None:
        """Edge case: zero direction should raise ValueError."""
        direction = torch.zeros(64)
        vectors = torch.randn(3, 64)
        with pytest.raises(ValueError, match="near-zero"):
            project_out(vectors, direction)


# ---------------------------------------------------------------------------
# cosine_sim
# ---------------------------------------------------------------------------

class TestCosineSim:
    def test_identical_vectors_give_one(self) -> None:
        v = torch.randn(64)
        result = cosine_sim(v, v)
        assert abs(result.item() - 1.0) < 1e-5

    def test_orthogonal_vectors_give_zero(self) -> None:
        a = torch.tensor([1.0, 0.0, 0.0])
        b = torch.tensor([0.0, 1.0, 0.0])
        result = cosine_sim(a, b)
        assert abs(result.item()) < 1e-6

    def test_opposite_vectors_give_negative_one(self) -> None:
        v = torch.randn(32)
        result = cosine_sim(v, -v)
        assert abs(result.item() + 1.0) < 1e-5

    def test_batch_computation(self) -> None:
        a = torch.randn(5, 64)
        b = torch.randn(5, 64)
        result = cosine_sim(a, b)
        assert result.shape == (5,)
        # All values should be between -1 and 1
        assert (result >= -1.0 - 1e-5).all()
        assert (result <= 1.0 + 1e-5).all()

    def test_scaled_vectors_same_similarity(self) -> None:
        a = torch.randn(32)
        b = torch.randn(32)
        sim_orig = cosine_sim(a, b)
        sim_scaled = cosine_sim(a * 5.0, b * 0.3)
        assert torch.allclose(sim_orig, sim_scaled, atol=1e-5)

    def test_zero_vector_returns_zero(self) -> None:
        a = torch.zeros(32)
        b = torch.randn(32)
        result = cosine_sim(a, b)
        assert abs(result.item()) < 1e-6


# ---------------------------------------------------------------------------
# normalize
# ---------------------------------------------------------------------------

class TestNormalize:
    def test_result_has_unit_norm(self) -> None:
        v = torch.randn(64)
        result = normalize(v)
        assert abs(result.norm().item() - 1.0) < 1e-5

    def test_preserves_direction(self) -> None:
        v = torch.tensor([3.0, 4.0])
        result = normalize(v)
        expected = torch.tensor([0.6, 0.8])
        assert torch.allclose(result, expected, atol=1e-5)

    def test_batched_normalize(self) -> None:
        v = torch.randn(5, 32)
        result = normalize(v)
        norms = result.norm(dim=-1)
        assert torch.allclose(norms, torch.ones(5), atol=1e-5)

    def test_already_normalized_unchanged(self) -> None:
        v = torch.randn(64)
        v = v / v.norm()
        result = normalize(v)
        assert torch.allclose(result, v, atol=1e-6)
