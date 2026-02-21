"""Tests for TransformerLens hook functions in arth.core.hooks."""

from __future__ import annotations

import pytest
import torch
from torch import Tensor

from arth.core.hooks import ablation_hook, patching_hook, steering_hook


# ---------------------------------------------------------------------------
# ablation_hook
# ---------------------------------------------------------------------------

class TestAblationHook:
    def test_returns_correct_shape(self, sample_direction: Tensor) -> None:
        hook_fn = ablation_hook(sample_direction)
        activation = torch.randn(2, 5, 64)  # (batch, seq, d_model)
        result = hook_fn(activation, hook=None)
        assert result.shape == activation.shape

    def test_removes_direction_component(self, sample_direction: Tensor) -> None:
        hook_fn = ablation_hook(sample_direction)
        activation = torch.randn(2, 5, 64)
        result = hook_fn(activation, hook=None)
        # The result should have near-zero projection onto the direction
        d_norm = sample_direction / sample_direction.norm()
        projection = (result @ d_norm).abs()
        assert projection.max().item() < 1e-5

    @pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
    def test_preserves_dtype(self, sample_direction: Tensor, dtype: torch.dtype) -> None:
        hook_fn = ablation_hook(sample_direction)
        activation = torch.randn(2, 5, 64, dtype=dtype)
        result = hook_fn(activation, hook=None)
        assert result.dtype == dtype

    def test_idempotent(self, sample_direction: Tensor) -> None:
        """Applying the ablation hook twice should give the same result as once."""
        hook_fn = ablation_hook(sample_direction)
        activation = torch.randn(2, 5, 64)
        once = hook_fn(activation, hook=None)
        twice = hook_fn(once, hook=None)
        assert torch.allclose(once, twice, atol=1e-5)

    def test_hook_signature(self, sample_direction: Tensor) -> None:
        """Hook should accept (activation, hook) and return a Tensor."""
        hook_fn = ablation_hook(sample_direction)
        activation = torch.randn(1, 3, 64)
        result = hook_fn(activation, hook="mock_hook_object")
        assert isinstance(result, Tensor)

    def test_does_not_modify_input_in_place(self, sample_direction: Tensor) -> None:
        hook_fn = ablation_hook(sample_direction)
        activation = torch.randn(2, 5, 64)
        original = activation.clone()
        hook_fn(activation, hook=None)
        assert torch.allclose(activation, original)

    def test_zero_direction_raises(self) -> None:
        with pytest.raises(ValueError, match="near-zero norm"):
            ablation_hook(torch.zeros(64))


# ---------------------------------------------------------------------------
# steering_hook
# ---------------------------------------------------------------------------

class TestSteeringHook:
    def test_all_position_adds_to_every_token(self, sample_direction: Tensor) -> None:
        hook_fn = steering_hook(sample_direction, scale=1.0, position="all")
        activation = torch.zeros(2, 5, 64)
        result = hook_fn(activation, hook=None)
        # Every position should now have the direction added
        expected = sample_direction.float()
        for b in range(2):
            for s in range(5):
                assert torch.allclose(result[b, s], expected, atol=1e-5)

    def test_last_position_only_modifies_last_token(self, sample_direction: Tensor) -> None:
        hook_fn = steering_hook(sample_direction, scale=1.0, position="last")
        activation = torch.zeros(2, 5, 64)
        result = hook_fn(activation, hook=None)
        # Non-last positions should be unchanged (zeros)
        assert torch.allclose(result[:, :-1, :], torch.zeros(2, 4, 64), atol=1e-6)
        # Last position should have the vector added
        expected = sample_direction.float()
        for b in range(2):
            assert torch.allclose(result[b, -1], expected, atol=1e-5)

    def test_scale_parameter(self, sample_direction: Tensor) -> None:
        hook_fn_1x = steering_hook(sample_direction, scale=1.0, position="all")
        hook_fn_3x = steering_hook(sample_direction, scale=3.0, position="all")
        activation = torch.zeros(1, 1, 64)
        result_1x = hook_fn_1x(activation, hook=None)
        result_3x = hook_fn_3x(activation, hook=None)
        assert torch.allclose(result_3x, result_1x * 3.0, atol=1e-5)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
    def test_preserves_dtype(self, sample_direction: Tensor, dtype: torch.dtype) -> None:
        hook_fn = steering_hook(sample_direction, scale=1.0, position="all")
        activation = torch.randn(1, 3, 64, dtype=dtype)
        result = hook_fn(activation, hook=None)
        assert result.dtype == dtype

    def test_hook_signature(self, sample_direction: Tensor) -> None:
        hook_fn = steering_hook(sample_direction)
        activation = torch.randn(1, 3, 64)
        result = hook_fn(activation, hook="mock_hook_object")
        assert isinstance(result, Tensor)

    def test_zero_vector_raises(self) -> None:
        with pytest.raises(ValueError, match="near-zero norm"):
            steering_hook(torch.zeros(64))


# ---------------------------------------------------------------------------
# patching_hook
# ---------------------------------------------------------------------------

class TestPatchingHook:
    def test_replaces_activation_entirely(self) -> None:
        clean = torch.ones(2, 5, 64) * 42.0
        hook_fn = patching_hook(clean, layer_idx=3)
        corrupted = torch.randn(2, 5, 64)
        result = hook_fn(corrupted, hook=None)
        assert torch.allclose(result, clean)

    def test_result_independent_of_input(self) -> None:
        clean = torch.randn(1, 3, 64)
        hook_fn = patching_hook(clean)
        result_a = hook_fn(torch.randn(1, 3, 64), hook=None)
        result_b = hook_fn(torch.randn(1, 3, 64), hook=None)
        assert torch.allclose(result_a, result_b)

    def test_hook_signature(self) -> None:
        clean = torch.randn(1, 3, 64)
        hook_fn = patching_hook(clean, layer_idx=0)
        result = hook_fn(torch.randn(1, 3, 64), hook="mock_hook_object")
        assert isinstance(result, Tensor)
