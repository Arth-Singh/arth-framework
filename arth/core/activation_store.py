"""Batch activation collection and management."""

from __future__ import annotations

import torch
from torch import Tensor

from arth.core.models import ContrastPair
from arth.core.model_backend import ModelBackend


class ActivationStore:
    """Collect and manage residual stream activations in batches."""

    def __init__(self, backend: ModelBackend, batch_size: int = 32) -> None:
        self.backend = backend
        self.batch_size = batch_size

    @torch.no_grad()
    def collect(
        self,
        prompts: list[str],
        layers: list[int] | None = None,
        position: str = "last",
    ) -> dict[int, Tensor]:
        """Collect activations for all prompts, processing in batches.

        Args:
            prompts: List of input strings.
            layers: Layer indices to collect (``None`` = all).
            position: How to reduce across the sequence dimension:
                - ``"last"``: take the last token position.
                - ``"all"``: keep all positions (concatenated across prompts
                  along batch dim; not recommended for variable-length inputs).
                - ``"mean"``: mean-pool across token positions.

        Returns:
            Dict mapping layer index to tensor of shape
            ``(n_prompts, d_model)`` for ``"last"`` / ``"mean"``, or
            ``(total_tokens, d_model)`` for ``"all"``.
        """
        if position not in ("last", "all", "mean"):
            raise ValueError(f"position must be 'last', 'all', or 'mean', got {position!r}")

        all_activations: dict[int, list[Tensor]] = {}

        for start in range(0, len(prompts), self.batch_size):
            batch = prompts[start : start + self.batch_size]
            batch_acts = self.backend.get_residual_stream(batch, layers=layers)

            for layer, acts in batch_acts.items():
                reduced = self._reduce_position(acts, position)
                all_activations.setdefault(layer, []).append(reduced)

        # Concatenate across batches
        return {
            layer: torch.cat(parts, dim=0)
            for layer, parts in all_activations.items()
        }

    @torch.no_grad()
    def collect_paired(
        self,
        pairs: list[ContrastPair],
        layers: list[int] | None = None,
        position: str = "last",
    ) -> tuple[dict[int, Tensor], dict[int, Tensor]]:
        """Collect activations for harmful and harmless prompts separately.

        Args:
            pairs: List of contrast pairs.
            layers: Layer indices to collect.
            position: Position reduction strategy (see :meth:`collect`).

        Returns:
            Tuple of ``(harmful_activations, harmless_activations)`` dicts,
            each mapping layer index to ``(n_pairs, d_model)`` tensors.
        """
        harmful_prompts = [p.harmful for p in pairs]
        harmless_prompts = [p.harmless for p in pairs]

        harmful_acts = self.collect(harmful_prompts, layers=layers, position=position)
        harmless_acts = self.collect(harmless_prompts, layers=layers, position=position)

        return harmful_acts, harmless_acts

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _reduce_position(activations: Tensor, position: str) -> Tensor:
        """Reduce a ``(batch, seq, d_model)`` tensor along the sequence dim.

        Args:
            activations: Tensor of shape ``(batch, seq, d_model)``.
            position: Reduction mode.

        Returns:
            Reduced tensor.
        """
        if position == "last":
            return activations[:, -1, :]  # (batch, d_model)
        if position == "mean":
            return activations.mean(dim=1)  # (batch, d_model)
        # "all" - flatten batch and seq into one dim
        return activations.reshape(-1, activations.shape[-1])  # (batch*seq, d_model)
