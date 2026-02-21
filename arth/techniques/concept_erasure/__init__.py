"""Concept erasure via LEACE projection."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor

from arth.core.activation_store import ActivationStore
from arth.core.models import TechniqueResult
from arth.utils.tensor_ops import normalize
from arth.utils.io import save_vector, load_vector
from arth.techniques.base import BaseTechnique


class ConceptErasure(BaseTechnique):
    """Erase the linear subspace distinguishing harmful from harmless activations (LEACE)."""

    @property
    def name(self) -> str:
        return "concept_erasure"

    @property
    def description(self) -> str:
        return "LEACE-based concept erasure to remove the safety-relevant linear subspace"

    def extract(self, backend, dataset, config=None) -> TechniqueResult:
        """Compute the LEACE projection matrix.

        Uses the closed-form solution: ``P = I - X^T (X X^T)^{-1} X``
        where X is the concept direction matrix distinguishing harmful
        from harmless activations.
        """
        output_dir = Path(config.output_dir) if config else Path("results")
        layers = config.layers if config else None
        batch_size = config.batch_size if config else 32

        store = ActivationStore(backend, batch_size=batch_size)
        harmful_acts, harmless_acts = store.collect_paired(
            dataset, layers=layers, position="last"
        )

        # Find best layer by concept direction magnitude
        best_layer = -1
        best_norm = -1.0
        directions: dict[int, Tensor] = {}

        for layer in harmful_acts:
            # Concept direction: mean difference
            direction = harmful_acts[layer].mean(dim=0) - harmless_acts[layer].mean(dim=0)
            directions[layer] = direction
            norm = direction.norm().item()
            if norm > best_norm:
                best_norm = norm
                best_layer = layer

        # Compute LEACE projection for the best layer
        X = directions[best_layer].unsqueeze(0).float()  # (1, d_model)
        # P = I - X^T (X X^T)^{-1} X
        XXT = X @ X.T  # (1, 1)
        XXT_inv = 1.0 / XXT  # scalar inverse for rank-1
        projection = torch.eye(X.shape[1], device=X.device) - X.T @ XXT_inv @ X  # (d_model, d_model)

        artifact_path = output_dir / f"concept_erasure_{backend.config.name.replace('/', '_')}.pt"
        save_vector(projection, artifact_path, metadata={
            "layer": best_layer,
            "concept_norm": best_norm,
            "model": backend.config.name,
            "n_pairs": len(dataset),
        })

        return TechniqueResult(
            technique=self.name,
            model_name=backend.config.name,
            artifact_path=artifact_path,
            metadata={
                "best_layer": best_layer,
                "concept_norm": best_norm,
                "projection_shape": list(projection.shape),
            },
        )

    def apply(self, backend, artifact_path, prompts, **kwargs) -> list[dict[str, str]]:
        """Apply LEACE projection to the residual stream during generation.

        Args:
            backend: ModelBackend instance.
            artifact_path: Path to saved projection matrix.
            prompts: Prompts to test.
            **kwargs:
                max_new_tokens (int): Max generation tokens.
        """
        projection, metadata = load_vector(artifact_path)
        projection = projection.to(backend.config.device).float()
        best_layer = metadata.get("layer", 0)
        max_new_tokens = kwargs.get("max_new_tokens", 128)

        # Generate original outputs
        original_outputs = backend.generate(prompts, max_new_tokens=max_new_tokens)

        # Hook that applies the LEACE projection
        def leace_hook(activation: Tensor, hook: object) -> Tensor:
            # activation: (batch, seq, d_model)
            original_dtype = activation.dtype
            act_float = activation.float()
            # Apply projection: P @ x for each position
            projected = torch.einsum("ij,...j->...i", projection, act_float)
            return projected.to(original_dtype)

        hook_name = f"blocks.{best_layer}.hook_resid_post"
        hooks = [(hook_name, leace_hook)]
        modified_outputs = backend.run_with_hooks(
            prompts, hooks=hooks, max_new_tokens=max_new_tokens
        )

        return [
            {"prompt": p, "original": o, "modified": m}
            for p, o, m in zip(prompts, original_outputs, modified_outputs)
        ]
