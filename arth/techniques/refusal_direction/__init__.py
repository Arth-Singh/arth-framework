"""Refusal direction extraction and ablation (Arditi et al.)."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor

from arth.core.activation_store import ActivationStore
from arth.core.hooks import ablation_hook
from arth.core.models import TechniqueResult
from arth.utils.tensor_ops import difference_in_means, normalize, cosine_sim
from arth.utils.io import save_vector, load_vector
from arth.techniques.base import BaseTechnique


class RefusalDirection(BaseTechnique):
    """Extract and ablate the refusal direction via difference-in-means."""

    @property
    def name(self) -> str:
        return "refusal_direction"

    @property
    def description(self) -> str:
        return "Extract refusal direction via difference-in-means and ablate it to bypass safety"

    def extract(self, backend, dataset, config=None) -> TechniqueResult:
        """Extract the refusal direction from contrast pairs.

        Computes the difference-in-means between harmful and harmless
        activations at every layer, then selects the layer whose direction
        has the largest L2 norm as the strongest refusal signal.
        """
        output_dir = Path(config.output_dir) if config else Path("results")
        layers = config.layers if config else None
        batch_size = config.batch_size if config else 32

        store = ActivationStore(backend, batch_size=batch_size)
        harmful_acts, harmless_acts = store.collect_paired(
            dataset, layers=layers, position="last"
        )

        best_layer = -1
        best_norm = -1.0
        directions: dict[int, Tensor] = {}

        for layer in harmful_acts:
            direction = difference_in_means(harmful_acts[layer], harmless_acts[layer])
            directions[layer] = direction
            norm = direction.norm().item()
            if norm > best_norm:
                best_norm = norm
                best_layer = layer

        refusal_dir = normalize(directions[best_layer])

        # Save the direction
        artifact_path = output_dir / f"refusal_direction_{backend.config.name.replace('/', '_')}.pt"
        save_vector(refusal_dir, artifact_path, metadata={
            "layer": best_layer,
            "norm": best_norm,
            "model": backend.config.name,
            "n_pairs": len(dataset),
        })

        return TechniqueResult(
            technique=self.name,
            model_name=backend.config.name,
            artifact_path=artifact_path,
            metadata={
                "best_layer": best_layer,
                "best_norm": best_norm,
                "all_norms": {l: d.norm().item() for l, d in directions.items()},
            },
        )

    def apply(self, backend, artifact_path, prompts, **kwargs) -> list[dict[str, str]]:
        """Ablate the refusal direction during generation.

        Args:
            backend: ModelBackend instance.
            artifact_path: Path to saved refusal direction.
            prompts: Prompts to test.
            **kwargs:
                weight_orthogonalization (bool): If True, modify model weights
                    directly instead of using runtime hooks.
                max_new_tokens (int): Max tokens for generation.
        """
        direction, metadata = load_vector(artifact_path)
        direction = direction.to(backend.config.device)
        best_layer = metadata.get("layer", 0)
        max_new_tokens = kwargs.get("max_new_tokens", 128)
        weight_orth = kwargs.get("weight_orthogonalization", False)

        # Generate original outputs
        original_outputs = backend.generate(prompts, max_new_tokens=max_new_tokens)

        if weight_orth:
            modified_outputs = self._apply_weight_orthogonalization(
                backend, direction, prompts, max_new_tokens
            )
        else:
            # Use ablation hook on the best layer
            hook_name = f"blocks.{best_layer}.hook_resid_post"
            hook_fn = ablation_hook(direction)
            hooks = [(hook_name, hook_fn)]
            modified_outputs = backend.run_with_hooks(
                prompts, hooks=hooks, max_new_tokens=max_new_tokens
            )

        return [
            {"prompt": p, "original": o, "modified": m}
            for p, o, m in zip(prompts, original_outputs, modified_outputs)
        ]

    def _apply_weight_orthogonalization(
        self, backend, direction: Tensor, prompts: list[str], max_new_tokens: int
    ) -> list[str]:
        """Modify model weights to project out the refusal direction, generate, then restore."""
        backend._ensure_loaded()
        model = backend.model
        d = normalize(direction).float()

        # Store original weights
        original_weights = {}
        for layer_idx in range(backend.n_layers):
            block = model.blocks[layer_idx]
            for name, param in block.named_parameters():
                if "W_O" in name or "W_out" in name:
                    original_weights[(layer_idx, name)] = param.data.clone()
                    # Project out refusal direction from output space: W' = W - (W d) d^T
                    # Works for any (..., d_model) shaped weight where last dim is d_model
                    W = param.data.float()
                    proj = (W @ d).unsqueeze(-1) * d
                    param.data = (W - proj).to(param.dtype)

        try:
            modified_outputs = backend.generate(prompts, max_new_tokens=max_new_tokens)
        finally:
            # Restore original weights
            for (layer_idx, name), orig in original_weights.items():
                block = model.blocks[layer_idx]
                for param_name, param in block.named_parameters():
                    if param_name == name:
                        param.data = orig
                        break

        return modified_outputs
