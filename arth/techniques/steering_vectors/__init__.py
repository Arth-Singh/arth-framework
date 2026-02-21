"""Steering vectors (RepE / Contrastive Activation Addition)."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor

from arth.core.activation_store import ActivationStore
from arth.core.hooks import steering_hook
from arth.core.models import TechniqueResult
from arth.utils.tensor_ops import difference_in_means, pca, normalize
from arth.utils.io import save_vector, load_vector
from arth.techniques.base import BaseTechnique


class SteeringVectors(BaseTechnique):
    """Compute and apply steering vectors via contrastive activation addition."""

    @property
    def name(self) -> str:
        return "steering_vectors"

    @property
    def description(self) -> str:
        return "Extract steering vectors via PCA on contrast pair activation diffs"

    def extract(self, backend, dataset, config=None) -> TechniqueResult:
        """Compute steering vector from contrast pairs.

        For each layer, compute the activation difference for each pair,
        then use PCA to find the principal steering direction.
        """
        output_dir = Path(config.output_dir) if config else Path("results")
        layers = config.layers if config else None
        batch_size = config.batch_size if config else 32

        store = ActivationStore(backend, batch_size=batch_size)
        harmful_acts, harmless_acts = store.collect_paired(
            dataset, layers=layers, position="last"
        )

        best_layer = -1
        best_variance = -1.0
        vectors: dict[int, Tensor] = {}
        variances: dict[int, float] = {}

        for layer in harmful_acts:
            # Per-pair activation differences
            diffs = harmful_acts[layer] - harmless_acts[layer]
            # PCA to find principal direction
            components, explained_var = pca(diffs, n_components=1)
            steering_dir = components[0]  # (d_model,)
            var_value = explained_var[0].item()

            vectors[layer] = steering_dir
            variances[layer] = var_value

            if var_value > best_variance:
                best_variance = var_value
                best_layer = layer

        steering_vec = normalize(vectors[best_layer])

        artifact_path = output_dir / f"steering_vector_{backend.config.name.replace('/', '_')}.pt"
        save_vector(steering_vec, artifact_path, metadata={
            "layer": best_layer,
            "explained_variance": best_variance,
            "model": backend.config.name,
            "n_pairs": len(dataset),
        })

        return TechniqueResult(
            technique=self.name,
            model_name=backend.config.name,
            artifact_path=artifact_path,
            metadata={
                "best_layer": best_layer,
                "explained_variance": best_variance,
                "all_variances": variances,
            },
        )

    def apply(self, backend, artifact_path, prompts, **kwargs) -> list[dict[str, str]]:
        """Apply steering vector during generation.

        Args:
            backend: ModelBackend instance.
            artifact_path: Path to saved steering vector.
            prompts: Prompts to test.
            **kwargs:
                scale (float): Steering vector multiplier (default 1.0).
                position (str): ``"all"`` or ``"last"`` token position.
                max_new_tokens (int): Max generation tokens.
        """
        vector, metadata = load_vector(artifact_path)
        vector = vector.to(backend.config.device)
        best_layer = metadata.get("layer", 0)
        scale = kwargs.get("scale", 1.0)
        position = kwargs.get("position", "all")
        max_new_tokens = kwargs.get("max_new_tokens", 128)

        # Generate original outputs
        original_outputs = backend.generate(prompts, max_new_tokens=max_new_tokens)

        # Generate with steering
        hook_name = f"blocks.{best_layer}.hook_resid_post"
        hook_fn = steering_hook(vector, scale=scale, position=position)
        hooks = [(hook_name, hook_fn)]
        modified_outputs = backend.run_with_hooks(
            prompts, hooks=hooks, max_new_tokens=max_new_tokens
        )

        return [
            {"prompt": p, "original": o, "modified": m}
            for p, o, m in zip(prompts, original_outputs, modified_outputs)
        ]
