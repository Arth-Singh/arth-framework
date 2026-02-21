"""Sparse Autoencoder (SAE) feature analysis for safety circuits."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from arth.core.activation_store import ActivationStore
from arth.core.models import TechniqueResult
from arth.utils.io import save_vector, load_vector
from arth.techniques.base import BaseTechnique


class SAEAnalysis(BaseTechnique):
    """Identify and manipulate safety-related sparse autoencoder features."""

    @property
    def name(self) -> str:
        return "sae_analysis"

    @property
    def description(self) -> str:
        return "Identify safety-related SAE features and ablate/amplify them"

    def extract(self, backend, dataset, config=None) -> TechniqueResult:
        """Identify SAE features differentially active for harmful vs harmless prompts.

        Requires ``sae-lens`` to be installed for full SAE loading. Falls back
        to a simple autoencoder analysis if sae-lens is unavailable.
        """
        output_dir = Path(config.output_dir) if config else Path("results")
        layers = config.layers if config else None
        batch_size = config.batch_size if config else 32

        store = ActivationStore(backend, batch_size=batch_size)
        harmful_acts, harmless_acts = store.collect_paired(
            dataset, layers=layers, position="last"
        )

        # Try to load SAE via sae-lens
        sae = self._try_load_sae(backend, layers=layers)

        if sae is not None:
            return self._extract_with_sae(
                backend, sae, harmful_acts, harmless_acts, output_dir
            )
        else:
            return self._extract_without_sae(
                backend, harmful_acts, harmless_acts, output_dir
            )

    def _try_load_sae(self, backend, layers: list[int] | None = None) -> Any | None:
        """Attempt to load a sparse autoencoder via sae-lens.

        Tries the first layer in *layers*, falling back to layer 0.
        """
        target_layer = layers[0] if layers else 0
        try:
            from sae_lens import SAE as SAELens
            sae = SAELens.from_pretrained(
                release=f"{backend.config.name}-res-jb",
                sae_id=f"blocks.{target_layer}.hook_resid_post",
                device=backend.config.device,
            )
            return sae
        except ImportError:
            return None
        except Exception:
            return None

    def _extract_with_sae(
        self, backend, sae, harmful_acts, harmless_acts, output_dir: Path
    ) -> TechniqueResult:
        """Extract safety-related features using an SAE from sae-lens."""
        # Pick layer that the SAE was trained on (typically in metadata)
        layer = list(harmful_acts.keys())[0]
        harmful = harmful_acts[layer].float()
        harmless = harmless_acts[layer].float()

        # Encode through SAE to get feature activations
        harmful_features = sae.encode(harmful)   # (n_harmful, n_features)
        harmless_features = sae.encode(harmless)  # (n_harmless, n_features)

        # Find differentially active features
        harmful_mean = harmful_features.mean(dim=0)
        harmless_mean = harmless_features.mean(dim=0)
        diff = harmful_mean - harmless_mean

        # Top features most associated with harmful prompts
        top_k = min(50, diff.shape[0])
        top_values, top_indices = torch.topk(diff.abs(), top_k)
        safety_feature_ids = top_indices.tolist()
        safety_feature_scores = {
            int(idx): diff[idx].item() for idx in top_indices
        }

        artifact_path = output_dir / f"sae_analysis_{backend.config.name.replace('/', '_')}.pt"
        save_vector(
            diff,
            artifact_path,
            metadata={
                "layer": layer,
                "safety_feature_ids": safety_feature_ids,
                "safety_feature_scores": safety_feature_scores,
                "model": backend.config.name,
                "sae_available": True,
            },
        )

        return TechniqueResult(
            technique=self.name,
            model_name=backend.config.name,
            artifact_path=artifact_path,
            metadata={
                "layer": layer,
                "top_safety_features": safety_feature_ids[:10],
                "sae_available": True,
            },
        )

    def _extract_without_sae(
        self, backend, harmful_acts, harmless_acts, output_dir: Path
    ) -> TechniqueResult:
        """Fallback: analyze activation differences without a trained SAE.

        Performs a simple differential activation analysis across dimensions.
        """
        best_layer = -1
        best_diff_norm = -1.0
        all_diffs: dict[int, Tensor] = {}

        for layer in harmful_acts:
            harmful_mean = harmful_acts[layer].float().mean(dim=0)
            harmless_mean = harmless_acts[layer].float().mean(dim=0)
            diff = harmful_mean - harmless_mean
            all_diffs[layer] = diff
            diff_norm = diff.norm().item()
            if diff_norm > best_diff_norm:
                best_diff_norm = diff_norm
                best_layer = layer

        diff = all_diffs[best_layer]
        top_k = min(50, diff.shape[0])
        top_values, top_indices = torch.topk(diff.abs(), top_k)
        safety_dim_ids = top_indices.tolist()

        artifact_path = output_dir / f"sae_analysis_{backend.config.name.replace('/', '_')}.pt"
        save_vector(
            diff,
            artifact_path,
            metadata={
                "layer": best_layer,
                "safety_dim_ids": safety_dim_ids,
                "model": backend.config.name,
                "sae_available": False,
            },
        )

        return TechniqueResult(
            technique=self.name,
            model_name=backend.config.name,
            artifact_path=artifact_path,
            metadata={
                "layer": best_layer,
                "top_safety_dims": safety_dim_ids[:10],
                "sae_available": False,
                "note": "sae-lens not available; using raw activation dimension analysis",
            },
        )

    def apply(self, backend, artifact_path, prompts, **kwargs) -> list[dict[str, str]]:
        """Ablate or amplify safety-related SAE features during generation.

        Args:
            backend: ModelBackend instance.
            artifact_path: Path to saved feature diff vector.
            prompts: Prompts to test.
            **kwargs:
                mode (str): ``"ablate"`` to zero out safety features,
                    ``"amplify"`` to increase them.
                scale (float): Scale factor for amplification.
                max_new_tokens (int): Max generation tokens.
        """
        diff_vector, metadata = load_vector(artifact_path)
        diff_vector = diff_vector.to(backend.config.device).float()
        layer = metadata.get("layer", 0)
        mode = kwargs.get("mode", "ablate")
        scale = kwargs.get("scale", 1.0)
        max_new_tokens = kwargs.get("max_new_tokens", 128)

        # Generate original outputs
        original_outputs = backend.generate(prompts, max_new_tokens=max_new_tokens)

        # Build manipulation hook
        if metadata.get("sae_available", False):
            safety_ids = metadata.get("safety_feature_ids", [])
        else:
            safety_ids = metadata.get("safety_dim_ids", [])

        def feature_manipulation_hook(activation: Tensor, hook: object) -> Tensor:
            act = activation.clone()
            original_dtype = act.dtype
            act = act.float()

            if mode == "ablate":
                # Zero out safety-related dimensions
                for dim_id in safety_ids:
                    if dim_id < act.shape[-1]:
                        act[..., dim_id] = 0.0
            elif mode == "amplify":
                for dim_id in safety_ids:
                    if dim_id < act.shape[-1]:
                        act[..., dim_id] *= scale

            return act.to(original_dtype)

        hook_name = f"blocks.{layer}.hook_resid_post"
        hooks = [(hook_name, feature_manipulation_hook)]
        modified_outputs = backend.run_with_hooks(
            prompts, hooks=hooks, max_new_tokens=max_new_tokens
        )

        return [
            {"prompt": p, "original": o, "modified": m}
            for p, o, m in zip(prompts, original_outputs, modified_outputs)
        ]
