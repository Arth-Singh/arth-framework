"""Activation patching (causal tracing) for safety-relevant components."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor

from arth.core.activation_store import ActivationStore
from arth.core.hooks import patching_hook
from arth.core.models import TechniqueResult
from arth.utils.io import save_vector, load_vector
from arth.techniques.base import BaseTechnique


class ActivationPatching(BaseTechnique):
    """Identify safety-critical components via activation patching / causal tracing."""

    @property
    def name(self) -> str:
        return "activation_patching"

    @property
    def description(self) -> str:
        return "Causal tracing via activation patching to identify safety-critical components"

    def extract(self, backend, dataset, config=None) -> TechniqueResult:
        """Run causal tracing to identify which components are most important for safety.

        1. Run clean forward pass and record activations + final logits.
        2. Run corrupted forward pass (noise added to embeddings).
        3. For each layer, patch in clean activations and measure how much
           the output recovers toward the clean logits.
        """
        output_dir = Path(config.output_dir) if config else Path("results")
        layers = config.layers if config else None
        batch_size = config.batch_size if config else 32
        noise_std = 0.1

        backend._ensure_loaded()
        model = backend.model

        # Gather prompts
        prompts = []
        for item in dataset:
            if hasattr(item, "harmful"):
                prompts.append(item.harmful)
            elif hasattr(item, "prompt"):
                prompts.append(item.prompt)
            elif isinstance(item, str):
                prompts.append(item)

        if layers is None:
            layers = list(range(backend.n_layers))

        store = ActivationStore(backend, batch_size=batch_size)

        # 1. Clean activations
        clean_acts = store.collect(prompts, layers=layers, position="last")
        # Clean logits for reference
        clean_logits = backend.get_logits(prompts)  # (batch, seq, vocab)
        clean_last_logits = clean_logits[:, -1, :]  # (batch, vocab)

        # 2. Corrupted pass -- add noise to embeddings via hook
        noise_seed = 42
        device = backend.config.device
        rng = torch.Generator(device=device if device != "cpu" else None).manual_seed(noise_seed)

        def embedding_noise_hook(activation: Tensor, hook: object) -> Tensor:
            noise = torch.randn(
                activation.shape, generator=rng, device=activation.device, dtype=activation.dtype
            ) * noise_std
            return activation + noise

        tokens = model.to_tokens(prompts, prepend_bos=True)
        embed_hook = ("hook_embed", embedding_noise_hook)
        _, corrupted_cache = model.run_with_cache(
            tokens,
            names_filter=[f"blocks.{l}.hook_resid_post" for l in layers],
            fwd_hooks=[embed_hook],
        )

        corrupted_logits = model(tokens, fwd_hooks=[embed_hook])
        corrupted_last_logits = corrupted_logits[:, -1, :]

        # Baseline KL divergence (corrupted vs clean)
        clean_probs = torch.softmax(clean_last_logits, dim=-1)
        corrupted_probs = torch.softmax(corrupted_last_logits, dim=-1)
        baseline_kl = torch.nn.functional.kl_div(
            corrupted_probs.log(), clean_probs, reduction="batchmean"
        ).item()

        # 3. Patch each layer and measure recovery
        importance_scores: dict[int, float] = {}

        for layer in layers:
            clean_act = clean_acts[layer]  # (n_prompts, d_model)
            # Expand to (n_prompts, 1, d_model) to broadcast over seq positions
            # We patch the last position only
            hook_name = f"blocks.{layer}.hook_resid_post"

            # Build a patching hook that replaces last token position with clean activation
            clean_for_patch = clean_act.unsqueeze(1)  # (batch, 1, d_model)

            def make_last_pos_patch_hook(clean_vals: Tensor):
                def hook_fn(activation: Tensor, hook: object) -> Tensor:
                    patched = activation.clone()
                    patched[:, -1:, :] = clean_vals
                    return patched
                return hook_fn

            patched_logits = model(
                tokens,
                fwd_hooks=[embed_hook, (hook_name, make_last_pos_patch_hook(clean_for_patch))],
            )
            patched_last_logits = patched_logits[:, -1, :]
            patched_probs = torch.softmax(patched_last_logits, dim=-1)

            patched_kl = torch.nn.functional.kl_div(
                patched_probs.log(), clean_probs, reduction="batchmean"
            ).item()

            # Recovery = how much patching this layer reduces the KL divergence
            recovery = (baseline_kl - patched_kl) / max(baseline_kl, 1e-8)
            importance_scores[layer] = recovery

        # Save importance map
        scores_tensor = torch.tensor(
            [importance_scores.get(l, 0.0) for l in layers]
        )
        artifact_path = output_dir / f"activation_patching_{backend.config.name.replace('/', '_')}.pt"
        save_vector(scores_tensor, artifact_path, metadata={
            "layers": layers,
            "importance_scores": importance_scores,
            "baseline_kl": baseline_kl,
            "noise_std": noise_std,
            "model": backend.config.name,
            "n_prompts": len(prompts),
        })

        return TechniqueResult(
            technique=self.name,
            model_name=backend.config.name,
            artifact_path=artifact_path,
            metadata={
                "importance_scores": importance_scores,
                "baseline_kl": baseline_kl,
                "most_important_layer": max(importance_scores, key=importance_scores.get),
            },
        )

    def apply(self, backend, artifact_path, prompts, **kwargs) -> list[dict[str, str]]:
        """Return the component importance map for each prompt.

        Activation patching is primarily an analysis technique.
        """
        _, metadata = load_vector(artifact_path)
        importance_scores = metadata.get("importance_scores", {})

        results = []
        for prompt in prompts:
            results.append({
                "prompt": prompt,
                "original": "(analysis technique -- no generation)",
                "modified": str(importance_scores),
            })
        return results

    def evaluate(self, results: list[dict]) -> dict[str, float]:
        """Return empty evaluation scores (analysis-only technique)."""
        return {"note": 0.0}
