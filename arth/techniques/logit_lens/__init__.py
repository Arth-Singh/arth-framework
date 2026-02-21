"""Logit lens analysis -- per-layer token probability projection."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor

from arth.core.activation_store import ActivationStore
from arth.core.models import TechniqueResult
from arth.utils.io import save_vector, load_vector
from arth.techniques.base import BaseTechnique

# Tokens commonly associated with refusal behavior
REFUSAL_TOKENS = ["I", "sorry", "cannot", "can", "'t", "not", "apolog", "unable"]


class LogitLens(BaseTechnique):
    """Project residual stream through the unembedding matrix at each layer."""

    @property
    def name(self) -> str:
        return "logit_lens"

    @property
    def description(self) -> str:
        return "Analyze per-layer token probabilities via logit lens to trace refusal formation"

    def extract(self, backend, dataset, config=None) -> TechniqueResult:
        """Compute per-layer logit lens projections for the given prompts.

        For each layer, project the residual stream through the unembedding
        matrix and track refusal-related token probabilities.
        """
        output_dir = Path(config.output_dir) if config else Path("results")
        layers = config.layers if config else None
        batch_size = config.batch_size if config else 32

        backend._ensure_loaded()
        model = backend.model

        # Resolve refusal token IDs
        refusal_token_ids = self._resolve_token_ids(backend, REFUSAL_TOKENS)

        # Collect prompts (use harmful prompts from contrast pairs if available)
        prompts = []
        for item in dataset:
            if hasattr(item, "harmful"):
                prompts.append(item.harmful)
            elif hasattr(item, "prompt"):
                prompts.append(item.prompt)
            elif isinstance(item, str):
                prompts.append(item)

        store = ActivationStore(backend, batch_size=batch_size)
        # Use "last" position to get single token activations per prompt
        activations = store.collect(prompts, layers=layers, position="last")

        # Get unembedding matrix
        W_U = model.W_U  # (d_model, vocab_size)

        per_layer_probs: dict[int, dict[str, float]] = {}

        for layer in sorted(activations.keys()):
            acts = activations[layer].float()  # (n_prompts, d_model)
            # Apply layer norm if available
            if hasattr(model, "ln_final"):
                acts = model.ln_final(acts)
            # Project through unembedding: (n_prompts, vocab_size)
            logits = acts @ W_U
            probs = torch.softmax(logits, dim=-1)
            # Average over prompts
            mean_probs = probs.mean(dim=0)  # (vocab_size,)

            layer_token_probs = {}
            for token_str, token_id in refusal_token_ids.items():
                layer_token_probs[token_str] = mean_probs[token_id].item()

            per_layer_probs[layer] = layer_token_probs

        # Save analysis as artifact
        artifact_path = output_dir / f"logit_lens_{backend.config.name.replace('/', '_')}.pt"
        save_vector(
            torch.zeros(1),  # placeholder tensor
            artifact_path,
            metadata={
                "per_layer_probs": {str(k): v for k, v in per_layer_probs.items()},
                "refusal_tokens": list(refusal_token_ids.keys()),
                "model": backend.config.name,
                "n_prompts": len(prompts),
            },
        )

        return TechniqueResult(
            technique=self.name,
            model_name=backend.config.name,
            artifact_path=artifact_path,
            metadata={
                "per_layer_probs": {str(k): v for k, v in per_layer_probs.items()},
                "refusal_tokens": list(refusal_token_ids.keys()),
                "n_prompts": len(prompts),
            },
        )

    def apply(self, backend, artifact_path, prompts, **kwargs) -> list[dict[str, str]]:
        """Return visualization data showing refusal probability across layers.

        Since logit lens is primarily an analysis technique, ``apply`` returns
        per-prompt layer-by-layer refusal probability data rather than
        modified completions.
        """
        _, metadata = load_vector(artifact_path)
        per_layer_probs = metadata.get("per_layer_probs", {})

        results = []
        for prompt in prompts:
            results.append({
                "prompt": prompt,
                "original": "(analysis technique -- no generation)",
                "modified": str(per_layer_probs),
            })
        return results

    def evaluate(self, results: list[dict]) -> dict[str, float]:
        """Logit lens evaluation returns empty scores (analysis-only technique)."""
        return {"note": 0.0}

    @staticmethod
    def _resolve_token_ids(backend, tokens: list[str]) -> dict[str, int]:
        """Map token strings to their vocabulary IDs."""
        tokenizer = backend.tokenizer
        resolved = {}
        for tok in tokens:
            ids = tokenizer.encode(tok, add_special_tokens=False)
            if ids:
                resolved[tok] = ids[0]
        return resolved
