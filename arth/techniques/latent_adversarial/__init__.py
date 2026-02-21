"""Latent Adversarial Training (LAT) -- PGD in activation space."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor

from arth.core.activation_store import ActivationStore
from arth.core.models import TechniqueResult
from arth.utils.tensor_ops import normalize
from arth.utils.io import save_vector, load_vector
from arth.techniques.base import BaseTechnique


class LatentAdversarial(BaseTechnique):
    """Find adversarial perturbation vectors in activation space via PGD."""

    @property
    def name(self) -> str:
        return "latent_adversarial"

    @property
    def description(self) -> str:
        return "PGD-based adversarial perturbation in activation space to bypass safety"

    def extract(self, backend, dataset, config=None) -> TechniqueResult:
        """Use projected gradient descent to find perturbation vectors that bypass safety.

        Optimizes a perturbation vector added to a target layer's residual
        stream that maximizes the probability of non-refusal continuations.

        Args:
            backend: ModelBackend instance.
            dataset: List of ContrastPair or prompts.
            config: Optional ExperimentConfig.
        """
        output_dir = Path(config.output_dir) if config else Path("results")
        batch_size = config.batch_size if config else 32

        backend._ensure_loaded()
        model = backend.model

        # PGD hyperparameters
        target_layer = backend.n_layers // 2  # attack middle layer
        epsilon = 4.0        # perturbation L2 budget
        step_size = 0.1      # PGD step size
        n_steps = 50         # number of PGD steps

        # Gather harmful prompts
        prompts = []
        for item in dataset:
            if hasattr(item, "harmful"):
                prompts.append(item.harmful)
            elif hasattr(item, "prompt"):
                prompts.append(item.prompt)
            elif isinstance(item, str):
                prompts.append(item)

        prompts = prompts[:batch_size]  # limit for memory
        tokens = model.to_tokens(prompts, prepend_bos=True)

        # Target tokens: encourage "Sure" / "Here" style continuations
        target_strs = ["Sure", "Here", "Of", "Okay"]
        target_ids = []
        for s in target_strs:
            ids = model.tokenizer.encode(s, add_special_tokens=False)
            if ids:
                target_ids.append(ids[0])

        if not target_ids:
            raise ValueError("Could not resolve any target tokens for PGD optimization")

        # Initialize perturbation
        d_model = backend.d_model
        perturbation = torch.zeros(d_model, device=backend.config.device, requires_grad=True)

        hook_name = f"blocks.{target_layer}.hook_resid_post"

        for step in range(n_steps):
            perturbation = perturbation.detach().requires_grad_(True)

            # Forward pass with perturbation hook — capture current perturbation via default arg
            def perturbation_hook(activation: Tensor, hook: object, _p=perturbation) -> Tensor:
                return activation + _p

            logits = model(tokens, fwd_hooks=[(hook_name, perturbation_hook)])
            # Get logits at the last position
            last_logits = logits[:, -1, :]  # (batch, vocab)
            # Maximize probability of target tokens
            target_logits = last_logits[:, target_ids]  # (batch, n_targets)
            loss = -target_logits.mean()

            loss.backward()

            with torch.no_grad():
                # PGD step
                grad = perturbation.grad
                if grad is None:
                    break
                perturbation = perturbation - step_size * normalize(grad)
                # Project back to epsilon-ball
                norm = perturbation.norm()
                if norm > epsilon:
                    perturbation = perturbation * (epsilon / norm)

            model.zero_grad()

        perturbation = perturbation.detach()

        artifact_path = output_dir / f"latent_adversarial_{backend.config.name.replace('/', '_')}.pt"
        save_vector(perturbation, artifact_path, metadata={
            "target_layer": target_layer,
            "epsilon": epsilon,
            "n_steps": n_steps,
            "step_size": step_size,
            "perturbation_norm": perturbation.norm().item(),
            "model": backend.config.name,
            "n_prompts": len(prompts),
        })

        return TechniqueResult(
            technique=self.name,
            model_name=backend.config.name,
            artifact_path=artifact_path,
            metadata={
                "target_layer": target_layer,
                "epsilon": epsilon,
                "perturbation_norm": perturbation.norm().item(),
                "n_steps": n_steps,
            },
        )

    def apply(self, backend, artifact_path, prompts, **kwargs) -> list[dict[str, str]]:
        """Add learned perturbation to activations during inference.

        Args:
            backend: ModelBackend instance.
            artifact_path: Path to saved perturbation vector.
            prompts: Prompts to test.
            **kwargs:
                scale (float): Perturbation multiplier (default 1.0).
                max_new_tokens (int): Max generation tokens.
        """
        perturbation, metadata = load_vector(artifact_path)
        perturbation = perturbation.to(backend.config.device).float()
        target_layer = metadata.get("target_layer", backend.n_layers // 2)
        scale = kwargs.get("scale", 1.0)
        max_new_tokens = kwargs.get("max_new_tokens", 128)

        # Generate original outputs
        original_outputs = backend.generate(prompts, max_new_tokens=max_new_tokens)

        # Hook that adds the perturbation
        scaled_perturbation = perturbation * scale

        def adversarial_hook(activation: Tensor, hook: object) -> Tensor:
            return activation + scaled_perturbation

        hook_name = f"blocks.{target_layer}.hook_resid_post"
        hooks = [(hook_name, adversarial_hook)]
        modified_outputs = backend.run_with_hooks(
            prompts, hooks=hooks, max_new_tokens=max_new_tokens
        )

        return [
            {"prompt": p, "original": o, "modified": m}
            for p, o, m in zip(prompts, original_outputs, modified_outputs)
        ]
