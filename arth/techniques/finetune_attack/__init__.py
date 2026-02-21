"""LoRA-based safety removal analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from arth.core.activation_store import ActivationStore
from arth.core.models import TechniqueResult
from arth.utils.io import save_vector, load_vector
from arth.techniques.base import BaseTechnique


class FinetuneAttack(BaseTechnique):
    """Analyze safety removal via minimal-rank LoRA fine-tuning."""

    @property
    def name(self) -> str:
        return "finetune_attack"

    @property
    def description(self) -> str:
        return "Analyze safety removal via LoRA fine-tuning at minimal rank"

    def extract(self, backend, dataset, config=None) -> TechniqueResult:
        """Identify which layers' LoRA adapters most affect safety behavior.

        If ``peft`` is available, trains minimal LoRA adapters and measures
        per-layer safety impact. Otherwise, falls back to gradient-based
        analysis of which layers are most sensitive to safety-relevant changes.
        """
        output_dir = Path(config.output_dir) if config else Path("results")
        layers = config.layers if config else None
        batch_size = config.batch_size if config else 32

        backend._ensure_loaded()

        peft_available = self._check_peft()

        if peft_available:
            return self._extract_with_peft(backend, dataset, output_dir, layers, batch_size)
        else:
            return self._extract_gradient_analysis(backend, dataset, output_dir, layers, batch_size)

    @staticmethod
    def _check_peft() -> bool:
        """Check if peft library is available."""
        try:
            import peft
            return True
        except ImportError:
            return False

    def _extract_with_peft(
        self, backend, dataset, output_dir: Path, layers: list[int] | None, batch_size: int
    ) -> TechniqueResult:
        """Train LoRA adapters and identify safety-critical layers."""
        from peft import LoraConfig, get_peft_model, TaskType

        model = backend.model

        # Gather harmful prompts and target completions
        prompts = []
        for item in dataset:
            if hasattr(item, "harmful"):
                prompts.append(item.harmful)
            elif isinstance(item, str):
                prompts.append(item)

        if layers is None:
            layers = list(range(backend.n_layers))

        # Per-layer LoRA impact analysis
        layer_impacts: dict[int, float] = {}

        for layer_idx in layers:
            # Configure LoRA for a single layer
            target_modules = [f"blocks.{layer_idx}.attn.W_Q", f"blocks.{layer_idx}.attn.W_K"]
            lora_config = LoraConfig(
                r=4,
                lora_alpha=16,
                target_modules=target_modules,
                lora_dropout=0.0,
                task_type=TaskType.CAUSAL_LM,
            )

            try:
                peft_model = get_peft_model(model, lora_config)
                # Measure initial refusal logits
                tokens = model.to_tokens(prompts[:batch_size], prepend_bos=True)
                with torch.no_grad():
                    base_logits = model(tokens)[:, -1, :]

                # Simple training loop (few steps)
                optimizer = torch.optim.Adam(
                    [p for p in peft_model.parameters() if p.requires_grad], lr=1e-4
                )
                target_strs = ["Sure", "Here", "Of course"]
                target_ids = []
                for s in target_strs:
                    ids = model.tokenizer.encode(s, add_special_tokens=False)
                    if ids:
                        target_ids.append(ids[0])

                if not target_ids:
                    layer_impacts[layer_idx] = 0.0
                    del peft_model
                    continue

                for _step in range(5):
                    logits = peft_model(tokens)[:, -1, :]
                    # Maximize target token probability
                    target_logits = logits[:, target_ids].mean()
                    loss = -target_logits
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                # Measure impact
                with torch.no_grad():
                    trained_logits = peft_model(tokens)[:, -1, :]
                    kl = torch.nn.functional.kl_div(
                        torch.log_softmax(trained_logits, dim=-1),
                        torch.softmax(base_logits, dim=-1),
                        reduction="batchmean",
                    ).item()
                layer_impacts[layer_idx] = kl

                # Remove adapter
                peft_model.disable_adapter_layers()
                del peft_model
            except Exception:
                layer_impacts[layer_idx] = 0.0

        impacts_tensor = torch.tensor([layer_impacts.get(l, 0.0) for l in layers])
        artifact_path = output_dir / f"finetune_attack_{backend.config.name.replace('/', '_')}.pt"
        save_vector(impacts_tensor, artifact_path, metadata={
            "layers": layers,
            "layer_impacts": layer_impacts,
            "model": backend.config.name,
            "peft_available": True,
            "lora_rank": 4,
        })

        return TechniqueResult(
            technique=self.name,
            model_name=backend.config.name,
            artifact_path=artifact_path,
            metadata={
                "layer_impacts": layer_impacts,
                "most_impactful_layer": max(layer_impacts, key=layer_impacts.get) if layer_impacts else -1,
                "peft_available": True,
            },
        )

    def _extract_gradient_analysis(
        self, backend, dataset, output_dir: Path, layers: list[int] | None, batch_size: int
    ) -> TechniqueResult:
        """Fallback: gradient-based analysis without peft.

        Measures how sensitive each layer's parameters are to safety-relevant
        changes by computing gradient norms with respect to target tokens.
        """
        model = backend.model

        prompts = []
        for item in dataset:
            if hasattr(item, "harmful"):
                prompts.append(item.harmful)
            elif isinstance(item, str):
                prompts.append(item)

        if layers is None:
            layers = list(range(backend.n_layers))

        prompts = prompts[:batch_size]
        tokens = model.to_tokens(prompts, prepend_bos=True)

        # Target tokens
        target_strs = ["Sure", "Here", "Of course"]
        target_ids = []
        for s in target_strs:
            ids = model.tokenizer.encode(s, add_special_tokens=False)
            if ids:
                target_ids.append(ids[0])

        if not target_ids:
            norms_tensor = torch.zeros(len(layers))
            artifact_path = output_dir / f"finetune_attack_{backend.config.name.replace('/', '_')}.pt"
            save_vector(norms_tensor, artifact_path, metadata={
                "layers": layers, "layer_grad_norms": {}, "model": backend.config.name,
                "peft_available": False, "note": "no target tokens resolved",
            })
            return TechniqueResult(
                technique=self.name, model_name=backend.config.name,
                artifact_path=artifact_path,
                metadata={"peft_available": False, "note": "no target tokens resolved"},
            )

        # Forward pass
        logits = model(tokens)
        last_logits = logits[:, -1, :]
        target_logits = last_logits[:, target_ids].mean()
        target_logits.backward()

        # Collect gradient norms per layer
        layer_grad_norms: dict[int, float] = {}
        for layer_idx in layers:
            block = model.blocks[layer_idx]
            total_grad_norm = 0.0
            n_params = 0
            for param in block.parameters():
                if param.grad is not None:
                    total_grad_norm += param.grad.norm().item()
                    n_params += 1
            layer_grad_norms[layer_idx] = total_grad_norm / max(n_params, 1)

        model.zero_grad()

        norms_tensor = torch.tensor([layer_grad_norms.get(l, 0.0) for l in layers])
        artifact_path = output_dir / f"finetune_attack_{backend.config.name.replace('/', '_')}.pt"
        save_vector(norms_tensor, artifact_path, metadata={
            "layers": layers,
            "layer_grad_norms": layer_grad_norms,
            "model": backend.config.name,
            "peft_available": False,
        })

        return TechniqueResult(
            technique=self.name,
            model_name=backend.config.name,
            artifact_path=artifact_path,
            metadata={
                "layer_grad_norms": layer_grad_norms,
                "most_sensitive_layer": max(layer_grad_norms, key=layer_grad_norms.get) if layer_grad_norms else -1,
                "peft_available": False,
                "note": "peft not available; using gradient-based sensitivity analysis",
            },
        )

    def apply(self, backend, artifact_path, prompts, **kwargs) -> list[dict[str, str]]:
        """Apply LoRA adapter or report layer sensitivity analysis.

        Args:
            backend: ModelBackend instance.
            artifact_path: Path to saved analysis artifact.
            prompts: Prompts to test.
            **kwargs:
                max_new_tokens (int): Max generation tokens.
        """
        _, metadata = load_vector(artifact_path)
        max_new_tokens = kwargs.get("max_new_tokens", 128)

        peft_available = metadata.get("peft_available", False)

        if peft_available:
            layer_impacts = metadata.get("layer_impacts", {})
            results = []
            for prompt in prompts:
                results.append({
                    "prompt": prompt,
                    "original": "(LoRA analysis -- see metadata for layer impacts)",
                    "modified": str(layer_impacts),
                })
            return results
        else:
            layer_norms = metadata.get("layer_grad_norms", {})
            results = []
            for prompt in prompts:
                results.append({
                    "prompt": prompt,
                    "original": "(gradient analysis -- peft not available)",
                    "modified": str(layer_norms),
                })
            return results

    def evaluate(self, results: list[dict]) -> dict[str, float]:
        """Return empty scores (analysis technique)."""
        return {"note": 0.0}
