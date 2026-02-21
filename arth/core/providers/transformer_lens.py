"""TransformerLens provider -- full mechanistic interpretability support.

This provider wraps ``transformer_lens.HookedTransformer`` and exposes
residual-stream activations, logits, and hook-based generation.  It is
the default provider for the Arth toolkit.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from arth.core.providers.base import BaseProvider


class TransformerLensProvider(BaseProvider):
    """Provider backed by TransformerLens ``HookedTransformer``.

    Supports:
      - Full residual-stream activation extraction
      - Raw logit tensors
      - Hook-based generation (``fwd_hooks``)
    """

    def __init__(self) -> None:
        self._model: Any | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(
        self,
        model_name: str,
        device: str = "cpu",
        dtype: str = "float16",
        **kwargs: Any,
    ) -> None:
        """Load the model via ``HookedTransformer.from_pretrained``.

        Args:
            model_name: HuggingFace model name or path.
            device: Target device.
            dtype: Weight dtype string (``"float16"``, ``"bfloat16"``, ``"float32"``).
            **kwargs: Extra keyword arguments.  Recognised keys:
                ``trust_remote_code`` (bool).
        """
        try:
            from transformer_lens import HookedTransformer
        except ImportError as exc:
            raise ImportError(
                "TransformerLens is required for the 'transformer_lens' provider. "
                "Install it with:  pip install transformer-lens"
            ) from exc

        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(dtype, torch.float16)
        trust_remote_code = kwargs.get("trust_remote_code", False)

        self._model = HookedTransformer.from_pretrained(
            model_name,
            device=device,
            dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
        )
        self._model.eval()

    def _ensure_loaded(self) -> None:
        if self._model is None:
            raise RuntimeError(
                "Model not loaded. Call provider.load(model_name, device, dtype) first."
            )

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        prompts: list[str],
        max_new_tokens: int = 128,
        **kwargs: Any,
    ) -> list[str]:
        self._ensure_loaded()
        assert self._model is not None

        tokens = self._model.to_tokens(prompts, prepend_bos=True)
        prompt_len = tokens.shape[1]
        output_tokens = self._model.generate(
            tokens, max_new_tokens=max_new_tokens, **kwargs
        )

        completions: list[str] = []
        for i in range(output_tokens.shape[0]):
            new_tokens = output_tokens[i, prompt_len:]
            text = self._model.tokenizer.decode(new_tokens, skip_special_tokens=True)
            completions.append(text)
        return completions

    # ------------------------------------------------------------------
    # Logits / Activations
    # ------------------------------------------------------------------

    @torch.no_grad()
    def get_logits(self, prompts: list[str]) -> Tensor:
        self._ensure_loaded()
        assert self._model is not None
        tokens = self._model.to_tokens(prompts, prepend_bos=True)
        return self._model(tokens)

    @torch.no_grad()
    def get_residual_stream(
        self,
        prompts: list[str],
        layers: list[int] | None = None,
    ) -> dict[int, Tensor]:
        self._ensure_loaded()
        assert self._model is not None

        if layers is None:
            layers = list(range(self._model.cfg.n_layers))

        names_filter = [f"blocks.{l}.hook_resid_post" for l in layers]
        tokens = self._model.to_tokens(prompts, prepend_bos=True)
        _, cache = self._model.run_with_cache(tokens, names_filter=names_filter)

        activations: dict[int, Tensor] = {}
        for layer in layers:
            key = f"blocks.{layer}.hook_resid_post"
            activations[layer] = cache[key]
        return activations

    # ------------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------------

    def run_with_hooks(
        self,
        prompts: list[str],
        hooks: list[tuple],
        max_new_tokens: int = 128,
    ) -> list[str]:
        """Run generation with TransformerLens hooks applied."""
        self._ensure_loaded()
        assert self._model is not None

        tokens = self._model.to_tokens(prompts, prepend_bos=True)
        prompt_len = tokens.shape[1]

        with torch.no_grad():
            output_tokens = self._model.generate(
                tokens, max_new_tokens=max_new_tokens, fwd_hooks=hooks
            )

        completions: list[str] = []
        for i in range(output_tokens.shape[0]):
            new_tokens = output_tokens[i, prompt_len:]
            text = self._model.tokenizer.decode(new_tokens, skip_special_tokens=True)
            completions.append(text)
        return completions

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "transformer_lens"

    @property
    def tokenizer(self) -> Any:
        self._ensure_loaded()
        assert self._model is not None
        return self._model.tokenizer

    @property
    def supports_activations(self) -> bool:
        return True

    @property
    def supports_logits(self) -> bool:
        return True

    @property
    def n_layers(self) -> int | None:
        if self._model is None:
            return None
        return self._model.cfg.n_layers

    @property
    def d_model(self) -> int | None:
        if self._model is None:
            return None
        return self._model.cfg.d_model
