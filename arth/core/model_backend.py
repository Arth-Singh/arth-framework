"""Model loading and inference wrapper using the provider system.

The :class:`ModelBackend` class is the primary entry-point for all model
operations in the Arth toolkit.  It delegates to a
:class:`~arth.core.providers.base.BaseProvider` selected via
``ModelConfig.provider``.

For backward compatibility, when no ``provider`` is specified (or it is
``"transformer_lens"``), behaviour is identical to the original
TransformerLens-only implementation.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from arth.core.models import ModelConfig
from arth.core.providers.base import BaseProvider


class ModelBackend:
    """Unified model backend that delegates to a pluggable provider.

    The provider is determined by ``config.provider`` and resolved via the
    provider registry.  All technique code interacts with this class,
    which forwards calls to the underlying provider instance.
    """

    def __init__(self, config: ModelConfig) -> None:
        self.config = config
        self._provider: BaseProvider | None = None
        self.model: Any | None = None  # kept for backward compat; populated by TL provider

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _resolve_provider(self) -> BaseProvider:
        """Instantiate the provider specified in the config."""
        from arth.core.providers.registry import get_provider

        provider = get_provider(self.config.provider)
        return provider

    def load(self) -> None:
        """Load the model through the configured provider."""
        if self._provider is None:
            self._provider = self._resolve_provider()

        extra_kwargs: dict[str, Any] = {
            "trust_remote_code": self.config.trust_remote_code,
        }
        if self.config.api_key is not None:
            extra_kwargs["api_key"] = self.config.api_key
        if self.config.base_url is not None:
            extra_kwargs["base_url"] = self.config.base_url
        if self.config.quantization is not None:
            extra_kwargs["quantization"] = self.config.quantization

        self._provider.load(
            model_name=self.config.name,
            device=self.config.device,
            dtype=self.config.dtype,
            **extra_kwargs,
        )

        # Backward compatibility: expose the inner model for TransformerLens
        if hasattr(self._provider, "_model"):
            self.model = self._provider._model

    def _ensure_loaded(self) -> None:
        if self._provider is None:
            self.load()

    # ------------------------------------------------------------------
    # Provider access
    # ------------------------------------------------------------------

    @property
    def provider(self) -> BaseProvider:
        """Return the underlying provider instance, loading if necessary."""
        self._ensure_loaded()
        assert self._provider is not None
        return self._provider

    # ------------------------------------------------------------------
    # Capability checks
    # ------------------------------------------------------------------

    @property
    def supports_activations(self) -> bool:
        """Whether the current provider can return residual-stream activations."""
        return self.provider.supports_activations

    @property
    def supports_logits(self) -> bool:
        """Whether the current provider can return raw logit tensors."""
        return self.provider.supports_logits

    def require_activations(self, operation: str = "This operation") -> None:
        """Raise a clear error if the provider does not support activations.

        Techniques that need activations should call this at the start of
        ``extract()`` to fail fast with a helpful message.
        """
        if not self.supports_activations:
            raise RuntimeError(
                f"{operation} requires residual-stream activations, but the "
                f"'{self.config.provider}' provider does not support them. "
                f"Use --provider transformer_lens for full mechanistic interpretability."
            )

    def require_logits(self, operation: str = "This operation") -> None:
        """Raise a clear error if the provider does not support logits."""
        if not self.supports_logits:
            raise RuntimeError(
                f"{operation} requires logit access, but the "
                f"'{self.config.provider}' provider does not support them. "
                f"Use --provider transformer_lens or --provider huggingface_local."
            )

    # ------------------------------------------------------------------
    # Activation extraction
    # ------------------------------------------------------------------

    @torch.no_grad()
    def get_residual_stream(
        self,
        prompts: list[str],
        layers: list[int] | None = None,
    ) -> dict[int, Tensor]:
        """Run prompts through the model and return residual stream activations.

        Args:
            prompts: List of input strings.
            layers: Layer indices to collect.  ``None`` means all layers.

        Returns:
            Dict mapping layer index to a tensor of shape
            ``(batch, seq_len, d_model)``.

        Raises:
            RuntimeError: If the provider does not support activations.
        """
        self.require_activations("get_residual_stream()")
        result = self.provider.get_residual_stream(prompts, layers)
        assert result is not None
        return result

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
        """Generate completions for the given prompts.

        Args:
            prompts: Input strings.
            max_new_tokens: Maximum number of new tokens to generate.
            **kwargs: Extra arguments forwarded to the provider.

        Returns:
            List of generated strings (completions only, prompt stripped).
        """
        self._ensure_loaded()
        return self.provider.generate(prompts, max_new_tokens=max_new_tokens, **kwargs)

    def run_with_hooks(
        self,
        prompts: list[str],
        hooks: list[tuple],
        max_new_tokens: int = 128,
    ) -> list[str]:
        """Run generation with TransformerLens hooks applied.

        Args:
            prompts: Input strings.
            hooks: List of ``(hook_name, hook_fn)`` tuples.
            max_new_tokens: Maximum number of new tokens to generate.

        Returns:
            List of generated strings (completions only).

        Raises:
            NotImplementedError: If the provider does not support hooks.
        """
        self._ensure_loaded()
        return self.provider.run_with_hooks(prompts, hooks, max_new_tokens=max_new_tokens)

    @torch.no_grad()
    def get_logits(self, prompts: list[str]) -> Tensor:
        """Get logit outputs for the given prompts.

        Args:
            prompts: Input strings.

        Returns:
            Tensor of shape ``(batch, seq_len, vocab_size)``.

        Raises:
            RuntimeError: If the provider does not support logits.
        """
        self.require_logits("get_logits()")
        result = self.provider.get_logits(prompts)
        assert result is not None
        return result

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_layers(self) -> int:
        """Number of transformer layers in the model."""
        self._ensure_loaded()
        n = self.provider.n_layers
        if n is None:
            # Fall back to config override
            if self.config.n_layers is not None:
                return self.config.n_layers
            raise AttributeError(
                f"Provider '{self.config.provider}' does not report the number of layers. "
                "Specify --n-layers or use a local provider."
            )
        return n

    @property
    def d_model(self) -> int:
        """Hidden dimension of the model."""
        self._ensure_loaded()
        d = self.provider.d_model
        if d is None:
            raise AttributeError(
                f"Provider '{self.config.provider}' does not report the hidden dimension. "
                "Use a local provider for model architecture details."
            )
        return d

    @property
    def tokenizer(self) -> Any:
        """The model's tokenizer."""
        self._ensure_loaded()
        tok = self.provider.tokenizer
        if tok is None:
            raise AttributeError(
                f"Provider '{self.config.provider}' does not expose a local tokenizer."
            )
        return tok
