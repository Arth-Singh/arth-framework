"""Abstract base class for all model providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from torch import Tensor


class BaseProvider(ABC):
    """Abstract base class for model providers.

    Every provider implements a uniform interface for model loading, text
    generation, and (optionally) logit/activation extraction.  Providers
    that wrap API-only services return ``None`` for methods that require
    direct model access.
    """

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @abstractmethod
    def load(self, model_name: str, device: str = "cpu", dtype: str = "float16", **kwargs: Any) -> None:
        """Load or connect to the model.

        Args:
            model_name: HuggingFace model name, API model ID, or path.
            device: Target device (``"cpu"``, ``"cuda"``, ``"auto"``).
            dtype: Weight dtype (``"float16"``, ``"bfloat16"``, ``"float32"``).
            **kwargs: Provider-specific options (api_key, base_url, quantization, etc.).
        """
        ...

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    @abstractmethod
    def generate(self, prompts: list[str], max_new_tokens: int = 128, **kwargs: Any) -> list[str]:
        """Generate text completions for the given prompts.

        Args:
            prompts: Input strings.
            max_new_tokens: Maximum number of new tokens to generate.
            **kwargs: Provider-specific generation parameters.

        Returns:
            List of generated completion strings (prompt text stripped).
        """
        ...

    # ------------------------------------------------------------------
    # Logits / Activations  (optional capabilities)
    # ------------------------------------------------------------------

    def get_logits(self, prompts: list[str]) -> Tensor | None:
        """Return logit tensors for the given prompts.

        Providers that do not support logit access should return ``None``.

        Returns:
            Tensor of shape ``(batch, seq_len, vocab_size)`` or ``None``.
        """
        return None

    def get_residual_stream(
        self,
        prompts: list[str],
        layers: list[int] | None = None,
    ) -> dict[int, Tensor] | None:
        """Return residual-stream activations per layer.

        Only mechanistic-interpretability providers (e.g. TransformerLens) can
        supply this.  All others return ``None``.

        Returns:
            Dict mapping layer index to tensor ``(batch, seq_len, d_model)``
            or ``None``.
        """
        return None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def name(self) -> str:
        """Short machine-readable identifier for this provider (e.g. ``'transformer_lens'``)."""
        ...

    @property
    def tokenizer(self) -> Any:
        """The provider's tokenizer, if available.

        Returns ``None`` for API-only providers that do not expose a local tokenizer.
        """
        return None

    @property
    def supports_activations(self) -> bool:
        """Whether the provider can return residual-stream activations."""
        return False

    @property
    def supports_logits(self) -> bool:
        """Whether the provider can return raw logit tensors."""
        return False

    # ------------------------------------------------------------------
    # Hooks  (only TransformerLens-style providers)
    # ------------------------------------------------------------------

    def run_with_hooks(
        self,
        prompts: list[str],
        hooks: list[tuple],
        max_new_tokens: int = 128,
    ) -> list[str]:
        """Run generation with TransformerLens-style hooks.

        Not all providers support this.  The default implementation raises
        :class:`NotImplementedError`.
        """
        raise NotImplementedError(
            f"Provider {self.name!r} does not support hook-based generation. "
            "Use the 'transformer_lens' provider for full mechanistic interpretability."
        )

    # ------------------------------------------------------------------
    # Model metadata (optional, providers override as available)
    # ------------------------------------------------------------------

    @property
    def n_layers(self) -> int | None:
        """Number of transformer layers, or ``None`` if unknown."""
        return None

    @property
    def d_model(self) -> int | None:
        """Hidden dimension of the model, or ``None`` if unknown."""
        return None

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"<{type(self).__name__} name={self.name!r}>"
