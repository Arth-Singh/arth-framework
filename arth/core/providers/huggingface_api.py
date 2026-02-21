"""HuggingFace Inference API provider -- remote inference via the HF Hub.

Uses ``huggingface_hub.InferenceClient`` to run any model hosted on the
HuggingFace Hub without downloading weights locally.  Requires an
``HF_TOKEN`` environment variable or an explicit ``api_key``.
"""

from __future__ import annotations

import os
from typing import Any

from arth.core.providers.base import BaseProvider


class HuggingFaceAPIProvider(BaseProvider):
    """Provider backed by the HuggingFace Inference API.

    Suitable for quick experimentation with any model on the Hub.
    Does **not** support activations or logits -- generation only.
    """

    def __init__(self) -> None:
        self._client: Any | None = None
        self._model_name: str | None = None

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
        """Connect to the HuggingFace Inference API.

        Args:
            model_name: HuggingFace model ID (e.g. ``"meta-llama/Llama-2-7b-chat-hf"``).
            device: Ignored (inference runs on HF servers).
            dtype: Ignored.
            **kwargs: Extra options:
                ``api_key`` (str): HF API token.  Falls back to ``HF_TOKEN`` env var.
                ``timeout`` (int): Request timeout in seconds (default 120).
        """
        try:
            from huggingface_hub import InferenceClient
        except ImportError as exc:
            raise ImportError(
                "The 'huggingface_api' provider requires huggingface_hub. "
                "Install it with:  pip install huggingface_hub"
            ) from exc

        api_key = kwargs.get("api_key") or os.environ.get("HF_TOKEN")
        if not api_key:
            raise ValueError(
                "HuggingFace API requires an API token. "
                "Set the HF_TOKEN environment variable or pass --api-key."
            )

        timeout = kwargs.get("timeout", 120)

        self._client = InferenceClient(
            model=model_name,
            token=api_key,
            timeout=timeout,
        )
        self._model_name = model_name

    def _ensure_loaded(self) -> None:
        if self._client is None:
            raise RuntimeError(
                "Client not initialised. Call provider.load(model_name) first."
            )

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(
        self,
        prompts: list[str],
        max_new_tokens: int = 128,
        **kwargs: Any,
    ) -> list[str]:
        self._ensure_loaded()
        assert self._client is not None

        completions: list[str] = []
        for prompt in prompts:
            try:
                response = self._client.text_generation(
                    prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=kwargs.get("temperature", 0.7),
                    top_p=kwargs.get("top_p"),
                    repetition_penalty=kwargs.get("repetition_penalty"),
                    do_sample=kwargs.get("do_sample", True),
                )
                # text_generation returns the generated text (str)
                completions.append(response)
            except Exception as exc:
                completions.append(f"[ERROR: {exc}]")
        return completions

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "huggingface_api"

    @property
    def supports_activations(self) -> bool:
        return False

    @property
    def supports_logits(self) -> bool:
        return False
