"""OpenAI-compatible API provider.

Works with OpenAI, Azure OpenAI, Together AI, Groq, Ollama, and any
endpoint that implements the OpenAI chat/completions API.  Uses the
``openai`` Python library if available, otherwise falls back to
``httpx`` for raw HTTP requests.
"""

from __future__ import annotations

import os
from typing import Any

from arth.core.providers.base import BaseProvider


class OpenAICompatProvider(BaseProvider):
    """Provider for any OpenAI-compatible API endpoint.

    Supports:
      - OpenAI (``https://api.openai.com/v1``)
      - Azure OpenAI (custom base_url)
      - Together AI (``https://api.together.xyz/v1``)
      - Groq (``https://api.groq.com/openai/v1``)
      - Ollama (``http://localhost:11434/v1``)
      - Any OpenAI-compatible server
    """

    _DEFAULT_BASE_URL = "https://api.openai.com/v1"

    def __init__(self) -> None:
        self._client: Any | None = None
        self._model_name: str | None = None
        self._base_url: str = self._DEFAULT_BASE_URL
        self._use_openai_lib: bool = False
        self._api_key: str | None = None
        self._supports_logprobs: bool = False

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
        """Connect to an OpenAI-compatible API.

        Args:
            model_name: Model ID (e.g. ``"gpt-4o"``, ``"meta-llama/Llama-3-70b"``).
            device: Ignored (inference runs remotely).
            dtype: Ignored.
            **kwargs: Extra options:
                ``api_key`` (str): API key.  Falls back to ``OPENAI_API_KEY`` env var.
                ``base_url`` (str): Custom base URL for the API.
                ``supports_logprobs`` (bool): Whether the API supports logprobs.
        """
        self._model_name = model_name
        self._base_url = kwargs.get("base_url") or self._DEFAULT_BASE_URL
        self._api_key = kwargs.get("api_key") or os.environ.get("OPENAI_API_KEY")
        self._supports_logprobs = kwargs.get("supports_logprobs", False)

        if not self._api_key:
            # Some local endpoints (Ollama) don't require a key
            if "localhost" not in self._base_url and "127.0.0.1" not in self._base_url:
                raise ValueError(
                    "OpenAI-compatible API requires an API key. "
                    "Set the OPENAI_API_KEY environment variable or pass --api-key."
                )
            self._api_key = "not-needed"

        # Try to use the openai library; fall back to httpx
        try:
            from openai import OpenAI

            self._client = OpenAI(
                api_key=self._api_key,
                base_url=self._base_url,
            )
            self._use_openai_lib = True
        except ImportError:
            # Fall back to httpx raw requests
            try:
                import httpx  # noqa: F401
            except ImportError as exc:
                raise ImportError(
                    "The 'openai_compat' provider requires either the openai library "
                    "or httpx. Install with:  pip install openai  (or)  pip install httpx"
                ) from exc
            self._use_openai_lib = False

    def _ensure_loaded(self) -> None:
        if self._model_name is None:
            raise RuntimeError(
                "Provider not initialised. Call provider.load(model_name) first."
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

        completions: list[str] = []
        for prompt in prompts:
            try:
                text = self._generate_single(prompt, max_new_tokens, **kwargs)
                completions.append(text)
            except Exception as exc:
                completions.append(f"[ERROR: {exc}]")
        return completions

    def _generate_single(
        self, prompt: str, max_new_tokens: int, **kwargs: Any
    ) -> str:
        """Generate a single completion."""
        temperature = kwargs.get("temperature", 0.7)
        top_p = kwargs.get("top_p", 1.0)
        use_chat = kwargs.get("use_chat", True)

        if self._use_openai_lib:
            return self._generate_openai_lib(
                prompt, max_new_tokens, temperature, top_p, use_chat
            )
        else:
            return self._generate_httpx(
                prompt, max_new_tokens, temperature, top_p, use_chat
            )

    def _generate_openai_lib(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        use_chat: bool,
    ) -> str:
        """Generate via the ``openai`` library."""
        assert self._client is not None

        if use_chat:
            response = self._client.chat.completions.create(
                model=self._model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            return response.choices[0].message.content or ""
        else:
            response = self._client.completions.create(
                model=self._model_name,
                prompt=prompt,
                max_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            return response.choices[0].text or ""

    def _generate_httpx(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        use_chat: bool,
    ) -> str:
        """Generate via raw ``httpx`` HTTP requests (fallback)."""
        import httpx

        headers = {
            "Content-Type": "application/json",
        }
        if self._api_key and self._api_key != "not-needed":
            headers["Authorization"] = f"Bearer {self._api_key}"

        if use_chat:
            url = f"{self._base_url.rstrip('/')}/chat/completions"
            payload = {
                "model": self._model_name,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_new_tokens,
                "temperature": temperature,
                "top_p": top_p,
            }
        else:
            url = f"{self._base_url.rstrip('/')}/completions"
            payload = {
                "model": self._model_name,
                "prompt": prompt,
                "max_tokens": max_new_tokens,
                "temperature": temperature,
                "top_p": top_p,
            }

        with httpx.Client(timeout=120.0) as client:
            resp = client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()

        if use_chat:
            return data["choices"][0]["message"]["content"]
        else:
            return data["choices"][0]["text"]

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "openai_compat"

    @property
    def supports_activations(self) -> bool:
        return False

    @property
    def supports_logits(self) -> bool:
        return self._supports_logprobs
