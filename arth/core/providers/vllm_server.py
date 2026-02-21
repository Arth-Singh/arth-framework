"""vLLM / TGI server provider -- connect to a running inference server.

Communicates with a vLLM or Text Generation Inference (TGI) server over
HTTP using the OpenAI-compatible ``/v1/completions`` endpoint that both
frameworks expose.  Supports logprobs when the server is configured to
return them.
"""

from __future__ import annotations

import os
from typing import Any

from arth.core.providers.base import BaseProvider


class VLLMServerProvider(BaseProvider):
    """Provider that connects to a running vLLM or TGI inference server.

    By default connects to ``http://localhost:8000``.  Override with
    ``base_url`` kwarg or ``--base-url`` CLI flag.
    """

    _DEFAULT_URL = "http://localhost:8000"

    def __init__(self) -> None:
        self._model_name: str | None = None
        self._base_url: str = self._DEFAULT_URL
        self._api_key: str | None = None

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
        """Configure connection to a vLLM/TGI server.

        Args:
            model_name: Model ID served by the inference server.  Must match
                the model the server was started with.
            device: Ignored (the server manages its own devices).
            dtype: Ignored.
            **kwargs: Extra options:
                ``base_url`` (str): Server URL (default ``http://localhost:8000``).
                ``api_key`` (str): API key if the server requires auth.
        """
        # Verify httpx is available
        try:
            import httpx  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "The 'vllm_server' provider requires httpx for HTTP requests. "
                "Install it with:  pip install httpx"
            ) from exc

        self._model_name = model_name
        self._base_url = (
            kwargs.get("base_url")
            or os.environ.get("VLLM_BASE_URL")
            or self._DEFAULT_URL
        )
        self._api_key = kwargs.get("api_key") or os.environ.get("VLLM_API_KEY")

    def _ensure_loaded(self) -> None:
        if self._model_name is None:
            raise RuntimeError(
                "Provider not initialised. Call provider.load(model_name) first."
            )

    def _headers(self) -> dict[str, str]:
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

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
        import httpx

        temperature = kwargs.get("temperature", 0.7)
        top_p = kwargs.get("top_p", 1.0)
        use_chat = kwargs.get("use_chat", False)  # vLLM default: raw completions

        completions: list[str] = []

        if use_chat:
            url = f"{self._base_url.rstrip('/')}/v1/chat/completions"
            for prompt in prompts:
                payload = {
                    "model": self._model_name,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_new_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                }
                try:
                    with httpx.Client(timeout=300.0) as client:
                        resp = client.post(url, json=payload, headers=self._headers())
                        resp.raise_for_status()
                        data = resp.json()
                    completions.append(data["choices"][0]["message"]["content"])
                except Exception as exc:
                    completions.append(f"[ERROR: {exc}]")
        else:
            # Batch-capable: vLLM /v1/completions accepts a list of prompts
            url = f"{self._base_url.rstrip('/')}/v1/completions"
            payload: dict[str, Any] = {
                "model": self._model_name,
                "prompt": prompts,
                "max_tokens": max_new_tokens,
                "temperature": temperature,
                "top_p": top_p,
            }

            # Request logprobs if asked
            logprobs_n = kwargs.get("logprobs")
            if logprobs_n is not None:
                payload["logprobs"] = logprobs_n

            try:
                with httpx.Client(timeout=300.0) as client:
                    resp = client.post(url, json=payload, headers=self._headers())
                    resp.raise_for_status()
                    data = resp.json()

                # vLLM returns one choice per prompt in the same order
                for choice in sorted(data["choices"], key=lambda c: c["index"]):
                    completions.append(choice["text"])
            except Exception as exc:
                completions = [f"[ERROR: {exc}]"] * len(prompts)

        return completions

    # ------------------------------------------------------------------
    # Logits  (via logprobs endpoint)
    # ------------------------------------------------------------------

    def get_logits(self, prompts: list[str]) -> None:
        """vLLM supports logprobs via the generate API, but not raw logit tensors.

        Use ``generate(..., logprobs=5)`` to get top-k logprobs in the
        response.  Full logit tensors are not available over HTTP.
        """
        return None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "vllm_server"

    @property
    def supports_activations(self) -> bool:
        return False

    @property
    def supports_logits(self) -> bool:
        # vLLM supports logprobs (partial logit info) but not full tensors
        return True
