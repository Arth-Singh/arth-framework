"""HuggingFace Local provider -- run any model locally via ``transformers``.

Supports quantisation (4-bit / 8-bit via bitsandbytes), flash-attention-2,
and ``device_map="auto"`` for multi-GPU inference.  Does **not** expose
residual-stream activations (no hook API), but does provide raw logit access.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from arth.core.providers.base import BaseProvider


class HuggingFaceLocalProvider(BaseProvider):
    """Provider backed by ``transformers.AutoModelForCausalLM``.

    Suitable for running any HuggingFace causal-LM checkpoint locally
    without requiring TransformerLens support.
    """

    def __init__(self) -> None:
        self._model: Any | None = None
        self._tokenizer: Any | None = None

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
        """Load a causal-LM model from HuggingFace.

        Args:
            model_name: HuggingFace model name or local path.
            device: Target device (``"cpu"``, ``"cuda"``, ``"auto"``).
            dtype: Weight dtype string.
            **kwargs: Extra options:
                ``trust_remote_code`` (bool),
                ``quantization`` (``"4bit"`` | ``"8bit"`` | ``None``),
                ``use_flash_attention`` (bool).
        """
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "The 'huggingface_local' provider requires the transformers library. "
                "Install it with:  pip install transformers"
            ) from exc

        trust_remote_code = kwargs.get("trust_remote_code", False)
        quantization = kwargs.get("quantization")
        use_flash_attn = kwargs.get("use_flash_attention", False)

        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(dtype, torch.float16)

        # --- Tokenizer -------------------------------------------------
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=trust_remote_code
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        # --- Model kwargs ----------------------------------------------
        model_kwargs: dict[str, Any] = {
            "trust_remote_code": trust_remote_code,
            "torch_dtype": torch_dtype,
        }

        # Quantization via bitsandbytes
        if quantization in ("4bit", "8bit"):
            model_kwargs["device_map"] = "auto"
            try:
                from transformers import BitsAndBytesConfig
            except ImportError as exc:
                raise ImportError(
                    f"Quantization ({quantization}) requires bitsandbytes. "
                    "Install with:  pip install bitsandbytes"
                ) from exc

            if quantization == "4bit":
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch_dtype,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
            else:
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_8bit=True,
                )
        elif device == "auto":
            model_kwargs["device_map"] = "auto"
        else:
            model_kwargs["device_map"] = None

        # Flash Attention 2
        if use_flash_attn:
            try:
                model_kwargs["attn_implementation"] = "flash_attention_2"
            except Exception:
                pass  # silently fall back if unavailable

        # --- Load model ------------------------------------------------
        self._model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

        # Move to device if we didn't use device_map
        if model_kwargs.get("device_map") is None and device != "cpu":
            self._model = self._model.to(device)

        self._model.eval()
        self._device = device

    def _ensure_loaded(self) -> None:
        if self._model is None or self._tokenizer is None:
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
        assert self._model is not None and self._tokenizer is not None

        inputs = self._tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True
        )
        # Move inputs to model device
        device = next(self._model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": kwargs.get("do_sample", False),
            "pad_token_id": self._tokenizer.pad_token_id,
        }
        # Forward any extra generation params
        for key in ("temperature", "top_p", "top_k", "repetition_penalty"):
            if key in kwargs:
                generation_kwargs[key] = kwargs[key]

        output_ids = self._model.generate(**inputs, **generation_kwargs)

        # Strip prompt tokens from each output
        prompt_lengths = inputs["input_ids"].shape[1]
        completions: list[str] = []
        for i in range(output_ids.shape[0]):
            new_tokens = output_ids[i, prompt_lengths:]
            text = self._tokenizer.decode(new_tokens, skip_special_tokens=True)
            completions.append(text)
        return completions

    # ------------------------------------------------------------------
    # Logits
    # ------------------------------------------------------------------

    @torch.no_grad()
    def get_logits(self, prompts: list[str]) -> Tensor:
        self._ensure_loaded()
        assert self._model is not None and self._tokenizer is not None

        inputs = self._tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True
        )
        device = next(self._model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        outputs = self._model(**inputs)
        return outputs.logits

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "huggingface_local"

    @property
    def tokenizer(self) -> Any:
        return self._tokenizer

    @property
    def supports_activations(self) -> bool:
        return False

    @property
    def supports_logits(self) -> bool:
        return True

    @property
    def n_layers(self) -> int | None:
        if self._model is None:
            return None
        config = getattr(self._model, "config", None)
        if config is None:
            return None
        # Different model architectures use different attribute names
        for attr in ("num_hidden_layers", "n_layer", "n_layers", "num_layers"):
            val = getattr(config, attr, None)
            if val is not None:
                return int(val)
        return None

    @property
    def d_model(self) -> int | None:
        if self._model is None:
            return None
        config = getattr(self._model, "config", None)
        if config is None:
            return None
        for attr in ("hidden_size", "d_model", "n_embd"):
            val = getattr(config, attr, None)
            if val is not None:
                return int(val)
        return None
