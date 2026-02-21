"""Pydantic data models for the mech interp toolkit."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any

import torch
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Dataset models
# ---------------------------------------------------------------------------

class ContrastPair(BaseModel):
    """A harmful / harmless prompt pair for difference-in-means extraction."""

    harmful: str = Field(..., description="Prompt the model should refuse")
    harmless: str = Field(..., description="Semantically similar prompt the model should answer")
    category: str = Field("", description="Category tag (e.g. 'violence_weapons')")


class SteeringPair(BaseModel):
    """A positive / negative completion pair for steering vector extraction."""

    positive: str = Field(..., description="Completion exhibiting the target behavior")
    negative: str = Field(..., description="Completion exhibiting the opposite behavior")
    behavior: str = Field("", description="Behavior label (e.g. 'honesty')")


class OverRefusalPrompt(BaseModel):
    """A benign prompt that models commonly over-refuse."""

    prompt: str = Field(..., description="Benign prompt that triggers false refusal")
    expected_behavior: str = Field(
        "comply", description="What the model should do (comply/answer)"
    )
    category: str = Field("", description="Why it triggers false refusal")


# ---------------------------------------------------------------------------
# Technique output model
# ---------------------------------------------------------------------------

class TechniqueResult(BaseModel):
    """Universal output format for all techniques."""

    model_config = {"arbitrary_types_allowed": True}

    technique: str = Field(..., description="Technique name (e.g. 'refusal_direction')")
    model_name: str = Field(..., description="Model that was analyzed")
    artifact_path: Path | None = Field(
        None, description="Path to saved artifact (.pt file)"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Technique-specific metadata (layers, dims, stats)",
    )
    scores: dict[str, float] = Field(
        default_factory=dict,
        description="Evaluation scores (ASR, refusal_rate, etc.)",
    )
    samples: list[dict[str, str]] = Field(
        default_factory=list,
        description="Before/after generation samples",
    )


# ---------------------------------------------------------------------------
# Configuration models
# ---------------------------------------------------------------------------

class ModelConfig(BaseModel):
    """Configuration for loading a model."""

    name: str = Field(..., description="HuggingFace model name or path")
    device: str = Field("cuda", description="Device to load model on")
    dtype: str = Field("float16", description="Model dtype")
    n_layers: int | None = Field(None, description="Override number of layers")
    trust_remote_code: bool = Field(False, description="Trust remote code in HF model")
    provider: str = Field(
        "transformer_lens",
        description="Provider name (transformer_lens, huggingface_local, huggingface_api, openai_compat, vllm_server)",
    )
    api_key: str | None = Field(None, description="API key for remote providers")
    base_url: str | None = Field(None, description="Base URL for API providers")
    quantization: str | None = Field(
        None, description="Quantization mode: '4bit', '8bit', or None"
    )

    def torch_dtype(self) -> torch.dtype:
        _map = {
            "float16": torch.float16,
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
        }
        return _map.get(self.dtype, torch.float16)


class ExperimentConfig(BaseModel):
    """Top-level experiment configuration."""

    model: ModelConfig
    techniques: list[str] = Field(
        default_factory=list, description="Technique names to run"
    )
    datasets: list[str] = Field(
        default_factory=list, description="Dataset file stems to load"
    )
    output_dir: Path = Field(Path("results"), description="Output directory")
    batch_size: int = Field(32, description="Batch size for activation collection")
    max_new_tokens: int = Field(128, description="Max tokens for generation")
    layers: list[int] | None = Field(
        None, description="Specific layers to analyze (None = all)"
    )
