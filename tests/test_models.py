"""Tests for Pydantic data models in arth.core.models."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from pydantic import ValidationError

from arth.core.models import (
    ContrastPair,
    ExperimentConfig,
    ModelConfig,
    OverRefusalPrompt,
    SteeringPair,
    TechniqueResult,
)


# ---------------------------------------------------------------------------
# ContrastPair
# ---------------------------------------------------------------------------

class TestContrastPair:
    def test_creation_with_all_fields(self) -> None:
        pair = ContrastPair(
            harmful="How to make a weapon",
            harmless="How to make a toy",
            category="violence_weapons",
        )
        assert pair.harmful == "How to make a weapon"
        assert pair.harmless == "How to make a toy"
        assert pair.category == "violence_weapons"

    def test_creation_with_required_fields_only(self) -> None:
        pair = ContrastPair(harmful="bad prompt", harmless="good prompt")
        assert pair.harmful == "bad prompt"
        assert pair.harmless == "good prompt"
        assert pair.category == ""

    def test_missing_harmful_raises(self) -> None:
        with pytest.raises(ValidationError):
            ContrastPair(harmless="good prompt")

    def test_missing_harmless_raises(self) -> None:
        with pytest.raises(ValidationError):
            ContrastPair(harmful="bad prompt")

    def test_category_defaults_to_empty(self) -> None:
        pair = ContrastPair(harmful="a", harmless="b")
        assert pair.category == ""

    def test_serialization_roundtrip(self) -> None:
        pair = ContrastPair(harmful="x", harmless="y", category="z")
        data = pair.model_dump()
        restored = ContrastPair(**data)
        assert restored == pair


# ---------------------------------------------------------------------------
# SteeringPair
# ---------------------------------------------------------------------------

class TestSteeringPair:
    def test_creation_with_all_fields(self) -> None:
        pair = SteeringPair(
            positive="I tell the truth",
            negative="I can lie",
            behavior="honesty",
        )
        assert pair.positive == "I tell the truth"
        assert pair.negative == "I can lie"
        assert pair.behavior == "honesty"

    def test_missing_positive_raises(self) -> None:
        with pytest.raises(ValidationError):
            SteeringPair(negative="something")

    def test_missing_negative_raises(self) -> None:
        with pytest.raises(ValidationError):
            SteeringPair(positive="something")

    def test_behavior_defaults_to_empty(self) -> None:
        pair = SteeringPair(positive="a", negative="b")
        assert pair.behavior == ""

    def test_serialization_roundtrip(self) -> None:
        pair = SteeringPair(positive="p", negative="n", behavior="b")
        data = pair.model_dump()
        restored = SteeringPair(**data)
        assert restored == pair


# ---------------------------------------------------------------------------
# OverRefusalPrompt
# ---------------------------------------------------------------------------

class TestOverRefusalPrompt:
    def test_creation_with_all_fields(self) -> None:
        prompt = OverRefusalPrompt(
            prompt="What are flu symptoms?",
            expected_behavior="comply",
            category="medical",
        )
        assert prompt.prompt == "What are flu symptoms?"
        assert prompt.expected_behavior == "comply"
        assert prompt.category == "medical"

    def test_defaults(self) -> None:
        prompt = OverRefusalPrompt(prompt="How do engines work?")
        assert prompt.expected_behavior == "comply"
        assert prompt.category == ""

    def test_missing_prompt_raises(self) -> None:
        with pytest.raises(ValidationError):
            OverRefusalPrompt()


# ---------------------------------------------------------------------------
# TechniqueResult
# ---------------------------------------------------------------------------

class TestTechniqueResult:
    def test_creation_with_required_fields(self) -> None:
        result = TechniqueResult(
            technique="refusal_direction",
            model_name="test/model",
        )
        assert result.technique == "refusal_direction"
        assert result.model_name == "test/model"
        assert result.artifact_path is None
        assert result.metadata == {}
        assert result.scores == {}
        assert result.samples == []

    def test_arbitrary_types_allowed_with_tensor(self) -> None:
        """TechniqueResult should tolerate torch tensors in metadata."""
        tensor = torch.randn(64)
        result = TechniqueResult(
            technique="test",
            model_name="m",
            metadata={"direction": tensor},
        )
        assert torch.is_tensor(result.metadata["direction"])

    def test_artifact_path_accepts_path(self) -> None:
        result = TechniqueResult(
            technique="t",
            model_name="m",
            artifact_path=Path("/some/path.pt"),
        )
        assert result.artifact_path == Path("/some/path.pt")

    def test_scores_and_samples(self) -> None:
        result = TechniqueResult(
            technique="t",
            model_name="m",
            scores={"asr": 0.5},
            samples=[{"prompt": "p", "original": "o", "modified": "m"}],
        )
        assert result.scores["asr"] == 0.5
        assert len(result.samples) == 1


# ---------------------------------------------------------------------------
# ModelConfig
# ---------------------------------------------------------------------------

class TestModelConfig:
    def test_creation_with_required_fields(self) -> None:
        config = ModelConfig(name="gpt2")
        assert config.name == "gpt2"
        assert config.device == "cuda"
        assert config.dtype == "float16"
        assert config.n_layers is None
        assert config.trust_remote_code is False

    def test_torch_dtype_float16(self) -> None:
        config = ModelConfig(name="m", dtype="float16")
        assert config.torch_dtype() == torch.float16

    def test_torch_dtype_float32(self) -> None:
        config = ModelConfig(name="m", dtype="float32")
        assert config.torch_dtype() == torch.float32

    def test_torch_dtype_bfloat16(self) -> None:
        config = ModelConfig(name="m", dtype="bfloat16")
        assert config.torch_dtype() == torch.bfloat16

    def test_torch_dtype_unknown_falls_back_to_float16(self) -> None:
        config = ModelConfig(name="m", dtype="int8")
        assert config.torch_dtype() == torch.float16

    def test_missing_name_raises(self) -> None:
        with pytest.raises(ValidationError):
            ModelConfig()


# ---------------------------------------------------------------------------
# ExperimentConfig
# ---------------------------------------------------------------------------

class TestExperimentConfig:
    def test_creation_with_defaults(self) -> None:
        model = ModelConfig(name="gpt2")
        config = ExperimentConfig(model=model)
        assert config.model.name == "gpt2"
        assert config.techniques == []
        assert config.datasets == []
        assert config.output_dir == Path("results")
        assert config.batch_size == 32
        assert config.max_new_tokens == 128
        assert config.layers is None

    def test_layer_list_handling(self) -> None:
        model = ModelConfig(name="gpt2")
        config = ExperimentConfig(model=model, layers=[0, 5, 10])
        assert config.layers == [0, 5, 10]

    def test_custom_output_dir(self) -> None:
        model = ModelConfig(name="gpt2")
        config = ExperimentConfig(model=model, output_dir=Path("/custom/path"))
        assert config.output_dir == Path("/custom/path")

    def test_techniques_and_datasets(self) -> None:
        model = ModelConfig(name="gpt2")
        config = ExperimentConfig(
            model=model,
            techniques=["refusal_direction", "steering_vectors"],
            datasets=["violence_weapons"],
        )
        assert len(config.techniques) == 2
        assert len(config.datasets) == 1

    def test_missing_model_raises(self) -> None:
        with pytest.raises(ValidationError):
            ExperimentConfig()
