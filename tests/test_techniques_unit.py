"""Unit tests for technique internals using mocked backends."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from arth.core.models import ExperimentConfig, ModelConfig, TechniqueResult
from arth.techniques.base import BaseTechnique
from arth.techniques import get_technique, list_techniques


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_experiment_config(tmp_path: Path) -> ExperimentConfig:
    """Create an ExperimentConfig pointing at a temp directory."""
    return ExperimentConfig(
        model=ModelConfig(name="mock-model", device="cpu", dtype="float32"),
        output_dir=tmp_path,
        batch_size=4,
        layers=[0, 1, 2, 3],
    )


# ---------------------------------------------------------------------------
# RefusalDirection
# ---------------------------------------------------------------------------

class TestRefusalDirectionUnit:
    def test_extract_returns_technique_result(
        self, mock_model_backend: MagicMock, sample_contrast_pairs, tmp_path: Path
    ) -> None:
        try:
            tech = get_technique("refusal_direction")
        except KeyError:
            pytest.skip("refusal_direction not available")

        config = _make_experiment_config(tmp_path)
        result = tech.extract(mock_model_backend, sample_contrast_pairs, config=config)
        assert isinstance(result, TechniqueResult)
        assert result.technique == "refusal_direction"
        assert result.model_name == "mock-model"
        assert result.artifact_path is not None
        assert result.artifact_path.exists()

    def test_extract_metadata_has_best_layer(
        self, mock_model_backend: MagicMock, sample_contrast_pairs, tmp_path: Path
    ) -> None:
        try:
            tech = get_technique("refusal_direction")
        except KeyError:
            pytest.skip("refusal_direction not available")

        config = _make_experiment_config(tmp_path)
        result = tech.extract(mock_model_backend, sample_contrast_pairs, config=config)
        assert "best_layer" in result.metadata
        assert "best_norm" in result.metadata

    def test_name_is_refusal_direction(self) -> None:
        try:
            tech = get_technique("refusal_direction")
        except KeyError:
            pytest.skip("refusal_direction not available")
        assert tech.name == "refusal_direction"


# ---------------------------------------------------------------------------
# LogitLens
# ---------------------------------------------------------------------------

class TestLogitLensUnit:
    def test_resolve_token_ids_maps_tokens(self, mock_model_backend: MagicMock) -> None:
        try:
            tech = get_technique("logit_lens")
        except KeyError:
            pytest.skip("logit_lens not available")

        # The mock tokenizer.encode returns [42] for any input
        tokens = ["I", "sorry", "cannot"]
        result = tech._resolve_token_ids(mock_model_backend, tokens)
        assert isinstance(result, dict)
        for tok in tokens:
            assert tok in result
            assert result[tok] == 42

    def test_name_is_logit_lens(self) -> None:
        try:
            tech = get_technique("logit_lens")
        except KeyError:
            pytest.skip("logit_lens not available")
        assert tech.name == "logit_lens"


# ---------------------------------------------------------------------------
# BaseTechnique.evaluate (default implementation)
# ---------------------------------------------------------------------------

class TestBaseTechniqueEvaluate:
    def test_evaluate_empty_results(self) -> None:
        """The default evaluate method on BaseTechnique should handle empty results."""
        techniques = list_techniques()
        if not techniques:
            pytest.skip("No techniques available")

        # Use a technique that inherits the default evaluate (not logit_lens which overrides)
        for name, tech in techniques.items():
            result = tech.evaluate([])
            assert isinstance(result, dict)
            break  # just test one

    def test_evaluate_with_refusal_bypass(self) -> None:
        """Default evaluate should detect when a refusal is bypassed."""
        try:
            tech = get_technique("refusal_direction")
        except KeyError:
            pytest.skip("refusal_direction not available")

        results = [
            {"original": "I'm sorry, I cannot help.", "modified": "Sure, here is the answer."},
            {"original": "I'm sorry, I won't do that.", "modified": "I'm sorry, I still cannot do that."},
        ]
        scores = tech.evaluate(results)
        assert "asr" in scores
        assert "refusal_rate_original" in scores
        assert "refusal_rate_modified" in scores
        # 2 original refusals, 1 bypassed => ASR = 0.5
        assert abs(scores["asr"] - 0.5) < 1e-6

    def test_evaluate_no_refusals(self) -> None:
        """When no originals are refusals, ASR should be 0."""
        try:
            tech = get_technique("refusal_direction")
        except KeyError:
            pytest.skip("refusal_direction not available")

        results = [
            {"original": "The answer is 42.", "modified": "Yes, 42 is correct."},
        ]
        scores = tech.evaluate(results)
        assert scores["asr"] == 0.0
        assert scores["refusal_rate_original"] == 0.0


# ---------------------------------------------------------------------------
# Technique name and description checks
# ---------------------------------------------------------------------------

EXPECTED_NAMES = {
    "refusal_direction",
    "steering_vectors",
    "logit_lens",
    "activation_patching",
    "concept_erasure",
    "sae_analysis",
    "latent_adversarial",
    "finetune_attack",
}


class TestTechniqueNameAndDescription:
    @pytest.mark.parametrize("expected_name", sorted(EXPECTED_NAMES))
    def test_name_matches_expected(self, expected_name: str) -> None:
        try:
            tech = get_technique(expected_name)
        except KeyError:
            pytest.skip(f"{expected_name} not available")
        assert tech.name == expected_name

    @pytest.mark.parametrize("expected_name", sorted(EXPECTED_NAMES))
    def test_description_is_non_empty(self, expected_name: str) -> None:
        try:
            tech = get_technique(expected_name)
        except KeyError:
            pytest.skip(f"{expected_name} not available")
        assert isinstance(tech.description, str)
        assert len(tech.description) > 0


# ---------------------------------------------------------------------------
# SteeringVectors unit test
# ---------------------------------------------------------------------------

class TestSteeringVectorsUnit:
    def test_extract_returns_technique_result(
        self, mock_model_backend: MagicMock, sample_contrast_pairs, tmp_path: Path
    ) -> None:
        try:
            tech = get_technique("steering_vectors")
        except KeyError:
            pytest.skip("steering_vectors not available")

        config = _make_experiment_config(tmp_path)
        result = tech.extract(mock_model_backend, sample_contrast_pairs, config=config)
        assert isinstance(result, TechniqueResult)
        assert result.technique == "steering_vectors"
        assert result.artifact_path is not None
        assert result.artifact_path.exists()

    def test_extract_metadata_has_variance(
        self, mock_model_backend: MagicMock, sample_contrast_pairs, tmp_path: Path
    ) -> None:
        try:
            tech = get_technique("steering_vectors")
        except KeyError:
            pytest.skip("steering_vectors not available")

        config = _make_experiment_config(tmp_path)
        result = tech.extract(mock_model_backend, sample_contrast_pairs, config=config)
        assert "best_layer" in result.metadata
        assert "explained_variance" in result.metadata
