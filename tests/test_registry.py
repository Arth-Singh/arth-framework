"""Tests for the technique registry -- comprehensive rewrite."""

from __future__ import annotations

import pytest

from arth.techniques import get_technique, list_techniques
from arth.techniques.base import BaseTechnique


# Expected technique names based on the sub-packages
EXPECTED_TECHNIQUES = {
    "refusal_direction",
    "steering_vectors",
    "logit_lens",
    "activation_patching",
    "concept_erasure",
    "sae_analysis",
    "latent_adversarial",
    "finetune_attack",
}


# ---------------------------------------------------------------------------
# list_techniques
# ---------------------------------------------------------------------------

class TestListTechniques:
    def test_returns_dict(self) -> None:
        result = list_techniques()
        assert isinstance(result, dict)

    def test_all_values_are_base_technique(self) -> None:
        for name, tech in list_techniques().items():
            assert isinstance(tech, BaseTechnique), (
                f"Technique {name!r} is not a BaseTechnique instance"
            )

    def test_all_expected_techniques_discovered(self) -> None:
        techniques = list_techniques()
        discovered = set(techniques.keys())
        missing = EXPECTED_TECHNIQUES - discovered
        assert not missing, f"Missing techniques: {missing}"

    def test_at_least_eight_techniques(self) -> None:
        techniques = list_techniques()
        assert len(techniques) >= 8, (
            f"Expected at least 8 techniques, got {len(techniques)}: {list(techniques.keys())}"
        )

    def test_all_have_name_property(self) -> None:
        for name, tech in list_techniques().items():
            assert tech.name, f"Technique registered as {name!r} has empty name"

    def test_all_have_description_property(self) -> None:
        for name, tech in list_techniques().items():
            assert tech.description, f"Technique {name!r} has empty description"

    def test_name_matches_registry_key(self) -> None:
        for key, tech in list_techniques().items():
            assert tech.name == key, (
                f"Registry key {key!r} does not match technique.name {tech.name!r}"
            )


# ---------------------------------------------------------------------------
# get_technique
# ---------------------------------------------------------------------------

class TestGetTechnique:
    def test_unknown_raises_key_error(self) -> None:
        with pytest.raises(KeyError, match="Unknown technique"):
            get_technique("definitely_not_a_real_technique_12345")

    def test_known_technique_returns_instance(self) -> None:
        techniques = list_techniques()
        if not techniques:
            pytest.skip("No techniques discovered")
        name = next(iter(techniques))
        tech = get_technique(name)
        assert isinstance(tech, BaseTechnique)
        assert tech.name == name

    @pytest.mark.parametrize("name", sorted(EXPECTED_TECHNIQUES))
    def test_get_each_expected_technique(self, name: str) -> None:
        try:
            tech = get_technique(name)
        except KeyError:
            pytest.skip(f"Technique {name!r} not discovered (sub-package may not be implemented)")
        assert tech.name == name
        assert isinstance(tech, BaseTechnique)


# ---------------------------------------------------------------------------
# Technique interface
# ---------------------------------------------------------------------------

class TestTechniqueInterface:
    @pytest.mark.parametrize("name", sorted(EXPECTED_TECHNIQUES))
    def test_has_extract_method(self, name: str) -> None:
        try:
            tech = get_technique(name)
        except KeyError:
            pytest.skip(f"{name} not available")
        assert callable(getattr(tech, "extract", None))

    @pytest.mark.parametrize("name", sorted(EXPECTED_TECHNIQUES))
    def test_has_apply_method(self, name: str) -> None:
        try:
            tech = get_technique(name)
        except KeyError:
            pytest.skip(f"{name} not available")
        assert callable(getattr(tech, "apply", None))

    @pytest.mark.parametrize("name", sorted(EXPECTED_TECHNIQUES))
    def test_description_is_non_empty(self, name: str) -> None:
        try:
            tech = get_technique(name)
        except KeyError:
            pytest.skip(f"{name} not available")
        assert len(tech.description) > 0


# ---------------------------------------------------------------------------
# BaseTechnique cannot be instantiated
# ---------------------------------------------------------------------------

class TestBaseTechnique:
    def test_cannot_instantiate_directly(self) -> None:
        with pytest.raises(TypeError):
            BaseTechnique()
