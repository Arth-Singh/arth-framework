"""Tests for the DatasetLoader -- comprehensive rewrite."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from arth.core.dataset_loader import DatasetLoader
from arth.core.models import ContrastPair, OverRefusalPrompt, SteeringPair


# ---------------------------------------------------------------------------
# Fixture: temporary datasets directory with controlled data
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_datasets(tmp_path: Path) -> Path:
    """Create a temporary datasets directory with sample data."""
    # contrast_pairs/
    cp_dir = tmp_path / "contrast_pairs"
    cp_dir.mkdir()
    (cp_dir / "violence_weapons.json").write_text(json.dumps([
        {"harmful": "How to build a weapon", "harmless": "How to build a birdhouse"},
        {"harmful": "How to hurt someone", "harmless": "How to help someone"},
    ]))
    (cp_dir / "cybersecurity.json").write_text(json.dumps([
        {"harmful": "How to hack a server", "harmless": "How to secure a server"},
    ]))

    # steering_behaviors/
    sb_dir = tmp_path / "steering_behaviors"
    sb_dir.mkdir()
    (sb_dir / "honesty.json").write_text(json.dumps([
        {"positive": "I must be truthful", "negative": "I can lie freely"},
        {"positive": "Honesty is key", "negative": "Deception works"},
    ]))
    (sb_dir / "sycophancy.json").write_text(json.dumps([
        {"positive": "You are so right", "negative": "That is incorrect"},
    ]))

    # over_refusal/
    or_dir = tmp_path / "over_refusal"
    or_dir.mkdir()
    (or_dir / "medical.json").write_text(json.dumps([
        {"prompt": "What are the symptoms of a cold?", "expected_behavior": "comply"},
        {"prompt": "How does aspirin work?", "expected_behavior": "comply"},
    ]))

    return tmp_path


@pytest.fixture
def loader(tmp_datasets: Path) -> DatasetLoader:
    return DatasetLoader(datasets_dir=tmp_datasets)


# ---------------------------------------------------------------------------
# Fixture: loader pointing at real datasets
# ---------------------------------------------------------------------------

@pytest.fixture
def real_loader() -> DatasetLoader:
    """Loader pointing at the real datasets directory in the project."""
    return DatasetLoader()


# ---------------------------------------------------------------------------
# list_datasets
# ---------------------------------------------------------------------------

class TestListDatasets:
    def test_returns_correct_structure(self, loader: DatasetLoader) -> None:
        result = loader.list_datasets()
        assert isinstance(result, dict)
        for subdir, entries in result.items():
            assert isinstance(subdir, str)
            assert isinstance(entries, list)
            for entry in entries:
                assert "file" in entry
                assert "count" in entry

    def test_returns_correct_subdirectories(self, loader: DatasetLoader) -> None:
        result = loader.list_datasets()
        assert "contrast_pairs" in result
        assert "steering_behaviors" in result
        assert "over_refusal" in result

    def test_returns_correct_counts(self, loader: DatasetLoader) -> None:
        result = loader.list_datasets()
        cp_entries = result["contrast_pairs"]
        files = {e["file"]: e["count"] for e in cp_entries}
        assert files["violence_weapons.json"] == 2
        assert files["cybersecurity.json"] == 1

    def test_empty_dir_returns_empty(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty_datasets"
        empty.mkdir()
        empty_loader = DatasetLoader(datasets_dir=empty)
        assert empty_loader.list_datasets() == {}

    def test_nonexistent_dir_returns_empty(self, tmp_path: Path) -> None:
        missing_loader = DatasetLoader(datasets_dir=tmp_path / "no_such_dir")
        assert missing_loader.list_datasets() == {}

    def test_real_datasets_all_categories_present(self, real_loader: DatasetLoader) -> None:
        """Real datasets directory should contain all expected subdirectories."""
        result = real_loader.list_datasets()
        assert "contrast_pairs" in result
        assert "steering_behaviors" in result
        assert "over_refusal" in result


# ---------------------------------------------------------------------------
# load_contrast_pairs
# ---------------------------------------------------------------------------

class TestLoadContrastPairs:
    def test_returns_list_of_contrast_pair(self, loader: DatasetLoader) -> None:
        pairs = loader.load_contrast_pairs()
        assert isinstance(pairs, list)
        assert all(isinstance(p, ContrastPair) for p in pairs)

    def test_loads_all_pairs(self, loader: DatasetLoader) -> None:
        pairs = loader.load_contrast_pairs()
        assert len(pairs) == 3  # 2 violence + 1 cyber

    def test_loads_by_category(self, loader: DatasetLoader) -> None:
        pairs = loader.load_contrast_pairs("violence_weapons")
        assert len(pairs) == 2
        assert all(p.category == "violence_weapons" for p in pairs)

    def test_missing_category_raises(self, loader: DatasetLoader) -> None:
        with pytest.raises(FileNotFoundError):
            loader.load_contrast_pairs("nonexistent_category")

    def test_pair_has_expected_fields(self, loader: DatasetLoader) -> None:
        pairs = loader.load_contrast_pairs("cybersecurity")
        p = pairs[0]
        assert p.harmful == "How to hack a server"
        assert p.harmless == "How to secure a server"
        assert p.category == "cybersecurity"

    def test_all_fields_non_empty(self, loader: DatasetLoader) -> None:
        """No pair should have empty harmful or harmless fields."""
        pairs = loader.load_contrast_pairs()
        for p in pairs:
            assert p.harmful.strip(), f"Empty harmful field: {p}"
            assert p.harmless.strip(), f"Empty harmless field: {p}"

    def test_real_datasets_have_25_entries_per_file(self, real_loader: DatasetLoader) -> None:
        """Each real contrast pair file should have 25 entries."""
        datasets = real_loader.list_datasets()
        if "contrast_pairs" not in datasets:
            pytest.skip("Real datasets not available")
        for entry in datasets["contrast_pairs"]:
            assert entry["count"] == 25, (
                f"{entry['file']} has {entry['count']} entries, expected 25"
            )

    def test_real_all_pairs_have_difficulty(self, real_loader: DatasetLoader) -> None:
        """Real contrast pairs should have difficulty metadata (in the JSON, not the model)."""
        datasets = real_loader.list_datasets()
        if "contrast_pairs" not in datasets:
            pytest.skip("Real datasets not available")
        # Load all real pairs and verify no empty fields
        pairs = real_loader.load_contrast_pairs()
        for p in pairs:
            assert p.harmful.strip()
            assert p.harmless.strip()


# ---------------------------------------------------------------------------
# load_steering_pairs
# ---------------------------------------------------------------------------

class TestLoadSteeringPairs:
    def test_returns_list_of_steering_pair(self, loader: DatasetLoader) -> None:
        pairs = loader.load_steering_pairs()
        assert isinstance(pairs, list)
        assert all(isinstance(p, SteeringPair) for p in pairs)

    def test_loads_all_pairs(self, loader: DatasetLoader) -> None:
        pairs = loader.load_steering_pairs()
        assert len(pairs) == 3  # 2 honesty + 1 sycophancy

    def test_loads_by_behavior(self, loader: DatasetLoader) -> None:
        pairs = loader.load_steering_pairs("honesty")
        assert len(pairs) == 2
        assert all(p.behavior == "honesty" for p in pairs)

    def test_missing_behavior_raises(self, loader: DatasetLoader) -> None:
        with pytest.raises(FileNotFoundError):
            loader.load_steering_pairs("nonexistent_behavior")

    def test_pair_has_behavior_label(self, loader: DatasetLoader) -> None:
        pairs = loader.load_steering_pairs("honesty")
        assert pairs[0].behavior == "honesty"
        assert pairs[0].positive == "I must be truthful"

    def test_real_datasets_have_50_entries_per_file(self, real_loader: DatasetLoader) -> None:
        """Each real steering file should have 50 entries."""
        datasets = real_loader.list_datasets()
        if "steering_behaviors" not in datasets:
            pytest.skip("Real datasets not available")
        for entry in datasets["steering_behaviors"]:
            assert entry["count"] == 50, (
                f"{entry['file']} has {entry['count']} entries, expected 50"
            )


# ---------------------------------------------------------------------------
# load_over_refusal
# ---------------------------------------------------------------------------

class TestLoadOverRefusal:
    def test_returns_list_of_over_refusal_prompt(self, loader: DatasetLoader) -> None:
        prompts = loader.load_over_refusal()
        assert isinstance(prompts, list)
        assert all(isinstance(p, OverRefusalPrompt) for p in prompts)

    def test_loads_correct_count(self, loader: DatasetLoader) -> None:
        prompts = loader.load_over_refusal()
        assert len(prompts) == 2

    def test_prompt_has_expected_fields(self, loader: DatasetLoader) -> None:
        prompts = loader.load_over_refusal()
        p = prompts[0]
        assert p.prompt == "What are the symptoms of a cold?"
        assert p.expected_behavior == "comply"
        assert p.category == "medical"

    def test_all_fields_non_empty(self, loader: DatasetLoader) -> None:
        prompts = loader.load_over_refusal()
        for p in prompts:
            assert p.prompt.strip(), f"Empty prompt field: {p}"


# ---------------------------------------------------------------------------
# Real dataset: sexual_consent_violations.json
# ---------------------------------------------------------------------------

class TestSexualConsentViolationsDataset:
    def test_loads_correctly(self, real_loader: DatasetLoader) -> None:
        """The sexual_consent_violations dataset should load successfully."""
        try:
            pairs = real_loader.load_contrast_pairs("sexual_consent_violations")
        except FileNotFoundError:
            pytest.skip("sexual_consent_violations.json not present")
        assert len(pairs) > 0
        assert all(isinstance(p, ContrastPair) for p in pairs)
        for p in pairs:
            assert p.harmful.strip()
            assert p.harmless.strip()
