"""Dataset loading and validation."""

from __future__ import annotations

import json
from pathlib import Path

from arth.core.models import ContrastPair, OverRefusalPrompt, SteeringPair

# Project root is three levels up from this file:
# arth-mech-interp/arth/core/dataset_loader.py -> arth-mech-interp/
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class DatasetLoader:
    """Load and validate datasets from the ``datasets/`` directory."""

    def __init__(self, datasets_dir: Path | None = None) -> None:
        self.datasets_dir = Path(datasets_dir) if datasets_dir else _PROJECT_ROOT / "datasets"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_contrast_pairs(self, category: str | None = None) -> list[ContrastPair]:
        """Load contrast pairs for refusal direction extraction.

        Args:
            category: If given, load only the file named ``<category>.json``
                inside ``contrast_pairs/``.  Otherwise load all JSON files.

        Returns:
            List of validated :class:`ContrastPair` objects.
        """
        pairs_dir = self.datasets_dir / "contrast_pairs"
        raw = self._load_json_dir(pairs_dir, category)
        pairs: list[ContrastPair] = []
        for file_stem, items in raw.items():
            for item in items:
                item.setdefault("category", file_stem)
                pairs.append(ContrastPair(**item))
        return pairs

    def load_steering_pairs(self, behavior: str | None = None) -> list[SteeringPair]:
        """Load steering behavior pairs.

        Args:
            behavior: If given, load only ``<behavior>.json``.

        Returns:
            List of validated :class:`SteeringPair` objects.
        """
        pairs_dir = self.datasets_dir / "steering_behaviors"
        raw = self._load_json_dir(pairs_dir, behavior)
        pairs: list[SteeringPair] = []
        for file_stem, items in raw.items():
            for item in items:
                item.setdefault("behavior", file_stem)
                pairs.append(SteeringPair(**item))
        return pairs

    def load_over_refusal(self) -> list[OverRefusalPrompt]:
        """Load over-refusal prompts.

        Returns:
            List of validated :class:`OverRefusalPrompt` objects.
        """
        over_dir = self.datasets_dir / "over_refusal"
        raw = self._load_json_dir(over_dir, name=None)
        prompts: list[OverRefusalPrompt] = []
        for file_stem, items in raw.items():
            for item in items:
                item.setdefault("category", file_stem)
                prompts.append(OverRefusalPrompt(**item))
        return prompts

    def list_datasets(self) -> dict[str, list[dict]]:
        """List all available datasets with file names and counts.

        Returns:
            Dict mapping subdirectory name to a list of dicts with keys
            ``"file"`` and ``"count"``.
        """
        result: dict[str, list[dict]] = {}
        if not self.datasets_dir.exists():
            return result
        for subdir in sorted(self.datasets_dir.iterdir()):
            if not subdir.is_dir():
                continue
            entries: list[dict] = []
            for json_file in sorted(subdir.glob("*.json")):
                with open(json_file) as f:
                    data = json.load(f)
                entries.append({"file": json_file.name, "count": len(data)})
            result[subdir.name] = entries
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_json_dir(
        directory: Path, name: str | None
    ) -> dict[str, list[dict]]:
        """Load JSON files from a directory.

        Args:
            directory: Path to the dataset subdirectory.
            name: If given, load only ``<name>.json``.

        Returns:
            Dict mapping file stem to list of dicts.
        """
        if not directory.exists():
            return {}

        if name is not None:
            target = directory / f"{name}.json"
            if not target.exists():
                raise FileNotFoundError(
                    f"Dataset file not found: {target}"
                )
            with open(target) as f:
                return {name: json.load(f)}

        result: dict[str, list[dict]] = {}
        for json_file in sorted(directory.glob("*.json")):
            with open(json_file) as f:
                result[json_file.stem] = json.load(f)
        return result
