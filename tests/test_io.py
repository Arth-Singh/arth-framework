"""Tests for I/O utilities in arth.utils.io."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from arth.utils.io import load_results, load_vector, save_results, save_vector


# ---------------------------------------------------------------------------
# save_vector / load_vector roundtrip
# ---------------------------------------------------------------------------

class TestSaveLoadVector:
    def test_roundtrip(self, tmp_path: Path) -> None:
        tensor = torch.randn(64)
        path = tmp_path / "test.pt"
        save_vector(tensor, path)
        loaded_tensor, loaded_meta = load_vector(path)
        assert torch.allclose(tensor, loaded_tensor)
        assert loaded_meta == {}

    def test_with_metadata(self, tmp_path: Path) -> None:
        tensor = torch.randn(128)
        metadata = {"layer": 5, "model": "test-model", "norm": 3.14}
        path = tmp_path / "test_meta.pt"
        save_vector(tensor, path, metadata=metadata)
        loaded_tensor, loaded_meta = load_vector(path)
        assert torch.allclose(tensor, loaded_tensor)
        assert loaded_meta == metadata

    def test_loads_to_cpu(self, tmp_path: Path) -> None:
        tensor = torch.randn(32)
        path = tmp_path / "cpu_test.pt"
        save_vector(tensor, path)
        loaded_tensor, _ = load_vector(path)
        assert loaded_tensor.device == torch.device("cpu")

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        tensor = torch.randn(16)
        path = tmp_path / "nested" / "deep" / "dir" / "test.pt"
        result_path = save_vector(tensor, path)
        assert result_path.exists()
        loaded_tensor, _ = load_vector(path)
        assert torch.allclose(tensor, loaded_tensor)

    def test_nonexistent_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_vector(tmp_path / "nonexistent.pt")

    @pytest.mark.parametrize(
        "shape",
        [(64,), (10, 64), (3, 5, 64), (2, 3, 4, 8)],
        ids=["1d", "2d", "3d", "4d"],
    )
    def test_various_tensor_shapes(self, tmp_path: Path, shape: tuple) -> None:
        tensor = torch.randn(shape)
        path = tmp_path / f"shape_{'x'.join(map(str, shape))}.pt"
        save_vector(tensor, path)
        loaded_tensor, _ = load_vector(path)
        assert loaded_tensor.shape == shape
        assert torch.allclose(tensor, loaded_tensor)

    def test_returns_path(self, tmp_path: Path) -> None:
        tensor = torch.randn(8)
        path = tmp_path / "ret.pt"
        result = save_vector(tensor, path)
        assert result == path


# ---------------------------------------------------------------------------
# save_results / load_results roundtrip
# ---------------------------------------------------------------------------

class TestSaveLoadResults:
    def test_roundtrip(self, tmp_path: Path) -> None:
        data = {"metric": 0.85, "name": "test_run", "items": [1, 2, 3]}
        path = tmp_path / "results.json"
        save_results(data, path)
        loaded = load_results(path)
        assert loaded == data

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        data = {"key": "value"}
        path = tmp_path / "nested" / "deep" / "results.json"
        result_path = save_results(data, path)
        assert result_path.exists()
        loaded = load_results(path)
        assert loaded == data

    def test_handles_path_objects(self, tmp_path: Path) -> None:
        data = {"artifact": Path("/some/path/file.pt"), "output": Path("results/")}
        path = tmp_path / "path_test.json"
        save_results(data, path)
        loaded = load_results(path)
        assert loaded["artifact"] == "/some/path/file.pt"
        assert loaded["output"] == "results"

    def test_handles_tensor_values(self, tmp_path: Path) -> None:
        data = {"vector": torch.tensor([1.0, 2.0, 3.0])}
        path = tmp_path / "tensor_test.json"
        save_results(data, path)
        loaded = load_results(path)
        assert loaded["vector"] == [1.0, 2.0, 3.0]

    def test_nonexistent_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_results(tmp_path / "nonexistent.json")

    def test_returns_path(self, tmp_path: Path) -> None:
        data = {"a": 1}
        path = tmp_path / "ret.json"
        result = save_results(data, path)
        assert result == path
