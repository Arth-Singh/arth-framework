"""Tests for the CLI in arth.cli."""

from __future__ import annotations

import argparse
from unittest.mock import patch

import pytest

from arth.cli import _resolve_device, main


# ---------------------------------------------------------------------------
# main with no args
# ---------------------------------------------------------------------------

class TestMainNoArgs:
    def test_no_args_exits_with_code_1(self) -> None:
        with pytest.raises(SystemExit) as exc_info:
            main([])
        assert exc_info.value.code == 1

    def test_no_args_prints_help(self, capsys: pytest.CaptureFixture) -> None:
        with pytest.raises(SystemExit):
            main([])
        captured = capsys.readouterr()
        assert "Arth" in captured.out or "usage" in captured.out.lower()


# ---------------------------------------------------------------------------
# list-techniques subcommand
# ---------------------------------------------------------------------------

class TestListTechniques:
    def test_runs_without_error(self, capsys: pytest.CaptureFixture) -> None:
        main(["list-techniques"])
        captured = capsys.readouterr()
        # Should print technique names or "No techniques discovered"
        assert len(captured.out) > 0

    def test_outputs_technique_names(self, capsys: pytest.CaptureFixture) -> None:
        main(["list-techniques"])
        captured = capsys.readouterr()
        # At least "Name" header or a known technique name should appear
        assert "Name" in captured.out or "refusal_direction" in captured.out or "No techniques" in captured.out


# ---------------------------------------------------------------------------
# list-datasets subcommand
# ---------------------------------------------------------------------------

class TestListDatasets:
    def test_runs_without_error(self, capsys: pytest.CaptureFixture) -> None:
        main(["list-datasets"])
        captured = capsys.readouterr()
        assert len(captured.out) > 0

    def test_outputs_dataset_info(self, capsys: pytest.CaptureFixture) -> None:
        main(["list-datasets"])
        captured = capsys.readouterr()
        # Should mention dataset directories or "No datasets"
        assert (
            "contrast_pairs" in captured.out
            or "steering_behaviors" in captured.out
            or "No datasets" in captured.out
        )


# ---------------------------------------------------------------------------
# extract subcommand argument validation
# ---------------------------------------------------------------------------

class TestExtractArgs:
    def test_missing_model_flag_raises(self) -> None:
        with pytest.raises(SystemExit) as exc_info:
            main(["extract", "refusal_direction"])
        assert exc_info.value.code == 2  # argparse error exit code

    def test_missing_technique_raises(self) -> None:
        with pytest.raises(SystemExit) as exc_info:
            main(["extract", "--model", "test/model"])
        assert exc_info.value.code == 2


# ---------------------------------------------------------------------------
# apply subcommand argument validation
# ---------------------------------------------------------------------------

class TestApplyArgs:
    def test_missing_artifact_flag_raises(self) -> None:
        with pytest.raises(SystemExit) as exc_info:
            main(["apply", "refusal_direction", "--model", "test/model", "--prompts", "hello"])
        assert exc_info.value.code == 2

    def test_missing_model_flag_raises(self) -> None:
        with pytest.raises(SystemExit) as exc_info:
            main(["apply", "refusal_direction", "--artifact", "a.pt", "--prompts", "hello"])
        assert exc_info.value.code == 2

    def test_missing_prompts_flag_raises(self) -> None:
        with pytest.raises(SystemExit) as exc_info:
            main(["apply", "refusal_direction", "--model", "m", "--artifact", "a.pt"])
        assert exc_info.value.code == 2


# ---------------------------------------------------------------------------
# _resolve_device
# ---------------------------------------------------------------------------

class TestResolveDevice:
    def test_explicit_device_returned(self) -> None:
        assert _resolve_device("cpu") == "cpu"
        assert _resolve_device("cuda") == "cuda"
        assert _resolve_device("cuda:1") == "cuda:1"

    def test_none_falls_back_to_cpu_without_cuda(self) -> None:
        with patch("torch.cuda.is_available", return_value=False):
            assert _resolve_device(None) == "cpu"

    def test_none_uses_cuda_when_available(self) -> None:
        with patch("torch.cuda.is_available", return_value=True):
            assert _resolve_device(None) == "cuda"


# ---------------------------------------------------------------------------
# Argument parsing for all subcommands
# ---------------------------------------------------------------------------

class TestArgumentParsing:
    def _parse(self, argv: list[str]) -> argparse.Namespace:
        """Manually construct the parser and parse to test argument structure."""
        parser = argparse.ArgumentParser()
        sub = parser.add_subparsers(dest="command")

        p_extract = sub.add_parser("extract")
        p_extract.add_argument("technique")
        p_extract.add_argument("--model", required=True)
        p_extract.add_argument("--dataset", default=None)
        p_extract.add_argument("--layers", default=None)
        p_extract.add_argument("--output-dir", default="results")
        p_extract.add_argument("--batch-size", type=int, default=32)
        p_extract.add_argument("--device", default=None)

        p_apply = sub.add_parser("apply")
        p_apply.add_argument("technique")
        p_apply.add_argument("--artifact", required=True)
        p_apply.add_argument("--model", required=True)
        p_apply.add_argument("--prompts", required=True)
        p_apply.add_argument("--max-tokens", type=int, default=128)
        p_apply.add_argument("--device", default=None)

        sub.add_parser("list-techniques")
        sub.add_parser("list-datasets")

        p_audit = sub.add_parser("audit")
        p_audit.add_argument("--model", required=True)
        p_audit.add_argument("--techniques", default=None)
        p_audit.add_argument("--output-dir", default="results")
        p_audit.add_argument("--device", default=None)

        return parser.parse_args(argv)

    def test_extract_all_flags(self) -> None:
        args = self._parse([
            "extract", "refusal_direction",
            "--model", "gpt2",
            "--dataset", "violence_weapons",
            "--layers", "0,5,10",
            "--output-dir", "/tmp/out",
            "--batch-size", "16",
            "--device", "cpu",
        ])
        assert args.command == "extract"
        assert args.technique == "refusal_direction"
        assert args.model == "gpt2"
        assert args.dataset == "violence_weapons"
        assert args.layers == "0,5,10"
        assert args.output_dir == "/tmp/out"
        assert args.batch_size == 16
        assert args.device == "cpu"

    def test_apply_all_flags(self) -> None:
        args = self._parse([
            "apply", "steering_vectors",
            "--artifact", "/path/to/artifact.pt",
            "--model", "gpt2",
            "--prompts", "Tell me something",
            "--max-tokens", "256",
            "--device", "cuda",
        ])
        assert args.command == "apply"
        assert args.technique == "steering_vectors"
        assert args.artifact == "/path/to/artifact.pt"
        assert args.prompts == "Tell me something"
        assert args.max_tokens == 256

    def test_layers_parsing_comma_separated(self) -> None:
        args = self._parse([
            "extract", "refusal_direction",
            "--model", "gpt2",
            "--layers", "0,5,10,15",
        ])
        layers_str = args.layers
        layers = [int(x.strip()) for x in layers_str.split(",")]
        assert layers == [0, 5, 10, 15]

    def test_audit_all_flags(self) -> None:
        args = self._parse([
            "audit",
            "--model", "gpt2",
            "--techniques", "refusal_direction,steering_vectors",
            "--output-dir", "/tmp/audit",
            "--device", "cpu",
        ])
        assert args.command == "audit"
        assert args.model == "gpt2"
        assert args.techniques == "refusal_direction,steering_vectors"

    def test_list_techniques_subcommand(self) -> None:
        args = self._parse(["list-techniques"])
        assert args.command == "list-techniques"

    def test_list_datasets_subcommand(self) -> None:
        args = self._parse(["list-datasets"])
        assert args.command == "list-datasets"
