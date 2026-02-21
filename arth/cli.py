"""Command-line interface for the Arth red-teaming toolkit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


# -------------------------------------------------------------------------
# Shared CLI arguments for provider configuration
# -------------------------------------------------------------------------

def _add_provider_args(parser: argparse.ArgumentParser) -> None:
    """Add provider-related flags to a subparser."""
    parser.add_argument(
        "--provider",
        default="transformer_lens",
        help=(
            "Model provider backend "
            "(transformer_lens, huggingface_local, huggingface_api, "
            "openai_compat, vllm_server). Default: transformer_lens"
        ),
    )
    parser.add_argument(
        "--api-key-env",
        default=None,
        metavar="ENV_VAR",
        help=(
            "Name of environment variable containing the API key for remote "
            "providers (e.g. --api-key-env OPENAI_API_KEY). Do NOT pass raw "
            "keys on the command line. Falls back to HF_TOKEN, OPENAI_API_KEY, "
            "ANTHROPIC_API_KEY automatically."
        ),
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="Base URL for API providers (e.g. http://localhost:11434/v1 for Ollama)",
    )
    parser.add_argument(
        "--quantization",
        default=None,
        choices=["4bit", "8bit"],
        help="Quantization mode for huggingface_local provider (4bit or 8bit)",
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="arth",
        description="Arth: Mechanistic Interpretability Red-Teaming Toolkit",
    )
    sub = parser.add_subparsers(dest="command")

    # -- extract ----------------------------------------------------------
    p_extract = sub.add_parser("extract", help="Extract technique artifact from a model")
    p_extract.add_argument("technique", help="Technique name (e.g. refusal_direction)")
    p_extract.add_argument("--model", required=True, help="HuggingFace model name or path")
    p_extract.add_argument("--dataset", default=None, help="Dataset category to load")
    p_extract.add_argument("--layers", default=None, help="Comma-separated layer indices")
    p_extract.add_argument("--output-dir", default="results", help="Output directory")
    p_extract.add_argument("--batch-size", type=int, default=32, help="Batch size")
    p_extract.add_argument("--device", default=None, help="Device (cpu/cuda)")
    _add_provider_args(p_extract)

    # -- apply ------------------------------------------------------------
    p_apply = sub.add_parser("apply", help="Apply an extracted artifact to prompts")
    p_apply.add_argument("technique", help="Technique name")
    p_apply.add_argument("--artifact", required=True, help="Path to .pt artifact file")
    p_apply.add_argument("--model", required=True, help="HuggingFace model name or path")
    p_apply.add_argument("--prompts", required=True, help="Prompt text or path to a text file (one prompt per line)")
    p_apply.add_argument("--max-tokens", type=int, default=128, help="Max new tokens to generate")
    p_apply.add_argument("--device", default=None, help="Device (cpu/cuda)")
    _add_provider_args(p_apply)

    # -- list-techniques --------------------------------------------------
    sub.add_parser("list-techniques", help="List all available techniques")

    # -- list-datasets ----------------------------------------------------
    sub.add_parser("list-datasets", help="List all available datasets")

    # -- list-providers ---------------------------------------------------
    sub.add_parser("list-providers", help="List all available model providers and their capabilities")

    # -- audit ------------------------------------------------------------
    p_audit = sub.add_parser("audit", help="Run a full audit across techniques")
    p_audit.add_argument("--model", required=True, help="HuggingFace model name or path")
    p_audit.add_argument("--techniques", default=None, help="Comma-separated technique names (default: all)")
    p_audit.add_argument("--output-dir", default="results", help="Output directory")
    p_audit.add_argument("--device", default=None, help="Device (cpu/cuda)")
    _add_provider_args(p_audit)

    # -- dashboard --------------------------------------------------------
    p_dash = sub.add_parser("dashboard", help="Launch the interactive dashboard")
    p_dash.add_argument("--port", type=int, default=8050, help="Port number")
    p_dash.add_argument("--results-dir", default="results", help="Results directory to visualize")

    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "list-techniques":
        _cmd_list_techniques()
    elif args.command == "list-datasets":
        _cmd_list_datasets()
    elif args.command == "list-providers":
        _cmd_list_providers()
    elif args.command == "extract":
        _cmd_extract(args)
    elif args.command == "apply":
        _cmd_apply(args)
    elif args.command == "audit":
        _cmd_audit(args)
    elif args.command == "dashboard":
        _cmd_dashboard(args)


# -------------------------------------------------------------------------
# Command implementations
# -------------------------------------------------------------------------


def _cmd_list_techniques() -> None:
    from arth.techniques import list_techniques

    techniques = list_techniques()
    if not techniques:
        print("No techniques discovered.")
        return
    print(f"{'Name':<25} Description")
    print("-" * 60)
    for name, tech in sorted(techniques.items()):
        print(f"{name:<25} {tech.description}")


def _cmd_list_datasets() -> None:
    from arth.core import DatasetLoader

    loader = DatasetLoader()
    datasets = loader.list_datasets()
    if not datasets:
        print("No datasets found.")
        return
    for subdir, entries in sorted(datasets.items()):
        print(f"\n{subdir}/")
        for entry in entries:
            print(f"  {entry['file']:<35} {entry['count']} items")


def _cmd_list_providers() -> None:
    """List all discovered providers with their capability flags."""
    from arth.core.providers import list_providers

    providers = list_providers()
    if not providers:
        print("No providers discovered. Check that dependencies are installed.")
        return

    print(f"{'Provider':<22} {'Activations':<14} {'Logits':<10} {'Hooks':<8}")
    print("-" * 60)
    for name, prov in sorted(providers.items()):
        act = "yes" if prov.supports_activations else "no"
        log = "yes" if prov.supports_logits else "no"
        try:
            prov.run_with_hooks([], [])
            hooks = "yes"
        except NotImplementedError:
            hooks = "no"
        except Exception:
            hooks = "yes"  # it raised something else, meaning it tried
        print(f"{name:<22} {act:<14} {log:<10} {hooks:<8}")

    print(
        "\nProviders with activations=yes support full mechanistic interpretability.\n"
        "Use --provider <name> to select a provider for extract/apply/audit commands."
    )


def _resolve_device(args_device: str | None) -> str:
    """Determine device, falling back to cpu if cuda unavailable."""
    if args_device is not None:
        return args_device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def _resolve_api_key(args: argparse.Namespace) -> str | None:
    """Resolve API key from an environment variable name (never from raw CLI input)."""
    import os

    env_var = getattr(args, "api_key_env", None)
    if env_var is not None:
        value = os.environ.get(env_var)
        if value is None:
            print(
                f"WARNING: --api-key-env specified '{env_var}' but that "
                f"environment variable is not set.",
                file=sys.stderr,
            )
        return value
    return None


def _build_model_config(args: argparse.Namespace) -> "ModelConfig":
    """Build a ModelConfig from parsed CLI arguments."""
    from arth.core import ModelConfig

    device = _resolve_device(args.device)
    return ModelConfig(
        name=args.model,
        device=device,
        provider=getattr(args, "provider", "transformer_lens"),
        api_key=_resolve_api_key(args),
        base_url=getattr(args, "base_url", None),
        quantization=getattr(args, "quantization", None),
    )


def _cmd_extract(args: argparse.Namespace) -> None:
    from arth.techniques import get_technique
    from arth.core import DatasetLoader, ModelBackend

    config = _build_model_config(args)
    backend = ModelBackend(config)

    technique = get_technique(args.technique)
    loader = DatasetLoader()

    # Load appropriate dataset based on technique
    if hasattr(technique, "dataset_type"):
        dataset_type = technique.dataset_type
    else:
        dataset_type = "contrast_pairs"

    if dataset_type == "steering_behaviors":
        dataset = loader.load_steering_pairs(args.dataset)
    elif dataset_type == "over_refusal":
        dataset = loader.load_over_refusal()
    else:
        dataset = loader.load_contrast_pairs(args.dataset)

    layers = None
    if args.layers:
        layers = [int(x.strip()) for x in args.layers.split(",")]

    from arth.core.models import ExperimentConfig

    exp_config = ExperimentConfig(
        model=config,
        output_dir=Path(args.output_dir),
        batch_size=args.batch_size,
        layers=layers,
    )

    print(f"Extracting {args.technique} from {args.model} (provider: {config.provider})...")
    result = technique.extract(backend, dataset, config=exp_config)
    print(f"Artifact saved to: {result.artifact_path}")
    print(f"Metadata: {json.dumps(result.metadata, indent=2, default=str)}")


def _cmd_apply(args: argparse.Namespace) -> None:
    from arth.techniques import get_technique
    from arth.core import ModelBackend
    from arth.eval import Scorer, compute_all_metrics

    config = _build_model_config(args)
    backend = ModelBackend(config)

    technique = get_technique(args.technique)
    artifact_path = Path(args.artifact)

    # Load prompts from file or use as direct text
    prompts_input = args.prompts
    if Path(prompts_input).is_file():
        prompts = [
            line.strip()
            for line in Path(prompts_input).read_text().splitlines()
            if line.strip()
        ]
    else:
        prompts = [prompts_input]

    print(f"Applying {args.technique} with artifact {artifact_path} (provider: {config.provider})...")
    results = technique.apply(
        backend, artifact_path, prompts, max_new_tokens=args.max_tokens
    )

    scorer = Scorer()
    scores = scorer.score_batch(results)
    metrics = compute_all_metrics(scores)

    print(f"\nResults ({len(results)} prompts):")
    print(f"  ASR:          {metrics['attack_success_rate']:.2%}")
    print(f"  Refusal Rate: {metrics['refusal_rate']:.2%}")
    print(f"  Refusal Delta:{metrics['refusal_delta']:+.2%}")
    print(f"  Coherence:    {metrics['coherence_score']:.3f}")

    for i, r in enumerate(results[:5]):
        print(f"\n--- Sample {i + 1} ---")
        print(f"Prompt:   {r['prompt'][:100]}")
        print(f"Original: {r['original'][:100]}")
        print(f"Modified: {r['modified'][:100]}")


def _cmd_audit(args: argparse.Namespace) -> None:
    from arth.techniques import get_technique, list_techniques
    from arth.core import DatasetLoader, ModelBackend
    from arth.eval import Scorer, compute_all_metrics, Reporter

    from arth.core.models import ExperimentConfig

    model_config = _build_model_config(args)
    backend = ModelBackend(model_config)
    loader = DatasetLoader()
    scorer = Scorer()
    reporter = Reporter()

    if args.techniques:
        technique_names = [t.strip() for t in args.techniques.split(",")]
    else:
        technique_names = list(list_techniques().keys())

    output_dir = Path(args.output_dir)
    exp_config = ExperimentConfig(model=model_config, output_dir=output_dir)
    all_results: dict = {"model": args.model, "techniques": {}, "samples": [], "metrics": {}}

    for tech_name in technique_names:
        print(f"\nRunning {tech_name}...")
        try:
            technique = get_technique(tech_name)
            # Load dataset appropriate for the technique
            dataset_type = getattr(technique, "dataset_type", "contrast_pairs")
            if dataset_type == "steering_behaviors":
                dataset = loader.load_steering_pairs()
            elif dataset_type == "over_refusal":
                dataset = loader.load_over_refusal()
            else:
                dataset = loader.load_contrast_pairs()
            result = technique.extract(backend, dataset, config=exp_config)

            if result.artifact_path:
                # Extract prompts robustly from any dataset type
                prompts = []
                for item in dataset[:10]:
                    if hasattr(item, "harmful"):
                        prompts.append(item.harmful)
                    elif hasattr(item, "prompt"):
                        prompts.append(item.prompt)
                    elif hasattr(item, "positive"):
                        prompts.append(item.positive)
                    elif isinstance(item, str):
                        prompts.append(item)
                apply_results = technique.apply(backend, result.artifact_path, prompts)
                scores = scorer.score_batch(apply_results)
                metrics = compute_all_metrics(scores)
                all_results["techniques"][tech_name] = metrics
                all_results["samples"].extend(scores)
                print(f"  ASR: {metrics['attack_success_rate']:.2%}")
            else:
                print(f"  No artifact produced.")
        except Exception as e:
            print(f"  Error: {e}")
            all_results["techniques"][tech_name] = {"error": str(e)}

    # Aggregate metrics across all techniques
    if all_results["samples"]:
        all_results["metrics"] = compute_all_metrics(all_results["samples"])

    # Save reports
    reporter.generate_json(all_results, output_dir / "audit_report.json")
    reporter.generate_html(all_results, output_dir / "audit_report.html")
    print(f"\nReports saved to {output_dir}/")


def _cmd_dashboard(args: argparse.Namespace) -> None:
    try:
        from arth.dashboard.app import create_app
    except ImportError:
        print(
            "Dashboard requires 'dash' and 'plotly'. Install with:\n"
            "  pip install dash plotly"
        )
        sys.exit(1)

    results_dir = Path(args.results_dir)
    app = create_app(results_dir)
    print(f"Starting dashboard on http://localhost:{args.port}")
    app.run(debug=False, port=args.port)


if __name__ == "__main__":
    main()
