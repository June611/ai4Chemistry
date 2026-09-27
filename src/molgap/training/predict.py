"""Predict with trained Chemprop checkpoints and keep outputs with the run."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from molgap.config import load_config, resolve_path, run_directory
from molgap.training.commands import build_predict_command
from molgap.utils.files import write_command_record


def discover_models(run_dir: Path) -> list[Path]:
    """Prefer exported .pt models, then best checkpoints, then all checkpoints."""
    model_root = run_dir / "chemprop"
    models = sorted(model_root.rglob("*.pt"))
    if models:
        return models
    models = sorted(model_root.rglob("best*.ckpt"))
    if models:
        return models
    return sorted(model_root.rglob("*.ckpt"))


def predict(
    config_path: str | Path,
    input_override: str | Path | None = None,
    model_overrides: list[str] | None = None,
    output_override: str | Path | None = None,
) -> Path:
    config, root = load_config(config_path)
    run_dir = run_directory(config, root)
    input_path = resolve_path(
        root,
        input_override or Path(config["data"]["processed_dir"]) / "test.csv",
    )
    if not input_path.is_file():
        raise FileNotFoundError(f"Prediction input does not exist: {input_path}")

    model_paths = (
        [resolve_path(root, path) for path in model_overrides]
        if model_overrides
        else discover_models(run_dir)
    )
    if not model_paths:
        raise FileNotFoundError(f"No .pt or .ckpt model found under {run_dir / 'chemprop'}")

    output_path = resolve_path(
        root,
        output_override or run_dir / "predictions" / "test_predictions.csv",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = build_predict_command(config, root, model_paths, input_path, output_path)
    write_command_record(run_dir / "predict_command.json", command, {"kind": "predict"})
    subprocess.run(command, cwd=root, check=True)
    return output_path


def build_parser(default_config: str = "configs/baseline.yaml") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=default_config, help="Experiment YAML file")
    parser.add_argument("--input", help="CSV to predict; defaults to processed test.csv")
    parser.add_argument("--model-path", action="append", dest="model_paths")
    parser.add_argument("--output", help="Prediction CSV path")
    return parser


def main(default_config: str = "configs/baseline.yaml") -> None:
    args = build_parser(default_config).parse_args()
    destination = predict(args.config, args.input, args.model_paths, args.output)
    print(destination)


if __name__ == "__main__":
    main()
