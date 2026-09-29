"""Build inspectable Chemprop CLI commands from project YAML."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from molgap.config import resolve_path, run_directory, task_weights


def _append(command: list[str], flag: str, value: Any) -> None:
    if value is not None:
        command.extend((flag, str(value)))


def _chemprop_executable() -> str:
    """Use the Chemprop entry point installed beside the active Python interpreter."""
    candidate = Path(sys.executable).with_name("chemprop")
    return str(candidate) if candidate.is_file() else "chemprop"


def build_train_command(config: dict[str, Any], root: Path) -> list[str]:
    """Build a Chemprop v2 training command without invoking a shell."""
    data = config["data"]
    model = config["model"]
    training = config["training"]
    processed_dir = resolve_path(root, data["processed_dir"])
    run_dir = run_directory(config, root)
    command = [
        _chemprop_executable(),
        "train",
        "--data-path",
        *(str(processed_dir / f"{name}.csv") for name in ("train", "val", "test")),
        "--task-type",
        "regression",
        "--smiles-columns",
        data["smiles_column"],
        "--target-columns",
        *data["targets"],
        "--output-dir",
        str(run_dir / "chemprop"),
        "--metrics",
        *config["evaluation"]["metrics"],
        "--logfile",
        str(run_dir / "train.log"),
    ]
    if len(data["targets"]) > 1:
        command.append("--show-individual-scores")
    command.extend(("--task-weights", *(str(value) for value in task_weights(config))))

    for flag, key in (
        ("--message-hidden-dim", "message_hidden_dim"),
        ("--depth", "depth"),
        ("--dropout", "dropout"),
        ("--ffn-hidden-dim", "ffn_hidden_dim"),
        ("--ffn-num-layers", "ffn_num_layers"),
        ("--activation", "activation"),
        ("--aggregation", "aggregation"),
    ):
        _append(command, flag, model.get(key))
    for flag, key in (
        ("--batch-size", "batch_size"),
        ("--epochs", "epochs"),
        ("--init-lr", "init_lr"),
        ("--max-lr", "max_lr"),
        ("--final-lr", "final_lr"),
        ("--warmup-epochs", "warmup_epochs"),
        ("--patience", "patience"),
        ("--num-workers", "num_workers"),
        ("--accelerator", "accelerator"),
        ("--devices", "devices"),
    ):
        _append(command, flag, training.get(key))
    _append(command, "--loss-function", config["loss"]["function"])
    if model["batch_norm"]:
        command.append("--batch-norm")
    # The three split files are supplied explicitly, so Chemprop does not use
    # --data-seed to choose split membership. Chemprop still uses it for the
    # shuffled training DataLoader; bind it to the training replicate seed.
    _append(command, "--data-seed", training.get("seed"))
    _append(command, "--pytorch-seed", training.get("seed"))
    return command


def build_predict_command(
    config: dict[str, Any],
    root: Path,
    model_paths: list[Path],
    input_path: Path,
    output_path: Path,
) -> list[str]:
    """Build a Chemprop v2 prediction command."""
    return [
        _chemprop_executable(),
        "predict",
        "--test-path",
        str(input_path),
        "--model-paths",
        *(str(path) for path in model_paths),
        "--preds-path",
        str(output_path),
        "--smiles-columns",
        config["data"]["smiles_column"],
        "--num-workers",
        str(config["training"].get("num_workers", 0)),
        "--accelerator",
        str(config["training"].get("accelerator", "auto")),
        "--devices",
        str(config["training"].get("devices", "auto")),
    ]
