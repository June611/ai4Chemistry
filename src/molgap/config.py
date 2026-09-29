"""Configuration loading and strict validation for training experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when an experiment configuration is incomplete or inconsistent."""


TARGET_WEIGHT_KEYS = {"homo": "homo", "lumo": "lumo", "delta_e": "gap"}
PHASE_TARGETS = {1: ["delta_e"], 2: ["homo", "lumo", "delta_e"], 3: ["homo", "lumo", "delta_e"]}


def find_project_root(start: Path | None = None) -> Path:
    """Find the closest parent containing pyproject.toml."""
    current = (start or Path.cwd()).resolve()
    if current.is_file():
        current = current.parent
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    raise ConfigError("Could not find project root (pyproject.toml). Run inside the repository.")


def load_config(path: str | Path) -> tuple[dict[str, Any], Path]:
    """Load YAML and return it with the project root used for relative paths."""
    config_path = Path(path).resolve()
    if not config_path.is_file():
        raise ConfigError(f"Configuration file does not exist: {config_path}")
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ConfigError("The configuration root must be a mapping.")
    root = find_project_root(config_path)
    validate_config(config)
    return config, root


def _require(mapping: dict[str, Any], keys: tuple[str, ...], section: str) -> None:
    missing = [key for key in keys if key not in mapping]
    if missing:
        raise ConfigError(f"Missing {section} setting(s): {', '.join(missing)}")


def validate_config(config: dict[str, Any]) -> None:
    """Validate the complete, phase-aware training configuration."""
    sections = ("experiment", "data", "model", "training", "loss", "evaluation", "output")
    for section in sections:
        if not isinstance(config.get(section), dict):
            raise ConfigError(f"Missing mapping section: {section}")

    experiment = config["experiment"]
    _require(experiment, ("name", "phase"), "experiment")
    if not isinstance(experiment["name"], str) or not experiment["name"].strip():
        raise ConfigError("experiment.name must be a non-empty string.")
    phase = experiment["phase"]
    if phase not in PHASE_TARGETS:
        raise ConfigError("experiment.phase must be 1, 2, or 3.")

    data = config["data"]
    _require(
        data,
        ("processed_dir", "smiles_column", "id_column", "targets", "label_unit", "split"),
        "data",
    )
    targets = data["targets"]
    if targets != PHASE_TARGETS[phase]:
        raise ConfigError(f"Phase {phase} requires data.targets: {PHASE_TARGETS[phase]}.")
    if str(data["label_unit"]).lower() != "hartree":
        raise ConfigError("data.label_unit must be hartree.")
    split = data["split"]
    if not isinstance(split, dict):
        raise ConfigError("data.split must be a mapping.")
    _require(split, ("method", "sizes", "seed"), "data.split")
    if split["method"] not in {"random", "scaffold_balanced"}:
        raise ConfigError("data.split.method must be random or scaffold_balanced.")
    sizes = split["sizes"]
    if not isinstance(sizes, list) or len(sizes) != 3 or abs(sum(sizes) - 1.0) > 1e-8:
        raise ConfigError("data.split.sizes must contain three positive fractions summing to 1.")
    if any(not isinstance(value, (int, float)) or value <= 0 for value in sizes):
        raise ConfigError("Every split fraction must be positive.")
    stratify = split.get("stratify")
    if stratify is not None:
        if split["method"] != "random" or not isinstance(stratify, dict):
            raise ConfigError("data.split.stratify must be a mapping used with random splitting.")
        if stratify.get("strategy") != "quantile":
            raise ConfigError("data.split.stratify.strategy must be quantile.")
        if stratify.get("column") not in targets:
            raise ConfigError("data.split.stratify.column must be one of data.targets.")
        if not isinstance(stratify.get("bins"), int) or stratify["bins"] < 2:
            raise ConfigError("data.split.stratify.bins must be an integer >= 2.")

    model = config["model"]
    _require(
        model,
        (
            "backend",
            "message_hidden_dim",
            "depth",
            "dropout",
            "activation",
            "aggregation",
            "batch_norm",
            "ffn_hidden_dim",
            "ffn_num_layers",
        ),
        "model",
    )
    expected_backend = "chemprop_python" if phase == 3 else "chemprop_cli"
    if model["backend"] != expected_backend:
        raise ConfigError(f"Phase {phase} requires model.backend: {expected_backend}.")
    if model["aggregation"] not in {"mean", "sum", "norm"}:
        raise ConfigError("model.aggregation must be mean, sum, or norm.")

    training = config["training"]
    _require(
        training,
        (
            "batch_size",
            "epochs",
            "warmup_epochs",
            "init_lr",
            "max_lr",
            "final_lr",
            "seed",
            "num_workers",
            "accelerator",
            "devices",
            "patience",
        ),
        "training",
    )
    if int(training["epochs"]) <= 0 or int(training["batch_size"]) <= 0:
        raise ConfigError("training.epochs and training.batch_size must be positive.")

    loss = config["loss"]
    _require(loss, ("function", "weights"), "loss")
    if loss["function"] != "mse":
        raise ConfigError("The initial three-phase study requires loss.function: mse.")
    weights = loss["weights"]
    if not isinstance(weights, dict) or set(weights) != {"homo", "lumo", "gap", "consistency"}:
        raise ConfigError("loss.weights must define homo, lumo, gap, and consistency.")
    if any(not isinstance(value, (int, float)) or value < 0 for value in weights.values()):
        raise ConfigError("All loss weights must be non-negative numbers.")
    required_weights = {
        1: {"homo": 0.0, "lumo": 0.0, "gap": 1.0, "consistency": 0.0},
        2: {"homo": 1.0, "lumo": 1.0, "gap": 1.0, "consistency": 0.0},
    }
    if phase in required_weights and weights != required_weights[phase]:
        raise ConfigError(f"Phase {phase} requires loss.weights: {required_weights[phase]}.")

    metrics = config["evaluation"].get("metrics")
    if not isinstance(metrics, list) or not metrics:
        raise ConfigError("evaluation.metrics must be a non-empty list.")
    unsupported = set(metrics) - {"mae", "rmse", "r2"}
    if unsupported:
        raise ConfigError(f"Unsupported evaluation metrics: {sorted(unsupported)}")
    _require(config["output"], ("root",), "output")


def task_weights(config: dict[str, Any]) -> list[float]:
    """Return loss weights ordered exactly like data.targets."""
    weights = config["loss"]["weights"]
    return [float(weights[TARGET_WEIGHT_KEYS[target]]) for target in config["data"]["targets"]]


def resolve_path(root: Path, value: str | Path) -> Path:
    """Resolve a config path relative to the project root."""
    path = Path(value)
    return path if path.is_absolute() else root / path


def run_directory(config: dict[str, Any], root: Path) -> Path:
    """Return the standard output directory for an experiment and seed."""
    output_root = resolve_path(root, config["output"]["root"])
    seed = int(config["training"]["seed"])
    return output_root / config["experiment"]["name"] / f"seed{seed}"
