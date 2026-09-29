"""Strict YAML contract for the SMILES Transformer experiment."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from molgap.config import ConfigError, find_project_root


def _require(mapping: dict[str, Any], keys: tuple[str, ...], section: str) -> None:
    missing = [key for key in keys if key not in mapping]
    if missing:
        raise ConfigError(f"Missing {section} setting(s): {', '.join(missing)}")


def validate_transformer_config(config: dict[str, Any]) -> None:
    """Reject incomplete or scientifically inconsistent Transformer configs."""
    sections = ("experiment", "data", "tokenizer", "model", "training", "evaluation", "output")
    for section in sections:
        if not isinstance(config.get(section), dict):
            raise ConfigError(f"Missing mapping section: {section}")

    experiment = config["experiment"]
    _require(experiment, ("name", "type"), "experiment")
    if experiment["type"] != "smiles_transformer":
        raise ConfigError("experiment.type must be smiles_transformer.")

    data = config["data"]
    _require(
        data,
        (
            "processed_dir",
            "chemprop_reference_config",
            "smiles_column",
            "id_column",
            "target_column",
            "label_unit",
            "split",
        ),
        "data",
    )
    if data["target_column"] != "delta_e" or str(data["label_unit"]).lower() != "hartree":
        raise ConfigError("The sequence baseline requires delta_e in Hartree.")
    split = data["split"]
    if not isinstance(split, dict):
        raise ConfigError("data.split must be a mapping.")
    _require(split, ("method", "sizes", "seed", "stratify"), "data.split")
    sizes = split["sizes"]
    if (
        split["method"] != "random"
        or not isinstance(sizes, list)
        or len(sizes) != 3
        or any(float(value) <= 0 for value in sizes)
        or abs(sum(map(float, sizes)) - 1.0) > 1e-8
    ):
        raise ConfigError("data.split must describe the existing random 3-way split.")
    stratify = split["stratify"]
    if not isinstance(stratify, dict) or stratify.get("strategy") != "quantile":
        raise ConfigError("data.split.stratify.strategy must be quantile.")
    if stratify.get("column") != "delta_e" or int(stratify.get("bins", 0)) < 2:
        raise ConfigError("Transformer splits must be stratified on delta_e with at least 2 bins.")

    tokenizer = config["tokenizer"]
    _require(
        tokenizer,
        ("pattern", "max_length", "overlength_policy", "vocabulary_source", "special_tokens"),
        "tokenizer",
    )
    if int(tokenizer["max_length"]) < 2:
        raise ConfigError(
            "tokenizer.max_length must include room for [CLS] and at least one token."
        )
    if tokenizer["overlength_policy"] != "error":
        raise ConfigError("tokenizer.overlength_policy must be error to prevent silent truncation.")
    if tokenizer["vocabulary_source"] != "train":
        raise ConfigError("tokenizer.vocabulary_source must be train to prevent leakage.")
    special_tokens = tokenizer["special_tokens"]
    if not isinstance(special_tokens, dict) or set(special_tokens) != {"pad", "unk", "cls"}:
        raise ConfigError("tokenizer.special_tokens must define exactly pad, unk, and cls.")
    if len(set(special_tokens.values())) != 3:
        raise ConfigError("Tokenizer special tokens must be distinct.")

    model = config["model"]
    _require(
        model,
        (
            "d_model",
            "num_layers",
            "num_heads",
            "dim_feedforward",
            "dropout",
            "activation",
            "pooling",
            "regression_hidden_dim",
            "min_parameters",
            "max_parameters",
        ),
        "model",
    )
    d_model = int(model["d_model"])
    layers = int(model["num_layers"])
    heads = int(model["num_heads"])
    if not 256 <= d_model <= 512 or not 4 <= layers <= 6 or heads != 8:
        raise ConfigError("Use d_model 256-512, 4-6 encoder layers, and 8 attention heads.")
    if d_model % heads:
        raise ConfigError("model.d_model must be divisible by model.num_heads.")
    if model["pooling"] not in {"cls", "mean"}:
        raise ConfigError("model.pooling must be cls or mean.")
    if model["activation"] not in {"relu", "gelu"}:
        raise ConfigError("model.activation must be relu or gelu.")
    if not 0 <= float(model["dropout"]) < 1:
        raise ConfigError("model.dropout must be in [0, 1).")
    if int(model["min_parameters"]) <= 0 or int(model["max_parameters"]) <= int(
        model["min_parameters"]
    ):
        raise ConfigError("The configured parameter budget is invalid.")

    training = config["training"]
    _require(
        training,
        (
            "batch_size",
            "epochs",
            "optimizer",
            "lr",
            "weight_decay",
            "warmup_ratio",
            "min_lr_ratio",
            "gradient_clip_val",
            "target_scaling",
            "early_stopping_patience",
            "early_stopping_min_delta",
            "seed",
            "accelerator",
            "devices",
            "precision",
            "num_workers",
        ),
        "training",
    )
    if training["optimizer"] != "adamw" or training["target_scaling"] != "standard":
        raise ConfigError("training requires AdamW and train-only standard target scaling.")
    if not 1e-4 <= float(training["lr"]) <= 5e-4:
        raise ConfigError("training.lr must be between 1e-4 and 5e-4.")
    if int(training["batch_size"]) <= 0 or int(training["epochs"]) <= 0:
        raise ConfigError("training.batch_size and training.epochs must be positive.")
    if not 0 <= float(training["warmup_ratio"]) < 1:
        raise ConfigError("training.warmup_ratio must be in [0, 1).")
    if not 0 < float(training["min_lr_ratio"]) <= 1:
        raise ConfigError("training.min_lr_ratio must be in (0, 1].")

    if config["evaluation"].get("metrics") != ["mae", "rmse", "mape", "r2"]:
        raise ConfigError(
            "evaluation.metrics must be exactly [mae, rmse, mape, r2] for model comparison."
        )
    _require(config["output"], ("root",), "output")


def load_transformer_config(path: str | Path) -> tuple[dict[str, Any], Path]:
    """Load and validate a Transformer YAML config and its project root."""
    config_path = Path(path).resolve()
    if not config_path.is_file():
        raise ConfigError(f"Configuration file does not exist: {config_path}")
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ConfigError("The configuration root must be a mapping.")
    validate_transformer_config(config)
    return config, find_project_root(config_path)
