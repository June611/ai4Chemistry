"""Configuration contract for SPE multi-task affinity ablations."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

from molgap.config import ConfigError, find_project_root
from molgap.transformer.config import validate_transformer_config

TARGETS = ("homo", "lumo", "delta_e")


def validate_spe_multitask_config(config: dict[str, Any]) -> None:
    """Validate shared Transformer settings and multi-task-specific controls."""
    proxy = copy.deepcopy(config)
    if not isinstance(proxy.get("experiment"), dict):
        raise ConfigError("Missing mapping section: experiment")
    if proxy["experiment"].get("type") != "spe_multitask_transformer":
        raise ConfigError("experiment.type must be spe_multitask_transformer.")
    proxy["experiment"]["type"] = "smiles_transformer"
    validate_transformer_config(proxy)

    data = config["data"]
    if tuple(data.get("target_columns", ())) != TARGETS:
        raise ConfigError("data.target_columns must be [homo, lumo, delta_e] in this order.")
    if tuple(data.get("targets", ())) != TARGETS:
        raise ConfigError("data.targets must be [homo, lumo, delta_e] in this order.")
    if config["tokenizer"].get("type") != "spe":
        raise ConfigError("SPE multitask experiments require tokenizer.type=spe.")

    model = config["model"]
    if type(model.get("use_affinity")) is not bool:
        raise ConfigError("model.use_affinity must be true or false.")
    if float(model.get("affinity_temperature", 0.0)) <= 0:
        raise ConfigError("model.affinity_temperature must be positive.")
    if float(model.get("affinity_identity_bias", -1.0)) < 0:
        raise ConfigError("model.affinity_identity_bias must be non-negative.")

    loss = config.get("loss")
    if not isinstance(loss, dict):
        raise ConfigError("Missing mapping section: loss")
    if loss.get("function") != "mse":
        raise ConfigError("loss.function must be mse.")
    weights = loss.get("task_weights")
    if not isinstance(weights, dict) or tuple(weights) != TARGETS:
        raise ConfigError("loss.task_weights must contain ordered homo, lumo, delta_e entries.")
    if any(float(weights[name]) <= 0 for name in TARGETS):
        raise ConfigError("All task weights must be positive.")
    if float(loss.get("consistency_weight", -1.0)) < 0:
        raise ConfigError("loss.consistency_weight must be non-negative.")


def load_spe_multitask_config(path: str | Path) -> tuple[dict[str, Any], Path]:
    path = Path(path).resolve()
    if not path.is_file():
        raise ConfigError(f"Configuration file does not exist: {path}")
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ConfigError("The configuration root must be a mapping.")
    validate_spe_multitask_config(config)
    return config, find_project_root(path)
