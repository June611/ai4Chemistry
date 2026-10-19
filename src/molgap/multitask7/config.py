"""Frozen Phase 1 configuration contract."""

from pathlib import Path

import yaml

from molgap.config import ConfigError, find_project_root
from molgap.data.qm9_properties import PROPERTY_UNITS

PROTOCOL_ID = "paired_split_qm9_neural_multitask7_v1"
PROTOCOL_ID_200 = "paired_split_qm9_neural_multitask7_epoch200_v1"
TASK_NAMES = tuple(PROPERTY_UNITS)
TASK_UNITS = dict(PROPERTY_UNITS)
VARIANTS = {
    "chemprop_mtl7": ("B", False, False),
    "chemprop_mtl7_affinity": ("C", True, False),
    "chemprop_mtl7_uncertainty": ("D1", False, True),
    "chemprop_mtl7_affinity_uncertainty": ("D2", True, True),
}
SEEDS = (42, 2026, 3407)


def validate_multitask7_config(config: dict) -> None:
    try:
        experiment, data, training, model = (
            config[k] for k in ("experiment", "data", "training", "model")
        )
        if experiment["protocol_id"] not in {PROTOCOL_ID, PROTOCOL_ID_200}:
            raise ConfigError("Unexpected multitask7 protocol")
        expected = VARIANTS[experiment["model_id"]]
        if tuple(data["targets"]) != TASK_NAMES or data["units"] != TASK_UNITS:
            raise ConfigError("Seven ordered targets and their original units are required")
        seed = training["seed"]
        if seed not in SEEDS or seed != data["split"]["seed"] or seed != training["split_seed"]:
            raise ConfigError("Initialization and split seeds must be paired")
        stratify = data["split"]["stratify"]
        if stratify != {"strategy": "quantile", "column": "delta_e", "bins": 200}:
            raise ConfigError("Reuse the existing gap-stratified partitions")
        budget = tuple(training[k] for k in ("batch_size", "epochs", "patience", "min_delta"))
        epochs = 200 if experiment["protocol_id"] == PROTOCOL_ID_200 else 100
        if budget != (256, epochs, 20, 0.0):
            raise ConfigError(f"Formal budget is frozen at 256/{epochs}/20/0")
        if (
            training["selection_metric"] != "delta_e_mae"
            or training["target_scaling"] != "standard"
        ):
            raise ConfigError("Select by original-unit gap MAE with train-only Z-score")
        if training["optimizer"] != "adam" or training["weight_decay"] != 0:
            raise ConfigError("Match the Chemprop Adam baseline")
        if tuple(training[k] for k in ("init_lr", "max_lr", "final_lr", "warmup_epochs")) != (
            0.0004,
            0.004,
            0.0004,
            2,
        ):
            raise ConfigError("Phase 1 preserves the baseline Noam-like LR schedule")
        if training["precision"] != "32-true" or not training["deterministic"]:
            raise ConfigError("Use the existing FP32, paired-seed numerical policy")
        if training["grad_clip"] is not None or training["devices"] != 1:
            raise ConfigError("Phase 1 uses one GPU per task without gradient clipping")
        if (model["use_affinity"], model["use_uncertainty"]) != expected[1:]:
            raise ConfigError("Model identifier and ablation flags disagree")
        if model["affinity_temperature"] != 1.0 or model["affinity_identity_bias"] != 2.0:
            raise ConfigError("Freeze the Phase 1 affinity settings")
        if model["head_layout"] != "extra_projection":
            raise ConfigError(
                "Reviewed Phase 1 uses the user-selected extra-projection C/D2 layout"
            )
        if training["num_workers"] < 0 or training["epochs"] <= training["warmup_epochs"]:
            raise ConfigError("Invalid workers or warmup")
        if (
            model["message_hidden_dim"],
            model["depth"],
            model["ffn_hidden_dim"],
            model["ffn_num_layers"],
        ) != (300, 3, 300, 1):
            raise ConfigError("Phase 1 preserves the reviewed Chemprop architecture")
    except (KeyError, TypeError) as exc:
        raise ConfigError(f"Incomplete/unsupported multitask7 configuration: {exc}") from exc


def load_multitask7_config(path: str | Path) -> tuple[dict, Path]:
    path = Path(path).resolve()
    config = yaml.safe_load(path.read_text())
    if not isinstance(config, dict):
        raise ConfigError("Configuration must be a mapping")
    validate_multitask7_config(config)
    return config, find_project_root(path)
