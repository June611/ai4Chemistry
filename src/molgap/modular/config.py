"""Executable opt-in protocol; planned YAMLs are deliberately rejected."""

from pathlib import Path

import yaml

from molgap.config import find_project_root
from molgap.modular.physics import TASKS7, TASKS10, UNITS10

PROTOCOL = "qm9_modular_fresh200_v1"
BACKBONES = {
    "chemprop",
    "dimenet_pp",
    "demol",
    "painn",
    "egnn",
    "torchmd_net",
    "transformer_m",
    "3dinfomax",
    "schnet",
    "esa",
    "spe_transformer",
    "smiles_transformer",
}
PHYSICS = dict(
    kbt_hartree=0.000944186,
    gap_tolerance_hartree=1e-4,
    enthalpy_tolerance_hartree=1e-6,
    scale_floor_hartree=1e-8,
    warmup_epochs=10,
    ramp_epochs=10,
    weight=0.1,
)


def validate(config):
    if (
        config.get("status") == "planned_not_executable"
        or config["experiment"]["protocol_id"] != PROTOCOL
    ):
        raise ValueError("An executable modular fresh200 config is required")
    t, data, module = config["training"], config["data"], config["module"]
    if config["backbone"]["model"] not in BACKBONES:
        raise ValueError("Unsupported molecular encoder")
    if config["backbone"]["model"] == "3dinfomax" and not config["source"]["model"]["params"].get(
        "require_pretrained"
    ):
        raise ValueError("3D Infomax must retain its QM9 train-partition pretraining")
    if module["variant"] in {"D1", "D2"}:
        from molgap.multitask7.config import VARIANTS

        if (
            config["backbone"]["model"] != "chemprop"
            or VARIANTS[config["source"]["experiment"]["model_id"]][0] != module["variant"]
        ):
            raise ValueError("D1/D2 must preserve the original Chemprop variant")
    expected = {
        "single_task": tuple(data["targets"]),
        "D1": TASKS7,
        "D2": TASKS7,
        "task_attention7": TASKS7,
        "physics_thermo10": TASKS10,
    }[module["variant"]]
    if module["variant"] == "single_task" and (len(expected) != 1 or expected[0] not in TASKS7):
        raise ValueError("Single-task runs require exactly one original QM9 property")
    if tuple(data["targets"]) != expected or data["units"] != {k: UNITS10[k] for k in expected}:
        raise ValueError("Task order/units differ from module contract")
    if (t["epochs"], t["batch_size"], t["patience"], t["checkpoint_every_n_epochs"]) != (
        200,
        256,
        20,
        50,
    ):
        raise ValueError("Formal budget must be 200 epochs / batch256 / patience20 / checkpoint50")
    if t["seed"] not in (42, 2026, 3407) or t["seed"] != data["split"]["seed"]:
        raise ValueError("Use three paired split/initialization seeds")
    if (
        t["selection_metric"] != f"val_{data['selection_target']}_mae"
        or t["target_scaling"] != "train_only_per_target_zscore"
        or t["supervised_loss"] != "mse"
    ):
        raise ValueError("Use per-target train Z-score/MSE and original-unit gap validation MAE")
    if data["selection_target"] != (
        expected[0] if module["variant"] == "single_task" else "delta_e"
    ):
        raise ValueError(
            "Joint checkpoints must select gap; single-task checkpoints select their property"
        )
    if data["split"]["stratify"] != dict(strategy="quantile", column="delta_e", bins=200):
        raise ValueError("Frozen gap-stratified splits are required")
    if module["hidden_dim"] % module["num_heads"] or not 0 <= module["dropout"] < 1:
        raise ValueError("Invalid task attention width/head count/dropout")
    if t["num_workers"] < 0 or t["min_delta"] != 0:
        raise ValueError("Use nonnegative workers and frozen min_delta=0")
    for k, v in module["physics"].items():
        if k not in PHYSICS or v < 0:
            raise ValueError("Invalid physical-loss settings")
    if set(module["physics"]) != set(PHYSICS) or module["physics"]["scale_floor_hartree"] <= 0:
        raise ValueError("All physical scales/tolerances/weights must be recorded")


def load(path):
    path = Path(path).resolve()
    config = yaml.safe_load(path.read_text())
    validate(config)
    return config, find_project_root(path)
