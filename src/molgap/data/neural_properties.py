"""Opt-in QM9 single-property protocol; legacy gap and multitask stay unchanged."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from molgap.data.qm9_properties import NEW_TARGETS, PROPERTY_UNITS, digest

NEURAL_PROPERTY_PROTOCOL = "paired_split_qm9_neural_single_property_v1"


def is_property_run(config: dict) -> bool:
    return config.get("experiment", {}).get("protocol_id") == NEURAL_PROPERTY_PROTOCOL


def validate_property_target(config: dict) -> tuple[str, str]:
    data = config["data"]
    targets = data.get("targets", [data.get("target_column")])
    if len(targets) != 1 or targets[0] not in NEW_TARGETS:
        raise ValueError("Neural property protocol requires one of the six new QM9 targets")
    target = targets[0]
    if data.get("target_column", target) != target:
        raise ValueError("target_column and targets disagree")
    unit = PROPERTY_UNITS[target]
    if data.get("label_unit") != unit:
        raise ValueError(f"{target} requires label_unit={unit}")
    if int(config["training"]["seed"]) != int(data["split"]["seed"]):
        raise ValueError("Initialization and split seeds must match")
    if data["split"].get("stratify", {}).get("column") != "delta_e":
        raise ValueError("Reuse the frozen gap-stratified split; do not repartition by new target")
    return target, unit


def validate_property_data(config: dict, root: Path) -> dict:
    """Check immutable augmented labels and the original split lineage."""
    validate_property_target(config)
    value = Path(config["data"]["property_data_root"])
    data_root = value if value.is_absolute() else root / value
    identity_path = data_root / "data_identity.json"
    identity = json.loads(identity_path.read_text())
    if identity.get("targets") != PROPERTY_UNITS or not identity.get(
        "preserved_original_membership_order_and_energy_labels"
    ):
        raise ValueError("Extended QM9 data identity/units are incompatible")
    seed = int(config["data"]["split"]["seed"])
    selected = data_root / "random" / f"seed{seed}"
    configured = Path(config["data"]["processed_dir"])
    configured = configured if configured.is_absolute() else root / configured
    if selected.resolve() != configured.resolve():
        raise ValueError("Property split path differs from its data identity")
    for split in ("train", "val", "test"):
        relative = f"random/seed{seed}/{split}.csv"
        if digest(data_root / relative) != identity["extended_files"].get(relative):
            raise ValueError(f"Extended property data changed: {relative}")
    source_manifest = f"data/processed/qm9_full/random/seed{seed}/split_manifest.csv"
    if digest(selected / "split_manifest.csv") != identity["source_files"].get(source_manifest):
        raise ValueError("Property split manifest differs from the original QM9 split")
    return {"path": str(identity_path), "sha256": digest(identity_path)}


def scaler_record(values, target: str, unit: str) -> dict:
    values = np.asarray(values, dtype=np.float64)
    mean, scale = float(values.mean()), float(values.std(ddof=0))
    if not np.isfinite(values).all() or not np.isfinite(scale) or scale <= 0:
        raise ValueError("Training labels require finite values and a positive standard deviation")
    result = dict(
        method="standard",
        fit_split="train",
        target=target,
        unit=unit,
        mean=mean,
        scale=scale,
        ddof=0,
        train_rows=len(values),
    )
    if unit == "hartree":
        result.update(mean_hartree=mean, scale_hartree=scale)
    return result


def write_scaler(run_dir: Path, record: dict, feature_method: str) -> None:
    (run_dir / "target_scaler.json").write_text(json.dumps(record, indent=2) + "\n")
    normalization = dict(
        features=feature_method,
        target="standard",
        fit_split="train",
        target_column=record["target"],
        label_unit=record["unit"],
    )
    (run_dir / "normalization.json").write_text(json.dumps(normalization, indent=2) + "\n")
