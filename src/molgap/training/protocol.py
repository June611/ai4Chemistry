"""Execution-boundary validation for frozen formal experiment protocols."""

from __future__ import annotations

from collections.abc import Mapping

from molgap.data.neural_properties import NEURAL_PROPERTY_PROTOCOL

PAIRED_SEEDS = frozenset({42, 2026, 3407})
GAP_PROTOCOL = "paired_split_gap_mae_v2"


def _matched_seed(config: Mapping) -> int:
    split_seed = int(config["data"]["split"]["seed"])
    training_seed = int(config["training"]["seed"])
    if split_seed != training_seed or split_seed not in PAIRED_SEEDS:
        raise ValueError(
            "Formal paired protocols require one matched split/training seed from {42, 2026, 3407}"
        )
    return split_seed


def validate_qm9_protocol(config: Mapping, *, property_run: bool) -> None:
    """Reject configs that claim a formal QM9 protocol but violate its budget."""
    protocol = config.get("experiment", {}).get("protocol_id")
    expected = NEURAL_PROPERTY_PROTOCOL if property_run else GAP_PROTOCOL
    if protocol != expected:
        raise ValueError(f"QM9 run requires protocol_id={expected}")
    _matched_seed(config)
    training = config["training"]
    early = training.get("early_stopping", {})
    if int(training["batch_size"]) != 256:
        raise ValueError("Formal QM9 training requires batch_size=256; never lower it silently")
    epochs = int(training["epochs"])
    if epochs < 1 or epochs > 100:
        raise ValueError("Formal QM9 training allows 1..100 epochs")
    if int(early.get("patience", -1)) != 20:
        raise ValueError("Formal QM9 training requires early-stopping patience=20")
    if early.get("monitor") != "val_mae":
        raise ValueError("Formal QM9 downstream selection must monitor val_mae")
    if property_run:
        if training.get("target_scaling") != "standard" or training.get("loss") != "mse":
            raise ValueError("Formal property runs require train-only standard scaling and MSE")
        if float(early.get("min_delta", -1)) != 0.0:
            raise ValueError("Formal property runs require min_delta=0")
    else:
        if float(early.get("min_delta", -1)) != 1e-5:
            raise ValueError("Formal gap runs require min_delta=1e-5 Hartree")
