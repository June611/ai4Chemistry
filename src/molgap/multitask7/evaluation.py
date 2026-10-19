"""Verify on-disk seven-property predictions and training-only scalers."""

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from molgap.multitask7.config import TASK_NAMES, TASK_UNITS, validate_multitask7_config
from molgap.multitask7.data import load_splits
from molgap.multitask7.metrics import MultiTask7Metrics


def verify_run(run_dir, root, *, allow_diagnostic=False):
    run_dir, root = Path(run_dir), Path(root)
    config = yaml.safe_load((run_dir / "config.yaml").read_text())
    validate_multitask7_config(config)
    metrics = json.loads((run_dir / "test_metrics.json").read_text())
    if metrics.get("diagnostic") and not allow_diagnostic:
        raise ValueError("Diagnostic results cannot satisfy a formal job")
    datasets, _ = load_splits(config, root)
    saved_scalers = json.loads((run_dir / "scalers.json").read_text())
    for task in TASK_NAMES:
        actual, expected = saved_scalers[task], datasets["train"].scalers[task]
        for key in ("unit", "method", "fit_split", "ddof", "train_rows", "target"):
            if actual[key] != expected[key]:
                raise ValueError(f"Training scaler metadata mismatch: {task}/{key}")
        for key in ("mean", "scale"):
            if not math.isclose(actual[key], expected[key], rel_tol=1e-12, abs_tol=1e-14):
                raise ValueError(f"Training scaler value mismatch: {task}/{key}")
    prediction = pd.read_csv(run_dir / "test_predictions.csv", float_precision="round_trip")
    truth = datasets["test"].df
    if metrics.get("diagnostic"):
        truth = truth.iloc[: len(prediction)]
    if prediction.sample_id.astype(str).tolist() != truth.sample_id.astype(str).tolist():
        raise ValueError("Test prediction identity/order differs from the fixed split")
    if (
        prediction.smiles.tolist() != truth.smiles.tolist()
        or prediction.sample_id.duplicated().any()
    ):
        raise ValueError("Test prediction SMILES/identities differ")
    calculated = MultiTask7Metrics(saved_scalers).compute(
        prediction[list(TASK_NAMES)].to_numpy(),
        truth[list(TASK_NAMES)].to_numpy(),
        normalized=False,
    )
    if metrics["units"] != TASK_UNITS or metrics["rows"] != len(prediction):
        raise ValueError("Wrong output units/coverage")
    for task in TASK_NAMES:
        for metric in ("mae", "mse", "rmse", "r2"):
            if not math.isclose(
                calculated[task][metric],
                metrics["targets"][task][metric],
                rel_tol=1e-8,
                abs_tol=1e-12,
            ):
                raise ValueError(f"Saved metrics do not match prediction CSV: {task}/{metric}")
    if not math.isclose(metrics["delta_e_mae"], calculated["delta_e"]["mae"], rel_tol=1e-8):
        raise ValueError("Primary metric does not select gap")
    if not (run_dir / "best_checkpoint.pt").is_file():
        raise ValueError("Best checkpoint missing")
    if config["model"]["use_affinity"]:
        matrix = np.asarray(json.loads((run_dir / "affinity_matrix.json").read_text())["matrix"])
        if matrix.shape != (7, 7) or not np.isfinite(matrix).all() or (matrix < 0).any():
            raise ValueError("Invalid affinity matrix")
        np.testing.assert_allclose(matrix.sum(1), 1, atol=1e-6)
    if config["model"]["use_uncertainty"]:
        parameters = json.loads((run_dir / "uncertainty_params.json").read_text())
        np.testing.assert_allclose(
            np.exp(-np.asarray(parameters["log_vars"])), parameters["precision_weights"], rtol=1e-6
        )
    return {
        "test_rows": len(prediction),
        "mae": calculated["delta_e"]["mae"],
        "all_seven_targets_verified": True,
    }
