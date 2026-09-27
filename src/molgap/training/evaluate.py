"""Evaluate Chemprop predictions with per-target and consistency metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from molgap.config import load_config, resolve_path, run_directory
from molgap.metrics import regression_metrics


def _prediction_column(frame: pd.DataFrame, target: str) -> str:
    for candidate in (target, f"{target}_pred", f"pred_{target}"):
        if candidate in frame:
            return candidate
    raise ValueError(
        f"Prediction CSV has no column for target '{target}'. Found: {list(frame.columns)}"
    )


def evaluate_files(
    truth_path: Path,
    prediction_path: Path,
    targets: list[str],
    smiles_column: str,
    id_column: str = "sample_id",
) -> dict[str, Any]:
    truth = pd.read_csv(truth_path)
    predictions = pd.read_csv(prediction_path)
    if id_column in truth and id_column in predictions:
        if truth[id_column].duplicated().any() or predictions[id_column].duplicated().any():
            raise ValueError(f"Duplicate {id_column} values prevent unambiguous evaluation.")
        truth_ids = set(truth[id_column].astype(str))
        prediction_ids = set(predictions[id_column].astype(str))
        if truth_ids != prediction_ids:
            raise ValueError(f"{id_column} sets differ between truth and prediction files.")
        predictions = (
            predictions.assign(**{id_column: predictions[id_column].astype(str)})
            .set_index(id_column)
            .loc[truth[id_column].astype(str)]
            .reset_index()
        )
    elif len(truth) != len(predictions):
        raise ValueError(f"Row-count mismatch: truth={len(truth)}, predictions={len(predictions)}")
    if smiles_column in predictions and not truth[smiles_column].astype(str).reset_index(
        drop=True
    ).equals(predictions[smiles_column].astype(str).reset_index(drop=True)):
        raise ValueError("SMILES values differ after prediction/truth alignment.")

    metrics: dict[str, Any] = {"rows": len(truth), "targets": {}}
    predicted_values: dict[str, np.ndarray] = {}
    for target in targets:
        column = _prediction_column(predictions, target)
        predicted_values[target] = predictions[column].to_numpy(dtype=float)
        metrics["targets"][target] = regression_metrics(truth[target], predicted_values[target])

    required = {"homo", "lumo", "delta_e"}
    if required.issubset(predicted_values):
        residual = predicted_values["delta_e"] - (
            predicted_values["lumo"] - predicted_values["homo"]
        )
        metrics["consistency"] = {
            "mae": float(np.mean(np.abs(residual))),
            "rmse": float(np.sqrt(np.mean(residual**2))),
        }
    return metrics


def evaluate(
    config_path: str | Path,
    truth_override: str | Path | None = None,
    prediction_override: str | Path | None = None,
    output_override: str | Path | None = None,
) -> dict[str, Any]:
    config, root = load_config(config_path)
    run_dir = run_directory(config, root)
    truth_path = resolve_path(
        root,
        truth_override or Path(config["data"]["processed_dir"]) / "test.csv",
    )
    prediction_path = resolve_path(
        root,
        prediction_override or run_dir / "predictions" / "test_predictions.csv",
    )
    output_path = resolve_path(root, output_override or run_dir / "metrics.json")
    metrics = evaluate_files(
        truth_path,
        prediction_path,
        list(config["data"]["targets"]),
        config["data"]["smiles_column"],
        config["data"]["id_column"],
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return metrics


def build_parser(default_config: str = "configs/baseline.yaml") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=default_config, help="Experiment YAML file")
    parser.add_argument("--truth", help="Ground-truth CSV; defaults to processed test.csv")
    parser.add_argument("--predictions", help="Prediction CSV")
    parser.add_argument("--output", help="Metrics JSON path")
    return parser


def main(default_config: str = "configs/baseline.yaml") -> None:
    args = build_parser(default_config).parse_args()
    metrics = evaluate(args.config, args.truth, args.predictions, args.output)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
