"""Train one CPU RDKit-29 regressor on an existing immutable split."""

from __future__ import annotations

import argparse
import json
import re
import time
from importlib.metadata import version
from itertools import combinations
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from molgap.config import find_project_root, resolve_path, run_directory
from molgap.data.rdkit_features import FEATURES, descriptor_matrix
from molgap.metrics import regression_metrics
from molgap.utils.files import write_run_metadata


def load_ml_config(path: str | Path) -> tuple[dict[str, Any], Path]:
    path = Path(path).resolve()
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    sections = ("experiment", "data", "normalization", "model", "training", "output")
    if not isinstance(config, dict) or any(
        not isinstance(config.get(section), dict) for section in sections
    ):
        raise ValueError(f"ML config requires mapping sections: {sections}")
    name = config["experiment"].get("name", "")
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", name):
        raise ValueError("experiment.name must be a simple directory name.")
    data = config["data"]
    if not data.get("processed_dir") or not config["output"].get("root"):
        raise ValueError("data.processed_dir and output.root are required.")
    if (data.get("id_column"), data.get("smiles_column"), data.get("targets")) != (
        "sample_id",
        "smiles",
        ["delta_e"],
    ):
        raise ValueError("ML requires sample_id, smiles, and targets=[delta_e].")
    if data.get("label_unit") != "hartree":
        raise ValueError("ML labels must use hartree.")
    if data.get("split", {}).get("seed") != 3407:
        raise ValueError("ML uses the existing split seed3407.")
    if config["normalization"] != {"features": "none", "target": "none"}:
        raise ValueError("Feature and target normalization must both be none.")
    training = config["training"]
    if training.get("seed") != 3407 or training.get("run_mode") != "single":
        raise ValueError("ML requires seed=3407 and run_mode=single.")
    jobs = training.get("n_jobs")
    if type(jobs) is not int or jobs < 1:
        raise ValueError("training.n_jobs must be a positive integer.")
    model = config["model"]
    allowed = {
        "xgboost": {
            "objective",
            "gamma",
            "learning_rate",
            "max_depth",
            "min_child_weight",
            "n_estimators",
            "tree_method",
            "device",
            "reg_alpha",
            "reg_lambda",
            "subsample",
            "colsample_bytree",
        },
        "random_forest": {
            "n_estimators",
            "max_depth",
            "min_samples_split",
            "min_samples_leaf",
            "max_features",
            "bootstrap",
            "ccp_alpha",
        },
    }
    backend, params = model.get("backend"), model.get("params")
    if backend not in allowed or not isinstance(params, dict):
        raise ValueError("model requires backend=xgboost/random_forest and a params mapping.")
    if set(params) - allowed[backend]:
        raise ValueError(f"Unsupported model parameters: {sorted(set(params) - allowed[backend])}")
    if backend == "xgboost" and (
        params.get("objective") != "reg:squarederror"
        or params.get("device") != "cpu"
        or params.get("tree_method") != "hist"
    ):
        raise ValueError("XGBoost requires CPU hist with reg:squarederror.")
    return config, find_project_root(path)


def read_splits(directory: Path) -> dict[str, pd.DataFrame]:
    frames = {}
    for split in ("train", "val", "test"):
        frame = pd.read_csv(
            directory / f"{split}.csv",
            dtype={"sample_id": str, "smiles": str},
            float_precision="round_trip",
        )
        required = ["sample_id", "smiles", "delta_e"]
        if not set(required).issubset(frame.columns) or len(frame) < 2:
            raise ValueError(f"{split} requires at least two rows with {required}.")
        if frame[required].isna().any().any():
            raise ValueError(f"Missing values in {split}.")
        if frame["sample_id"].str.strip().eq("").any() or frame["sample_id"].duplicated().any():
            raise ValueError(f"Empty or duplicate sample IDs in {split}.")
        if not np.isfinite(frame["delta_e"].to_numpy(dtype=float)).all():
            raise ValueError(f"Nonfinite labels in {split}.")
        frames[split] = frame[required].copy()
    for left, right in combinations(frames, 2):
        for column in ("sample_id", "smiles"):
            if set(frames[left][column]) & set(frames[right][column]):
                raise ValueError(f"Overlapping {column} between {left} and {right}.")
    return frames


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def load_estimator(run_dir: str | Path) -> XGBRegressor | RandomForestRegressor:
    """Reload trusted local artifacts; callers must use the saved feature order."""
    run_dir = Path(run_dir)
    config = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    if config["model"]["backend"] == "xgboost":
        model = XGBRegressor()
        model.load_model(run_dir / "model.json")
        return model
    if config["model"]["backend"] == "random_forest":
        return joblib.load(run_dir / "model.joblib")
    raise ValueError("Unknown saved model backend.")


def train_ml(config_path: str | Path, *, dry_run: bool = False) -> dict[str, Any]:
    """Fit once, save once, predict test once; metrics.json marks a completed run."""
    config_path = Path(config_path).resolve()
    config, root = load_ml_config(config_path)
    run_dir = run_directory(config, root)
    if run_dir.exists() and not dry_run:
        raise FileExistsError(f"Output already exists: {run_dir}. Use a new experiment name.")
    directory = resolve_path(root, config["data"]["processed_dir"])
    frames = read_splits(directory)
    plan = {
        "run_dir": str(run_dir),
        "backend": config["model"]["backend"],
        "split_rows": {split: len(frame) for split, frame in frames.items()},
        "features": list(FEATURES),
        "normalization": config["normalization"],
        "training_seed": 3407,
        "run_mode": "single",
        "label_unit": "hartree",
    }
    if dry_run:
        return {**plan, "descriptor_calculation": "deferred until training"}
    started = time.perf_counter()
    matrices, identities = {}, {}
    for split, frame in frames.items():
        print(f"Calculating RDKit-29: {split} ({len(frame)} molecules)", flush=True)
        matrices[split], canonical = descriptor_matrix(
            frame["smiles"].tolist(), frame["sample_id"].tolist()
        )
        identities[split] = set(canonical)
    for left, right in combinations(identities, 2):
        if identities[left] & identities[right]:
            raise ValueError(f"Overlapping canonical molecules between {left} and {right}.")
    feature_seconds = time.perf_counter() - started
    backend = config["model"]["backend"]
    params = {
        **config["model"]["params"],
        "random_state": 3407,
        "n_jobs": config["training"]["n_jobs"],
    }
    model = XGBRegressor(**params) if backend == "xgboost" else RandomForestRegressor(**params)
    run_dir.mkdir(parents=True, exist_ok=False)
    write_run_metadata(run_dir, config, config_path, root)
    environment_path = run_dir / "environment.json"
    environment = json.loads(environment_path.read_text(encoding="utf-8"))
    environment["packages"] = {
        name: version(name)
        for name in ("rdkit", "numpy", "pandas", "scikit-learn", "xgboost", "joblib")
    }
    environment["run_mode"] = "single"
    _write_json(environment_path, environment)
    _write_json(run_dir / "feature_names.json", list(FEATURES))
    _write_json(
        run_dir / "normalization.json",
        {
            **config["normalization"],
            "label_unit": "hartree",
            "scaler": None,
        },
    )
    print(f"Fitting {backend} once on train only", flush=True)
    started = time.perf_counter()
    model.fit(matrices["train"], frames["train"]["delta_e"].to_numpy(dtype=float))
    fit_seconds = time.perf_counter() - started
    if backend == "xgboost":
        model.save_model(run_dir / "model.json")
    else:
        joblib.dump(model, run_dir / "model.joblib", compress=3)
    predictions, metrics = {}, {}
    started = time.perf_counter()
    for split in ("val", "test"):
        prediction = np.asarray(model.predict(matrices[split]), dtype=float)
        if prediction.shape != (len(frames[split]),) or not np.isfinite(prediction).all():
            raise ValueError(f"Invalid predictions for {split}.")
        predictions[split] = prediction
        metrics[split] = {
            "rows": len(prediction),
            "unit": "hartree",
            "targets": {"delta_e": regression_metrics(frames[split]["delta_e"], prediction)},
        }
    prediction_seconds = time.perf_counter() - started
    prediction_dir = run_dir / "predictions"
    prediction_dir.mkdir()
    output = frames["test"].copy()
    output["delta_e_pred"] = predictions["test"]
    output.to_csv(prediction_dir / "test_predictions.csv", index=False)
    _write_json(run_dir / "validation_metrics.json", metrics["val"])
    _write_json(
        run_dir / "training_timing.json",
        {
            "feature_seconds": feature_seconds,
            "fit_seconds": fit_seconds,
            "prediction_seconds": prediction_seconds,
        },
    )
    _write_json(run_dir / "metrics.json", metrics["test"])
    return {**plan, "metrics": metrics["test"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--dry-run", action="store_true", help="Validate config and split tables only"
    )
    args = parser.parse_args()
    print(json.dumps(train_ml(args.config, dry_run=args.dry_run), indent=2))


if __name__ == "__main__":
    main()
