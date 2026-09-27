"""Train the Phase 3 physics-consistent Chemprop model through its Python API."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch
from chemprop.data import MoleculeDatapoint, MoleculeDataset, build_dataloader
from chemprop.featurizers import SimpleMoleculeMolGraphFeaturizer
from chemprop.models.utils import save_model
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from molgap.config import load_config, resolve_path, run_directory
from molgap.models import build_phase3_model
from molgap.training.evaluate import evaluate_files
from molgap.utils.files import write_run_metadata


def _read_split(path: Path, config: dict[str, Any]) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = [
        config["data"]["id_column"],
        config["data"]["smiles_column"],
        *config["data"]["targets"],
    ]
    missing = [column for column in required if column not in frame]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    if frame[config["data"]["id_column"]].duplicated().any():
        raise ValueError(f"{path} contains duplicate sample IDs.")
    if not np.isfinite(frame[config["data"]["targets"]].to_numpy(dtype=float)).all():
        raise ValueError(f"{path} contains missing or non-finite targets.")
    return frame


def _dataset(frame: pd.DataFrame, config: dict[str, Any]) -> MoleculeDataset:
    smiles_column = config["data"]["smiles_column"]
    id_column = config["data"]["id_column"]
    targets = config["data"]["targets"]
    datapoints = [
        MoleculeDatapoint.from_smi(
            str(row[smiles_column]),
            y=row[targets].to_numpy(dtype=float),
            name=str(row[id_column]),
        )
        for _, row in frame.iterrows()
    ]
    return MoleculeDataset(datapoints, SimpleMoleculeMolGraphFeaturizer())


def _dry_run_plan(config: dict[str, Any], root: Path) -> dict[str, Any]:
    processed_dir = resolve_path(root, config["data"]["processed_dir"])
    return {
        "phase": 3,
        "backend": "chemprop_python",
        "input_files": [str(processed_dir / f"{name}.csv") for name in ("train", "val", "test")],
        "targets": config["data"]["targets"],
        "heads": ["homo", "lumo", "delta_e"],
        "loss": config["loss"],
        "run_dir": str(run_directory(config, root)),
    }


def train_phase3(
    config_path: str | Path,
    dry_run: bool = False,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Train, export, predict, and evaluate the configured Phase 3 experiment."""
    config_path = Path(config_path).resolve()
    config, root = load_config(config_path)
    plan = _dry_run_plan(config, root)
    if dry_run:
        print(json.dumps(plan, indent=2, ensure_ascii=False))
        return plan

    processed_dir = resolve_path(root, config["data"]["processed_dir"])
    paths = {name: processed_dir / f"{name}.csv" for name in ("train", "val", "test")}
    missing = [path for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing processed splits: " + ", ".join(map(str, missing)))
    frames = {name: _read_split(path, config) for name, path in paths.items()}
    id_column = config["data"]["id_column"]
    id_sets = {name: set(frame[id_column].astype(str)) for name, frame in frames.items()}
    split_pairs = (("train", "val"), ("train", "test"), ("val", "test"))
    if any(id_sets[left] & id_sets[right] for left, right in split_pairs):
        raise ValueError("A sample_id occurs in more than one split.")

    run_dir = run_directory(config, root)
    if run_dir.exists() and any(run_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"Training artifacts already exist in {run_dir}; pass --overwrite.")
    if run_dir.exists() and overwrite:
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    write_run_metadata(run_dir, config, config_path, root)

    seed = int(config["training"]["seed"])
    pl.seed_everything(seed, workers=True)
    datasets = {name: _dataset(frame, config) for name, frame in frames.items()}
    scaler = datasets["train"].normalize_targets()
    training = config["training"]
    loaders = {
        name: build_dataloader(
            dataset,
            batch_size=int(training["batch_size"]),
            num_workers=int(training["num_workers"]),
            seed=seed,
            shuffle=name == "train",
        )
        for name, dataset in datasets.items()
    }
    model = build_phase3_model(config, scaler)
    checkpoint = ModelCheckpoint(
        dirpath=run_dir / "checkpoints",
        filename="best",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
    )
    callbacks = [
        checkpoint,
        EarlyStopping(monitor="val_loss", mode="min", patience=int(training["patience"])),
    ]
    trainer = pl.Trainer(
        accelerator=training["accelerator"],
        devices=training["devices"],
        max_epochs=int(training["epochs"]),
        callbacks=callbacks,
        logger=CSVLogger(run_dir / "logs", name="lightning"),
        deterministic=True,
    )
    trainer.fit(model, loaders["train"], loaders["val"])
    # Lightning 2.5 defaults to weights_only=True under PyTorch 2.6, but its own
    # checkpoints contain Chemprop metric objects. This checkpoint was created by
    # this run, so restore the trusted full checkpoint explicitly.
    trainer.test(model, loaders["test"], ckpt_path="best", weights_only=False)
    batches = trainer.predict(model, loaders["test"], ckpt_path="best", weights_only=False)
    predictions = torch.cat(batches, dim=0).detach().cpu().numpy()

    checkpoint_payload = torch.load(
        checkpoint.best_model_path, map_location="cpu", weights_only=False
    )
    model.load_state_dict(checkpoint_payload["state_dict"])
    model_dir = run_dir / "chemprop"
    model_dir.mkdir()
    save_model(model_dir / "model.pt", model, output_columns=config["data"]["targets"])

    prediction_dir = run_dir / "predictions"
    prediction_dir.mkdir()
    prediction_frame = frames["test"][[id_column, config["data"]["smiles_column"]]].copy()
    for index, target in enumerate(config["data"]["targets"]):
        prediction_frame[target] = predictions[:, index]
    prediction_path = prediction_dir / "test_predictions.csv"
    prediction_frame.to_csv(prediction_path, index=False, float_format="%.10g")
    metrics = evaluate_files(
        paths["test"],
        prediction_path,
        config["data"]["targets"],
        config["data"]["smiles_column"],
        id_column,
    )
    with (run_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return {**plan, "best_checkpoint": checkpoint.best_model_path, "metrics": metrics}
