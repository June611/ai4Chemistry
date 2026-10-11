"""Train SPE Transformer multi-task affinity and physics-consistency ablations."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import time
from pathlib import Path
from typing import Any

import lightning.pytorch as pl
import numpy as np
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, TQDMProgressBar
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from molgap.config import run_directory
from molgap.metrics.regression import regression_metrics
from molgap.tracking.inference import measure_prediction
from molgap.transformer.dataset import (
    PreparedMultiTaskTransformerData,
    prepare_multitask_transformer_data,
    write_data_artifacts,
)
from molgap.transformer.model import count_trainable_parameters
from molgap.transformer.multitask import SpeAffinityMultiTaskRegressor
from molgap.transformer.multitask_config import TARGETS, load_spe_multitask_config
from molgap.utils.files import write_run_metadata


def gap_consistency_residual(predictions: torch.Tensor) -> torch.Tensor:
    """Return predicted gap minus predicted LUMO-HOMO in Hartree."""
    if predictions.ndim != 2 or predictions.shape[1] != 3:
        raise ValueError("Predictions must have shape [batch, 3] ordered homo, lumo, delta_e.")
    return predictions[:, 2] - (predictions[:, 1] - predictions[:, 0])


class SpeMultiTaskLightningModule(pl.LightningModule):
    """Optimize task-normalized MSE plus the established physical gap residual."""

    def __init__(
        self,
        model: SpeAffinityMultiTaskRegressor,
        target_means: tuple[float, ...],
        target_scales: tuple[float, ...],
        training_config: dict[str, Any],
        loss_config: dict[str, Any],
    ) -> None:
        super().__init__()
        self.model = model
        self.training_config = dict(training_config)
        self.consistency_weight = float(loss_config["consistency_weight"])
        weights = [float(loss_config["task_weights"][name]) for name in TARGETS]
        self.register_buffer("target_means", torch.tensor(target_means, dtype=torch.float32))
        self.register_buffer("target_scales", torch.tensor(target_scales, dtype=torch.float32))
        self.register_buffer("task_weights", torch.tensor(weights, dtype=torch.float32))

    def normalized_predictions(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        return self.model(input_ids, attention_mask)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Return HOMO, LUMO, and delta_e predictions in Hartree."""
        normalized = self.normalized_predictions(input_ids, attention_mask)
        return normalized * self.target_scales + self.target_means

    def _losses(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, ...]:
        normalized = self.normalized_predictions(batch["input_ids"], batch["attention_mask"])
        normalized_targets = (batch["target"] - self.target_means) / self.target_scales
        per_task = (normalized - normalized_targets).square().mean(dim=0)
        supervised = (per_task * self.task_weights).sum() / self.task_weights.sum()
        physical = normalized * self.target_scales + self.target_means
        consistency = gap_consistency_residual(physical).square().mean()
        total = supervised + self.consistency_weight * consistency
        return total, supervised, consistency

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        del batch_idx
        total, supervised, consistency = self._losses(batch)
        self.log("train_loss", total, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_supervised_loss", supervised, on_step=False, on_epoch=True)
        self.log("train_consistency_mse_hartree2", consistency, on_step=False, on_epoch=True)
        return total

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        del batch_idx
        total, supervised, consistency = self._losses(batch)
        physical = self(batch["input_ids"], batch["attention_mask"])
        gap_mae = (physical[:, 2] - batch["target"][:, 2]).abs().mean()
        batch_size = len(physical)
        self.log(
            "val_delta_e_mae",
            gap_mae,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
        )
        self.log("val_loss", total, on_step=False, on_epoch=True, batch_size=batch_size)
        self.log(
            "val_supervised_loss",
            supervised,
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )
        self.log(
            "val_consistency_mse_hartree2",
            consistency,
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )

    def predict_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> torch.Tensor:
        del batch_idx, dataloader_idx
        return self(batch["input_ids"], batch["attention_mask"])

    def configure_optimizers(self) -> dict[str, Any]:
        optimizer = AdamW(
            self.parameters(),
            lr=float(self.training_config["lr"]),
            weight_decay=float(self.training_config["weight_decay"]),
        )
        total_steps = max(1, int(self.trainer.estimated_stepping_batches))
        warmup_steps = int(total_steps * float(self.training_config["warmup_ratio"]))
        min_ratio = float(self.training_config["min_lr_ratio"])

        def schedule(step: int) -> float:
            if warmup_steps > 0 and step < warmup_steps:
                return max((step + 1) / warmup_steps, 1 / warmup_steps)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            progress = min(max(progress, 0.0), 1.0)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_ratio + (1.0 - min_ratio) * cosine

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": LambdaLR(optimizer, schedule),
                "interval": "step",
                "frequency": 1,
            },
        }


def _make_loaders(
    prepared: PreparedMultiTaskTransformerData, config: dict[str, Any]
) -> dict[str, DataLoader[dict[str, torch.Tensor]]]:
    training = config["training"]
    workers = int(training["num_workers"])
    return {
        name: DataLoader(
            dataset,
            batch_size=int(training["batch_size"]),
            shuffle=name == "train",
            num_workers=workers,
            pin_memory=bool(training.get("pin_memory", True)),
            persistent_workers=workers > 0,
            generator=(
                torch.Generator().manual_seed(int(training["seed"])) if name == "train" else None
            ),
        )
        for name, dataset in prepared.datasets.items()
    }


def _build_model(
    config: dict[str, Any], prepared: PreparedMultiTaskTransformerData
) -> tuple[SpeAffinityMultiTaskRegressor, int]:
    model = SpeAffinityMultiTaskRegressor(
        vocab_size=len(prepared.tokenizer.vocabulary),
        max_length=prepared.tokenizer.max_length,
        config=config["model"],
    )
    parameters = count_trainable_parameters(model)
    minimum = int(config["model"]["min_parameters"])
    maximum = int(config["model"]["max_parameters"])
    if not minimum <= parameters <= maximum:
        raise ValueError(
            f"Transformer has {parameters:,} trainable parameters, outside configured "
            f"budget [{minimum:,}, {maximum:,}]."
        )
    return model, parameters


def _plan(
    config: dict[str, Any],
    root: Path,
    prepared: PreparedMultiTaskTransformerData,
    parameters: int,
) -> dict[str, Any]:
    return {
        "experiment": config["experiment"]["name"],
        "selection": {
            "target": "delta_e",
            "metric": "mae",
            "unit": "hartree",
            "mode": "min",
            "monitor": "val_delta_e_mae",
            "patience": config["training"]["early_stopping_patience"],
            "min_delta": config["training"]["early_stopping_min_delta"],
        },
        "pipeline": (
            "canonical SMILES -> train-only SPE -> Transformer encoder -> "
            "three task representations -> affinity fusion -> homo/lumo/delta_e heads"
        ),
        "ablation": {
            "use_affinity": bool(config["model"]["use_affinity"]),
            "consistency_weight": float(config["loss"]["consistency_weight"]),
        },
        "physics": {
            "identity": "delta_e = lumo - homo",
            "residual": "pred_delta_e - (pred_lumo - pred_homo)",
        },
        "verified_same_split_as_chemprop": prepared.split_identity[
            "verified_same_split_as_chemprop"
        ],
        "split_manifest_sha256": prepared.split_identity["manifest"]["sha256"],
        "split_rows": {name: len(frame) for name, frame in prepared.frames.items()},
        "vocabulary_size": len(prepared.tokenizer.vocabulary),
        "trainable_parameters": parameters,
        "targets": list(prepared.target_columns),
        "target_unit": "hartree",
        "metrics": config["evaluation"]["metrics"],
        "training": config["training"],
        "run_dir": str(run_directory(config, root)),
    }


def _metrics(
    frame: Any, predictions: np.ndarray, target_columns: tuple[str, ...]
) -> dict[str, Any]:
    direct = {
        name: regression_metrics(frame[name].to_numpy(dtype=float), predictions[:, index])
        for index, name in enumerate(target_columns)
    }
    derived_gap = predictions[:, 1] - predictions[:, 0]
    residual = predictions[:, 2] - derived_gap
    return {
        "unit": "hartree",
        "targets": direct,
        "derived_gap_from_orbitals": regression_metrics(
            frame["delta_e"].to_numpy(dtype=float), derived_gap
        ),
        "gap_consistency": {
            "mae": float(np.mean(np.abs(residual))),
            "rmse": float(np.sqrt(np.mean(np.square(residual)))),
            "mean_signed_error": float(np.mean(residual)),
        },
    }


def train_spe_multitask(
    config_path: str | Path, dry_run: bool = False, overwrite: bool = False
) -> dict[str, Any]:
    """Validate, train, select by validation gap MAE, and evaluate once on test."""
    config_path = Path(config_path).resolve()
    config, root = load_spe_multitask_config(config_path)
    training = config["training"]
    prepared = prepare_multitask_transformer_data(config, root)
    pl.seed_everything(int(training["seed"]), workers=True)
    base_model, parameters = _build_model(config, prepared)
    plan = _plan(config, root, prepared, parameters)
    if dry_run:
        print(json.dumps(plan, indent=2, ensure_ascii=False))
        return plan

    run_dir = run_directory(config, root)
    if run_dir.exists() and any(run_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"Training artifacts already exist in {run_dir}; pass --overwrite.")
    if run_dir.exists() and overwrite:
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    write_run_metadata(run_dir, config, config_path, root)
    write_data_artifacts(prepared, run_dir)
    (run_dir / "model_summary.json").write_text(
        json.dumps({**plan, "model": config["model"], "loss": config["loss"]}, indent=2) + "\n"
    )

    loaders = _make_loaders(prepared, config)
    lightning_model = SpeMultiTaskLightningModule(
        base_model,
        prepared.target_means,
        prepared.target_scales,
        training,
        config["loss"],
    )
    checkpoint = ModelCheckpoint(
        dirpath=run_dir / "checkpoints",
        filename="best-{epoch:03d}-{val_delta_e_mae:.6f}",
        monitor="val_delta_e_mae",
        mode="min",
        save_top_k=1,
    )
    trainer = pl.Trainer(
        accelerator=training["accelerator"],
        devices=training["devices"],
        precision=training["precision"],
        max_epochs=int(training["epochs"]),
        callbacks=[
            checkpoint,
            EarlyStopping(
                monitor="val_delta_e_mae",
                mode="min",
                patience=int(training["early_stopping_patience"]),
                min_delta=float(training["early_stopping_min_delta"]),
            ),
            TQDMProgressBar(refresh_rate=1),
        ],
        logger=[
            TensorBoardLogger(run_dir / "logs", name="spe_multitask"),
            CSVLogger(run_dir / "logs", name="spe_multitask_csv", flush_logs_every_n_steps=1),
        ],
        deterministic=True,
        gradient_clip_val=float(training["gradient_clip_val"]),
        log_every_n_steps=int(training.get("log_every_n_steps", 50)),
        enable_progress_bar=True,
    )
    fit_started = time.perf_counter()
    trainer.fit(lightning_model, loaders["train"], loaders["val"])
    fit_seconds = time.perf_counter() - fit_started
    with measure_prediction(
        run_dir,
        len(prepared.frames["test"]),
        "test_loader_checkpoint_restore_and_forward",
        model=lightning_model,
    ):
        prediction_batches = trainer.predict(
            lightning_model, loaders["test"], ckpt_path="best", weights_only=False
        )
        predictions = torch.cat(prediction_batches).detach().cpu().numpy().astype(float)
    checkpoint_payload = torch.load(
        checkpoint.best_model_path, map_location="cpu", weights_only=False
    )
    lightning_model.load_state_dict(checkpoint_payload["state_dict"])
    affinity = lightning_model.model.affinity_matrix().detach().cpu().numpy()

    scaler = {
        "fit_split": "train",
        "method": "standard",
        "columns": list(prepared.target_columns),
        "mean_hartree": dict(zip(prepared.target_columns, prepared.target_means, strict=True)),
        "scale_hartree": dict(zip(prepared.target_columns, prepared.target_scales, strict=True)),
    }
    portable = {
        "format_version": 1,
        "model_type": "spe_affinity_multitask_transformer",
        "state_dict": lightning_model.model.state_dict(),
        "model_config": config["model"],
        "tokenizer": prepared.tokenizer.to_dict(),
        "target_scaler": scaler,
        "target_columns": list(prepared.target_columns),
        "label_unit": "hartree",
        "split_manifest_sha256": prepared.split_identity["manifest"]["sha256"],
    }
    torch.save(portable, run_dir / "model.pt")
    (run_dir / "affinity_matrix.json").write_text(
        json.dumps(
            {
                "enabled": bool(config["model"]["use_affinity"]),
                "task_order": list(TARGETS),
                "row_semantics": "target task receives weighted source task representations",
                "matrix": affinity.tolist(),
            },
            indent=2,
        )
        + "\n"
    )

    test_frame = prepared.frames["test"]
    id_column = config["data"]["id_column"]
    smiles_column = config["data"]["smiles_column"]
    result = test_frame[[id_column, smiles_column, *prepared.target_columns]].copy()
    for index, target in enumerate(prepared.target_columns):
        result[f"{target}_pred"] = predictions[:, index]
    result["delta_e_from_orbitals_pred"] = predictions[:, 1] - predictions[:, 0]
    result["gap_consistency_residual"] = (
        predictions[:, 2] - result["delta_e_from_orbitals_pred"].to_numpy()
    )
    prediction_dir = run_dir / "predictions"
    prediction_dir.mkdir()
    result.to_csv(prediction_dir / "test_predictions.csv", index=False, float_format="%.10g")

    metrics = {
        **_metrics(test_frame, predictions, prepared.target_columns),
        "best_checkpoint": checkpoint.best_model_path,
        "best_validation_delta_e_mae": float(checkpoint.best_model_score.detach().cpu().item()),
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n")
    timing = {
        "fit_seconds": fit_seconds,
        "epochs_completed": int(trainer.fit_loop.epoch_progress.current.completed),
    }
    (run_dir / "training_timing.json").write_text(json.dumps(timing, indent=2) + "\n")
    return {
        **plan,
        "best_checkpoint": checkpoint.best_model_path,
        "metrics": metrics,
        "timing": timing,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/spe_multitask_affinity_consistency.yaml",
        help="Experiment YAML file",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    train_spe_multitask(args.config, dry_run=args.dry_run, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
