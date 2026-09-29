"""Train the canonical-SMILES Transformer delta_e baseline."""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Any

import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torchmetrics.regression import MeanAbsoluteError, R2Score

from molgap.config import run_directory
from molgap.metrics.regression import regression_metrics
from molgap.transformer.config import load_transformer_config
from molgap.transformer.dataset import (
    PreparedTransformerData,
    prepare_transformer_data,
    write_data_artifacts,
)
from molgap.transformer.model import SmilesTransformerRegressor, count_trainable_parameters
from molgap.utils.files import write_run_metadata


class TransformerLightningModule(pl.LightningModule):
    """Optimize normalized MSE while logging MAE/R² in physical Hartree units."""

    def __init__(
        self,
        model: SmilesTransformerRegressor,
        target_mean: float,
        target_scale: float,
        training_config: dict[str, Any],
    ) -> None:
        super().__init__()
        self.model = model
        self.training_config = dict(training_config)
        self.register_buffer("target_mean", torch.tensor(float(target_mean), dtype=torch.float32))
        self.register_buffer("target_scale", torch.tensor(float(target_scale), dtype=torch.float32))
        self.criterion = nn.MSELoss()
        self.val_mae = MeanAbsoluteError()
        self.val_r2 = R2Score()
        self.test_mae = MeanAbsoluteError()
        self.test_r2 = R2Score()

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Return predictions in Hartree."""
        normalized = self.model(input_ids, attention_mask)
        return normalized * self.target_scale + self.target_mean

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        del batch_idx
        predictions = self.model(batch["input_ids"], batch["attention_mask"])
        normalized_target = (batch["target"] - self.target_mean) / self.target_scale
        loss = self.criterion(predictions, normalized_target)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        del batch_idx
        predictions = self(batch["input_ids"], batch["attention_mask"])
        self.val_mae.update(predictions, batch["target"])
        self.val_r2.update(predictions, batch["target"])
        self.log("val_mae", self.val_mae, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_r2", self.val_r2, on_step=False, on_epoch=True)

    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        del batch_idx
        predictions = self(batch["input_ids"], batch["attention_mask"])
        self.test_mae.update(predictions, batch["target"])
        self.test_r2.update(predictions, batch["target"])
        self.log("test_mae", self.test_mae, on_step=False, on_epoch=True)
        self.log("test_r2", self.test_r2, on_step=False, on_epoch=True)

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

        scheduler = LambdaLR(optimizer, schedule)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }


def _make_loaders(
    prepared: PreparedTransformerData, config: dict[str, Any]
) -> dict[str, DataLoader[dict[str, torch.Tensor]]]:
    training = config["training"]
    seed = int(training["seed"])
    generator = torch.Generator().manual_seed(seed)
    workers = int(training["num_workers"])
    return {
        name: DataLoader(
            dataset,
            batch_size=int(training["batch_size"]),
            shuffle=name == "train",
            num_workers=workers,
            pin_memory=bool(training.get("pin_memory", True)),
            persistent_workers=workers > 0,
            generator=generator if name == "train" else None,
        )
        for name, dataset in prepared.datasets.items()
    }


def _build_model(
    config: dict[str, Any], prepared: PreparedTransformerData
) -> tuple[SmilesTransformerRegressor, int]:
    model = SmilesTransformerRegressor(
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
    config: dict[str, Any], root: Path, prepared: PreparedTransformerData, parameters: int
) -> dict[str, Any]:
    return {
        "experiment": config["experiment"]["name"],
        "pipeline": (
            "canonical SMILES -> regex tokenizer -> Transformer encoder "
            "-> regression head -> delta_e"
        ),
        "verified_same_split_as_chemprop": prepared.split_identity[
            "verified_same_split_as_chemprop"
        ],
        "split_manifest_sha256": prepared.split_identity["manifest"]["sha256"],
        "split_rows": {name: len(frame) for name, frame in prepared.frames.items()},
        "vocabulary_size": len(prepared.tokenizer.vocabulary),
        "max_observed_tokens_without_cls": max(
            details["max_tokens_without_cls"]
            for details in prepared.tokenizer_report["splits"].values()
        ),
        "max_length_including_cls": prepared.tokenizer.max_length,
        "unknown_tokens_by_split": {
            name: details["unknown_tokens"]
            for name, details in prepared.tokenizer_report["splits"].items()
        },
        "trainable_parameters": parameters,
        "target": {"column": "delta_e", "unit": "hartree"},
        "metrics": config["evaluation"]["metrics"],
        "run_dir": str(run_directory(config, root)),
    }


def train_transformer(
    config_path: str | Path, dry_run: bool = False, overwrite: bool = False
) -> dict[str, Any]:
    """Validate, train, checkpoint, predict, and evaluate the sequence baseline."""
    config_path = Path(config_path).resolve()
    config, root = load_transformer_config(config_path)
    prepared = prepare_transformer_data(config, root)
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
    with (run_dir / "model_summary.json").open("w", encoding="utf-8") as handle:
        json.dump({**plan, "model": config["model"]}, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    training = config["training"]
    pl.seed_everything(int(training["seed"]), workers=True)
    loaders = _make_loaders(prepared, config)
    lightning_model = TransformerLightningModule(
        model=base_model,
        target_mean=prepared.target_mean,
        target_scale=prepared.target_scale,
        training_config=training,
    )
    checkpoint = ModelCheckpoint(
        dirpath=run_dir / "checkpoints",
        filename="best-{epoch:03d}-{val_mae:.6f}",
        monitor="val_mae",
        mode="min",
        save_top_k=1,
    )
    callbacks = [
        checkpoint,
        EarlyStopping(
            monitor="val_mae",
            mode="min",
            patience=int(training["early_stopping_patience"]),
            min_delta=float(training["early_stopping_min_delta"]),
        ),
    ]
    trainer = pl.Trainer(
        accelerator=training["accelerator"],
        devices=training["devices"],
        precision=training["precision"],
        max_epochs=int(training["epochs"]),
        callbacks=callbacks,
        logger=TensorBoardLogger(run_dir / "logs", name="transformer"),
        deterministic=True,
        gradient_clip_val=float(training["gradient_clip_val"]),
        log_every_n_steps=int(training.get("log_every_n_steps", 50)),
    )
    trainer.fit(lightning_model, loaders["train"], loaders["val"])
    trainer.test(lightning_model, loaders["test"], ckpt_path="best", weights_only=False)
    prediction_batches = trainer.predict(
        lightning_model, loaders["test"], ckpt_path="best", weights_only=False
    )
    predictions = torch.cat(prediction_batches).detach().cpu().numpy().astype(float)

    checkpoint_payload = torch.load(
        checkpoint.best_model_path, map_location="cpu", weights_only=False
    )
    lightning_model.load_state_dict(checkpoint_payload["state_dict"])
    portable = {
        "format_version": 1,
        "model_type": "smiles_transformer_regressor",
        "state_dict": lightning_model.model.state_dict(),
        "model_config": config["model"],
        "tokenizer": prepared.tokenizer.to_dict(),
        "target_scaler": {
            "mean_hartree": prepared.target_mean,
            "scale_hartree": prepared.target_scale,
        },
        "target_column": config["data"]["target_column"],
        "label_unit": "hartree",
        "split_manifest_sha256": prepared.split_identity["manifest"]["sha256"],
    }
    torch.save(portable, run_dir / "model.pt")

    target = config["data"]["target_column"]
    id_column = config["data"]["id_column"]
    smiles_column = config["data"]["smiles_column"]
    prediction_dir = run_dir / "predictions"
    prediction_dir.mkdir()
    prediction_frame = prepared.frames["test"][[id_column, smiles_column, target]].copy()
    prediction_frame[f"{target}_pred"] = predictions
    prediction_path = prediction_dir / "test_predictions.csv"
    prediction_frame.to_csv(prediction_path, index=False, float_format="%.10g")

    computed = regression_metrics(prediction_frame[target], predictions)
    metrics = {
        "target": target,
        "unit": "hartree",
        "n": computed["n"],
        "mae": computed["mae"],
        "rmse": computed["rmse"],
        "mape": computed["mape"],
        "mape_n": computed["mape_n"],
        "mape_excluded_zero_targets": computed["mape_excluded_zero_targets"],
        "r2": computed["r2"],
        "best_checkpoint": checkpoint.best_model_path,
        "best_validation_mae": float(checkpoint.best_model_score.detach().cpu().item()),
    }
    with (run_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return {**plan, "best_checkpoint": checkpoint.best_model_path, "metrics": metrics}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/transformer.yaml", help="Experiment YAML file")
    parser.add_argument(
        "--dry-run", action="store_true", help="Validate data identity/tokenization/model only"
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace this run's artifacts")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    train_transformer(args.config, dry_run=args.dry_run, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
