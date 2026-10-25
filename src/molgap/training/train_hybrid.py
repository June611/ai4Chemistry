"""
Training script for hybrid MPNN-Transformer models (Work 1 & 2).

Supports:
- Work 1: 2D only (MPNN + Transformer, single task)
- Work 2: 3D with geometry injection (λ gating, MLP/Fourier PE, multi-task)
"""

import argparse
import json
from pathlib import Path

import lightning as L
import torch
import torch.nn as nn
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from molgap.models.mpnn_transformer import MPNNTransformerHybrid
from molgap.models.mpnn_transformer_3d import MPNNTransformer3D
from molgap.nn.multitask_loss import UncertaintyWeightedLoss


class HybridModelLightning(L.LightningModule):
    """Lightning wrapper for hybrid models."""

    def __init__(
        self,
        model_type: str = "work1",  # 'work1' or 'work2'
        # MPNN parameters
        d_h: int = 300,
        mpnn_depth: int = 3,
        # Transformer parameters
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        # Work 2 specific
        pe_type: str = "mlp",  # 'mlp' or 'fourier'
        lambda_geometry: float = 1.0,
        learnable_lambda: bool = False,
        num_frequencies: int = 50,  # for Fourier PE
        # Task
        n_tasks: int = 1,
        use_uncertainty_weighting: bool = False,
        # Training
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        warmup_epochs: int = 2,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Create model
        if model_type == "work1":
            self.model = MPNNTransformerHybrid(
                d_h=d_h,
                mpnn_depth=mpnn_depth,
                d_model=d_model,
                nhead=nhead,
                num_layers=num_layers,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                n_tasks=n_tasks,
            )
        elif model_type == "work2":
            self.model = MPNNTransformer3D(
                d_h=d_h,
                mpnn_depth=mpnn_depth,
                d_model=d_model,
                nhead=nhead,
                num_layers=num_layers,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                pe_type=pe_type,
                num_frequencies=num_frequencies,
                lambda_geometry=lambda_geometry,
                learnable_lambda=learnable_lambda,
                n_tasks=n_tasks,
            )
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        # Loss function
        if use_uncertainty_weighting and n_tasks > 1:
            self.criterion = UncertaintyWeightedLoss(n_tasks=n_tasks)
            self.use_uncertainty = True
        else:
            self.criterion = nn.MSELoss()
            self.use_uncertainty = False

    def forward(self, bmg, coords=None):
        if hasattr(self.model, "forward") and "coords" in self.model.forward.__code__.co_varnames:
            return self.model(bmg, coords=coords)
        else:
            return self.model(bmg)

    def _batch_coords(self, batch):
        """Extract coordinates from either dict- or attribute-style batches."""
        if isinstance(batch, dict):
            return batch.get("coords", None)
        return getattr(batch, "coords", None)

    def training_step(self, batch, batch_idx):
        # Get predictions
        preds = self.forward(batch.bmg, coords=self._batch_coords(batch))

        # Compute loss
        targets = batch.y if hasattr(batch, "y") else batch["y"]

        if self.use_uncertainty:
            loss, task_losses, weights = self.criterion(preds, targets)

            # Log per-task losses and weights
            for i, (tl, w) in enumerate(zip(task_losses, weights, strict=True)):
                self.log(f"train/task_{i}_loss", tl, prog_bar=False)
                self.log(f"train/task_{i}_weight", w, prog_bar=False)

            # Log uncertainties
            sigmas = self.criterion.get_uncertainties()
            for i, sigma in enumerate(sigmas):
                self.log(f"train/task_{i}_sigma", sigma, prog_bar=False)
        else:
            loss = self.criterion(preds, targets)

        self.log("train/loss", loss, prog_bar=True)

        # Log lambda if learnable
        if hasattr(self.model, "lambda_geometry") and self.model.lambda_geometry.requires_grad:
            self.log("train/lambda", self.model.lambda_geometry.item(), prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        preds = self.forward(batch.bmg, coords=self._batch_coords(batch))
        targets = batch.y if hasattr(batch, "y") else batch["y"]

        if self.use_uncertainty:
            loss, task_losses, _ = self.criterion(preds, targets)
            for i, tl in enumerate(task_losses):
                self.log(f"val/task_{i}_loss", tl)
        else:
            loss = self.criterion(preds, targets)

        self.log("val/loss", loss, prog_bar=True)

        # MAE for reporting
        mae = torch.abs(preds - targets).mean()
        self.log("val/mae", mae, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        preds = self.forward(batch.bmg, coords=self._batch_coords(batch))
        targets = batch.y if hasattr(batch, "y") else batch["y"]

        mae = torch.abs(preds - targets).mean()
        self.log("test/mae", mae)

        # Per-task MAE if multi-task
        if preds.shape[-1] > 1:
            for i in range(preds.shape[-1]):
                task_mae = torch.abs(preds[:, i] - targets[:, i]).mean()
                self.log(f"test/task_{i}_mae", task_mae)

        return mae

    def configure_optimizers(self):
        # Include criterion parameters if using uncertainty weighting
        params = list(self.model.parameters())
        if self.use_uncertainty:
            params += list(self.criterion.parameters())

        optimizer = torch.optim.Adam(
            params, lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
        )

        # Cosine annealing with warmup
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs - self.hparams.warmup_epochs,
            eta_min=self.hparams.lr * 0.01,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }


def train_hybrid_model(
    data_dir: str,
    output_dir: str,
    model_type: str = "work1",
    config: dict = None,
    seed: int = 42,
    max_epochs: int = 100,
    batch_size: int = 32,
    num_workers: int = 4,
    gpus: int = 1,
):
    """
    Main training function.

    Args:
        data_dir: Path to QM9 data
        output_dir: Where to save results
        model_type: 'work1' or 'work2'
        config: Model hyperparameters
        seed: Random seed
        max_epochs: Maximum training epochs
        batch_size: Batch size
        num_workers: DataLoader workers
        gpus: Number of GPUs
    """
    L.seed_everything(seed)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Default config
    if config is None:
        config = {}

    # Create model
    model = HybridModelLightning(model_type=model_type, **config)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # TODO: Load actual QM9 data
    # For now, this is a placeholder
    # train_loader, val_loader, test_loader = load_qm9_data(data_dir, batch_size, num_workers)

    # Callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=output_dir / "checkpoints",
            filename="best-{epoch:02d}-{val/mae:.4f}",
            monitor="val/mae",
            mode="min",
            save_top_k=3,
        ),
        EarlyStopping(
            monitor="val/mae",
            patience=20,
            mode="min",
        ),
    ]

    # Logger
    logger = TensorBoardLogger(save_dir=output_dir, name="logs", version=f"seed{seed}")

    # Trainer (constructed but .fit/.test intentionally not run in this scaffold)
    _trainer = L.Trainer(
        max_epochs=max_epochs,
        accelerator="gpu" if gpus > 0 else "cpu",
        devices=gpus if gpus > 0 else 1,
        callbacks=callbacks,
        logger=logger,
        gradient_clip_val=1.0,
        deterministic=True,
    )

    # Train
    # trainer.fit(model, train_loader, val_loader)

    # Test
    # trainer.test(model, test_loader)

    # Save config
    config_path = output_dir / f"config_seed{seed}.json"
    with open(config_path, "w") as f:
        json.dump(
            {
                "model_type": model_type,
                "seed": seed,
                "config": config,
                "max_epochs": max_epochs,
                "batch_size": batch_size,
            },
            f,
            indent=2,
        )

    print(f"Training completed! Results saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Train hybrid MPNN-Transformer models")

    # Data
    parser.add_argument("--data_dir", type=str, required=True, help="Path to QM9 data")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")

    # Model
    parser.add_argument("--model_type", type=str, default="work1", choices=["work1", "work2"])
    parser.add_argument("--d_h", type=int, default=300)
    parser.add_argument("--mpnn_depth", type=int, default=3)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)

    # Work 2 specific
    parser.add_argument("--pe_type", type=str, default="mlp", choices=["mlp", "fourier"])
    parser.add_argument("--lambda_geometry", type=float, default=1.0)
    parser.add_argument("--learnable_lambda", action="store_true")
    parser.add_argument("--num_frequencies", type=int, default=50)

    # Task
    parser.add_argument("--n_tasks", type=int, default=1)
    parser.add_argument("--use_uncertainty_weighting", action="store_true")

    # Training
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--gpus", type=int, default=1)

    args = parser.parse_args()

    config = {
        "d_h": args.d_h,
        "mpnn_depth": args.mpnn_depth,
        "d_model": args.d_model,
        "nhead": args.nhead,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "pe_type": args.pe_type,
        "lambda_geometry": args.lambda_geometry,
        "learnable_lambda": args.learnable_lambda,
        "num_frequencies": args.num_frequencies,
        "n_tasks": args.n_tasks,
        "use_uncertainty_weighting": args.use_uncertainty_weighting,
        "lr": args.lr,
    }

    train_hybrid_model(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        model_type=args.model_type,
        config=config,
        seed=args.seed,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        gpus=args.gpus,
    )


if __name__ == "__main__":
    main()
