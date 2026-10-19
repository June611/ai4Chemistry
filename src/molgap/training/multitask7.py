"""Train independent Chemprop seven-task Phase 1 variants."""

import argparse
import json
import os
import time
from pathlib import Path

import lightning.pytorch as pl
import pandas as pd
import torch
from chemprop.schedulers import build_NoamLike_LRSched
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from molgap.multitask7.config import TASK_NAMES, TASK_UNITS, load_multitask7_config
from molgap.multitask7.data import load_splits, make_graph_loaders
from molgap.multitask7.metrics import MultiTask7Metrics
from molgap.multitask7.models import ChempropSevenTaskNetwork
from molgap.tracking.inference import measure_prediction
from molgap.tracking.records import utc_now, write_json
from molgap.utils.files import write_run_metadata


class MultiTask7LightningModule(pl.LightningModule):
    def __init__(self, config, scalers):
        super().__init__()
        self.config = config
        self.network = ChempropSevenTaskNetwork(config)
        self.register_buffer(
            "target_scales", torch.tensor([scalers[t]["scale"] for t in TASK_NAMES])
        )
        self.register_buffer("val_abs", torch.zeros(7, dtype=torch.float64), persistent=False)
        self.register_buffer("val_square", torch.zeros(7, dtype=torch.float64), persistent=False)
        self.val_rows = 0

    def forward(self, batch):
        return self.network(batch.bmg, batch.V_d)

    def training_step(self, batch, batch_idx):
        prediction = self(batch)
        loss = self.network.head.loss_fn(prediction, batch.Y)
        if not torch.isfinite(loss):
            raise ValueError("Non-finite training objective")
        self.log("train_loss", loss, on_step=False, on_epoch=True, batch_size=len(batch.Y))
        return loss

    def on_validation_epoch_start(self):
        self.val_abs.zero_()
        self.val_square.zero_()
        self.val_rows = 0

    def validation_step(self, batch, batch_idx):
        errors = (self(batch).double() - batch.Y.double()) * self.target_scales.double()
        self.val_abs += errors.abs().sum(0)
        self.val_square += errors.square().sum(0)
        self.val_rows += len(batch.Y)

    def on_validation_epoch_end(self):
        if not self.val_rows:
            raise ValueError("Empty validation split")
        for i, name in enumerate(TASK_NAMES):
            self.log(f"val_{name}_mae", self.val_abs[i] / self.val_rows, prog_bar=name == "delta_e")
            self.log(f"val_{name}_rmse", torch.sqrt(self.val_square[i] / self.val_rows))

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self(batch)

    def configure_optimizers(self):
        t = self.config["training"]
        optimizer = torch.optim.Adam(
            self.parameters(), lr=t["init_lr"], weight_decay=t["weight_decay"]
        )
        if self.trainer.train_dataloader is None:
            _ = self.trainer.estimated_stepping_batches
        steps = int(self.trainer.num_training_batches)
        # Diagnostic max-epochs overrides do not compress the formal 100-epoch LR schedule.
        schedule = build_NoamLike_LRSched(
            optimizer,
            t["warmup_epochs"] * steps,
            (t["epochs"] - t["warmup_epochs"]) * steps,
            t["init_lr"],
            t["max_lr"],
            t["final_lr"],
        )
        return dict(optimizer=optimizer, lr_scheduler=dict(scheduler=schedule, interval="step"))


class EpochRecorder(pl.Callback):
    def __init__(self, run_dir):
        self.run_dir = Path(run_dir)
        self.history = []
        self.started = None

    def on_train_epoch_start(self, trainer, module):
        self.started = time.perf_counter()

    def on_train_start(self, trainer, module):
        barrier = os.environ.get("QM9_MTL7_CALIBRATION_BARRIER")
        if barrier:
            if not json.loads((self.run_dir / "run_plan.json").read_text())["diagnostic"]:
                raise ValueError("A calibration barrier cannot affect formal training")
            barrier = Path(barrier)
            barrier.mkdir(parents=True, exist_ok=True)
            (barrier / f"{os.getpid()}.ready").touch()
            deadline = time.monotonic() + 300
            while len(list(barrier.glob("*.ready"))) < int(
                os.environ["QM9_MTL7_CALIBRATION_COUNT"]
            ):
                if time.monotonic() > deadline:
                    raise TimeoutError("Calibration replicas did not all become ready")
                time.sleep(0.2)

    def on_train_epoch_end(self, trainer, module):
        if module.device.type == "cuda":
            torch.cuda.synchronize(module.device)
        row = dict(
            epoch=int(trainer.current_epoch) + 1,
            global_step=int(trainer.global_step),
            epoch_seconds=time.perf_counter() - self.started,
            recorded_at=utc_now(),
            lr=trainer.optimizers[0].param_groups[0]["lr"],
        )
        row.update({k: float(v) for k, v in trainer.callback_metrics.items()})
        self.history.append(row)
        pd.DataFrame(self.history).to_csv(self.run_dir / "training_log.csv", index=False)
        diagnostics = dict(
            epoch=row["epoch"],
            affinity=module.network.head.get_affinity_matrix(),
            uncertainty=module.network.head.get_uncertainty_params(),
        )
        with (self.run_dir / "task_dynamics.jsonl").open("a") as stream:
            stream.write(json.dumps(diagnostics, allow_nan=False) + "\n")
        print(
            f"EPOCH {row['epoch']} seconds={row['epoch_seconds']:.3f} "
            f"val_delta_e_mae={row.get('val_delta_e_mae')}",
            flush=True,
        )


def train(config_path, *, dry_run=False, max_epochs=None, output_dir=None, limit_rows=None):
    started = time.perf_counter()
    config, root = load_multitask7_config(config_path)
    if max_epochs is not None and not 1 <= max_epochs <= config["training"]["epochs"]:
        raise ValueError(f"Diagnostic epochs must be between 1 and {config['training']['epochs']}")
    if limit_rows is not None and limit_rows < 2:
        raise ValueError("Diagnostic subsets need at least two rows")
    diagnostic = max_epochs is not None or limit_rows is not None
    if diagnostic and output_dir is None:
        raise ValueError("Diagnostic runs require an explicit isolated output directory")
    run_dir = (
        Path(output_dir)
        if output_dir
        else (
            Path(config["output"]["root"])
            / config["experiment"]["model_id"]
            / f"seed{config['training']['seed']}"
        )
    )
    if not run_dir.is_absolute():
        run_dir = root / run_dir
    datasets, identity = load_splits(config, root)
    plan = dict(
        model=config["experiment"]["model_id"],
        targets=list(TASK_NAMES),
        units=TASK_UNITS,
        training=config["training"],
        diagnostic=diagnostic,
        diagnostic_max_epochs=max_epochs,
        split_sizes={s: len(ds) for s, ds in datasets.items()},
        data_identity=identity,
        selection="original-unit validation delta_e MAE",
        run_dir=str(run_dir),
    )
    if dry_run:
        print(json.dumps(plan, indent=2))
        return plan
    if run_dir.exists():
        raise FileExistsError(f"Existing output retained: {run_dir}")
    run_dir.mkdir(parents=True)
    scalers = datasets["train"].scalers
    if limit_rows is not None:
        for ds in datasets.values():
            ds.df = ds.df.iloc[:limit_rows].copy()
            ds.targets = ds.targets[:limit_rows]
    write_run_metadata(run_dir, config, Path(config_path), root)
    write_json(run_dir / "run_plan.json", plan)
    write_json(run_dir / "scalers.json", scalers)
    write_json(run_dir / "split_identity.json", identity)
    t = config["training"]
    pl.seed_everything(t["seed"], workers=True)
    # Construct before loaders so graph caching never changes paired initialization.
    model = MultiTask7LightningModule(config, scalers)
    parameter_count = sum(p.numel() for p in model.parameters())
    loaders = make_graph_loaders(datasets, config)
    checkpoint = ModelCheckpoint(
        dirpath=run_dir / "checkpoints",
        filename="best",
        monitor="val_delta_e_mae",
        mode="min",
        save_top_k=1,
        save_last=True,
    )
    recorder = EpochRecorder(run_dir)
    extra_callbacks = []
    if t["epochs"] == 200:
        from molgap.training.terminal_checkpoint import TerminalFitCheckpoint

        extra_callbacks.append(TerminalFitCheckpoint())
    trainer = pl.Trainer(
        accelerator=t["accelerator"],
        devices=t["devices"],
        precision=t["precision"],
        max_epochs=max_epochs or t["epochs"],
        deterministic=t["deterministic"],
        callbacks=[
            recorder,
            EarlyStopping(
                monitor="val_delta_e_mae",
                mode="min",
                patience=t["patience"],
                min_delta=t["min_delta"],
            ),
            checkpoint,
        ]
        + extra_callbacks,
        logger=CSVLogger(run_dir / "logs", name="lightning"),
        num_sanity_val_steps=0,
        enable_progress_bar=False,
        enable_model_summary=False,
        log_every_n_steps=50,
    )
    fit_started = time.perf_counter()
    trainer.fit(model, loaders["train"], loaders["val"])
    fit_seconds = time.perf_counter() - fit_started
    # Trusted, locally produced checkpoint. Retain terminal diagnostics separately.
    write_json(
        run_dir / "last_task_parameters.json",
        dict(
            affinity=model.network.head.get_affinity_matrix(),
            uncertainty=model.network.head.get_uncertainty_params(),
        ),
    )
    payload = torch.load(checkpoint.best_model_path, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    torch.save(
        dict(
            state_dict=payload["state_dict"],
            config=config,
            scalers=scalers,
            epoch=payload["epoch"],
            global_step=payload["global_step"],
        ),
        run_dir / "best_checkpoint.pt",
    )
    with measure_prediction(
        run_dir, len(datasets["test"]), "test_loader_best_checkpoint_forward", model=model
    ):
        batches = trainer.predict(model, loaders["test"])
        normalized = torch.cat(batches).detach().cpu().numpy()
    evaluator = MultiTask7Metrics(scalers)
    original = evaluator.inverse_transform(normalized)
    predictions = datasets["test"].df[["sample_id", "smiles"]].copy()
    for i, name in enumerate(TASK_NAMES):
        predictions[name] = original[:, i]
    predictions.to_csv(run_dir / "test_predictions.csv", index=False, float_format="%.17g")
    metrics = evaluator.compute(original, datasets["test"].targets, normalized=False)
    report = dict(
        targets=metrics,
        delta_e_mae=metrics["delta_e"]["mae"],
        rows=len(original),
        units=TASK_UNITS,
        protocol_id=config["experiment"]["protocol_id"],
        diagnostic=diagnostic,
    )
    write_json(run_dir / "test_metrics.json", report)
    write_json(run_dir / "metrics.json", report)
    affinity, uncertainty = (
        model.network.head.get_affinity_matrix(),
        model.network.head.get_uncertainty_params(),
    )
    if affinity:
        affinity["checkpoint"] = "best_by_validation_delta_e_mae"
        write_json(run_dir / "affinity_matrix.json", affinity)
        pd.DataFrame(affinity["matrix"], index=TASK_NAMES, columns=TASK_NAMES).to_csv(
            run_dir / "affinity_matrix.csv"
        )
    if uncertainty:
        uncertainty["checkpoint"] = "best_by_validation_delta_e_mae"
        write_json(run_dir / "uncertainty_params.json", uncertainty)
    timing = dict(
        fit_seconds=fit_seconds,
        pipeline_seconds=time.perf_counter() - started,
        epochs_completed=int(trainer.fit_loop.epoch_progress.current.completed),
        best_epoch=int(payload["epoch"]) + 1,
        best_val_delta_e_mae=float(checkpoint.best_model_score),
        parameters=parameter_count,
        diagnostic=diagnostic,
    )
    write_json(run_dir / "training_timing.json", timing)
    # Verify actual saved CSVs rather than in-memory arrays before declaring success.
    from molgap.multitask7.evaluation import verify_run

    verify_run(run_dir, root, allow_diagnostic=diagnostic)
    print(
        json.dumps(dict(run_dir=str(run_dir), timing=timing, delta_e=metrics["delta_e"])),
        flush=True,
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--limit-rows", type=int, help="Diagnostic subset only; never a formal experiment"
    )
    args = parser.parse_args()
    train(
        args.config,
        dry_run=args.dry_run,
        max_epochs=args.max_epochs,
        output_dir=args.output_dir,
        limit_rows=args.limit_rows,
    )


if __name__ == "__main__":
    main()
