"""Audited 100->200 Lightning continuation, separate from fresh-200 comparisons.

Default is preflight only. Only trusted local checkpoints are accepted. Keeps the
original 100-epoch LR schedule and its terminal LR; does not restart warmup.
"""

import argparse
import copy
import json
import sys
import time
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def validate_payload(payload):
    if not all(
        payload.get(k) for k in ("state_dict", "optimizer_states", "lr_schedulers", "loops")
    ):
        raise ValueError("Full weights/optimizer/scheduler/loop checkpoint required")
    completed = (
        payload["loops"]
        .get("fit_loop", {})
        .get("epoch_progress", {})
        .get("total", {})
        .get("completed")
    )
    # Lightning versions differ in whether terminal save stores epoch99 or epoch100.
    if (payload.get("epoch"), completed) not in {
        (99, 99),
        (99, 100),
        (100, 100),
    } or not payload.get("global_step"):
        raise ValueError("Strict 100->200 requires a completed epoch100 checkpoint")
    if any(s.get("last_epoch") != payload["global_step"] for s in payload["lr_schedulers"]):
        raise ValueError("Scheduler progress differs from global_step")


def build_plan(source, checkpoint, output, backend):
    import torch
    import yaml

    from molgap.config import find_project_root
    from molgap.utils.files import sha256

    source, checkpoint, output = map(lambda p: Path(p).resolve(), (source, checkpoint, output))
    if output.exists():
        raise FileExistsError("New continuation directory required")
    if source not in checkpoint.parents:
        raise ValueError("Checkpoint must belong to the named source run")
    root = find_project_root(source)
    cfg = yaml.safe_load((source / "config.yaml").read_text())
    if cfg["training"]["epochs"] != 100 or cfg["training"]["batch_size"] != 256:
        raise ValueError("Source must use the original batch256/100-epoch protocol")
    if cfg["training"]["seed"] != cfg["data"]["split"]["seed"]:
        raise ValueError("Split/init seed mismatch")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    validate_payload(payload)
    best_states = [
        v
        for v in payload.get("callbacks", {}).values()
        if isinstance(v, dict) and v.get("best_model_score") is not None
    ]
    if len(best_states) != 1:
        raise ValueError("A unique historical best-checkpoint callback is required")
    historical = Path(best_states[0]["best_model_path"]).resolve()
    if not historical.is_file() or source not in historical.parents:
        raise ValueError("Historical best checkpoint must remain available in the source run")
    env = json.loads((source / "environment.json").read_text())
    inputs = env.get("inputs", {})
    if not all(k in inputs for k in ("train.csv", "val.csv", "test.csv")):
        raise ValueError("Source input fingerprints are missing")
    for item in inputs.values():
        if sha256(Path(item["path"])) != item["sha256"]:
            raise ValueError("Source data identity changed")
    import math

    import pandas as pd

    rows = len(pd.read_csv(inputs["train.csv"]["path"], usecols=[cfg["data"]["id_column"]]))
    if payload["global_step"] != 100 * math.ceil(rows / 256):
        raise ValueError("Checkpoint does not contain 100 full epochs of optimizer updates")
    plan = dict(
        backend=backend,
        source_run=str(source),
        checkpoint=str(checkpoint),
        checkpoint_sha256=sha256(checkpoint),
        historical_best_checkpoint=str(historical),
        historical_best_sha256=sha256(historical),
        historical_best_score=float(best_states[0]["best_model_score"]),
        output_dir=str(output),
        epochs=200,
        parent_epochs=100,
        batch_size=256,
        lr_policy="retain_100epoch_schedule_then_hold_final",
        evaluation="original-unit validation MAE; gap MAE for seven-task",
        continuity="optimizer/scheduler/loop resume; historical RNG/worker state incomplete",
        comparison_group="continuation_100_plus_100_not_fresh200",
        inputs=inputs,
    )
    return plan, cfg, root, payload


def build_components(backend, cfg, source, root, payload):
    import numpy as np
    import torch

    if backend == "chemprop":
        from chemprop.data import build_dataloader
        from chemprop.models import MPNN

        from molgap.training.gap_metric import GapMetricScale
        from molgap.training.phase3 import _dataset
        from molgap.transformer.dataset import _read_and_validate_splits

        frames, _ = _read_and_validate_splits(cfg, root)
        datasets = {s: _dataset(f, cfg) for s, f in frames.items()}
        scaler = datasets["train"].normalize_targets()
        datasets["val"].normalize_targets(scaler)
        # Use the original architecture, transforms and criterion from this trusted checkpoint.
        model = MPNN.load_from_checkpoint(str(source["checkpoint"]), map_location="cpu")
        transform = model.predictor.output_transform
        np.testing.assert_allclose(
            transform.mean.cpu().numpy().reshape(-1), scaler.mean_, rtol=1e-5
        )
        np.testing.assert_allclose(
            transform.scale.cpu().numpy().reshape(-1), scaler.scale_, rtol=1e-5
        )
        t = cfg["training"]
        loaders = {
            s: build_dataloader(
                ds,
                batch_size=256,
                num_workers=t["num_workers"],
                seed=t["seed"],
                shuffle=s == "train",
                drop_last=False,
            )
            for s, ds in datasets.items()
        }
        monitor = "val/" + t["selection_metric"]
        return model, loaders, frames, monitor, [GapMetricScale()], lambda x: x
    if backend == "sequence":
        from molgap.training.transformer import (
            TransformerLightningModule,
            _build_model,
            _make_loaders,
        )
        from molgap.transformer.dataset import prepare_transformer_data

        prepared = prepare_transformer_data(cfg, root)
        original = Path(source["source_run"])
        # Vocabulary order and target scaling must match the checkpoint's original run.
        tokenizer = original / "vocabulary.json"
        if not tokenizer.exists():
            raise ValueError("Original vocabulary.json required for sequence continuation")
        # Use the same deterministic train-only preparation and check the saved buffers.
        base, _ = _build_model(cfg, prepared)
        model = TransformerLightningModule(
            base, prepared.target_mean, prepared.target_scale, cfg["training"]
        )
        for key in ("target_mean", "target_scale"):
            torch.testing.assert_close(model.state_dict()[key], payload["state_dict"][key])
        if prepared.tokenizer.to_dict() != json.loads(tokenizer.read_text()):
            raise ValueError("Tokenizer identity differs from the source run")
        return model, _make_loaders(prepared, cfg), prepared.frames, "val_mae", [], lambda x: x
    if backend == "multitask7":
        from molgap.multitask7.data import load_splits, make_graph_loaders
        from molgap.multitask7.metrics import MultiTask7Metrics
        from molgap.training.multitask7 import MultiTask7LightningModule

        datasets, _ = load_splits(cfg, root)
        scalers = datasets["train"].scalers
        saved = json.loads((Path(source["source_run"]) / "scalers.json").read_text())
        if saved != scalers:
            raise ValueError("Seven-task scaler identity differs")
        model = MultiTask7LightningModule(cfg, scalers)
        evaluator = MultiTask7Metrics(scalers)
        return (
            model,
            make_graph_loaders(datasets, cfg),
            {s: ds.df for s, ds in datasets.items()},
            "val_delta_e_mae",
            [],
            evaluator.inverse_transform,
        )
    raise ValueError("Unknown backend")


def keep_original_schedule(model):
    """Build the original schedule while allowing Trainer to execute to epoch200."""
    original_configure = model.configure_optimizers

    def configure(self):
        horizon = self.trainer.fit_loop.max_epochs
        self.trainer.fit_loop.max_epochs = 100
        try:
            return original_configure()
        finally:
            self.trainer.fit_loop.max_epochs = horizon

    model.configure_optimizers = types.MethodType(configure, model)


def execute(plan, cfg, root, payload):
    import lightning.pytorch as pl
    import numpy as np
    import torch
    import yaml
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
    from lightning.pytorch.loggers import CSVLogger

    from molgap.metrics.regression import regression_metrics
    from molgap.tracking.records import write_json
    from molgap.training.terminal_checkpoint import TerminalFitCheckpoint

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; refusing CPU fallback")
    pl.seed_everything(cfg["training"]["seed"], workers=True)
    model, loaders, frames, monitor, callbacks, inverse = build_components(
        plan["backend"], cfg, plan, root, payload
    )
    # Optimizer/scheduler are constructed using the original schedule horizon (100).
    # Trainer's execution limit is 200. Restored LambdaLR holds its terminal rate.
    keep_original_schedule(model)
    output = Path(plan["output_dir"])
    output.mkdir(parents=True)
    recorded = copy.deepcopy(cfg)
    recorded["training"]["epochs"] = 200
    recorded["continuation"] = plan
    (output / "config.yaml").write_text(yaml.safe_dump(recorded, sort_keys=False))
    write_json(output / "continuation.json", plan)
    cp = ModelCheckpoint(
        dirpath=output / "checkpoints",
        filename="best",
        monitor=monitor,
        mode="min",
        save_top_k=1,
        save_last=True,
    )
    t = cfg["training"]
    patience = t.get("patience", t.get("early_stopping_patience", 20))
    callbacks += [
        EarlyStopping(
            monitor,
            mode="min",
            patience=patience,
            min_delta=t.get("min_delta", t.get("early_stopping_min_delta", 0.0)),
        ),
        cp,
        TerminalFitCheckpoint(),
    ]
    trainer = pl.Trainer(
        max_epochs=200,
        accelerator="gpu",
        devices=1,
        precision="32-true",
        deterministic=True,
        gradient_clip_val=float(t.get("gradient_clip_val") or t.get("grad_clip") or 0.0),
        callbacks=callbacks,
        logger=CSVLogger(output / "logs"),
        num_sanity_val_steps=0,
        enable_progress_bar=False,
    )
    started = time.perf_counter()
    trainer.fit(model, loaders["train"], loaders["val"], ckpt_path=plan["checkpoint"])
    elapsed = time.perf_counter() - started
    # Retain the best over both the original and the additional training period.
    source_scores = [
        float(v["best_model_score"])
        for v in payload.get("callbacks", {}).values()
        if isinstance(v, dict) and v.get("best_model_score") is not None
    ]
    if not source_scores:
        raise ValueError("Historical best score missing; cannot compare the full 1..200 range")
    chosen = (
        cp.best_model_path
        if cp.best_model_score is not None and float(cp.best_model_score) < min(source_scores)
        else plan["historical_best_checkpoint"]
    )
    model.load_state_dict(torch.load(chosen, map_location="cpu", weights_only=False)["state_dict"])
    predictions = inverse(torch.cat(trainer.predict(model, loaders["test"])).cpu().numpy())
    if predictions.ndim == 1:
        predictions = predictions[:, None]
    targets = cfg["data"].get("targets", [cfg["data"].get("target_column")])
    frame = frames["test"]
    if len(frame) != len(predictions) or not np.isfinite(predictions).all():
        raise ValueError("Invalid original-unit predictions")
    result = frame[["sample_id", "smiles", *targets]].copy()
    for i, target in enumerate(targets):
        result[target + "_pred"] = predictions[:, i]
    result.to_csv(output / "test_predictions.csv", index=False, float_format="%.17g")
    write_json(
        output / "metrics.json",
        dict(
            targets={
                target: regression_metrics(frame[target], predictions[:, i])
                for i, target in enumerate(targets)
            },
            selected_checkpoint=chosen,
            training_plan=plan,
        ),
    )
    write_json(
        output / "training_timing.json",
        dict(
            additional_fit_seconds=elapsed,
            terminal_epoch=trainer.current_epoch,
            parent_epochs=100,
            max_epochs=200,
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend", choices=("chemprop", "sequence", "multitask7"), required=True)
    p.add_argument("--source-run", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--execute", action="store_true")
    p.add_argument(
        "--verify-components",
        action="store_true",
        help="Build original model/data on CPU without fitting",
    )
    args = p.parse_args()
    plan, cfg, root, payload = build_plan(
        args.source_run, args.checkpoint, args.output_dir, args.backend
    )
    print(json.dumps(plan, indent=2))
    if args.verify_components:
        build_components(args.backend, cfg, plan, root, payload)
        print("Model/data/scaler/tokenizer preflight passed; no fitting performed")
    if args.execute:
        execute(plan, cfg, root, payload)
