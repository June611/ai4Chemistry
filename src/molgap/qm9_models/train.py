"""Train fixed-split molecular adapters with validation-only selection and audited outputs."""

from __future__ import annotations

import argparse
import copy
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm

from molgap.config import find_project_root, run_directory
from molgap.data.neural_properties import (
    is_property_run,
    scaler_record,
    validate_property_target,
    write_scaler,
)
from molgap.metrics.regression import regression_metrics
from molgap.qm9_models.batch import collate_molecules
from molgap.qm9_models.data import load_data
from molgap.qm9_models.factory import build_model
from molgap.qm9_models.policy import require_enabled
from molgap.qm9_models.runtime import make_loader, move_batch
from molgap.tracking.inference import measure_prediction
from molgap.training.protocol import GAP_PROTOCOL, validate_qm9_protocol
from molgap.utils.files import sha256, write_run_metadata


def _validate_infomax_checkpoint(config: dict, root: Path, identity: dict, geometry: dict) -> None:
    if config["model"]["name"] != "3dinfomax":
        return
    checkpoint_value = config["model"]["params"].get("pretrain_checkpoint")
    if not checkpoint_value:
        return
    checkpoint_path = Path(checkpoint_value)
    if not checkpoint_path.is_absolute():
        checkpoint_path = root / checkpoint_path
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"3D Infomax checkpoint does not exist: {checkpoint_path}")
    expected_hash = config["model"]["params"].get("pretrain_checkpoint_sha256")
    if expected_hash and sha256(checkpoint_path) != expected_hash:
        raise ValueError("3D Infomax checkpoint content changed after task preparation")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    expected_manifest = identity["manifest"]["sha256"]
    if checkpoint.get("split_manifest_sha256") != expected_manifest:
        raise ValueError(
            "3D Infomax checkpoint was not pretrained on this split's training partition"
        )
    if checkpoint.get("geometry_sha256") != geometry["geometry_sha256"]:
        raise ValueError("3D Infomax checkpoint geometry differs from the selected split artifact")
    expected_seed = int(config["data"]["split"]["seed"])
    if checkpoint.get("training_seed") != expected_seed:
        raise ValueError(
            "3D Infomax checkpoint pretraining seed differs from the selected split seed"
        )


def run(
    config_path: Path,
    *,
    smoke: bool = False,
    dry_run: bool = False,
    seed_override: int | None = None,
    split_seed_override: int | None = None,
    pretrain_checkpoint: Path | None = None,
) -> dict:
    try:
        root = find_project_root(config_path)
    except ValueError:
        # Queue snapshots may deliberately live outside the repository.
        root = find_project_root()
    config = yaml.safe_load(config_path.read_text())
    config = copy.deepcopy(config)
    training = config["training"]
    property_run = is_property_run(config)
    target, unit = validate_property_target(config) if property_run else ("delta_e", "hartree")
    if property_run and (training["target_scaling"] != "standard" or training["loss"] != "mse"):
        raise ValueError("Neural property experiments require standard target scaling and MSE")
    if not property_run:
        config["experiment"]["protocol_id"] = GAP_PROTOCOL
    if not dry_run:
        require_enabled(config["model"]["name"], root)
    if training["early_stopping"].get("monitor") != "val_mae":
        raise ValueError("QM9 downstream selection must monitor raw Hartree gap val_mae")
    if split_seed_override is not None:
        if property_run:
            raise ValueError("Use the frozen per-seed property config, not --split-seed")
        split_seed = int(split_seed_override)
        reference = (
            "configs/baseline.yaml"
            if split_seed == 3407
            else f"configs/qm9_models/split_references/seed{split_seed}.yaml"
        )
        if not (root / reference).is_file():
            raise ValueError(f"Unsupported split seed {split_seed}: missing {reference}")
        config["data"].update(
            {
                "processed_dir": f"data/processed/qm9_full/random/seed{split_seed}",
                "geometry": f"data/geometry/qm9_full_seed{split_seed}.npz",
                "chemprop_reference_config": reference,
            }
        )
        config["data"]["split"]["seed"] = split_seed
    if seed_override is not None:
        training["seed"] = seed_override
    split_seed = int(config["data"]["split"]["seed"])
    seed = int(training["seed"])
    if seed != split_seed:
        raise ValueError(
            "The paired protocol requires training seed to equal split seed "
            f"(training={seed}, split={split_seed})."
        )
    config["experiment"]["model_id"] = config["model"]["name"]
    training["run_mode"] = "paired_split_seed"
    training["split_seed"] = split_seed
    training["seed_pairing"] = "matched"
    if not smoke and training.get("formal_protocol", False):
        validate_qm9_protocol(config, property_run=property_run)
    if pretrain_checkpoint is not None:
        if config["model"]["name"] != "3dinfomax":
            raise ValueError("--pretrain-checkpoint is only valid for 3dinfomax")
        config["model"]["params"]["pretrain_checkpoint"] = str(pretrain_checkpoint)
    if smoke:
        config["experiment"]["name"] += "_smoke"
        training["epochs"] = 1
        training["num_workers"] = config.get("smoke", {}).get("num_workers", 0)
        training["batch_size"] = min(
            config.get("smoke", {}).get("batch_size", 8), training["batch_size"]
        )
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    accelerator = training["accelerator"]
    if accelerator == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(
        "cuda" if accelerator in {"auto", "cuda"} and torch.cuda.is_available() else "cpu"
    )
    frames, datasets, identity, geometry = load_data(config, root, smoke)
    _validate_infomax_checkpoint(config, root, identity, geometry)
    model = build_model(config["model"]["name"], config["model"]["params"], root, device)
    plan = {
        "model": config["model"]["name"],
        "protocol_id": config["experiment"]["protocol_id"],
        "split_seed": split_seed,
        "training_seed": seed,
        "budget": {
            "batch_size": training["batch_size"],
            "epochs": training["epochs"],
            "early_stopping_patience": training["early_stopping"]["patience"],
        },
        "smoke": smoke,
        "selection": {
            "target": target,
            "metric": "mae",
            "unit": unit,
            "mode": "min",
            "monitor": "val_mae",
            "patience": training["early_stopping"]["patience"],
            "min_delta": training["early_stopping"]["min_delta"],
        },
        "split_rows": {k: len(v) for k, v in frames.items()},
        "parameters": sum(p.numel() for p in model.parameters()),
        "split_identity": identity,
        "geometry_sha256": geometry["geometry_sha256"],
        "topology_sha256": geometry.get("topology_sha256"),
    }
    if config["model"]["name"] == "3dinfomax":
        pretrain_path = Path(config["model"]["params"]["pretrain_checkpoint"])
        plan["pretrain_checkpoint_sha256"] = sha256(
            pretrain_path if pretrain_path.is_absolute() else root / pretrain_path
        )
    if dry_run:
        print(json.dumps(plan, indent=2))
        return plan
    run_dir = run_directory(config, root)
    if run_dir.exists():
        raise FileExistsError(f"Refusing to overwrite run: {run_dir}")
    run_dir.mkdir(parents=True)
    write_run_metadata(run_dir, config, config_path, root)
    (run_dir / "split_identity.json").write_text(json.dumps(identity, indent=2) + "\n")
    (run_dir / "model_summary.json").write_text(json.dumps(plan, indent=2) + "\n")
    (run_dir / "upstream_sources.json").write_text(
        (root / "configs/qm9_models/sources.lock.json").read_text()
    )
    if config.get("provenance", {}).get("sources_file"):
        (run_dir / "reproduction_sources.json").write_text(
            (root / config["provenance"]["sources_file"]).read_text()
        )
    loaders = {
        name: make_loader(
            dataset,
            training=training,
            device=device,
            batch_size=training["batch_size"],
            shuffle=name == "train",
            collate_fn=collate_molecules,
            generator=torch.Generator().manual_seed(seed),
            smoke=smoke,
        )
        for name, dataset in datasets.items()
    }
    target_scaling = training["target_scaling"]
    if target_scaling not in {"mad", "standard"}:
        raise ValueError("training.target_scaling must be 'mad' or 'standard'")
    targets = frames["train"][target].to_numpy()
    mean = float(targets.mean())
    scale = float(np.abs(targets - mean).mean() if target_scaling == "mad" else targets.std(ddof=0))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Training-only target scale must be finite and positive")
    target_scaler = {
        "method": target_scaling,
        "fit_split": "train",
        "target": target,
        "unit": unit,
        "mean_hartree": mean,
        "scale_hartree": scale,
        "train_rows": len(targets),
    }
    if property_run:
        target_scaler = scaler_record(targets, target, unit)
        write_scaler(run_dir, target_scaler, "native_graph_encoding_and_geometry")
    else:
        (run_dir / "target_scaler.json").write_text(json.dumps(target_scaler, indent=2) + "\n")
    optimizer_class = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW}[training["optimizer"]]
    optimizer = optimizer_class(
        model.parameters(), lr=training["lr"], weight_decay=training["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=training["epochs"], eta_min=training["lr"] * 0.01
    )
    loss_fn = {"mae": torch.nn.functional.l1_loss, "mse": torch.nn.functional.mse_loss}[
        training["loss"]
    ]
    auxiliary_weight = float(training.get("auxiliary_bond_weight", 0.0))
    if auxiliary_weight < 0 or (auxiliary_weight and not hasattr(model, "auxiliary_loss")):
        raise ValueError("Invalid auxiliary bond weight/model combination")

    # Get early stopping params with defaults
    early_stopping = training.get("early_stopping", {})
    min_delta = early_stopping.get("min_delta", 0.0)
    patience = early_stopping.get("patience", 100)

    best, stale, history = float("inf"), 0, []
    selection_reference = float("inf")
    started = time.perf_counter()
    for epoch in range(training["epochs"]):
        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        epoch_started = time.perf_counter()
        model.train()
        supervised_total = torch.zeros((), device=device)
        auxiliary_total = torch.zeros((), device=device)
        total = torch.zeros((), device=device)
        batches = 0
        for raw in tqdm(loaders["train"], desc=f"Epoch {epoch + 1}", mininterval=2):
            batch = move_batch(raw, device)
            optimizer.zero_grad()
            prediction = model(batch)
            expected = (batch["target"] - mean) / scale
            if prediction.shape != expected.shape:
                raise ValueError(
                    f"Model returned {tuple(prediction.shape)} for target {tuple(expected.shape)}"
                )
            loss = loss_fn(prediction, expected)
            supervised_total += loss.detach() * len(prediction)
            if auxiliary_weight:
                auxiliary = model.auxiliary_loss()
                loss = loss + auxiliary_weight * auxiliary
                auxiliary_total += auxiliary.detach() * len(prediction)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), training["gradient_clip_val"])
            optimizer.step()
            total += loss.detach() * len(prediction)
            batches += 1
        if device.type == "cuda":
            torch.cuda.synchronize()
        train_seconds = time.perf_counter() - epoch_started
        model.eval()
        prediction_batches = []
        with torch.no_grad():
            for raw in loaders["val"]:
                prediction_batches.append(model(move_batch(raw, device)) * scale + mean)
        predictions = torch.cat(prediction_batches).cpu().numpy()
        if len(predictions) != len(frames["val"]) or not np.isfinite(predictions).all():
            raise ValueError("Validation predictions are missing or nonfinite")
        val_mae = regression_metrics(frames["val"][target], predictions)["mae"]
        if val_mae < best:
            best = val_mae
            torch.save(
                {
                    "model_type": config["model"]["name"],
                    "model_config": config["model"]["params"],
                    "state_dict": model.state_dict(),
                    "target_scaler": target_scaler,
                    "epoch": epoch + 1,
                    "best_validation_mae": best,
                    "split_manifest_sha256": identity["manifest"]["sha256"],
                    "geometry_sha256": geometry["geometry_sha256"],
                    "topology_sha256": geometry.get("topology_sha256"),
                },
                run_dir / "model.pt",
            )
        if val_mae < selection_reference - min_delta:
            selection_reference, stale = val_mae, 0
        else:
            stale += 1
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": total.item() / len(datasets["train"]),
                "supervised_loss": supervised_total.item() / len(datasets["train"]),
                "auxiliary_bond_loss": auxiliary_total.item() / len(datasets["train"]),
                "val_mae": val_mae,
                "train_batches": batches,
                "train_seconds": train_seconds,
                "epoch_seconds": time.perf_counter() - epoch_started,
                "cuda_peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20
                if device.type == "cuda"
                else None,
                "cuda_peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20
                if device.type == "cuda"
                else None,
            }
        )
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
        print(
            f"Epoch {epoch + 1}: val_{target if property_run else 'gap'}_mae={val_mae:.8g} "
            f"{unit if property_run else 'Hartree'} "
            f"train_seconds={train_seconds:.1f}",
            flush=True,
        )
        scheduler.step()
        if stale >= patience:
            break
    fit_seconds = time.perf_counter() - started
    best_state = torch.load(run_dir / "model.pt", map_location=device, weights_only=False)
    model.load_state_dict(best_state["state_dict"])
    model.eval()
    prediction_batches = []
    with (
        measure_prediction(run_dir, len(frames["test"]), "test_loader_and_forward", model=model),
        torch.no_grad(),
    ):
        for raw in loaders["test"]:
            prediction_batches.append(model(move_batch(raw, device)) * scale + mean)
    predictions = torch.cat(prediction_batches).cpu().numpy()
    if not np.isfinite(predictions).all():
        raise ValueError("Nonfinite test predictions")
    result = frames["test"][["sample_id", "smiles", target]].copy()
    result[f"{target}_pred"] = predictions
    (run_dir / "predictions").mkdir()
    result.to_csv(run_dir / "predictions/test_predictions.csv", index=False)
    metrics = {
        "target": target,
        "unit": unit,
        "smoke": smoke,
        **regression_metrics(result[target], predictions),
        "best_validation_mae": best,
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    (run_dir / "training_timing.json").write_text(
        json.dumps({"fit_seconds": fit_seconds, "epochs_completed": len(history)}, indent=2) + "\n"
    )
    print(json.dumps(metrics, indent=2))
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--seed", type=int, help="Override training seed; must match the split seed"
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        help="Use a prepared QM9 data split and matching reordered geometry artifact",
    )
    parser.add_argument(
        "--pretrain-checkpoint",
        type=Path,
        help="QM9 3D-Infomax pretraining checkpoint for downstream PNA transfer",
    )
    args = parser.parse_args()
    run(
        args.config,
        smoke=args.smoke,
        dry_run=args.dry_run,
        seed_override=args.seed,
        split_seed_override=args.split_seed,
        pretrain_checkpoint=args.pretrain_checkpoint,
    )


if __name__ == "__main__":
    main()
