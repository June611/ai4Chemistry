"""Fresh-200 shared-head training with full epoch-boundary recovery, never test selection."""

import argparse
import copy
import csv
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from molgap.config import run_directory
from molgap.modular.backbones import MolecularEncoder, move_inputs
from molgap.modular.config import load
from molgap.modular.data import load_labels
from molgap.modular.heads import IndependentTaskHead, TaskAttentionHead
from molgap.modular.inputs import make_inputs, make_loaders
from molgap.modular.physics import TASKS7, QM9PhysicsLoss
from molgap.training.terminal_checkpoint import loader_state, restore_loader, restore_rng, rng_state
from molgap.utils.files import write_run_metadata

# Set before the first CUDA matmul, including direct Python/CLI entrypoints.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def atomic_checkpoint(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


class ModularNetwork(nn.Module):
    def __init__(self, config, root, device, vocab_size=None):
        super().__init__()
        self.variant = config["module"]["variant"]
        m = config["module"]
        name = config["backbone"]["model"]
        if self.variant in {"D1", "D2"}:
            from molgap.multitask7.models import ChempropSevenTaskNetwork

            self.encoder = ChempropSevenTaskNetwork(config["source"])
            self.head = None
        elif self.variant == "single_task" and name == "chemprop":
            from chemprop.nn.ffn import MLP

            self.encoder = MolecularEncoder(config, root, device)
            old = config["source"]["model"]
            self.head = MLP.build(
                self.encoder.output_dim,
                1,
                old["ffn_hidden_dim"],
                old["ffn_num_layers"],
                old["dropout"],
                old["activation"],
            )
        elif self.variant == "single_task":
            from molgap.qm9_models.factory import build_model

            self.encoder = build_model(
                config["backbone"]["model"], config["source"]["model"]["params"], root, device
            )
            self.head = None
        else:
            self.encoder = MolecularEncoder(config, root, device, vocab_size)
            cls = TaskAttentionHead if self.variant == "task_attention7" else IndependentTaskHead
            kwargs = dict(
                input_dim=self.encoder.output_dim,
                n_tasks=len(config["data"]["targets"]),
                hidden_dim=m["hidden_dim"],
                dropout=m["dropout"],
            )
            if self.variant == "task_attention7":
                kwargs["num_heads"] = m["num_heads"]
            self.head = cls(**kwargs)

    def forward(self, inputs):
        if self.variant in {"D1", "D2"}:
            return self.encoder(inputs["graph"], inputs.get("V_d"))
        value = self.encoder(inputs)
        return value.reshape(-1, 1) if self.head is None else self.head(value)


def make_optimizer(model, config, steps):
    t = config["training"]
    name = config["backbone"]["model"]
    original = config["source"]["training"]
    if name == "chemprop":
        from chemprop.schedulers import build_NoamLike_LRSched

        optimizer = torch.optim.Adam(model.parameters(), lr=original["init_lr"])
        scheduler = build_NoamLike_LRSched(
            optimizer,
            original["warmup_epochs"] * steps,
            (t["epochs"] - original["warmup_epochs"]) * steps,
            original["init_lr"],
            original["max_lr"],
            original["final_lr"],
        )
        return optimizer, scheduler, "step"
    optimizer_cls = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW}[original["optimizer"]]
    optimizer = optimizer_cls(
        model.parameters(), lr=original["lr"], weight_decay=original["weight_decay"]
    )
    if name in {"smiles_transformer", "spe_transformer"}:
        total = t["epochs"] * steps
        warm = int(original["warmup_ratio"] * total)

        def factor(step):
            if step < warm:
                return max((step + 1) / max(1, warm), 1 / max(1, warm))
            p = min(1.0, max(0.0, (step - warm) / max(1, total - warm)))
            return original["min_lr_ratio"] + (1 - original["min_lr_ratio"]) * 0.5 * (
                1 + math.cos(math.pi * p)
            )

        return optimizer, torch.optim.lr_scheduler.LambdaLR(optimizer, factor), "step"
    return (
        optimizer,
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=t["epochs"], eta_min=original["lr"] * 0.01
        ),
        "epoch",
    )


def original_predictions(z, scalers, tasks):
    return np.asarray(z, dtype=np.float64) * np.array(
        [scalers[t]["scale"] for t in tasks]
    ) + np.array([scalers[t]["mean"] for t in tasks])


def metrics(prediction, truth, tasks):
    from molgap.metrics.regression import regression_metrics

    if prediction.shape != truth.shape or not np.isfinite(prediction).all():
        raise ValueError("Nonfinite or misaligned property predictions")
    out = {}
    for j, t in enumerate(tasks):
        out[t] = regression_metrics(truth[:, j], prediction[:, j])
    return out


@torch.no_grad()
def evaluate(model, loader, frame, tasks, scalers, device):
    model.eval()
    outputs = []
    ids = []
    attention = []
    for inputs, _, names in loader:
        out = model(move_inputs(inputs, device))
        outputs.append(out.cpu().numpy())
        ids.extend(names)
        if isinstance(model.head, TaskAttentionHead):
            attention.append(model.head.last_attention.sum(0).cpu())
    if ids != frame.sample_id.astype(str).tolist():
        raise ValueError("Evaluation predictions lost row identity/order")
    pred = original_predictions(np.concatenate(outputs), scalers, tasks)
    values = metrics(pred, frame[list(tasks)].to_numpy(dtype=np.float64), tasks)
    return pred, values, None if not attention else torch.stack(attention).sum(0) / len(ids)


def train(
    config_path,
    *,
    resume=None,
    output_dir=None,
    diagnostic_epochs=None,
    limit_rows=None,
    device_override=None,
    dry_run=False,
):
    config, root = load(config_path)
    config = copy.deepcopy(config)
    diagnostic = diagnostic_epochs is not None or limit_rows is not None
    if diagnostic and output_dir is None:
        raise ValueError("Diagnostics require an isolated --output-dir")
    if device_override == "cpu" and not diagnostic and not dry_run:
        raise ValueError("CPU execution is diagnostic only")
    if diagnostic_epochs is not None and not 1 <= diagnostic_epochs <= 200:
        raise ValueError("Invalid diagnostic epoch cap")
    if limit_rows is not None and limit_rows < 2:
        raise ValueError("At least two rows per split required")
    device = torch.device(device_override or ("cpu" if dry_run else "cuda"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; refusing silent CPU fallback")
    t = config["training"]
    tasks = tuple(config["data"]["targets"])
    selection_target = config["data"]["selection_target"]
    seed = t["seed"]
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    frames, scalers, identity = load_labels(config, root)
    resume_payload = None
    if resume:
        resume_payload = torch.load(resume, map_location="cpu", weights_only=False)
        if resume_payload.get("format") != "molgap_modular_epoch_v1":
            raise ValueError("A new full training checkpoint is required")
        saved_scalers = resume_payload["run_context"]["scalers"]
        if set(saved_scalers) != set(scalers):
            raise ValueError("Resume scaler task set differs")
        for task, fitted in scalers.items():
            saved = saved_scalers[task]
            for key, value in fitted.items():
                if key in {"mean", "scale", "mean_hartree", "scale_hartree"}:
                    same = math.isclose(value, saved[key], rel_tol=1e-12, abs_tol=1e-14)
                else:
                    same = value == saved[key]
                if not same:
                    raise ValueError(f"Resume scaler differs: {task}/{key}")
        # Reuse original coefficients, rather than refitting with a possibly
        # different NumPy/CPU reduction order on the destination machine.
        scalers = copy.deepcopy(saved_scalers)
    # Fit every scaler on the complete frozen training partition, even for a diagnostic prefix.
    if limit_rows:
        frames = {s: f.head(limit_rows).copy() for s, f in frames.items()}
    run = Path(output_dir).resolve() if output_dir else run_directory(config, root)
    if run.exists() and not resume and not dry_run:
        raise FileExistsError(f"Retain existing run: {run}")
    inputs, kind, tokenizer = make_inputs(
        config,
        root,
        frames,
        identity,
        saved_tokenizer=resume_payload["run_context"]["data_identity"].get("tokenizer")
        if resume
        else None,
    )
    loaders = make_loaders(config, inputs, kind, frames, scalers)
    model = ModularNetwork(
        config, root, device, None if tokenizer is None else len(tokenizer.vocabulary)
    ).to(device)
    optimizer, scheduler, interval = make_optimizer(model, config, len(loaders["train"]))
    criterion = (
        QM9PhysicsLoss(scalers, config["module"]["physics"]).to(device)
        if len(tasks) == 10
        else None
    )
    context = dict(
        config=config,
        data_identity=identity,
        scalers=scalers,
        limit_rows=limit_rows,
        diagnostic=diagnostic,
        visible_gpu_count=torch.cuda.device_count() if device.type == "cuda" else 0,
        numerical_policy=dict(
            precision="float32",
            matmul_allow_tf32=False,
            cublas_workspace_config=os.environ["CUBLAS_WORKSPACE_CONFIG"],
            torch_version=str(torch.__version__),
        ),
    )
    if dry_run:
        from_epoch = None
        if resume:
            saved = resume_payload
            expected_gpu_count = saved["run_context"]["visible_gpu_count"]
            context["visible_gpu_count"] = expected_gpu_count
            if saved.get("format") != "molgap_modular_epoch_v1" or saved["run_context"] != context:
                raise ValueError(
                    "Checkpoint protocol/data/config/scalers/diagnostic context differs"
                )
            if saved["early_stopped"]:
                raise ValueError("Run already early-stopped; do not reset its patience silently")
            if saved["completed_epochs"] >= (diagnostic_epochs or t["epochs"]):
                raise ValueError("Checkpoint already reached the requested epoch cap")
            if not saved.get("optimizer_states") or not saved.get("lr_schedulers"):
                raise ValueError("A full optimizer/scheduler checkpoint is required")
            model.load_state_dict(saved["state_dict"])
            from_epoch = saved["completed_epochs"]
        return dict(
            run_dir=str(run),
            variant=config["module"]["variant"],
            targets=tasks,
            parameters=sum(p.numel() for p in model.parameters()),
            data_identity=identity,
            training=t,
            executable=True,
            resume_from_epoch=from_epoch,
            diagnostic_epoch_cap=diagnostic_epochs,
            gpu_topology_verified=(
                expected_gpu_count == 0 or torch.cuda.device_count() == expected_gpu_count
            )
            if resume
            else False,
        )
    start = 0
    step = 0
    best = float("inf")
    best_epoch = 0
    bad = 0
    history = []
    accumulated = 0.0
    best_weights = None
    checkpoints = run / "checkpoints"
    limit = diagnostic_epochs or t["epochs"]
    if resume:
        payload = resume_payload
        if payload.get("format") != "molgap_modular_epoch_v1" or payload["run_context"] != context:
            raise ValueError("Checkpoint protocol/data/config/scalers/diagnostic context differs")
        if payload["early_stopped"]:
            raise ValueError("Run already early-stopped; do not reset its patience silently")
        if payload["completed_epochs"] >= limit:
            raise ValueError("Checkpoint already reached the requested epoch cap")
        if "best_inference_state" not in payload:
            raise ValueError("Checkpoint lacks historical best weights")
        model.load_state_dict(payload["state_dict"])
        optimizer.load_state_dict(payload["optimizer_states"][0])
        scheduler.load_state_dict(payload["lr_schedulers"][0])
        for state in optimizer.state.values():
            for key, value in state.items():
                if isinstance(value, torch.Tensor) and key != "step":
                    state[key] = value.to(device)
        for s, loader in loaders.items():
            restore_loader(loader, payload["loader_states"][s])
        start = payload["completed_epochs"]
        step = payload["global_step"]
        best = payload["best_score"]
        best_epoch = payload["best_epoch"]
        bad = payload["bad_epochs"]
        history = payload["history"]
        accumulated = payload["fit_seconds"]
        best_weights = payload["best_inference_state"]
        # Preserve abandoned future history when rewinding to an older periodic checkpoint.
        backup = run / "resume_backups" / str(time.time_ns())
        backup.mkdir(parents=True)
        for name in ("training_log.csv", "COMPLETED.json", "training_timing.json"):
            if (run / name).exists():
                import shutil

                shutil.copy2(run / name, backup / name)
        # A newer branch's best must never masquerade as the resumed branch's
        # best. Historical best weights are embedded in every full checkpoint;
        # the authoritative prediction export is best_checkpoint.pt.
        if (checkpoints / "best.ckpt").exists():
            import shutil

            shutil.move(str(checkpoints / "best.ckpt"), str(backup / "best.ckpt"))
        restore_rng(payload["rng_snapshot"])
    else:
        run.mkdir(parents=True)
        write_run_metadata(run, config, Path(config_path), root)
        write_json(run / "scalers.json", scalers)
        write_json(run / "split_identity.json", identity)
        if tokenizer:
            write_json(run / "vocabulary.json", tokenizer.to_dict())
    checkpoints.mkdir(exist_ok=True)
    if start >= limit:
        raise ValueError("Checkpoint already reached the requested epoch cap")
    fit_started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    early_stopped = False

    def save(epoch, stopped=False):
        return dict(
            format="molgap_modular_epoch_v1",
            state_dict=model.state_dict(),
            optimizer_states=[optimizer.state_dict()],
            lr_schedulers=[scheduler.state_dict()],
            epoch=epoch - 1,
            completed_epochs=epoch,
            global_step=step,
            loops=dict(fit_loop=dict(epoch_progress=dict(total=dict(completed=epoch)))),
            rng_snapshot=rng_state(include_cuda=device.type == "cuda"),
            loader_states={s: loader_state(v) for s, v in loaders.items()},
            run_context=context,
            best_score=best,
            best_epoch=best_epoch,
            best_inference_state=best_weights,
            bad_epochs=bad,
            history=history,
            early_stopped=stopped,
            fit_seconds=accumulated + time.perf_counter() - fit_started,
        )

    for epoch in range(start + 1, limit + 1):
        began = time.perf_counter()
        model.train()
        total = 0.0
        seen = 0
        components = {}
        diagnostic_values = {}
        for batch, labels, _ in loaders["train"]:
            optimizer.zero_grad(set_to_none=True)
            prediction = model(move_inputs(batch, device))
            labels = labels.to(device)
            if criterion:
                loss, parts, diag = criterion(prediction, labels, epoch)
            elif config["module"]["variant"] in {"D1", "D2"}:
                loss = model.encoder.head.loss_fn(prediction, labels)
                parts = dict(
                    supervised=(prediction - labels).square().mean(), weighted_objective=loss
                )
                diag = {}
            else:
                loss = (prediction - labels).square().mean()
                parts = dict(supervised=loss)
                diag = {}
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite objective")
            loss.backward()
            parameters = [p for p in model.parameters() if p.grad is not None]
            if any(not torch.isfinite(p.grad).all() for p in parameters):
                raise ValueError("Nonfinite model gradient")
            clip = config["source"]["training"].get("gradient_clip_val")
            if clip:
                torch.nn.utils.clip_grad_norm_(parameters, clip, error_if_nonfinite=True)
            optimizer.step()
            step += 1
            if interval == "step":
                scheduler.step()
            n = len(labels)
            total += float(loss.detach()) * n
            seen += n
            for k, v in parts.items():
                components[k] = components.get(k, 0.0) + float(v.detach()) * n
            for k, v in diag.items():
                diagnostic_values[k] = diagnostic_values.get(k, 0.0) + v * n
        if interval == "epoch":
            scheduler.step()
        _, validation, _ = evaluate(model, loaders["val"], frames["val"], tasks, scalers, device)
        score = validation[selection_target]["mae"]
        improved = score < best - t["min_delta"]
        if improved:
            best = score
            best_epoch = epoch
            bad = 0
            best_weights = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        early_stopped = bad >= t["patience"]
        record = dict(
            epoch=epoch,
            global_step=step,
            epoch_seconds=time.perf_counter() - began,
            train_loss=total / seen,
            lr=optimizer.param_groups[0]["lr"],
            bad_epochs=bad,
        )
        record.update({f"train_{k}": v / seen for k, v in components.items()})
        record.update({k: v / seen for k, v in diagnostic_values.items()})
        for task, values in validation.items():
            for metric in ("mae", "rmse"):
                record[f"val_{task}_{metric}"] = values[metric]
        history.append(record)
        with (run / "training_log.csv").open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(record))
            writer.writeheader()
            writer.writerows(history)
        payload = save(epoch, early_stopped)
        if improved:
            atomic_checkpoint(checkpoints / "best.ckpt", payload)
        atomic_checkpoint(checkpoints / "last.ckpt", payload)
        if epoch % t["checkpoint_every_n_epochs"] == 0:
            atomic_checkpoint(checkpoints / f"epoch{epoch:04d}.ckpt", payload)
        print(json.dumps(record), flush=True)
        if early_stopped:
            break
    atomic_checkpoint(checkpoints / "terminal_fit.ckpt", save(epoch, early_stopped))
    fit_seconds = accumulated + time.perf_counter() - fit_started
    fit_peak_allocated = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
    fit_peak_reserved = torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None
    model.load_state_dict(best_weights)
    from molgap.tracking.inference import measure_prediction

    with measure_prediction(
        run,
        len(frames["test"]),
        "test_loader_forward_and_original_unit_metrics",
        model=model if device.type == "cuda" else None,
    ):
        prediction, values, attention = evaluate(
            model, loaders["test"], frames["test"], tasks, scalers, device
        )
    import pandas as pd

    saved = frames["test"][["sample_id", "smiles"]].copy()
    for j, task in enumerate(tasks):
        saved[task] = prediction[:, j]
        saved["true_" + task] = frames["test"][task].to_numpy()
    saved.to_csv(run / "test_predictions.csv", index=False, float_format="%.17g")
    (run / "predictions").mkdir(exist_ok=True)
    link = run / "predictions/test_predictions.csv"
    if not link.exists():
        link.symlink_to("../test_predictions.csv")
    reread = pd.read_csv(run / "test_predictions.csv", float_precision="round_trip")
    recalculated = metrics(
        reread[list(tasks)].to_numpy(), reread[["true_" + x for x in tasks]].to_numpy(), tasks
    )
    if recalculated != values:
        raise ValueError("On-disk prediction metrics differ")
    report = dict(
        targets={
            k: dict(v, unit=config["data"]["units"][k]) for k, v in values.items() if k in TASKS7
        },
        auxiliary_targets={k: v for k, v in values.items() if k not in TASKS7},
        selection_target=selection_target,
        best_validation_mae=best,
        protocol_id=config["experiment"]["protocol_id"],
        diagnostic=diagnostic,
        units={k: config["data"]["units"][k] for k in tasks},
        rows=len(saved),
        best_epoch=best_epoch,
    )
    if "delta_e" in values:
        report["delta_e_mae"] = values["delta_e"]["mae"]
    if criterion:
        report["physical_weight_at_best"] = criterion.weight(best_epoch)
        report["physics_active_at_best"] = criterion.weight(best_epoch) > 0
    write_json(run / "test_metrics.json", report)
    write_json(run / "metrics.json", report)
    write_json(
        run / "training_timing.json",
        dict(
            fit_seconds=fit_seconds,
            epochs_completed=epoch,
            best_epoch=best_epoch,
            best_validation_mae=best,
            parameters=sum(p.numel() for p in model.parameters()),
            peak_gpu_allocated_bytes=fit_peak_allocated,
            peak_gpu_reserved_bytes=fit_peak_reserved,
            diagnostic=diagnostic,
        ),
    )
    if attention is not None:
        write_json(
            run / "attention.json",
            dict(
                task_names=list(tasks),
                matrix_per_head=attention.tolist(),
                checkpoint="best_by_validation_gap_mae",
                aggregation="sample_mean_on_test; heads_retained",
            ),
        )
    if model.variant in {"D1", "D2"}:
        write_json(run / "uncertainty_params.json", model.encoder.head.get_uncertainty_params())
        affinity = model.encoder.head.get_affinity_matrix()
        if affinity:
            write_json(run / "affinity_matrix.json", affinity)
    atomic_checkpoint(
        run / "best_checkpoint.pt",
        dict(state_dict=model.state_dict(), config=config, scalers=scalers, epoch=best_epoch),
    )
    write_json(
        run / "COMPLETED.json",
        dict(
            diagnostic=diagnostic,
            epochs_completed=epoch,
            best_epoch=best_epoch,
            early_stopped=early_stopped,
        ),
    )
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--resume", type=Path)
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--diagnostic-epochs", type=int)
    p.add_argument("--limit-rows", type=int)
    p.add_argument("--device", choices=["cpu", "cuda"])
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    print(
        json.dumps(
            train(
                a.config,
                resume=a.resume,
                output_dir=a.output_dir,
                diagnostic_epochs=a.diagnostic_epochs,
                limit_rows=a.limit_rows,
                device_override=a.device,
                dry_run=a.dry_run,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
