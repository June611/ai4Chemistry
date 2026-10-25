"""Disk-backed six-property experiments using the existing independent GPU queues."""

from __future__ import annotations

import argparse
import copy
import json
import signal
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from molgap.config import find_project_root, load_config, run_directory
from molgap.data.neural_properties import (
    NEURAL_PROPERTY_PROTOCOL,
    validate_property_data,
    validate_property_target,
)
from molgap.data.qm9_properties import NEW_TARGETS, SEEDS, digest, prepare_property_data
from molgap.scheduling.queue import check_config, verify
from molgap.tracking.records import read_json, utc_now, write_json
from molgap.tracking.resources import process_identity

MODELS = (
    "chemprop_single_gap",
    "smiles_transformer",
    "spe_transformer",
    "schnet",
    "3dinfomax",
    "egnn",
    "painn",
    "dimenet_pp",
    "esa",
    "torchmd_net",
)
TRANSFORMERS = {"smiles_transformer", "spe_transformer"}
METRICS = ("mae", "mse", "rmse", "r2")
LABELS = {
    "chemprop_single_gap": "Chemprop (2D)",
    "schnet": "SchNet (3D)",
    "3dinfomax": r"3D Infomax$^{\dagger}$",
    "egnn": "EGNN (3D)",
    "painn": "PaiNN (3D)",
    "dimenet_pp": "DimeNet++ (3D)",
    "smiles_transformer": "SMILES Transformer (2D)",
    "spe_transformer": "SPE Transformer (2D)",
    "esa": "ESA (2D graph)",
    "torchmd_net": "TorchMD-Net (3D)",
}


def _path(root: Path, value) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _yaml(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(value, sort_keys=False))


def validate_config(path: Path, model: str) -> dict:
    if model == "chemprop_single_gap":
        return load_config(path)[0]
    if model in TRANSFORMERS:
        from molgap.transformer.config import load_transformer_config

        return load_transformer_config(path)[0]
    cfg = yaml.safe_load(path.read_text())
    validate_property_target(cfg)
    if cfg["model"]["name"] != model:
        raise ValueError("Task model name differs from the selected backbone")
    if cfg["training"]["target_scaling"] != "standard" or cfg["training"]["loss"] != "mse":
        raise ValueError("Property models require train-only standard scaling and MSE")
    return cfg


def _checkpoint(root, value, seed, cfg, directory):
    """Validate supplied Infomax weights before reusing them for six independent heads."""
    if value == "auto":
        bases = (
            root / "artifacts/auto_training",
            root / "outputs/paired_gap_mae_v2/3dinfomax_pretrain",
        )
        candidates = sorted(
            path
            for base in bases
            if base.exists()
            for path in base.rglob("best_checkpoint.pt")
            if f"seed{seed}" in str(path) and ".attempt_" not in str(path)
        )
        for path in candidates:
            try:
                return _checkpoint(root, str(path), seed, cfg, directory)
            except (ValueError, KeyError, FileNotFoundError):
                continue
        return _checkpoint(root, None, seed, cfg, directory)
    if not value:
        return directory / "runs/3dinfomax_pretrain" / f"seed{seed}/best_checkpoint.pt"
    from molgap.qm9_models.train import _validate_infomax_checkpoint

    path = _path(root, value)
    cfg = copy.deepcopy(cfg)
    cfg["model"]["params"]["pretrain_checkpoint"] = str(path)
    geom = _path(root, cfg["data"]["geometry"])
    metadata = read_json(geom.with_suffix(".json"))
    if metadata.get("geometry_sha256") != digest(geom):
        raise ValueError("Pretraining reuse: geometry hash does not match metadata")
    identity = {
        "manifest": {
            "sha256": digest(_path(root, cfg["data"]["processed_dir"]) / "split_manifest.csv")
        }
    }
    _validate_infomax_checkpoint(cfg, root, identity, metadata)
    return path


def prepare(
    root: Path,
    tasks_path: Path,
    directory: Path,
    *,
    models=None,
    targets=None,
    seeds=None,
    gpus=(0,),
) -> dict:
    tasks = yaml.safe_load(tasks_path.read_text())
    if tasks.get("protocol_id") != NEURAL_PROPERTY_PROTOCOL:
        raise ValueError("Unexpected property task protocol")
    if tasks["defaults"] != dict(batch_size=256, epochs=100, patience=20, min_delta=0.0):
        raise ValueError("Formal task budget is frozen at 256/100/20 and min_delta=0")
    models = list(tasks["models"] if models is None else models)
    targets = list(tasks["targets"] if targets is None else targets)
    seeds = list(tasks["seeds"] if seeds is None else seeds)
    for selected, allowed in ((models, MODELS), (targets, NEW_TARGETS), (seeds, SEEDS)):
        if not selected or len(set(selected)) != len(selected) or not set(selected) <= set(allowed):
            raise ValueError("Empty, duplicate or unsupported experiment selection")
    if not gpus or len(set(gpus)) != len(gpus) or any(g < 0 for g in gpus):
        raise ValueError("GPU indices must be unique non-negative integers")
    directory = directory.resolve()
    if directory.exists():
        raise FileExistsError("Plan is immutable; use a fresh batch directory")
    directory.mkdir(parents=True)
    _yaml(directory / "tasks.yaml", tasks)
    bases = {}
    template_sources = {}
    for model in models:
        source = _path(root, tasks["models"][model]["config"])
        content = source.read_bytes()
        snapshot_path = directory / "templates" / f"{model}.yaml"
        snapshot_path.parent.mkdir(exist_ok=True)
        snapshot_path.write_bytes(content)
        bases[model] = yaml.safe_load(content)
        template_sources[model] = dict(
            path=str(source), snapshot=str(snapshot_path), sha256=digest(snapshot_path)
        )
    refs = {}
    pretrain_paths = {}
    groups = []
    for seed in seeds:
        for model in models:
            spec = tasks["models"][model]
            cfg = copy.deepcopy(bases[model])
            data, training = cfg["data"], cfg["training"]
            data.update(
                processed_dir=f"{tasks['data_root']}/random/seed{seed}",
                property_data_root=tasks["data_root"],
                target_column=targets[0],
                targets=[targets[0]],
            )
            data["split"]["seed"] = seed
            training.update(seed=seed, split_seed=seed, seed_pairing="matched")
            # Keep geometry units, array ordering, and all architecture parameters unchanged.
            if "geometry" in data:
                data["geometry"] = f"data/geometry/qm9_full_seed{seed}.npz"
            if seed not in refs:
                ref = directory / "references" / f"seed{seed}.yaml"
                ref_data = copy.deepcopy(data)
                ref_data["targets"] = ["delta_e", *NEW_TARGETS]
                _yaml(ref, {"data": ref_data})
                refs[seed] = ref
            data["chemprop_reference_config"] = str(refs[seed])
            jobs = []
            cfg["experiment"].update(protocol_id=NEURAL_PROPERTY_PROTOCOL, model_id=model)
            for target in targets:
                current = copy.deepcopy(cfg)
                current["experiment"]["name"] = target
                from molgap.data.qm9_properties import PROPERTY_UNITS

                current["data"].update(
                    target_column=target,
                    targets=[target],
                    label_unit=PROPERTY_UNITS[target],
                )
                if model == "chemprop_single_gap":
                    current["loss"]["weights"] = {target: 1.0}
                current["training"].update(
                    batch_size=tasks["defaults"]["batch_size"],
                    epochs=tasks["defaults"]["epochs"],
                    target_scaling="standard",
                    num_workers=tasks["resources"]["workers_per_job"],
                    formal_protocol=True,
                )
                t = current["training"]
                if model == "chemprop_single_gap":
                    t.update(
                        patience=tasks["defaults"]["patience"],
                        selection_metric="property_mae",
                        early_stopping_min_delta=tasks["defaults"]["min_delta"],
                    )
                elif model in TRANSFORMERS:
                    t.update(
                        early_stopping_patience=tasks["defaults"]["patience"],
                        early_stopping_min_delta=tasks["defaults"]["min_delta"],
                    )
                else:
                    t.update(
                        loss="mse",
                        early_stopping=dict(
                            monitor="val_mae",
                            patience=tasks["defaults"]["patience"],
                            min_delta=tasks["defaults"]["min_delta"],
                        ),
                    )
                validate_property_data(current, root)
                current["output"]["root"] = str(directory / "runs" / model)
                path = directory / "configs" / f"{model}_{target}_seed{seed}.yaml"
                deps = []
                if model == "3dinfomax":
                    value = tasks["pretraining"]["checkpoints"].get(seed)
                    if seed not in pretrain_paths:
                        pretrain_paths[seed] = _checkpoint(root, value, seed, current, directory)
                    checkpoint = pretrain_paths[seed]
                    current["model"]["params"]["pretrain_checkpoint"] = str(checkpoint)
                    if checkpoint.is_file():
                        current["model"]["params"]["pretrain_checkpoint_sha256"] = digest(
                            checkpoint
                        )
                    else:
                        deps = [f"3dinfomax_pretrain_seed{seed}"]
                _yaml(path, current)
                validate_config(path, model)
                kind = (
                    "chemprop"
                    if model == "chemprop_single_gap"
                    else ("transformer" if model in TRANSFORMERS else "qm9")
                )
                module = {
                    "chemprop": "training.train",
                    "transformer": "training.transformer",
                    "qm9": "qm9_models.train",
                }[kind]
                commands = [[sys.executable, "-m", "molgap." + module, "--config", str(path)]]
                if kind == "chemprop":
                    commands.extend(
                        [
                            [
                                sys.executable,
                                "-m",
                                "molgap.training." + action,
                                "--config",
                                str(path),
                            ]
                            for action in ("predict", "evaluate")
                        ]
                    )
                jobs.append(
                    dict(
                        id=f"{model}_{target}_seed{seed}",
                        model=model,
                        seed=seed,
                        target=target,
                        unit=PROPERTY_UNITS[target],
                        kind=kind,
                        config=str(path),
                        config_sha256=digest(path),
                        run_dir=str(run_directory(current, root)),
                        commands=commands,
                        gpu_mib=spec["gpu_mib"],
                        ram_mib=spec["ram_mib"],
                        cpu_slots=tasks["resources"]["threads_per_job"] + t["num_workers"],
                        exclusive=True,
                        priority=spec["estimated_seconds"],
                        depends_on=deps,
                        status="pending",
                        diagnostic=False,
                        local_property_report=True,
                        inference_uses_3d=spec["inference_uses_3d"],
                        pretraining_uses_3d=spec["pretraining_uses_3d"],
                    )
                )
            # Keep the six Infomax tasks with their one seed-specific pretraining dependency.
            if model == "3dinfomax":
                if jobs[0]["depends_on"]:
                    pre = yaml.safe_load(_path(root, tasks["pretraining"]["config"]).read_text())
                    pre["data"].update(
                        processed_dir=f"data/processed/qm9_full/random/seed{seed}",
                        geometry=f"data/geometry/qm9_full_seed{seed}.npz",
                        chemprop_reference_config=(
                            "configs/baseline.yaml"
                            if seed == 3407
                            else f"configs/qm9_models/split_references/seed{seed}.yaml"
                        ),
                    )
                    pre["data"]["split"]["seed"] = seed
                    pre["pretraining"].update(
                        seed=seed,
                        batch_size=256,
                        epochs=100,
                        patience=20,
                        num_workers=tasks["resources"]["workers_per_job"],
                    )
                    pre["output"]["root"] = str(directory / "runs/3dinfomax_pretrain")
                    pp = directory / "configs" / f"3dinfomax_pretrain_seed{seed}.yaml"
                    _yaml(pp, pre)
                    jobs.insert(
                        0,
                        dict(
                            id=f"3dinfomax_pretrain_seed{seed}",
                            model="3dinfomax_pretrain",
                            seed=seed,
                            kind="pretrain",
                            config=str(pp),
                            config_sha256=digest(pp),
                            run_dir=str(directory / "runs/3dinfomax_pretrain" / f"seed{seed}"),
                            commands=[
                                [
                                    sys.executable,
                                    "-m",
                                    "molgap.qm9_models.pretrain_infomax",
                                    "--config",
                                    str(pp),
                                ]
                            ],
                            gpu_mib=22528,
                            ram_mib=24576,
                            cpu_slots=(
                                tasks["resources"]["threads_per_job"]
                                + tasks["resources"]["workers_per_job"]
                            ),
                            exclusive=True,
                            priority=10**9,
                            depends_on=[],
                            status="pending",
                            diagnostic=False,
                            local_property_report=True,
                        ),
                    )
                groups.append(
                    (sum(j.get("priority", 0) for j in jobs if j["kind"] != "pretrain"), jobs)
                )
            else:
                groups.extend((spec["estimated_seconds"], [job]) for job in jobs)
    lanes = [[] for _ in gpus]
    loads = [0 for _ in gpus]
    for cost, jobs in sorted(groups, key=lambda item: -item[0]):
        lane = loads.index(min(loads))
        lanes[lane].extend(jobs)
        loads[lane] += cost
    lane_paths = []
    for gpu, jobs in zip(gpus, lanes, strict=True):
        lane = directory / "lanes" / f"gpu{gpu}"
        lane.mkdir(parents=True)
        policy = dict(
            gpu=gpu,
            cpu_reserve=tasks["resources"]["cpu_reserve"],
            ram_reserve_mib=tasks["resources"]["ram_reserve_mib"],
            disk_reserve_gib=tasks["resources"]["disk_reserve_gib"],
            gpu_reserve_mib=512,
            exclusive_gpu_reserve_mib=512,
            max_gpu_jobs=1,
            max_cpu_jobs=0,
            launch_spacing_seconds=5,
            sample_seconds=tasks["resources"]["sample_seconds"],
            gpu_admission_utilization=95,
            threads_per_job=tasks["resources"]["threads_per_job"],
        )
        write_json(lane / "plan.json", dict(root=str(root), jobs=jobs, policy=policy))
        lane_paths.append(str(lane))
    plan = dict(
        protocol_id=NEURAL_PROPERTY_PROTOCOL,
        root=str(root),
        created_at=utc_now(),
        tasks_sha256=digest(directory / "tasks.yaml"),
        template_sources=template_sources,
        models=models,
        targets=targets,
        seeds=seeds,
        downstream_jobs=len(models) * len(targets) * len(seeds),
        pretraining_jobs=sum(j["kind"] == "pretrain" for lane in lanes for j in lane),
        lanes=lane_paths,
        gpus=list(gpus),
        historical_seconds_per_lane=loads,
        estimate_warning="Historical gap timing only; property losses and machines differ",
    )
    write_json(directory / "plan.json", plan)
    return plan


def lane_states(directory: Path):
    plan = read_json(directory / "plan.json")
    for lane in plan["lanes"]:
        lane = Path(lane)
        state = read_json(lane / "runtime/state.json") or read_json(lane / "plan.json")
        yield lane, state


def _assert_stopped(directory: Path):
    for lane, state in lane_states(directory):
        controller = read_json(lane / "runtime/controller.json")
        entries = [controller] + [j for j in state["jobs"] if j["status"] == "running"]
        for item in entries:
            identity = item.get("identity", item.get("process_identity"))
            if item.get("pid") and identity and process_identity(item["pid"]) == identity:
                raise RuntimeError("Controller/worker is alive; do not mutate a live queue")


def run(directory: Path) -> int:
    import fcntl

    plan = read_json(directory / "plan.json")
    root = Path(plan["root"])
    _assert_stopped(directory)
    # Account for simultaneous independent lanes, not just one lane's reservations.
    from molgap.scheduling.hardware import snapshot

    maxima = []
    for _lane, state in lane_states(directory):
        pending = [j for j in state["jobs"] if j["status"] == "pending"]
        if pending:
            maxima.append(
                (max(j["ram_mib"] for j in pending), max(j["cpu_slots"] for j in pending))
            )
        for job in state["jobs"]:
            check_config(job)
    first = read_json(Path(plan["lanes"][0]) / "plan.json")["policy"]
    if not maxima:
        return int(
            any(
                j["status"] != "completed"
                for _, state in lane_states(directory)
                for j in state["jobs"]
            )
        )
    for gpu in plan["gpus"]:
        observed = snapshot(root, gpu)
        if observed["gpu_free_mib"] is None:
            raise RuntimeError(f"GPU {gpu} is unavailable; no task was launched")
    hw = snapshot(root, plan["gpus"][0])
    if sum(x[0] for x in maxima) + first["ram_reserve_mib"] > hw["ram_available_mib"]:
        raise RuntimeError("Insufficient host RAM for these concurrent lanes; prepare fewer GPUs")
    if sum(x[1] for x in maxima) + first["cpu_reserve"] > hw["cpus"]:
        raise RuntimeError("Insufficient CPU quota for these lanes; reduce workers or GPU lanes")
    children = []

    def stop(signum, frame):
        for child in children:
            if child.poll() is None:
                child.terminate()

    with (directory / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        previous = {s: signal.signal(s, stop) for s in (signal.SIGINT, signal.SIGTERM)}
        try:
            for lane in plan["lanes"]:
                children.append(
                    subprocess.Popen(
                        [
                            sys.executable,
                            "-m",
                            "molgap.scheduling",
                            "run",
                            "--directory",
                            lane,
                        ],
                        cwd=root,
                    )
                )
            codes = [child.wait() for child in children]
        finally:
            stop(None, None)
            for child in children:
                child.wait()
            for sig, handler in previous.items():
                signal.signal(sig, handler)
    return int(any(codes))


def retry(directory: Path, identifiers: list[str]) -> list[str]:
    import fcntl

    with (directory / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _retry_stopped(directory, identifiers)


def _retry_stopped(directory: Path, identifiers: list[str]) -> list[str]:
    _assert_stopped(directory)
    selected = [
        j for _, state in lane_states(directory) for j in state["jobs"] if j["id"] in identifiers
    ]
    if len(set(identifiers)) != len(identifiers) or {j["id"] for j in selected} != set(identifiers):
        raise ValueError("Unknown or duplicate retry IDs")
    for job in selected:
        if job["status"] not in {"failed", "blocked", "interrupted", "running"}:
            raise ValueError("Only unsuccessful/interrupted jobs may be retried")
        check_config(job)
    changed = []
    for lane, state in lane_states(directory):
        for job in state["jobs"]:
            if job["id"] not in identifiers:
                continue
            if job["status"] not in {"failed", "blocked", "interrupted", "running"}:
                raise ValueError("Only unsuccessful/interrupted jobs may be retried")
            check_config(job)
            output = Path(job["run_dir"])
            if output.exists():
                archived = output.with_name(output.name + ".attempt_" + utc_now().replace(":", ""))
                output.rename(archived)
            for key in (
                "pid",
                "process_identity",
                "returncode",
                "error",
                "verification",
                "peaks",
                "latest",
                "started_at",
                "ended_at",
                "wait_reason",
                "gpu_util_sum",
                "gpu_util_samples",
                "gpu_util_mean_pct",
            ):
                job.pop(key, None)
            job["status"] = "pending"
            changed.append(job["id"])
        write_json(lane / "runtime/state.json", state)
    if set(changed) != set(identifiers):
        raise ValueError("Some retry IDs are unknown")
    return changed


def summarize(directory: Path, *, strict=False) -> Path:
    plan = read_json(directory / "plan.json")
    root = Path(plan["root"])
    output = root / "outputs/summaries/qm9_results/batches" / directory.name
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for _lane, state in lane_states(directory):
        for job in state["jobs"]:
            if job["kind"] == "pretrain":
                continue
            row = {
                k: job.get(k)
                for k in (
                    "id",
                    "model",
                    "target",
                    "unit",
                    "seed",
                    "status",
                    "inference_uses_3d",
                    "pretraining_uses_3d",
                    "run_dir",
                )
            }
            if job["status"] == "completed":
                verify({**job, "root": str(root)})
                payload = read_json(Path(job["run_dir"]) / "metrics.json")
                values = payload.get("targets", {}).get(job["target"], payload)
                row.update({m: values[m] for m in METRICS})
                timing = read_json(Path(job["run_dir"]) / "training_timing.json")
                row.update(
                    fit_seconds=timing.get("fit_seconds", timing.get("fit_pipeline_seconds")),
                    epochs_completed=timing.get("epochs_completed"),
                    **job.get("peaks", {}),
                )
            rows.append(row)
    runs = pd.DataFrame(rows)
    runs.to_csv(output / "runs.csv", index=False)
    completed = runs[runs.status.eq("completed")]
    summaries = []
    if len(completed):
        for (model, target, unit), frame in completed.groupby(["model", "target", "unit"]):
            if frame.seed.duplicated().any():
                raise ValueError("Duplicate successful seeds in the same property")
            row = dict(
                model=model,
                target=target,
                unit=unit,
                n_seeds=len(frame),
                seeds=",".join(str(x) for x in sorted(frame.seed)),
                complete_three_seeds=set(frame.seed) == set(SEEDS),
                target_scaling="standard",
                protocol_id=NEURAL_PROPERTY_PROTOCOL,
            )
            for metric in (*METRICS, "fit_seconds", "epochs_completed"):
                if metric in frame:
                    row[metric + "_mean"] = frame[metric].mean()
                    row[metric + "_std"] = frame[metric].std(ddof=1)
            for metric in ("rss_mib", "pss_mib", "gpu_process_mib", "gpu_used_mib"):
                if metric in frame:
                    row[metric + "_max"] = frame[metric].max()
            summaries.append(row)
    summary = pd.DataFrame(summaries)
    summary.to_csv(output / "summary.csv", index=False)
    if len(summary):
        for target, frame in summary.groupby("target"):
            frame.sort_values("rmse_mean", ascending=False).to_csv(
                output / f"{target}_comparison.csv",
                index=False,
            )
    full = len(completed) == plan["downstream_jobs"]
    write_json(
        output / "verification.json",
        dict(
            expected=plan["downstream_jobs"],
            completed=len(completed),
            original_unit_predictions_recomputed=True,
            complete=full,
            protocol_id=NEURAL_PROPERTY_PROTOCOL,
            generated_at=utc_now(),
        ),
    )
    (output / "README.md").write_text(
        f"# QM9 六性质神经网络单任务结果\n\n完成 {len(completed)}/{plan['downstream_jobs']}。\n\n"
        "runs.csv 为逐 seed 指标和采样资源峰值；summary.csv 为按性质分组的均值及样本标准差。\n"
        "各性质使用原单位；r2 目标是电子空间范围，r2 指标是回归决定系数。\n"
        "*_comparison.csv 在同一性质内按 RMSE 降序排列；不同单位不能直接混排。\n"
        "历史 gap 使用不同机器和部分不同 loss/scaler，不能据此作严格速度/训练协议比较。\n"
    )
    if strict and not full:
        raise ValueError(f"Incomplete batch: {len(completed)}/{plan['downstream_jobs']}")
    return output


def fill_latex(summary_path: Path, source: Path, destination: Path):
    summary = pd.read_csv(summary_path)
    expected = {(m, t) for m in MODELS for t in NEW_TARGETS}
    if len(summary) != 60 or set(zip(summary.model, summary.target, strict=True)) != expected:
        raise ValueError("LaTeX export requires all 60 unique model/property summaries")
    if not summary.complete_three_seeds.all() or not summary.n_seeds.eq(3).all():
        raise ValueError("LaTeX export requires all three paired seeds")
    stats = {(r.model, r.target): r for r in summary.itertuples()}
    columns = ("homo", "lumo", "delta_e", "alpha", "mu", "r2", "zpve")
    reverse = {v: k for k, v in LABELS.items()}
    lines, filled = [], set()
    for line in source.read_text().splitlines():
        if " & " in line:
            fields = line.removesuffix(r" \\").split(" & ")
            if fields[0] in reverse:
                if len(fields) != 9:
                    raise ValueError("Unexpected LaTeX column count")
                model = reverse[fields[0]]
                for target in NEW_TARGETS:
                    row = stats[model, target]
                    if not np.isfinite([row.rmse_mean, row.rmse_std]).all():
                        raise ValueError("Nonfinite LaTeX metrics")
                    from molgap.data.qm9_properties import PROPERTY_UNITS

                    if row.unit != PROPERTY_UNITS[target]:
                        raise ValueError("LaTeX target unit mismatch")
                    fields[columns.index(target) + 1] = (
                        rf"\shortstack{{${row.rmse_mean:.6f}$\\$\pm {row.rmse_std:.6f}$}}"
                    )
                    filled.add((model, target))
                line = " & ".join(fields) + r" \\"
        lines.append(line)
    if filled != expected:
        raise ValueError("Some model rows were not found in the LaTeX source")
    text = "\n".join(lines) + "\n"
    text = text.replace(
        "RDKit29 baselines with unscaled labels.",
        "RDKit29 baselines with unscaled labels. Neural baselines use training-set "
        "Z-score and MSE; their predictions are restored to physical units before evaluation.",
    )
    if destination.exists():
        raise FileExistsError("Choose a fresh LaTeX output; existing tables are never overwritten")
    destination.write_text(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=("data", "prepare", "run", "status", "retry", "summarize", "verify", "latex"),
    )
    parser.add_argument("--tasks", type=Path, default=Path("configs/qm9_six_properties/tasks.yaml"))
    parser.add_argument(
        "--directory", type=Path, default=Path("artifacts/qm9_six_properties/run01")
    )
    parser.add_argument("--models", nargs="+", choices=MODELS)
    parser.add_argument("--targets", nargs="+", choices=NEW_TARGETS)
    parser.add_argument("--seeds", type=int, nargs="+", choices=SEEDS)
    parser.add_argument("--gpus", type=int, nargs="+", default=[0])
    parser.add_argument("--jobs", nargs="+")
    parser.add_argument("--source-tex", type=Path)
    parser.add_argument("--output-tex", type=Path)
    args = parser.parse_args()
    root = find_project_root()
    directory = args.directory.resolve()
    if args.action == "data":
        tasks = yaml.safe_load(args.tasks.read_text())
        print(json.dumps(prepare_property_data(root, _path(root, tasks["data_root"])), indent=2))
    elif args.action == "prepare":
        print(
            json.dumps(
                prepare(
                    root,
                    args.tasks,
                    directory,
                    models=args.models,
                    targets=args.targets,
                    seeds=args.seeds,
                    gpus=args.gpus,
                ),
                indent=2,
            )
        )
    elif args.action == "run":
        raise SystemExit(run(directory))
    elif args.action == "status":
        for lane, state in lane_states(directory):
            for job in state["jobs"]:
                print(
                    lane.name,
                    job["id"],
                    job["status"],
                    job.get("error") or job.get("wait_reason", ""),
                )
    elif args.action == "retry":
        if not args.jobs:
            parser.error("retry requires explicit --jobs IDs")
        print(json.dumps(retry(directory, args.jobs)))
    elif args.action == "latex":
        if not args.source_tex or not args.output_tex:
            parser.error("latex requires --source-tex and a fresh --output-tex")
        report = summarize(directory, strict=True)
        fill_latex(report / "summary.csv", args.source_tex, args.output_tex)
    else:
        print(summarize(directory, strict=args.action == "verify"))


if __name__ == "__main__":
    main()
