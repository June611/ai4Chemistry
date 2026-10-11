"""Expand existing paired-seed recipes, without altering the statistical protocol."""

import copy
import csv
import hashlib
import sys
from pathlib import Path

import yaml

from molgap.experiments.multiseed import expand_matrix
from molgap.qm9_models.policy import paused_models
from molgap.tracking.records import utc_now, write_json


def create_plan(root: Path, directory: Path, policy: dict, selected=None, seeds=None, epochs=None):
    directory = directory.resolve()
    if (directory / "plan.json").exists():
        raise FileExistsError("Plan exists; use run to resume it or choose another --directory")
    directory.mkdir(parents=True, exist_ok=True)
    resources = {r["model"]: r for r in csv.DictReader((root / policy["resource_csv"]).open())}
    resources["3dinfomax_pretrain"] = {
        "gpu_peak_gib": "2.195",
        "ram_peak_pss_gib": "2.267",
        "fit_seconds": "101",
    }
    disabled = paused_models(root)
    active = set(resources)
    if selected:
        if set(selected) & set(disabled):
            raise ValueError(
                f"Paused models cannot be scheduled: {sorted(set(selected) & set(disabled))}"
            )
        if set(selected) - active:
            raise ValueError(f"Unknown or paused models: {sorted(set(selected) - active)}")
        active = set(selected)
        if "3dinfomax" in active:
            active.add("3dinfomax_pretrain")
    jobs = []
    seeds = seeds or policy["seeds"]
    if len(set(seeds)) != len(seeds) or set(seeds) - {3407, 42, 2026}:
        raise ValueError("Use distinct existing split seeds: 3407,42,2026")
    for seed in seeds:
        suffix = "" if seed == 3407 else f"_seed{seed}"
        sequence = {}
        for matrix_name in ("multiseed", "spe_affinity_ablation_multiseed"):
            matrix_path = root / "configs" / f"{matrix_name}{suffix}.yaml"
            matrix = yaml.safe_load(matrix_path.read_text())
            names = active & {m["name"] for m in matrix["models"]}
            if not names:
                continue
            matrix["name"] = f"{matrix_name}_seed{seed}"
            matrix["output"]["root"] = str(directory / "expansion")
            copied = directory / f"{matrix_name}{suffix}.yaml"
            copied.write_text(yaml.safe_dump(matrix, sort_keys=False))
            _, _, _, _, specs = expand_matrix(copied, selected_models=names)
            sequence.update({s.model: (s.trainer, s.config_path) for s in specs})
        for model in sorted(active):
            if model in disabled:
                continue
            if model in sequence:
                kind, source = sequence[model]
            elif model in {"training_mean", "random_forest_rdkit29", "xgboost_rdkit29"}:
                kind, source = "ml", root / "configs" / f"{model}{suffix}.yaml"
            else:
                kind = "pretrain" if model == "3dinfomax_pretrain" else "qm9"
                source = root / "configs/qm9_models" / f"{model}.yaml"
            cfg = copy.deepcopy(yaml.safe_load(source.read_text()))
            cfg.setdefault("experiment", {}).update(
                name=model, model_id=model, protocol_id="paired_split_gap_mae_v2"
            )
            settings = cfg["pretraining" if kind == "pretrain" else "training"]
            settings["seed"] = seed
            if kind in {"qm9", "pretrain"}:
                cfg["data"].update(
                    processed_dir=f"data/processed/qm9_full/random/seed{seed}",
                    geometry=f"data/geometry/qm9_full_seed{seed}.npz",
                    chemprop_reference_config="configs/baseline.yaml"
                    if seed == 3407
                    else f"configs/qm9_models/split_references/seed{seed}.yaml",
                )
                cfg["data"]["split"]["seed"] = seed
            if kind != "pretrain":
                settings.update(
                    split_seed=seed, seed_pairing="matched", run_mode="paired_split_seed"
                )
            if kind != "ml":
                if settings["batch_size"] != 256 or settings["epochs"] != 100:
                    raise ValueError(f"Unfrozen budget in {source}")
                patience = settings.get(
                    "patience",
                    settings.get(
                        "early_stopping_patience",
                        settings.get("early_stopping", {}).get("patience"),
                    ),
                )
                if patience != 20:
                    raise ValueError(f"Unfrozen early-stopping patience in {source}")
                settings["num_workers"] = policy["workers_per_job"]
                if epochs is not None:
                    settings["epochs"] = epochs
                    cfg["smoke_suite"] = {
                        "purpose": "scheduler_diagnostic",
                        "epochs": epochs,
                        "full_fixed_split": True,
                    }
                    if (
                        kind == "chemprop"
                        and cfg["model"]["backend"] == "chemprop_cli"
                        and epochs <= settings["warmup_epochs"]
                    ):
                        settings["warmup_epochs"] = 0
            outputs = directory / "runs"
            cfg["output"]["root"] = str(outputs / model) if kind == "pretrain" else str(outputs)
            run_dir = outputs / model / f"seed{seed}"
            if model == "3dinfomax":
                cfg["model"]["params"]["pretrain_checkpoint"] = str(
                    outputs / "3dinfomax_pretrain" / f"seed{seed}" / "best_checkpoint.pt"
                )
            path = directory / "configs" / f"{model}_seed{seed}.yaml"
            path.parent.mkdir(exist_ok=True)
            path.write_text(yaml.safe_dump(cfg, sort_keys=False))
            module = {
                "qm9": "qm9_models.train",
                "pretrain": "qm9_models.pretrain_infomax",
                "chemprop": "training.train",
                "transformer": "training.transformer",
                "spe_multitask": "training.spe_multitask",
                "ml": "training.ml",
            }[kind]
            commands = [[sys.executable, "-m", "molgap." + module, "--config", str(path)]]
            if kind == "chemprop":
                commands += [
                    [sys.executable, "-m", "molgap.training." + action, "--config", str(path)]
                    for action in ("predict", "evaluate")
                ]
            row = resources[model]
            gpu = float(row["gpu_peak_gib"]) * 1024
            # Exclusive jobs reserve 1GiB beyond the measured peak in admission.
            # A second inflation would forbid ALIGNN after driver-reserved memory.
            exclusive = model in policy["exclusive_models"]
            gpu = gpu if exclusive else max(gpu * 1.2, gpu + 512) if gpu else 0
            cpu = (
                int(cfg["model"].get("params", {}).get("n_jobs", 1))
                if kind == "ml"
                else policy["workers_per_job"] + policy["threads_per_job"]
            )
            jobs.append(
                {
                    "id": f"{model}_seed{seed}",
                    "model": model,
                    "kind": kind,
                    "seed": seed,
                    "config": str(path),
                    "config_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "run_dir": str(run_dir),
                    "commands": commands,
                    "gpu_mib": gpu,
                    "ram_mib": max(float(row["ram_peak_pss_gib"]) * 1.5 * 1024, 2048),
                    "cpu_slots": cpu,
                    "exclusive": exclusive,
                    "priority": float(row["fit_seconds"]),
                    "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                    "depends_on": [f"3dinfomax_pretrain_seed{seed}"]
                    if model == "3dinfomax"
                    else [],
                    "status": "pending",
                    "diagnostic": epochs is not None,
                }
            )
    payload = {
        "created_at": utc_now(),
        "root": str(root),
        "policy": policy,
        "jobs": jobs,
        "paused_models": sorted(disabled),
    }
    write_json(directory / "plan.json", payload)
    return payload
