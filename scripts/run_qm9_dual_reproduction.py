"""Prepare/run the two new QM9 models using the existing isolated task queue.

One exclusive task per GPU; a failed seed does not cancel later independent
seeds. Formal admission requires successful full-epoch batch256 smoke evidence.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import yaml

from molgap.config import find_project_root, run_directory
from molgap.scheduling.hardware import snapshot
from molgap.tracking.records import read_json, utc_now, write_json

MODELS = {0: "demol", 1: "transformer_m"}
SEEDS = (3407, 42, 2026)


def prepare(root, directory):
    if directory.exists():
        raise FileExistsError("Use a new directory; existing evidence will not be overwritten")
    directory.mkdir(parents=True)
    for phase in ("smoke", "formal"):
        for gpu, model in MODELS.items():
            lane = directory / f"{phase}_gpu{gpu}"
            (lane / "configs").mkdir(parents=True)
            jobs = []
            for seed in SEEDS[:1] if phase == "smoke" else SEEDS:
                source = root / f"configs/qm9_models/{model}.yaml"
                config = copy.deepcopy(yaml.safe_load(source.read_text()))
                tr = config["training"]
                if (tr["batch_size"], tr["epochs"], tr["early_stopping"]["patience"]) != (
                    256,
                    100,
                    20,
                ):
                    raise ValueError("Only the frozen 256/100/20 recipe may be launched")
                config["data"].update(
                    processed_dir=f"data/processed/qm9_full/random/seed{seed}",
                    geometry=f"data/geometry/qm9_full_seed{seed}.npz",
                    chemprop_reference_config="configs/baseline.yaml"
                    if seed == 3407
                    else f"configs/qm9_models/split_references/seed{seed}.yaml",
                )
                config["data"]["split"]["seed"] = seed
                tr.update(seed=seed, num_workers=2)
                config["output"]["root"] = str(lane / "runs")
                path = lane / "configs" / f"{model}_seed{seed}.yaml"
                path.write_text(yaml.safe_dump(config, sort_keys=False))
                effective = copy.deepcopy(config)
                command = [sys.executable, "-m", "molgap.qm9_models.train", "--config", str(path)]
                if phase == "smoke":
                    command.append("--smoke")
                    effective["experiment"]["name"] += "_smoke"
                jobs.append(
                    dict(
                        id=f"{model}_seed{seed}",
                        model=model,
                        seed=seed,
                        kind="qm9",
                        config=str(path),
                        config_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                        run_dir=str(run_directory(effective, root)),
                        commands=[command],
                        gpu_mib=23000,
                        ram_mib=8192,
                        cpu_slots=6,
                        exclusive=True,
                        priority=10000 - seed / 10000,
                        depends_on=[],
                        status="pending",
                        diagnostic=phase == "smoke",
                    )
                )
            policy = dict(
                gpu=gpu,
                cpu_reserve=1,
                ram_reserve_mib=4096,
                gpu_reserve_mib=512,
                exclusive_gpu_reserve_mib=512,
                disk_reserve_gib=5,
                max_gpu_jobs=1,
                max_cpu_jobs=0,
                launch_spacing_seconds=5,
                sample_seconds=5,
                gpu_admission_utilization=95,
                threads_per_job=4,
            )
            write_json(
                lane / "plan.json",
                dict(
                    root=str(root), policy=policy, jobs=jobs, created_at=utc_now(), paused_models=[]
                ),
            )
    write_json(
        directory / "manifest.json",
        dict(
            models=MODELS,
            seeds=SEEDS,
            external_pretraining=False,
            source_root=str(root),
            created_at=utc_now(),
        ),
    )


def smoke_lane(directory, gpu):
    manifest = read_json(directory / "manifest.json")
    overrides = manifest.get("validated_smoke_sources", {})
    return Path(overrides.get(str(gpu), directory / f"smoke_gpu{gpu}"))


def validate_smoke(directory):
    for gpu in MODELS:
        lane = smoke_lane(directory, gpu)
        state = read_json(lane / "runtime/state.json")
        jobs = state.get("jobs", [])
        if (
            len(jobs) != 1
            or jobs[0]["status"] != "completed"
            or jobs[0]["model"] != MODELS[gpu]
            or jobs[0]["seed"] != 3407
        ):
            raise RuntimeError(f"GPU {gpu}: successful smoke evidence is required")
        run = Path(jobs[0]["run_dir"])
        plan = read_json(run / "model_summary.json")
        import pandas as pd

        history = pd.read_csv(run / "history.csv")
        if (
            plan["budget"]["batch_size"] != 256
            or len(history) != 1
            or history.iloc[0]["train_batches"] != 419
            or plan["split_rows"] != {"train": 107038, "val": 13380, "test": 13380}
        ):
            raise RuntimeError("Smoke must cover the original full epoch at batch256")


def balance_formal(directory):
    """Assign independent seeds by measured epoch cost, longest tasks first."""
    import pandas as pd

    manifest = read_json(directory / "manifest.json")
    if manifest.get("formal_balanced"):
        return
    validate_smoke(directory)
    seconds = {}
    for gpu, model in MODELS.items():
        state = read_json(smoke_lane(directory, gpu) / "runtime/state.json")
        run = Path(state["jobs"][0]["run_dir"])
        seconds[model] = float(pd.read_csv(run / "history.csv").iloc[0]["epoch_seconds"])
    plans, pending = {}, []
    for gpu in MODELS:
        lane = directory / f"formal_gpu{gpu}"
        if (lane / "runtime").exists():
            raise RuntimeError("Cannot rebalance a formal queue that has already started")
        plan = read_json(lane / "plan.json")
        plans[gpu] = plan
        pending.extend(plan["jobs"])
        plan["jobs"] = []
    loads = {gpu: 0.0 for gpu in MODELS}
    assignment = []
    for index, job in enumerate(sorted(pending, key=lambda j: -seconds[j["model"]])):
        gpu = min(loads, key=loads.get)
        cost = seconds[job["model"]] * 100
        loads[gpu] += cost
        config = yaml.safe_load(Path(job["config"]).read_text())
        lane = directory / f"formal_gpu{gpu}"
        config["output"]["root"] = str(lane / "runs")
        path = lane / "configs" / f"{job['id']}.yaml"
        path.write_text(yaml.safe_dump(config, sort_keys=False))
        job.update(
            config=str(path),
            config_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            run_dir=str(run_directory(config, Path(plans[gpu]["root"]))),
            commands=[[sys.executable, "-m", "molgap.qm9_models.train", "--config", str(path)]],
            priority=cost - index * 1e-6,
            estimated_fit_seconds=cost,
        )
        plans[gpu]["jobs"].append(job)
        assignment.append(dict(job=job["id"], gpu=gpu, estimated_fit_seconds=cost))
    for gpu, plan in plans.items():
        write_json(directory / f"formal_gpu{gpu}/plan.json", plan)
    manifest.update(
        formal_balanced=True,
        measured_epoch_seconds=seconds,
        estimated_lane_seconds=loads,
        assignment=assignment,
        estimate_scope="one full smoke epoch x100; not a convergence guarantee",
    )
    write_json(directory / "manifest.json", manifest)


def run_lane(directory, phase, gpu):
    from molgap.scheduling import queue

    plan = read_json(directory / f"{phase}_gpu{gpu}" / "plan.json")
    root = Path(plan["root"])
    original = queue.sample_resources

    def per_card(pid):
        result = original(pid)
        result["gpu_device_mib"] = snapshot(root, gpu)["gpu_used_mib"]
        return result

    queue.sample_resources = per_card
    return queue.run_queue(directory / f"{phase}_gpu{gpu}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", required=True, type=Path)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--phase", choices=("smoke", "formal"))
    parser.add_argument("--gpu", type=int, choices=(0, 1))
    args = parser.parse_args()
    directory = args.directory.resolve()
    root = find_project_root()
    if args.prepare:
        prepare(root, directory)
    if args.phase == "formal":
        validate_smoke(directory)
        if args.gpu is None:
            balance_formal(directory)
    if args.phase and args.gpu is not None:
        raise SystemExit(run_lane(directory, args.phase, args.gpu))
    elif args.phase:
        children = [
            subprocess.Popen(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--directory",
                    str(directory),
                    "--phase",
                    args.phase,
                    "--gpu",
                    str(gpu),
                ]
            )
            for gpu in MODELS
        ]
        codes = [p.wait() for p in children]
        print(json.dumps(dict(phase=args.phase, lane_exit_codes=codes)), flush=True)
        raise SystemExit(0 if all(c == 0 for c in codes) else 1)


if __name__ == "__main__":
    main()
