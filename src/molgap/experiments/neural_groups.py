"""Editable high/medium/low queues and verified report exchange between machines."""

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from molgap.config import find_project_root
from molgap.data.neural_properties import NEURAL_PROPERTY_PROTOCOL
from molgap.data.qm9_properties import NEW_TARGETS, PROPERTY_UNITS, SEEDS, digest
from molgap.experiments import neural_properties as queues
from molgap.tracking.records import read_json, utc_now, write_json

GROUPS = ("high", "medium", "low")
PROFILE_ROOT = Path("configs/qm9_six_properties/groups")


def audit_profiles(root: Path) -> dict:
    """Audit manifests without loading data, models, or starting any job."""
    seen, result = set(), {}
    for group in GROUPS:
        profile = root / PROFILE_ROOT / f"{group}.yaml"
        cfg = yaml.safe_load(profile.read_text())
        models = set(cfg["models"])
        if cfg.get("queue_group") != group or cfg.get("protocol_id") != NEURAL_PROPERTY_PROTOCOL:
            raise ValueError(f"Group identity/protocol mismatch: {group}")
        if not models or not models <= set(queues.MODELS) or seen & models:
            raise ValueError(f"Empty, unknown or overlapping models in {group}")
        if set(cfg["targets"]) != set(NEW_TARGETS) or len(cfg["targets"]) != 6:
            raise ValueError("Every source group must contain the same six properties")
        if set(cfg["seeds"]) != set(SEEDS) or len(cfg["seeds"]) != 3:
            raise ValueError("Every source group must contain three paired seeds")
        if cfg["defaults"] != dict(batch_size=256, epochs=100, patience=20, min_delta=0.0):
            raise ValueError("Group budgets must remain 256/100/20, min_delta=0")
        evidence = cfg["scheduling_reference"]
        source = root / evidence["source_csv"]
        if digest(source) != evidence["sha256"]:
            raise ValueError("Smoke reference hash changed")
        with source.open(encoding="utf-8-sig") as stream:
            rows = {r["model"]: r for r in csv.DictReader(stream)}
        for model, spec in cfg["models"].items():
            row = rows[model]
            if row["status"] != "success" or float(row["epochs"]) != 1:
                raise ValueError(f"No successful one-epoch reference for {model}")
            if float(row["batch_size"]) != 256:
                raise ValueError("Smoke batch size differs")
            if not math.isclose(float(row["fit_seconds"]), spec["smoke_fit_seconds"]):
                raise ValueError(f"Smoke timing does not match reference: {model}")
            if row["fit_timing_scope"] != spec["smoke_fit_scope"]:
                raise ValueError(f"Smoke timing scope mismatch: {model}")
            if not math.isfinite(spec["estimated_seconds"]) or spec["estimated_seconds"] <= 0:
                raise ValueError("Scheduling weights must be positive and finite")
        seen.update(models)
        result[group] = dict(
            models=list(cfg["models"]),
            downstream_jobs=len(models) * 18,
            source_profile=str(profile.relative_to(root)),
            source_sha256=digest(profile),
        )
    if seen != set(queues.MODELS):
        raise ValueError("The three groups must cover all ten models exactly once")
    return dict(groups=result, downstream_jobs=180, overlapping_models=False)


def check_batch(root: Path, directory: Path, group: str) -> dict:
    """Use the frozen group identity on resume, never a newly edited source YAML."""
    plan = read_json(directory / "plan.json")
    snapshot = yaml.safe_load((directory / "tasks.yaml").read_text())
    if snapshot.get("queue_group") != group:
        raise ValueError("This directory belongs to another group")
    if Path(plan["root"]).resolve() != root.resolve():
        raise ValueError("Prepare a fresh plan on the destination machine")
    if digest(directory / "tasks.yaml") != plan["tasks_sha256"]:
        raise ValueError("Frozen task snapshot changed")
    if set(plan["models"]) != set(snapshot["models"]):
        raise ValueError("Frozen plan and group models disagree")
    expected = {(m, t, s) for m in plan["models"] for t in plan["targets"] for s in plan["seeds"]}
    jobs = [
        j
        for lane in plan["lanes"]
        for j in read_json(Path(lane) / "plan.json")["jobs"]
        if j["kind"] != "pretrain"
    ]
    actual = [(j["model"], j["target"], j["seed"]) for j in jobs]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError("Group queue has duplicated or missing downstream jobs")
    return plan


def prepare_group(
    root: Path, group: str, directory: Path, *, gpus=(0,), seeds=None, targets=None
) -> dict:
    audit_profiles(root)
    # Distinct basenames prevent collisions in the existing per-batch report exporter.
    if not directory.name.startswith(group + "_"):
        raise ValueError(f"Batch directory basename must start with {group}_")
    plan = queues.prepare(
        root,
        root / PROFILE_ROOT / f"{group}.yaml",
        directory,
        seeds=seeds,
        targets=targets,
        gpus=gpus,
    )
    check_batch(root, directory, group)
    return plan


def start_group(
    root: Path, group: str, directory: Path, *, gpus=None, seeds=None, targets=None
) -> int:
    if not (directory / "plan.json").exists():
        prepare_group(root, group, directory, gpus=gpus or [0], seeds=seeds, targets=targets)
    plan = check_batch(root, directory, group)
    for name, selected in (("gpus", gpus), ("seeds", seeds), ("targets", targets)):
        if selected is not None and list(selected) != plan[name]:
            raise ValueError(f"Frozen {name} differ; prepare a fresh batch directory")
    return queues.run(directory)


def export_group(root: Path, group: str, directory: Path, *, strict=False) -> Path:
    plan = check_batch(root, directory, group)
    output = queues.summarize(directory, strict=strict)
    tasks = yaml.safe_load((directory / "tasks.yaml").read_text())
    data_root = Path(tasks["data_root"])
    if not data_root.is_absolute():
        data_root = root / data_root
    write_json(
        output / "group_report.json",
        dict(
            group=group,
            models=plan["models"],
            seeds=plan["seeds"],
            targets=plan["targets"],
            protocol_id=NEURAL_PROPERTY_PROTOCOL,
            defaults=tasks["defaults"],
            data_identity_sha256=digest(data_root / "data_identity.json"),
            created_at=utc_now(),
            files={
                name: digest(output / name)
                for name in ("runs.csv", "summary.csv", "verification.json")
            },
        ),
    )
    return output


def merge_reports(reports: list[Path], output: Path) -> Path:
    """Merge server-verified reports without server paths, weights or GPUs."""
    if len(reports) != 3 or output.exists():
        raise ValueError("Supply three group reports and a fresh output directory")
    all_runs, all_summary, groups, models, data_hashes = [], [], set(), set(), set()
    for report in reports:
        meta = read_json(report / "group_report.json")
        group = meta["group"]
        if group not in GROUPS or group in groups or models & set(meta["models"]):
            raise ValueError("Duplicate group or overlapping models")
        if meta["protocol_id"] != NEURAL_PROPERTY_PROTOCOL or meta["defaults"] != dict(
            batch_size=256, epochs=100, patience=20, min_delta=0.0
        ):
            raise ValueError("Group protocols/budgets disagree")
        if set(meta["seeds"]) != set(SEEDS) or set(meta["targets"]) != set(NEW_TARGETS):
            raise ValueError("Merge requires six properties and three seeds")
        for name in ("runs.csv", "summary.csv", "verification.json"):
            if digest(report / name) != meta["files"][name]:
                raise ValueError(f"Transferred report changed: {name}")
        if not read_json(report / "verification.json").get("complete"):
            raise ValueError("Group has unfinished experiments")
        runs = pd.read_csv(report / "runs.csv")
        summary = pd.read_csv(report / "summary.csv")
        expected = {(m, t, s) for m in meta["models"] for t in NEW_TARGETS for s in SEEDS}
        actual = list(zip(runs.model, runs.target, runs.seed, strict=True))
        if (
            len(actual) != len(expected)
            or set(actual) != expected
            or not runs.status.eq("completed").all()
        ):
            raise ValueError("Transferred group runs are incomplete or duplicated")
        for target, unit in zip(runs.target, runs.unit, strict=True):
            if unit != PROPERTY_UNITS[target]:
                raise ValueError("Property unit mismatch")
        expected_summary = {(m, t) for m in meta["models"] for t in NEW_TARGETS}
        actual_summary = list(zip(summary.model, summary.target, strict=True))
        if len(actual_summary) != len(expected_summary) or set(actual_summary) != expected_summary:
            raise ValueError("Transferred summaries are incomplete or duplicated")
        for row in summary.itertuples():
            subset = runs[runs.model.eq(row.model) & runs.target.eq(row.target)]
            if row.n_seeds != 3 or row.complete_three_seeds not in (True,):
                raise ValueError("Summary does not declare three completed paired seeds")
            if row.unit != PROPERTY_UNITS[row.target] or row.target_scaling != "standard":
                raise ValueError("Summary units/scaling disagree")
            for metric in queues.METRICS:
                values = subset[metric].to_numpy(dtype=float)
                if not np.isfinite(values).all() or not np.allclose(
                    [values.mean(), values.std(ddof=1)],
                    [getattr(row, metric + "_mean"), getattr(row, metric + "_std")],
                    rtol=1e-9,
                    atol=1e-12,
                ):
                    raise ValueError("Summary does not match per-seed results")
        all_runs.append(runs.assign(source_group=group))
        all_summary.append(summary.assign(source_group=group))
        groups.add(group)
        models.update(meta["models"])
        data_hashes.add(meta["data_identity_sha256"])
    if models != set(queues.MODELS) or len(data_hashes) != 1:
        raise ValueError("Ten-model coverage or QM9 data identity differs between machines")
    output.mkdir(parents=True)
    runs = pd.concat(all_runs, ignore_index=True)
    summary = pd.concat(all_summary, ignore_index=True)
    runs.to_csv(output / "runs.csv", index=False)
    summary.to_csv(output / "summary.csv", index=False)
    for target, frame in summary.groupby("target"):
        frame.sort_values("rmse_mean", ascending=False).to_csv(
            output / f"{target}_comparison.csv",
            index=False,
        )
    write_json(
        output / "verification.json",
        dict(
            groups=sorted(groups),
            completed=180,
            summaries=60,
            complete=True,
            protocol_id=NEURAL_PROPERTY_PROTOCOL,
            data_identity_sha256=next(iter(data_hashes)),
            source_reports=[str(p) for p in reports],
            generated_at=utc_now(),
            checks="Source report hashes; 180 unique runs; units; mean/sample-SD recomputed",
        ),
    )
    (output / "README.md").write_text(
        "# QM9 三组跨机合并结果\n\n180 个逐 seed 结果，60 个模型/性质摘要。\n"
        "分组报告在来源机器验证预测；本地合并核对文件哈希、数据身份、单位及统计量。\n"
        "各 *_comparison.csv 在同一性质内按 RMSE 降序排列，资源/时间字段保留来源记录。\n"
    )
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("group", choices=(*GROUPS, "merge"))
    parser.add_argument(
        "action",
        nargs="?",
        default="start",
        choices=("start", "prepare", "run", "status", "retry", "summarize", "verify", "audit"),
    )
    parser.add_argument("--directory", type=Path)
    parser.add_argument("--gpus", type=int, nargs="+")
    parser.add_argument("--seeds", type=int, nargs="+", choices=SEEDS)
    parser.add_argument("--targets", nargs="+", choices=NEW_TARGETS)
    parser.add_argument("--jobs", nargs="+")
    parser.add_argument("--reports", type=Path, nargs="+")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--source-tex", type=Path)
    parser.add_argument("--output-tex", type=Path)
    args = parser.parse_args()
    root = find_project_root()
    if args.group == "merge":
        if not args.reports:
            parser.error("merge requires --reports high_report medium_report low_report")
        if bool(args.source_tex) != bool(args.output_tex):
            parser.error("LaTeX export requires both --source-tex and --output-tex")
        output = args.output or root / "outputs/summaries/qm9_results/batches" / (
            "qm9_six_groups_" + datetime.now(UTC).strftime("%Y%m%d_%H%M%S_UTC")
        )
        print(merge_reports(args.reports, output))
        if args.source_tex:
            queues.fill_latex(output / "summary.csv", args.source_tex, args.output_tex)
        return
    directory = (
        args.directory or root / "artifacts/qm9_six_properties" / f"{args.group}_formal01"
    ).resolve()
    if args.action == "audit":
        print(json.dumps(audit_profiles(root), indent=2))
    elif args.action == "prepare":
        print(
            json.dumps(
                prepare_group(
                    root,
                    args.group,
                    directory,
                    gpus=args.gpus or [0],
                    seeds=args.seeds,
                    targets=args.targets,
                ),
                indent=2,
            )
        )
    elif args.action == "start":
        raise SystemExit(
            start_group(
                root, args.group, directory, gpus=args.gpus, seeds=args.seeds, targets=args.targets
            )
        )
    else:
        check_batch(root, directory, args.group)
        if args.gpus or args.seeds or args.targets:
            parser.error(
                "GPU/seeds/targets options belong to prepare/start, not frozen queue actions"
            )
        if args.action == "run":
            raise SystemExit(queues.run(directory))
        elif args.action == "retry":
            if not args.jobs:
                parser.error("retry requires explicit --jobs IDs")
            print(queues.retry(directory, args.jobs))
        elif args.action in {"summarize", "verify"}:
            print(export_group(root, args.group, directory, strict=args.action == "verify"))
        else:
            for lane, state in queues.lane_states(directory):
                for job in state["jobs"]:
                    print(
                        lane.name,
                        job["id"],
                        job["status"],
                        job.get("error") or job.get("wait_reason", ""),
                    )


if __name__ == "__main__":
    main()
