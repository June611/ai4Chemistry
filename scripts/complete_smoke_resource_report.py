#!/usr/bin/env python3
"""Wait for smoke completion, recheck missing resource coverage, and merge CSV evidence."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import yaml


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runner-pid", type=int, required=True)
    parser.add_argument(
        "--merge-only", action="store_true", help="Merge already completed measurements"
    )
    args = parser.parse_args()
    directory = args.output.resolve()
    root = Path(__file__).resolve().parents[1]
    while True:
        report = json.loads((directory / "status.json").read_text())
        if report.get("finished"):
            break
        if not Path(f"/proc/{args.runner_pid}").exists():
            raise RuntimeError("Original runner stopped without completing the smoke report")
        time.sleep(5)
    # Allow the first monitor to export final states before reading its CSV.
    time.sleep(3)
    original = read_csv(directory / "resources.csv")
    final_report = copy.deepcopy(report)
    retry_models = [
        entry["model"]
        for entry in report["models"]
        if entry["kind"] == "chemprop"
        and entry["status"] == "failed"
        and any(
            "The number of epochs should be higher" in line for line in entry.get("log_tail", [])
        )
    ]
    retry_directory = directory.with_name(directory.name + "_warmup_recheck")
    retried_rows = {}
    if retry_models:
        if not args.merge_only:
            subprocess.run(
                [
                    sys.executable,
                    str(root / "scripts/run_one_epoch_smoke.py"),
                    "--output",
                    str(retry_directory),
                    "--models",
                    *retry_models,
                ],
                cwd=root,
                check=True,
            )
            time.sleep(3)
        retried_rows = {row["model"]: row for row in read_csv(retry_directory / "resources.csv")}
        retry_report = json.loads((retry_directory / "status.json").read_text())
        retried_entries = {entry["model"]: entry for entry in retry_report["models"]}
        for index, entry in enumerate(final_report["models"]):
            if entry["model"] in retried_entries:
                corrected = copy.deepcopy(retried_entries[entry["model"]])
                corrected["initial_attempt"] = entry
                corrected["diagnostic_adjustment"] = "CLI warmup_epochs=0 because total epochs=1"
                final_report["models"][index] = corrected
    required = [
        row["model"] for row in original if row["coverage"] != "full" and row["status"] != "blocked"
    ]
    replacement = {}
    hardware_path = directory / "resource_hardware.json"
    hardware = json.loads(hardware_path.read_text()) if hardware_path.is_file() else {}
    recheck = directory.with_name(directory.name + "_resource_recheck")
    if required:
        if not args.merge_only:
            subprocess.run(
                [
                    sys.executable,
                    str(root / "scripts/run_one_epoch_smoke.py"),
                    "--output",
                    str(recheck),
                    "--models",
                    *required,
                ],
                cwd=root,
                check=True,
            )
            time.sleep(3)
        replacement = {row["model"]: row for row in read_csv(recheck / "resources.csv")}
    memory_fields = (
        "gpu_process_peak_mib",
        "gpu_device_peak_mib",
        "ram_tree_peak_rss_mib",
        "ram_tree_peak_pss_mib",
        "sample_count",
        "sample_interval_seconds",
        "coverage",
    )
    merged = []
    for row in original:
        measured = retried_rows.get(row["model"], replacement.get(row["model"], row))
        row["initial_status"] = row["status"]
        row["initial_error"] = row["error"]
        if row["model"] in retried_rows:
            row.update(
                {key: measured[key] for key in ("status", "error", "run_dir", "pipeline_seconds")}
            )
        row.update({key: measured[key] for key in memory_fields})
        row["measurement_status"] = measured["status"]
        row["measurement_run_dir"] = measured["run_dir"]
        row["measurement_recheck"] = str(
            row["model"] in replacement or row["model"] in retried_rows
        ).lower()
        row["measurement_pipeline_seconds"] = measured["pipeline_seconds"]
        gpu_mib = measured["gpu_process_peak_mib"]
        pss_mib = measured["ram_tree_peak_pss_mib"]
        row["gpu_peak_gib"] = round(float(gpu_mib) / 1024, 3) if gpu_mib else ""
        row["ram_peak_pss_gib"] = round(float(pss_mib) / 1024, 3) if pss_mib else ""
        row["gpu_type"] = hardware.get("gpu", "")
        limits = hardware.get("cgroup_limits", {})
        quota = limits.get("/sys/fs/cgroup/cpu.max", "").split()
        row["cpu_quota_cores"] = (
            float(quota[0]) / float(quota[1]) if quota and quota[0] != "max" else ""
        )
        ram_limit = limits.get("/sys/fs/cgroup/memory.max", "")
        row["ram_cgroup_limit_gib"] = (
            round(int(ram_limit) / 1024**3, 2) if ram_limit.isdigit() else ""
        )
        config_directory = retry_directory if row["model"] in retried_rows else directory
        config = yaml.safe_load((config_directory / "configs" / f"{row['model']}.yaml").read_text())
        settings = config.get("training", config.get("pretraining", {}))
        row["num_workers"] = settings.get("num_workers", "N/A")
        row["cpu_n_jobs"] = settings.get("n_jobs", "N/A")
        row["precision"] = settings.get("precision", "N/A" if row["kind"] == "ml" else "float32")
        timing_path = Path(row["run_dir"]) / (
            "timing.json" if row["kind"] == "pretrain" else "training_timing.json"
        )
        timing = json.loads(timing_path.read_text()) if timing_path.is_file() else {}
        timing_key = next(
            (key for key in ("fit_seconds", "fit_pipeline_seconds", "seconds") if key in timing),
            None,
        )
        row["fit_seconds"] = round(timing[timing_key], 2) if timing_key else ""
        row["fit_timing_scope"] = timing_key or "not_completed"
        effective = next(
            entry for entry in final_report["models"] if entry["model"] == row["model"]
        )
        row["gap_mae_hartree"] = effective.get("checks", {}).get("gap_mae_hartree", "")
        row["training_rows"] = 50000 if row["kind"] == "pretrain" else 107038
        row["validation_rows"] = 13380
        row["test_rows"] = effective.get("checks", {}).get("test_rows", 0)
        merged.append(row)
    destination = directory / "resources_complete.csv"
    with destination.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(merged[0]))
        writer.writeheader()
        writer.writerows(merged)
    completion = {
        "completed_at": datetime.now(UTC).isoformat(),
        "resource_csv": str(destination),
        "resource_rechecked_models": required,
        "resource_recheck_directory": str(recheck) if required else None,
        "diagnostic_warmup_retried_models": retry_models,
        "inconsistent_outcomes": [
            row["model"] for row in merged if row["status"] != row["measurement_status"]
        ],
        "missing_full_coverage": [
            row["model"]
            for row in merged
            if row["coverage"] != "full" and row["status"] != "blocked"
        ],
        "downstream_success": [
            entry["model"]
            for entry in final_report["models"]
            if entry["kind"] != "pretrain" and entry["status"] == "success"
        ],
        "downstream_failed_or_blocked": [
            entry["model"]
            for entry in final_report["models"]
            if entry["kind"] != "pretrain" and entry["status"] != "success"
        ],
    }
    final_report["original_report"] = str(directory / "status.json")
    final_report["initial_suite_finished"] = final_report["finished"]
    final_report["finished"] = completion["completed_at"]
    (directory / "final_status.json").write_text(json.dumps(final_report, indent=2) + "\n")
    lines = [
        "# QM9 final one-epoch smoke results",
        "",
        "Seed/split3407; neural batches recorded in resources_complete.csv, 1epoch; "
        "full fixed QM9 split.",
        "CLI Chemprop uses warmup0 for this diagnostic only. Initial failed attempts are retained.",
        "",
        "| Model | Final status | GPU peak GiB | RAM peak PSS GiB | Note |",
        "|---|---|---:|---:|---|",
    ]
    for row in merged:
        note = row["error"].replace("|", "/").replace("\n", " ")
        if row["model"] in retried_rows:
            note = (
                "Passed after diagnostic CLI warmup0 retry" if row["status"] == "success" else note
            )
        lines.append(
            f"| {row['model']} | {row['status']} | {row['gpu_peak_gib']} | "
            f"{row['ram_peak_pss_gib']} | {note} |"
        )
    (directory / "final_results.md").write_text("\n".join(lines) + "\n")
    (directory / "completion.json").write_text(json.dumps(completion, indent=2) + "\n")
    print(json.dumps(completion, indent=2), flush=True)


if __name__ == "__main__":
    main()
