#!/usr/bin/env python3
"""Sample a sequential smoke runner and export approximate model resource peaks."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path


def process_memory(root_pid: int) -> tuple[set[int], float, float]:
    """Sum RSS/PSS for the runner and its descendants; PSS apportions shared pages."""
    parents = {}
    for path in Path("/proc").glob("[0-9]*/stat"):
        try:
            fields = path.read_text().rsplit(")", 1)[1].split()
            parents[int(path.parent.name)] = int(fields[1])
        except (OSError, ValueError, IndexError):
            continue
    selected = {root_pid}
    while True:
        children = {pid for pid, parent in parents.items() if parent in selected}
        if children <= selected:
            break
        selected |= children
    rss, pss = 0.0, 0.0
    for pid in selected:
        try:
            for line in Path(f"/proc/{pid}/smaps_rollup").read_text().splitlines():
                if line.startswith("Rss:"):
                    rss += float(line.split()[1]) / 1024
                elif line.startswith("Pss:"):
                    pss += float(line.split()[1]) / 1024
        except (OSError, ValueError):
            continue
    return selected, rss, pss


def gpu_memory(pids: set[int]) -> tuple[float | None, float | None]:
    try:
        processes = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=5,
        )
        model = 0.0
        for line in processes.splitlines():
            pid, memory = line.split(",")
            if int(pid.strip()) in pids:
                model += float(memory.strip())
        device = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True,
            timeout=5,
        )
        return model, sum(float(line) for line in device.splitlines())
    except (OSError, ValueError, subprocess.SubprocessError):
        return None, None


def export(directory: Path, report: dict, samples: dict, interval: float) -> None:
    columns = (
        "model",
        "kind",
        "status",
        "split_seed",
        "training_seed",
        "batch_size",
        "epochs",
        "pipeline_seconds",
        "gpu_process_peak_mib",
        "gpu_device_peak_mib",
        "ram_tree_peak_rss_mib",
        "ram_tree_peak_pss_mib",
        "sample_count",
        "sample_interval_seconds",
        "coverage",
        "error",
        "run_dir",
    )
    temporary = directory / "resources.csv.tmp"
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for entry in report["models"]:
            measurement = samples.get(entry["model"], {})
            writer.writerow(
                {
                    "model": entry["model"],
                    "kind": entry["kind"],
                    "status": entry["status"],
                    "split_seed": 3407,
                    "training_seed": 3407,
                    "batch_size": "N/A"
                    if entry["kind"] == "ml"
                    else entry.get("batch_size", report.get("batch_size", 256)),
                    "epochs": "N/A" if entry["kind"] == "ml" else entry.get("epochs", 1),
                    "pipeline_seconds": round(entry.get("seconds", 0), 2),
                    **{
                        key: round(measurement[key], 2) if key in measurement else ""
                        for key in (
                            "gpu_process_peak_mib",
                            "gpu_device_peak_mib",
                            "ram_tree_peak_rss_mib",
                            "ram_tree_peak_pss_mib",
                        )
                    },
                    "sample_count": measurement.get("sample_count", 0),
                    "sample_interval_seconds": interval,
                    "coverage": measurement.get("coverage", "not_measured"),
                    "error": entry.get("error", ""),
                    "run_dir": entry["run_dir"],
                }
            )
    temporary.replace(directory / "resources.csv")
    (directory / "resources.json").write_text(
        json.dumps(
            {
                "updated": datetime.now(UTC).isoformat(),
                "interval_seconds": interval,
                "memory_scope": "runner and recursive subprocess tree; "
                "RSS sums shared pages, PSS apportions them",
                "gpu_scope": "sum of nvidia-smi compute process allocations for that tree; "
                "device total recorded separately",
                "limitations": "sampled approximate peaks; sub-interval spikes may be missed; "
                "failed runs measure attempted execution only",
                "models": samples,
            },
            indent=2,
        )
        + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runner-pid", type=int, required=True)
    parser.add_argument("--interval", type=float, default=1.0)
    args = parser.parse_args()
    directory = args.output.resolve()
    saved = directory / "resources.json"
    samples = json.loads(saved.read_text())["models"] if saved.is_file() else {}
    initial = json.loads((directory / "status.json").read_text())
    previous_pending = {
        entry["model"] for entry in initial["models"] if entry["status"] == "pending"
    }
    while True:
        report = json.loads((directory / "status.json").read_text())
        active = [entry for entry in report["models"] if entry["status"] == "running"]
        for entry in active:
            name = entry["model"]
            measurement = samples.setdefault(
                name,
                {
                    "sample_count": 0,
                    "coverage": "full" if name in previous_pending else "partial",
                    "ram_tree_peak_rss_mib": 0.0,
                    "ram_tree_peak_pss_mib": 0.0,
                },
            )
            pids, rss, pss = process_memory(args.runner_pid)
            gpu, device = gpu_memory(pids)
            measurement["sample_count"] += 1
            for key, value in (
                ("ram_tree_peak_rss_mib", rss),
                ("ram_tree_peak_pss_mib", pss),
                ("gpu_process_peak_mib", gpu),
                ("gpu_device_peak_mib", device),
            ):
                if value is not None:
                    measurement[key] = max(value, measurement.get(key, 0))
        previous_pending |= {
            entry["model"] for entry in report["models"] if entry["status"] == "pending"
        }
        export(directory, report, samples, args.interval)
        if report.get("finished"):
            return
        if not Path(f"/proc/{args.runner_pid}").exists():
            raise RuntimeError("Smoke runner disappeared before completing its report")
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
