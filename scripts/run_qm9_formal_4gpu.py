"""Run the formal unified QM9 queue on four independent GPU lanes.

This runner is resumable and intentionally executes only training_queue.csv.
It does not recursively discover YAML files and does not reduce batch size.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QUEUE = ROOT / "docs/features/qm9-epoch200-modular-study/training_queue.csv"
DEFAULT_OUT = ROOT / "artifacts/qm9_epoch200/formal_4gpu"
# Conservative relative costs from the preflight estimate; only lane balance
# depends on these values, training configuration remains in each YAML.
PER_EPOCH_SECONDS = {
    "chemprop": 12.0,
    "schnet": 82.1,
    "3dinfomax": 139.6,
    "egnn": 76.0,
    "painn": 90.4,
    "dimenet_pp": 147.2,
    "smiles_transformer": 19.1,
    "spe_transformer": 19.4,
    "esa": 123.9,
    "torchmd_net": 123.6,
}


def estimate(row: dict[str, str]) -> float:
    multiplier = 1.05 if row["variant"] != "single_task" else 1.0
    return PER_EPOCH_SECONDS[row["model"]] * 170 * multiplier


def load_rows() -> list[dict[str, str]]:
    rows = list(csv.DictReader(QUEUE.open()))
    if len(rows) != 42 or len({row["id"] for row in rows}) != 42:
        raise ValueError("Expected the complete unique 42-job formal queue")
    return rows


def assign(rows: list[dict[str, str]], lane_count: int) -> list[list[dict[str, str]]]:
    lanes: list[list[dict[str, str]]] = [[] for _ in range(lane_count)]
    loads = [0.0] * lane_count
    for row in sorted(rows, key=estimate, reverse=True):
        lane = min(range(lane_count), key=loads.__getitem__)
        lanes[lane].append(row)
        loads[lane] += estimate(row)
    return lanes


def run_row(row: dict[str, str], output_root: Path, gpu: str) -> bool:
    run = output_root / "runs" / row["id"]
    completed = run / "COMPLETED.json"
    if completed.is_file():
        try:
            payload = json.loads(completed.read_text())
            if not payload.get("diagnostic", False):
                print(f"[{gpu}] SKIP completed {row['id']}", flush=True)
                return True
        except (OSError, json.JSONDecodeError):
            pass
    command = [
        sys.executable,
        "-m",
        "molgap.modular.train",
        "--config",
        str(ROOT / row["config"]),
        "--output-dir",
        str(run),
    ]
    last = run / "checkpoints" / "last.ckpt"
    if last.is_file() and not completed.is_file():
        command.extend(["--resume", str(last)])
        action = "RESUME"
    else:
        action = "START"
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = gpu
    env.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    env.setdefault("OMP_NUM_THREADS", "2")
    env.setdefault("MKL_NUM_THREADS", "2")
    log = output_root / "logs" / f"{row['id']}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    print(f"[{gpu}] {action} {row['id']} -> {log}", flush=True)
    with log.open("a") as stream:
        stream.write(f"\n=== {action} {row['id']} ===\n")
        stream.flush()
        result = subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
    ok = result.returncode == 0 and completed.is_file()
    print(f"[{gpu}] {'DONE' if ok else 'FAILED'} {row['id']} rc={result.returncode}", flush=True)
    return ok


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--lane", type=int)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument(
        "--jobs",
        help="Comma-separated job ids; keep only these in the selected lane (aux runner).",
    )
    parser.add_argument(
        "--skip-jobs",
        help="Comma-separated job ids to remove from the selected lane (rebalanced away).",
    )
    args = parser.parse_args()
    gpus = [x.strip() for x in args.gpus.split(",") if x.strip()]
    if not gpus or len(set(gpus)) != len(gpus):
        parser.error("--gpus must contain unique GPU indices")
    rows = load_rows()
    lanes = assign(rows, len(gpus))
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "queue_snapshot.csv").write_text(QUEUE.read_text())
    manifest = []
    for index, lane in enumerate(lanes):
        manifest.append(
            {
                "lane": index,
                "gpu": gpus[index],
                "estimated_seconds": sum(estimate(row) for row in lane),
                "jobs": [row["id"] for row in lane],
            }
        )
    rebalanced = bool(args.jobs or args.skip_jobs)
    if not rebalanced:
        # Keep the original full-queue plan as the authoritative record;
        # aux/rebalanced invocations must not overwrite it.
        (args.output_root / "lane_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if args.plan_only:
        for item in manifest:
            print(json.dumps(item, ensure_ascii=False))
        return 0
    if args.lane is not None and not 0 <= args.lane < len(lanes):
        parser.error("--lane is outside the GPU lane range")
    selected = lanes if args.lane is None else [lanes[args.lane]]
    gpu_offset = 0 if args.lane is None else args.lane
    keep = {x.strip() for x in args.jobs.split(",") if x.strip()} if args.jobs else None
    skip = {x.strip() for x in args.skip_jobs.split(",") if x.strip()} if args.skip_jobs else set()
    all_ids = {row["id"] for row in rows}
    unknown = ((keep or set()) | skip) - all_ids
    if unknown:
        parser.error(f"unknown job ids: {sorted(unknown)}")
    failed = 0
    for lane in selected:
        gpu = gpus[gpu_offset]
        effective = [
            row for row in lane if (keep is None or row["id"] in keep) and row["id"] not in skip
        ]
        if rebalanced:
            print(f"[{gpu}] effective jobs: {[row['id'] for row in effective]}", flush=True)
        for row in effective:
            if not run_row(row, args.output_root, gpu):
                failed += 1
        gpu_offset += 1
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
