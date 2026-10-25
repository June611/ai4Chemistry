"""Run an extension job chain on one GPU after a formal-queue trigger completes.

Waits for a formal job's COMPLETED.json and a free GPU, then runs the given
extension jobs sequentially through the same run_row logic as the formal
runner (skip completed, resume last.ckpt, matched env, per-job logs).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from run_qm9_formal_4gpu import run_row  # noqa: E402

FORMAL_RUNS = ROOT / "artifacts/qm9_epoch200/formal_4gpu/runs"


def gpu_busy(gpu: str, threshold: int) -> bool:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", gpu],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    return int(out) >= threshold


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", required=True)
    parser.add_argument(
        "--trigger",
        required=True,
        help="formal job id whose COMPLETED.json gates the chain",
    )
    parser.add_argument(
        "--trigger-root",
        type=Path,
        default=FORMAL_RUNS,
        help="runs directory containing the trigger job (formal or extension)",
    )
    parser.add_argument(
        "--mem-threshold",
        type=int,
        default=1000,
        help="GPU is free below this used-MiB (raise when foreign tenants hold small allocations)",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--jobs",
        required=True,
        help="comma-separated id=relative/config/path entries, run in order",
    )
    args = parser.parse_args()

    trigger = args.trigger_root / args.trigger / "COMPLETED.json"
    print(f"[{args.gpu}] waiting for trigger {trigger}", flush=True)
    while not trigger.is_file():
        time.sleep(60)
    while gpu_busy(args.gpu, args.mem_threshold):
        time.sleep(60)
    time.sleep(30)
    print(f"[{args.gpu}] free at {time.strftime('%F %T')}; chain start", flush=True)

    failed = 0
    for entry in args.jobs.split(","):
        entry = entry.strip()
        if not entry:
            continue
        job_id, config = entry.split("=", 1)
        row = {"id": job_id, "config": config}
        if not run_row(row, args.output_root, args.gpu):
            failed += 1
            print(f"[{args.gpu}] FAILED {job_id}; chain continues", flush=True)
    print(f"[{args.gpu}] chain done at {time.strftime('%F %T')} failed={failed}", flush=True)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
