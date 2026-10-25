"""Run isolated CUDA smoke jobs from the real fresh200 queue; never launch formal runs."""

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# Unified paper queue (2026-10-20): 10 neural baselines x 3 seeds plus
# 2 SOTA paths x 2 modules x 3 matched seeds.
EXPECTED_QUEUE_JOBS = 42


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def source_digest(root):
    h = hashlib.sha256()
    paths = sorted((root / "src/molgap").rglob("*.py")) + [Path(__file__).resolve()]
    for path in paths:
        h.update(str(path.relative_to(root)).encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def audit_run(run, rows):
    """Check actual metrics, updates and full saved state before optional smoke cleanup."""
    import numpy as np
    import pandas as pd
    import torch

    from molgap.modular.config import validate
    from molgap.modular.train import metrics

    report = json.loads((run / "test_metrics.json").read_text())
    timing = json.loads((run / "training_timing.json").read_text())
    completed = json.loads((run / "COMPLETED.json").read_text())
    if not report["diagnostic"] or not completed["diagnostic"]:
        raise ValueError("Smoke output must be diagnostic")
    if timing["epochs_completed"] != 1 or timing["peak_gpu_allocated_bytes"] <= 0:
        raise ValueError("Expected one actual CUDA training epoch")
    pred = pd.read_csv(run / "test_predictions.csv", float_precision="round_trip")
    tasks = list(report["units"])
    if len(pred) != rows or pred.sample_id.duplicated().any():
        raise ValueError("Prediction row count/identity differs")
    recalculated = metrics(
        pred[tasks].to_numpy(), pred[["true_" + t for t in tasks]].to_numpy(), tasks
    )
    for task, actual in recalculated.items():
        saved = report["targets"].get(task, report["auxiliary_targets"].get(task))
        for key, value in actual.items():
            if not math.isfinite(value) or not np.isclose(
                value, saved[key], rtol=1e-12, atol=1e-14
            ):
                raise ValueError(f"On-disk metric mismatch: {task}/{key}")
    checkpoint = run / "checkpoints/last.ckpt"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = payload["run_context"]["config"]
    validate(config)
    expected_steps = math.ceil(rows / config["training"]["batch_size"])
    if payload["completed_epochs"] != 1 or payload["global_step"] != expected_steps:
        raise ValueError("Training did not consume the full smoke prefix")
    for key in (
        "optimizer_states",
        "lr_schedulers",
        "rng_snapshot",
        "loader_states",
        "best_inference_state",
    ):
        if not payload.get(key):
            raise ValueError(f"Missing full checkpoint state: {key}")
    for name, tensor in payload["state_dict"].items():
        if tensor.is_floating_point() and not torch.isfinite(tensor).all():
            raise ValueError(f"Nonfinite checkpoint parameter: {name}")
    if payload["run_context"]["limit_rows"] != rows:
        raise ValueError("Unexpected diagnostic subset")
    if set(payload["loader_states"]) != {"train", "val", "test"}:
        raise ValueError("Loader states incomplete")
    if any(
        x["fit_split"] != "train" or x["train_rows"] <= rows
        for x in payload["run_context"]["scalers"].values()
    ):
        raise ValueError("Scalers must use the original full training partition")
    files = [run / "best_checkpoint.pt"] + sorted((run / "checkpoints").glob("*.ckpt"))
    for name in ("best.ckpt", "last.ckpt", "terminal_fit.ckpt"):
        if not (run / "checkpoints" / name).is_file():
            raise ValueError(f"Missing checkpoint: {name}")
    return dict(
        config_sha256=digest(run / "config.yaml"),
        targets=tasks,
        diagnostic=True,
        epoch=1,
        optimizer_steps=expected_steps,
        rows_per_split=rows,
        metrics_recalculated_from_csv=True,
        full_checkpoint_state_verified=True,
        checkpoint_files=[
            dict(path=str(p.relative_to(run)), size_bytes=p.stat().st_size, sha256=digest(p))
            for p in files
        ],
        timing=timing,
    )


def worker(args):
    from molgap.modular.train import train

    run = args.worker_run.resolve()
    try:
        train(args.config, output_dir=run, diagnostic_epochs=1, limit_rows=args.rows)
        audit = audit_run(run, args.rows)
        write_json(run / "smoke_audit.json", audit)
        if args.prune_successful_checkpoints:
            # Explicit smoke-only policy: never prune failed/unaudited or formal runs.
            for record in audit["checkpoint_files"]:
                (run / record["path"]).unlink()
            audit["checkpoint_retention"] = (
                "verified then removed; diagnostic only, user authorized"
            )
            write_json(run / "smoke_audit.json", audit)
    except Exception:
        traceback.print_exc()
        return 1
    return 0


def rss_tree(pid):
    """Linux sampled sum RSS, including current DataLoader workers; shared pages may repeat."""
    total, pending, seen = 0, [pid], set()
    while pending:
        current = pending.pop()
        if current in seen:
            continue
        seen.add(current)
        base = Path("/proc") / str(current)
        try:
            for line in (base / "status").read_text().splitlines():
                if line.startswith("VmRSS:"):
                    total += int(line.split()[1]) * 1024
            pending.extend(
                int(p) for p in (base / "task" / str(current) / "children").read_text().split()
            )
        except (OSError, ValueError):
            pass
    return total


def save_results(root, records, expected_ids):
    latest = {r["id"]: r for r in records}
    ordered = [latest[j] for j in expected_ids if j in latest]
    write_json(
        root / "results.json",
        dict(
            expected=len(expected_ids),
            passed=sum(r["status"] == "passed" for r in ordered),
            failed=sum(r["status"] != "passed" for r in ordered),
            finished=len(ordered),
            attempts=records,
            latest=ordered,
            formal_training_started=False,
        ),
    )
    with (root / "results.csv").open("w") as stream:
        keys = [
            "id",
            "model",
            "variant",
            "seed",
            "attempt",
            "status",
            "exit_code",
            "elapsed_seconds",
            "epoch_seconds",
            "peak_gpu_allocated_bytes",
            "peak_gpu_reserved_bytes",
            "sampled_process_tree_rss_bytes",
            "config",
            "config_sha256",
            "source_sha256",
            "run_dir",
            "log",
            "failure_tail",
        ]
        writer = csv.DictWriter(stream, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(ordered)


def campaign(args):
    root = Path(__file__).resolve().parents[1]
    queue = list(
        csv.DictReader(
            (root / "docs/features/qm9-epoch200-modular-study/training_queue.csv").open()
        )
    )
    if len(queue) != EXPECTED_QUEUE_JOBS or len({r["id"] for r in queue}) != EXPECTED_QUEUE_JOBS:
        raise ValueError(
            f"Expected the complete unique {EXPECTED_QUEUE_JOBS}-job paper-scope queue"
        )
    selected = [
        r
        for r in queue
        if (not args.model or r["model"] in args.model) and (not args.job or r["id"] in args.job)
    ]
    if not selected:
        raise ValueError("No matching jobs")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    marker = output / "campaign.json"
    if marker.exists():
        old = json.loads(marker.read_text())
        if old["rows_per_split"] != args.rows or old["root"] != str(root):
            raise ValueError("Campaign data/root differs")
    else:
        write_json(
            marker,
            dict(
                root=str(root),
                rows_per_split=args.rows,
                epochs=1,
                batch_size=256,
                formal_training=False,
                prune_successful_checkpoints=args.prune_successful_checkpoints,
            ),
        )
    records = (
        json.loads((output / "results.json").read_text())["attempts"]
        if (output / "results.json").exists()
        else []
    )
    latest = {r["id"]: r for r in records}
    source_hash = source_digest(root)
    for row in selected:
        old = latest.get(row["id"])
        if (
            old
            and old["status"] == "passed"
            and old.get("source_sha256") == source_hash
            and old["config_sha256"] == digest(root / row["config"])
            and not args.rerun_passed
        ):
            continue
        attempt = sum(r["id"] == row["id"] for r in records) + 1
        run = output / "runs" / row["id"] / f"attempt_{attempt:03d}"
        log = output / "logs" / f"{row['id']}_attempt{attempt:03d}.log"
        log.parent.mkdir(exist_ok=True)
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker-run",
            str(run),
            "--config",
            str(root / row["config"]),
            "--rows",
            str(args.rows),
        ]
        if args.prune_successful_checkpoints:
            command.append("--prune-successful-checkpoints")
        env = dict(
            os.environ,
            PYTHONPATH=str(root / "src"),
            CUBLAS_WORKSPACE_CONFIG=os.environ.get("CUBLAS_WORKSPACE_CONFIG", ":4096:8"),
            OMP_NUM_THREADS="2",
            MKL_NUM_THREADS="2",
            MPLCONFIGDIR=str(output / "mpl"),
        )
        start, peak = time.perf_counter(), 0
        with log.open("w") as stream:
            process = subprocess.Popen(
                command, cwd=root, stdout=stream, stderr=subprocess.STDOUT, env=env
            )
            while process.poll() is None:
                peak = max(peak, rss_tree(process.pid))
                time.sleep(1)
        record = dict(
            id=row["id"],
            model=row["model"],
            variant=row["variant"],
            seed=int(row["seed"]),
            attempt=attempt,
            status="failed",
            exit_code=process.returncode,
            elapsed_seconds=time.perf_counter() - start,
            sampled_process_tree_rss_bytes=peak,
            config=row["config"],
            config_sha256=digest(root / row["config"]),
            source_sha256=source_hash,
            run_dir=str(run),
            log=str(log),
        )
        if process.returncode == 0 and (run / "smoke_audit.json").is_file():
            audit = json.loads((run / "smoke_audit.json").read_text())
            record.update(
                status="passed",
                **{
                    k: audit["timing"][k]
                    for k in ("peak_gpu_allocated_bytes", "peak_gpu_reserved_bytes")
                },
            )
            history = list(csv.DictReader((run / "training_log.csv").open()))
            record["epoch_seconds"] = float(history[-1]["epoch_seconds"])
        else:
            record["failure_tail"] = log.read_text(errors="replace")[-4000:]
        records.append(record)
        latest[row["id"]] = record
        save_results(output, records, [r["id"] for r in queue])
        print(
            json.dumps(
                {k: record[k] for k in ("id", "attempt", "status", "exit_code", "elapsed_seconds")}
            ),
            flush=True,
        )
    return 0 if all(latest.get(r["id"], {}).get("status") == "passed" for r in selected) else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/qm9_epoch200/smoke_paper_scope")
    )
    parser.add_argument("--rows", type=int, default=500)
    parser.add_argument("--model", action="append")
    parser.add_argument("--job", action="append")
    parser.add_argument("--rerun-passed", action="store_true")
    parser.add_argument("--prune-successful-checkpoints", action="store_true")
    parser.add_argument("--worker-run", type=Path)
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    if args.rows < 257:
        parser.error("Use at least 257 rows to exercise the second batch")
    if args.worker_run and not args.config:
        parser.error("Worker requires --config")
    return worker(args) if args.worker_run else campaign(args)


if __name__ == "__main__":
    raise SystemExit(main())
