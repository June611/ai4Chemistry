"""Persisted queue: independent pipelines, resource sampling, dependency and artifact checks."""

import csv
import fcntl
import hashlib
import json
import math
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import yaml

from molgap.scheduling.hardware import admission, snapshot
from molgap.tracking.records import build_record, read_json, utc_now, write_json
from molgap.tracking.resources import process_identity, sample_resources


def verify(job):
    run = Path(job["run_dir"])
    if job["kind"] == "pretrain":
        if not (run / "best_checkpoint.pt").is_file():
            raise ValueError("Missing pretraining checkpoint")
        history = json.loads((run / "history.json").read_text())
        if not history:
            raise ValueError("Missing pretraining history")
        return {"pretrain_epochs": len(history)}
    cfg = yaml.safe_load(Path(job["config"]).read_text())
    root = Path(job["root"])
    reference = root / cfg["data"]["processed_dir"] / "test.csv"
    prediction = run / "predictions/test_predictions.csv"
    id_column = cfg["data"].get("id_column", "sample_id")
    with reference.open() as stream:
        truth_rows = list(csv.DictReader(stream))
    expected = {r[id_column]: float(r["delta_e"]) for r in truth_rows}
    if len(expected) != len(truth_rows):
        raise ValueError("Duplicate fixed test identities")
    keys = list(expected)
    errors, seen = [], set()
    with prediction.open() as stream:
        reader = csv.DictReader(stream)
        prediction_column = next(
            (c for c in ("delta_e_pred", "pred_delta_e", "delta_e") if c in reader.fieldnames), None
        )
        if prediction_column is None:
            raise ValueError("Missing gap prediction column")
        for index, row in enumerate(reader):
            if index >= len(keys):
                raise ValueError("Too many test predictions")
            key = row.get(id_column)
            if key is None:
                key = keys[index]
                if row.get("smiles") != truth_rows[index].get("smiles"):
                    raise ValueError("Prediction SMILES order differs from the fixed test split")
            value = float(row[prediction_column])
            if key not in expected or key in seen or not math.isfinite(value):
                raise ValueError("Prediction identities or finite values failed verification")
            errors.append(value - expected[key])
            seen.add(key)
    if seen != set(expected):
        raise ValueError("Test predictions do not cover the fixed test split")
    payload = read_json(run / "metrics.json")
    metrics = payload.get("targets", {}).get("delta_e", payload)
    mae = sum(abs(e) for e in errors) / len(errors)
    if not math.isclose(mae, float(metrics["mae"]), rel_tol=1e-5, abs_tol=1e-8):
        raise ValueError("Saved gap MAE does not match test predictions")
    suffixes = {".pt", ".ckpt", ".joblib", ".json"} if job["kind"] == "ml" else {".pt", ".ckpt"}
    if not any(
        p.is_file() and p.suffix in suffixes and (job["kind"] != "ml" or p.name.startswith("model"))
        for p in run.rglob("*")
    ):
        raise ValueError("Missing saved model")
    return {"test_rows": len(errors), "mae": mae}


def worker(directory, identifier):
    plan = read_json(directory / "plan.json")
    job = next(j for j in plan["jobs"] if j["id"] == identifier)
    check_config(job)
    for command in job["commands"]:
        result = subprocess.run(command, cwd=plan["root"])
        if result.returncode:
            return result.returncode
    return 0


def check_config(job):
    digest = job.get("config_sha256")
    if digest and hashlib.sha256(Path(job["config"]).read_bytes()).hexdigest() != digest:
        raise ValueError(f"Frozen configuration changed: {job['id']}; create a fresh plan")


def publish_snapshot(runtime, directory, jobs, hw, queue_id, *, terminal=None, stopped=False):
    records = []
    for job in jobs:
        if job["status"] == "pending":
            continue
        try:
            record = build_record(
                Path(job["run_dir"]), {**job, "run_id": f"{queue_id}:{job['id']}"}
            )
        except (OSError, ValueError, TypeError) as exc:
            # Trainers may be halfway through a non-atomic history/metric write.
            # Observation failures must never terminate a training pipeline.
            job["monitor_warning"] = str(exc)
            continue
        if job["kind"] == "pretrain":
            record["fields"].update(目标="contrastive_pretraining", 单位="dimensionless")
        records.append({**record, "run_dir": job["run_dir"], "kind": job["kind"]})
    if records:
        temporary_csv = runtime / "summary.csv.tmp"
        with temporary_csv.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]["fields"]))
            writer.writeheader()
            writer.writerows(r["fields"] for r in records)
        temporary_csv.replace(runtime / "summary.csv")
    write_json(runtime / "state.json", {"jobs": jobs, "hardware": hw, "updated_at": utc_now()})
    write_json(
        runtime / "report.json",
        {
            "records": records,
            "hardware": hw,
            "queue_directory": str(directory),
            "terminal": terminal
            if terminal is not None
            else not any(j["status"] in {"pending", "running"} for j in jobs),
            "stopped": stopped,
            "updated_at": utc_now(),
        },
    )


def run_queue(directory: Path):
    directory = directory.resolve()
    plan = read_json(directory / "plan.json")
    policy, root = plan["policy"], Path(plan["root"])
    runtime = directory / "runtime"
    runtime.mkdir(exist_ok=True)
    with (runtime / "queue.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("This queue is already running") from None
        write_json(
            runtime / "controller.json",
            {
                "pid": os.getpid(),
                "identity": process_identity(os.getpid()),
                "host": socket.gethostname(),
            },
        )
        saved = read_json(runtime / "state.json")
        jobs = saved.get("jobs", plan["jobs"])
        for job in jobs:
            job["root"] = str(root)
            check_config(job)
            if job["status"] == "running":
                if (
                    process_identity(job["pid"]) == job.get("process_identity")
                    and job.get("process_identity") is not None
                ):
                    raise RuntimeError(
                        f"An earlier worker is alive: {job['id']}; stop it before restarting"
                    )
                job.update(
                    status="interrupted",
                    error="Previous controller/worker stopped; partial output retained",
                )
        processes = {}
        logs = {}
        stop = [False]

        def handle_signal(signum, frame):
            stop[0] = True

        previous = {s: signal.signal(s, handle_signal) for s in (signal.SIGTERM, signal.SIGINT)}
        last_launch = -float("inf")
        util_window = []
        hw = {}
        queue_id = hashlib.sha256(f"{socket.gethostname()}:{directory}".encode()).hexdigest()[:16]
        try:
            while True:
                hw = snapshot(root, policy["gpu"])
                hw["sampled_at"] = utc_now()
                util_window.append(hw["gpu_util_pct"])
                util_window = util_window[-3:]
                admission_hw = dict(hw)
                if all(v is not None for v in util_window):
                    admission_hw["gpu_util_pct"] = max(
                        hw["gpu_util_pct"], sum(util_window) / len(util_window)
                    )
                with (runtime / "hardware.jsonl").open("a") as stream:
                    stream.write(json.dumps(hw) + "\n")
                for identifier, process in list(processes.items()):
                    job = next(j for j in jobs if j["id"] == identifier)
                    try:
                        measured = sample_resources(process.pid)
                    except (OSError, ValueError, subprocess.SubprocessError) as exc:
                        measured = {"sample_error": type(exc).__name__}
                    job["latest"] = {**measured, **hw}
                    job["sample_count"] = job.get("sample_count", 0) + 1
                    job["elapsed_seconds"] = time.time() - job["started_unix"]
                    for name, value in job["latest"].items():
                        if isinstance(value, (int, float)):
                            job.setdefault("peaks", {})[name] = max(
                                value, job.get("peaks", {}).get(name, value)
                            )
                    if hw["gpu_util_pct"] is not None:
                        job["gpu_util_sum"] = job.get("gpu_util_sum", 0) + hw["gpu_util_pct"]
                        job["gpu_util_samples"] = job.get("gpu_util_samples", 0) + 1
                        job["gpu_util_mean_pct"] = job["gpu_util_sum"] / job["gpu_util_samples"]
                    with (runtime / f"{identifier}.jsonl").open("a") as stream:
                        stream.write(
                            json.dumps({"sampled_at": hw["sampled_at"], **job["latest"]}) + "\n"
                        )
                    code = process.poll()
                    if code is not None:
                        job.update(
                            status="completed" if code == 0 else "failed",
                            returncode=code,
                            ended_at=utc_now(),
                        )
                        if code == 0:
                            try:
                                job["verification"] = verify(job)
                            except Exception as exc:
                                job.update(status="failed", error=f"Artifact verification: {exc}")
                        else:
                            job["error"] = f"Pipeline exit {code}; see {job['log']}"
                        logs.pop(identifier).close()
                        del processes[identifier]
                        print(f"{job['id']}: {job['status']}", flush=True)
                by_id = {j["id"]: j for j in jobs}
                running = [j for j in jobs if j["status"] == "running"]
                for job in sorted(jobs, key=lambda j: -j["priority"]):
                    if job["status"] != "pending" or stop[0]:
                        continue
                    deps = [by_id[d]["status"] for d in job["depends_on"]]
                    if any(s in {"failed", "blocked", "interrupted"} for s in deps):
                        job.update(
                            status="blocked", error="Pretraining dependency failed or interrupted"
                        )
                        continue
                    if any(s != "completed" for s in deps):
                        job["wait_reason"] = "pretraining dependency"
                        continue
                    # Hardware capacity cannot improve by waiting for another queue task.
                    if job["cpu_slots"] > hw["cpus"] - policy["cpu_reserve"]:
                        job.update(status="blocked", error="CPU capacity smaller than job budget")
                        continue
                    if job["ram_mib"] > hw["ram_total_mib"] - policy["ram_reserve_mib"]:
                        job.update(status="blocked", error="RAM capacity smaller than job budget")
                        continue
                    reserve = (
                        policy["exclusive_gpu_reserve_mib"]
                        if job["exclusive"]
                        else policy["gpu_reserve_mib"]
                    )
                    if (
                        job["gpu_mib"]
                        and hw["gpu_total_mib"] is not None
                        and job["gpu_mib"] > hw["gpu_total_mib"] - reserve
                    ):
                        job.update(status="blocked", error="GPU smaller than measured reservation")
                        continue
                    if time.monotonic() - last_launch < policy["launch_spacing_seconds"]:
                        job["wait_reason"] = "startup spacing"
                        continue
                    reason = admission(job, running, admission_hw, policy)
                    job["wait_reason"] = reason
                    if reason:
                        continue
                    if Path(job["run_dir"]).exists():
                        job.update(
                            status="blocked",
                            error="Existing output retained; choose a fresh queue directory",
                        )
                        continue
                    log = runtime / f"{job['id']}.log"
                    env = {
                        **os.environ,
                        "OMP_NUM_THREADS": str(policy["threads_per_job"]),
                        "MKL_NUM_THREADS": str(policy["threads_per_job"]),
                        "CUDA_VISIBLE_DEVICES": str(policy["gpu"]) if job["gpu_mib"] else "",
                        "PYTHONUNBUFFERED": "1",
                    }
                    try:
                        logs[job["id"]] = log.open("a")
                        process = subprocess.Popen(
                            [
                                sys.executable,
                                "-m",
                                "molgap.scheduling",
                                "worker",
                                "--directory",
                                str(directory),
                                "--job",
                                job["id"],
                            ],
                            cwd=root,
                            env=env,
                            stdout=logs[job["id"]],
                            stderr=subprocess.STDOUT,
                            start_new_session=True,
                        )
                    except OSError as exc:
                        handle = logs.pop(job["id"], None)
                        if handle is not None:
                            handle.close()
                        job.update(
                            status="failed", error=f"Worker launch: {exc}", ended_at=utc_now()
                        )
                        print(f"{job['id']}: failed to start", flush=True)
                        continue
                    job.update(
                        status="running",
                        pid=process.pid,
                        process_identity=process_identity(process.pid),
                        host=socket.gethostname(),
                        started_at=utc_now(),
                        started_unix=time.time(),
                        log=str(log),
                        coverage="whole_pipeline_sampled",
                        interval_seconds=policy["sample_seconds"],
                    )
                    processes[job["id"]] = process
                    running.append(job)
                    last_launch = time.monotonic()
                    print(f"Started {job['id']}", flush=True)
                    write_json(
                        runtime / "state.json",
                        {"jobs": jobs, "hardware": hw, "updated_at": utc_now()},
                    )
                    break  # Refresh live observations before admitting the next job.
                publish_snapshot(runtime, directory, jobs, hw, queue_id)
                if stop[0] or not any(j["status"] in {"pending", "running"} for j in jobs):
                    break
                time.sleep(policy["sample_seconds"])
        finally:
            for identifier, process in processes.items():
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=5)
                except ProcessLookupError:
                    pass
                logs[identifier].close()
                next(j for j in jobs if j["id"] == identifier).update(
                    status="interrupted", ended_at=utc_now()
                )
            write_json(
                runtime / "state.json", {"jobs": jobs, "hardware": hw, "updated_at": utc_now()}
            )
            publish_snapshot(runtime, directory, jobs, hw, queue_id, terminal=True, stopped=stop[0])
            for signum, handler in previous.items():
                signal.signal(signum, handler)
        return 1 if any(j["status"] in {"failed", "blocked", "interrupted"} for j in jobs) else 0
