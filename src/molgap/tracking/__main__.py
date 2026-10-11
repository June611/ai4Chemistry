"""Collect local telemetry, supervise training, or backfill a Feishu Bitable."""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import os
import queue
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

from molgap.tracking.feishu import FeishuClient, FeishuError, Settings
from molgap.tracking.records import (
    SCHEMA,
    build_record,
    find_runs,
    read_json,
    utc_now,
    write_json,
)
from molgap.tracking.resources import process_identity, sample_resources


class SyncWorker:
    """Network requests never delay memory sampling or the training process."""

    def __init__(self, client: FeishuClient, run_dir: Path):
        self.client, self.run_dir = client, run_dir
        self.pending: queue.Queue = queue.Queue(maxsize=1)
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def submit(self, record: dict) -> None:
        with contextlib.suppress(queue.Empty):
            self.pending.get_nowait()
        self.pending.put_nowait(record)

    def _run(self) -> None:
        while (record := self.pending.get()) is not None:
            try:
                self.client.sync(self.run_dir, record)
                write_json(
                    self.run_dir / "tracking/sync_status.json",
                    {"status": "synced", "updated_at": utc_now()},
                )
            except Exception as exc:
                # API errors are sanitized; unexpected errors are identified by type only.
                message = str(exc) if isinstance(exc, FeishuError) else type(exc).__name__
                write_json(
                    self.run_dir / "tracking/sync_status.json",
                    {"status": "pending", "error": message, "updated_at": utc_now()},
                )
                print(f"Feishu sync pending: {message}", file=sys.stderr, flush=True)

    def close(self) -> None:
        self.pending.put(None)
        self.thread.join()


def monitor(
    run_dir: Path,
    pid: int,
    interval: float,
    sync_interval: float,
    client: FeishuClient | None,
    child: subprocess.Popen | None = None,
) -> int:
    directory = run_dir / "tracking"
    identity = process_identity(pid)
    if identity is None and child is None:
        raise ValueError("Training PID is not running")
    state = read_json(directory / "state.json")
    if not state or state.get("pid_identity") != identity:
        state = {
            "host": socket.gethostname(),
            "pid": pid,
            "pid_identity": identity,
            "status": "running",
            "started_at": utc_now(),
            "sample_count": 0,
            "peaks": {},
            "interval_seconds": interval,
            "coverage": "from_launch_sampled" if child else "attached_partial_sampled",
        }
        state["run_id"] = hashlib.sha256(f"{state['host']}:{run_dir}".encode()).hexdigest()[:32]
    else:
        state["coverage"] = "resumed_partial_sampled"
    state["interval_seconds"] = interval
    started = time.monotonic() - state.get("elapsed_seconds", 0)
    next_sync = 0.0
    worker = SyncWorker(client, run_dir) if client else None
    lock = None
    interrupted = False
    try:
        while True:
            tick = time.monotonic()
            alive = process_identity(pid) == identity if identity else False
            exit_code = child.poll() if child else None
            if child and exit_code is not None:
                alive = False
            state["elapsed_seconds"] = round(tick - started, 3)
            state["updated_at"] = utc_now()
            if alive:
                state["status"] = "running"
                latest = sample_resources(pid)
                state["latest"] = latest
                state["sample_count"] += 1
                for key in ("rss_mib", "pss_mib", "gpu_process_mib", "gpu_device_mib"):
                    if latest[key] is not None:
                        state["peaks"][key] = max(latest[key], state["peaks"].get(key, 0))
            else:
                state["status"] = (
                    "failed"
                    if exit_code is not None and exit_code != 0
                    else "completed"
                    if (run_dir / "metrics.json").is_file()
                    else "exited_unverified"
                )
                state["ended_at"] = utc_now()
                state["exit_code"] = exit_code
            # Do not precreate the output: ML trainers require a fresh run directory.
            if (run_dir / "config.yaml").is_file() or not alive:
                directory.mkdir(parents=True, exist_ok=True)
                if lock is None:
                    lock = (directory / ".monitor.lock").open("a")
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                write_json(directory / "state.json", state)
                if alive:
                    with (directory / "samples.jsonl").open("a", encoding="utf-8") as handle:
                        handle.write(
                            json.dumps(
                                {
                                    "at": state["updated_at"],
                                    "elapsed_seconds": state["elapsed_seconds"],
                                    **state["latest"],
                                },
                                allow_nan=False,
                            )
                            + "\n"
                        )
                try:
                    record = build_record(run_dir, state)
                except (OSError, ValueError):
                    if not alive:
                        raise
                else:
                    write_json(directory / "summary.json", record)
                    if worker and (tick >= next_sync or not alive):
                        worker.submit(record)
                        next_sync = tick + sync_interval
            if not alive:
                return exit_code if exit_code is not None else 0
            time.sleep(max(0, interval - (time.monotonic() - tick)))
    except KeyboardInterrupt:
        interrupted = True
        if child:
            # The launch path owns a separate process group, including Chemprop workers.
            with contextlib.suppress(ProcessLookupError):
                os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(child.pid, signal.SIGKILL)
                child.wait()
        state["status"] = "interrupted" if child else "monitor_stopped"
        state["updated_at"] = utc_now()
        write_json(directory / "state.json", state)
        record = build_record(run_dir, state)
        write_json(directory / "summary.json", record)
        if worker:
            worker.submit(record)
        return 130
    finally:
        if worker:
            worker.close()
        if lock:
            lock.close()
        if not interrupted:
            print(f"Local tracking saved: {directory}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("schema", help="Print the required fields without network access")
    sub.add_parser("check", help="Read-only authentication and field validation")
    sub.add_parser("init-table", help="Add missing fields to the configured existing table")
    sync = sub.add_parser("sync", help="Backfill completed or locally tracked runs")
    sync.add_argument("paths", nargs="+", type=Path)
    sync.add_argument("--dry-run", action="store_true", help="Print rows without network access")
    for action in ("watch", "run"):
        command = sub.add_parser(
            action,
            help="Attach to a PID"
            if action == "watch"
            else "Launch and monitor one training command",
        )
        command.add_argument("--run-dir", required=True, type=Path)
        command.add_argument("--interval", type=float, default=5)
        command.add_argument("--sync-interval", type=float, default=60)
        command.add_argument("--local-only", action="store_true")
        if action == "watch":
            command.add_argument("--pid", required=True, type=int)
        else:
            command.add_argument("command", nargs=argparse.REMAINDER, help="Command after --")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        if args.action == "schema":
            print(json.dumps(SCHEMA, indent=2, ensure_ascii=False))
            return
        if args.action == "sync":
            records = [(run, build_record(run)) for run in find_runs(args.paths)]
            # Validate all local artifacts before sending any rows.
            client = None if args.dry_run else FeishuClient(Settings.from_env())
            if client:
                client.ensure_schema()
            for run, record in records:
                if client:
                    write_json(run / "tracking/summary.json", record)
                    client.sync(run, record)
                else:
                    print(json.dumps(record, ensure_ascii=False, allow_nan=False))
            return
        if args.action in {"init-table", "check"}:
            client = FeishuClient(Settings.from_env())
            client.ensure_schema(create=args.action == "init-table")
            print("Feishu authentication and table fields verified")
            return
        if args.interval <= 0 or args.sync_interval <= 0:
            raise ValueError("Sampling and synchronization intervals must be positive")
        run_dir = args.run_dir.resolve()
        client = None if args.local_only else FeishuClient(Settings.from_env())
        if args.action == "watch":
            if args.pid <= 0:
                raise ValueError("PID must be positive")
            code = monitor(run_dir, args.pid, args.interval, args.sync_interval, client)
        else:
            command = args.command[1:] if args.command[:1] == ["--"] else args.command
            if not command:
                raise ValueError("Provide a training command after --")
            if run_dir.exists():
                raise ValueError(
                    "run requires a fresh output directory; use watch for existing runs"
                )
            child = subprocess.Popen(command, start_new_session=True)
            code = monitor(run_dir, child.pid, args.interval, args.sync_interval, client, child)
        raise SystemExit(code)
    except (FeishuError, OSError, ValueError) as exc:
        parser.exit(1, f"{exc}\n")


if __name__ == "__main__":
    main()
