"""Local Lark CLI bridge for local or remote queue snapshots and prediction attachments."""

import argparse
import hashlib
import json
import shlex
import subprocess
import time
from pathlib import Path

import yaml

from molgap.tracking.feishu import FeishuError
from molgap.tracking.lark_cli import LarkCLIClient
from molgap.tracking.records import read_json, write_json


def ssh_args(remote):
    return [
        "ssh",
        "-F",
        "/dev/null",
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        "UserKnownHostsFile=/tmp/chempro-qm9-known_hosts",
        "-o",
        "ControlMaster=auto",
        "-o",
        "ControlPersist=8h",
        "-o",
        "ControlPath=/tmp/chempro-qm9-v4-%C",
        "-o",
        "ConnectTimeout=15",
        "-o",
        "ServerAliveInterval=15",
        "-p",
        str(remote["port"]),
        remote["host"],
    ]


def remote_file(remote, path):
    # Paths originate from the queue, but still quote them and constrain to the project.
    resolved = Path(path).resolve()
    if not resolved.is_relative_to(Path(remote["root"]).resolve()):
        raise ValueError("Remote artifact must be inside the configured project")
    return subprocess.check_output(
        [*ssh_args(remote), "cat -- " + shlex.quote(str(resolved))], timeout=60
    )


def sync_report(report, client, cache, remote=None):
    failures = []
    for record in report["records"]:
        try:
            record_id, receipt = client.sync_record(record)
            if record["fields"]["状态"] != "completed" or record.get("kind") == "pretrain":
                continue
            path = Path(record["run_dir"]) / "predictions/test_predictions.csv"
            if remote:
                mirrored = (
                    cache
                    / "predictions"
                    / (hashlib.sha256(record["run_id"].encode()).hexdigest()[:16] + ".csv")
                )
                if not mirrored.exists():
                    mirrored.parent.mkdir(parents=True, exist_ok=True)
                    data = remote_file(remote, path)
                    temporary = mirrored.with_suffix(".tmp")
                    temporary.write_bytes(data)
                    temporary.replace(mirrored)
                path = mirrored
            relative = path.resolve().relative_to(Path.cwd().resolve())
            client.attach_prediction(record_id, receipt, relative)
        except (FeishuError, OSError, ValueError, subprocess.SubprocessError) as exc:
            failures.append({"run_id": record["run_id"], "error": str(exc)})
    hw = report.get("hardware", {})
    if hw:
        try:
            client.upsert(
                {
                    "采样ID": report["queue_directory"] + ":" + hw["sampled_at"][:16],
                    "采样时间": hw["sampled_at"],
                    "队列": report["queue_directory"],
                    "GPU型号": hw.get("gpu_name"),
                    "GPU利用率_pct": hw.get("gpu_util_pct"),
                    "整卡显存_MiB": hw.get("gpu_used_mib"),
                    "系统可用内存_MiB": hw.get("ram_available_mib"),
                    "数据盘剩余_GiB": hw.get("disk_free_gib"),
                },
                "采样ID",
                client.destination["hardware_table_id"],
            )
        except (FeishuError, OSError, subprocess.SubprocessError) as exc:
            failures.append({"run_id": "hardware", "error": str(exc)})
    write_json(
        cache / "sync_status.json",
        {
            "updated_at": report.get("updated_at"),
            "failures": failures,
            "records": len(report["records"]),
        },
    )
    return failures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path("artifacts/auto_training/formal"))
    parser.add_argument("--destination", type=Path, default=Path("configs/feishu_training.json"))
    parser.add_argument("--policy", type=Path, default=Path("configs/auto_training.yaml"))
    parser.add_argument("--remote", action="store_true")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--interval", type=float, default=60)
    args = parser.parse_args()
    if args.interval < 5:
        parser.error("Use a sync interval >=5 seconds")
    remote = yaml.safe_load(args.policy.read_text())["remote"] if args.remote else None
    cache = Path("artifacts/auto_training/feishu_bridge") / args.directory.name
    client = LarkCLIClient(read_json(args.destination), cache / "receipts")
    while True:
        try:
            if remote:
                path = Path(remote["root"]) / args.directory / "runtime/report.json"
                report = json.loads(remote_file(remote, path))
            else:
                report = read_json(args.directory / "runtime/report.json")
            if not report:
                raise FileNotFoundError("Queue report not ready")
            write_json(cache / "report.json", report)
            failures = sync_report(report, client, cache, remote)
            print(
                f"Synced snapshot: {len(report['records'])} runs, {len(failures)} errors",
                flush=True,
            )
            if args.once or (report.get("terminal") and not failures):
                raise SystemExit(1 if failures else 0)
        except (FeishuError, OSError, ValueError, subprocess.SubprocessError) as exc:
            print(f"Sync deferred; local/server artifacts retained: {exc}", flush=True)
            if args.once:
                raise SystemExit(1) from None
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
