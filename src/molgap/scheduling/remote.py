"""Start the server queue in the background, then synchronize via the local CLI."""

import argparse
import csv
import shlex
import subprocess
import sys
from pathlib import Path

import yaml

from molgap.config import find_project_root
from molgap.qm9_models.policy import paused_models
from molgap.tracking.bridge import ssh_args
from molgap.tracking.feishu import FeishuError
from molgap.tracking.lark_cli import LarkCLIClient
from molgap.tracking.records import read_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", default="artifacts/auto_training/formal")
    parser.add_argument("--config", default="configs/auto_training.yaml")
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--diagnostic-epochs", type=int)
    args = parser.parse_args()
    for path in (args.directory, args.config):
        if Path(path).is_absolute() or ".." in Path(path).parts:
            parser.error("Use repository-relative queue and policy paths")
    if args.seeds and (
        len(set(args.seeds)) != len(args.seeds) or set(args.seeds) - {3407, 42, 2026}
    ):
        parser.error("Use distinct existing split seeds: 3407,42,2026")
    if args.diagnostic_epochs is not None and args.diagnostic_epochs < 1:
        parser.error("Diagnostic epochs must be positive")
    policy = yaml.safe_load(Path(args.config).read_text())
    root = find_project_root(Path(args.config).resolve())
    if args.models:
        with (root / policy["resource_csv"]).open() as stream:
            available = {row["model"] for row in csv.DictReader(stream)} | {"3dinfomax_pretrain"}
        forbidden = (set(args.models) - available) | (set(args.models) & set(paused_models(root)))
        if forbidden:
            parser.error(f"Unknown or paused models: {sorted(forbidden)}")
    # Catch a missing destination or CLI authorization before starting expensive jobs.
    destination = read_json(Path("configs/feishu_training.json"))
    if not all(destination.get(k) for k in ("base_token", "table_id", "hardware_table_id")):
        parser.error("Configure configs/feishu_training.json before remote training")
    try:
        LarkCLIClient(destination, Path("artifacts/auto_training/feishu_preflight")).ensure_schema()
    except FeishuError as exc:
        parser.error(str(exc))
    remote = policy["remote"]
    command = [
        "bash",
        "scripts/train_all_auto.sh",
        "--directory",
        args.directory,
        "--config",
        args.config,
    ]
    if args.models:
        command += ["--models", *args.models]
    if args.seeds:
        command += ["--seeds", *map(str, args.seeds)]
    if args.diagnostic_epochs:
        command += ["--diagnostic-epochs", str(args.diagnostic_epochs)]
    log = args.directory + "/controller.log"
    resumed = [
        "bash",
        "scripts/train_all_auto.sh",
        "--directory",
        args.directory,
        "--config",
        args.config,
    ]
    script = (
        f"cd {shlex.quote(remote['root'])} && mkdir -p {shlex.quote(args.directory)} && "
        f"if test -f {shlex.quote(args.directory + '/plan.json')}; then "
        f"nohup {shlex.join(resumed)} >> {shlex.quote(log)} 2>&1 < /dev/null & "
        f"else nohup {shlex.join(command)} >> {shlex.quote(log)} 2>&1 < /dev/null & fi"
    )
    subprocess.run([*ssh_args(remote), script], check=True, timeout=30)
    print(
        "Server queue requested. Closing the local bridge does not stop server training.",
        flush=True,
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "molgap.tracking.bridge",
            "--remote",
            "--directory",
            args.directory,
            "--policy",
            args.config,
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
