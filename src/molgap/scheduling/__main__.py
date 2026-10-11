"""Plan/run/status/stop a QM9 resource-aware queue."""

import argparse
import json
import os
import signal
from pathlib import Path

import yaml

from molgap.config import find_project_root
from molgap.scheduling.plan import create_plan
from molgap.scheduling.queue import run_queue, worker
from molgap.tracking.records import read_json
from molgap.tracking.resources import process_identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "run", "status", "stop", "worker"))
    parser.add_argument("--directory", type=Path, default=Path("artifacts/auto_training/formal"))
    parser.add_argument("--config", type=Path, default=Path("configs/auto_training.yaml"))
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--diagnostic-epochs", type=int)
    parser.add_argument("--job")
    args = parser.parse_args()
    if args.diagnostic_epochs is not None and args.diagnostic_epochs < 1:
        parser.error("Diagnostic epochs must be positive")
    if args.action == "worker":
        raise SystemExit(worker(args.directory, args.job))
    root = find_project_root(args.config.resolve())
    if args.action == "plan":
        result = create_plan(
            root,
            args.directory,
            yaml.safe_load(args.config.read_text()),
            args.models,
            args.seeds,
            args.diagnostic_epochs,
        )
        print(
            json.dumps(
                {
                    "jobs": len(result["jobs"]),
                    "paused_models": result["paused_models"],
                    "directory": str(args.directory.resolve()),
                },
                ensure_ascii=False,
            )
        )
    elif args.action == "run":
        if not (args.directory / "plan.json").exists():
            create_plan(
                root,
                args.directory,
                yaml.safe_load(args.config.read_text()),
                args.models,
                args.seeds,
                args.diagnostic_epochs,
            )
        elif args.models or args.seeds or args.diagnostic_epochs:
            parser.error(
                "An existing plan is immutable; use a fresh --directory to change selections"
            )
        raise SystemExit(run_queue(args.directory))
    elif args.action == "status":
        state = read_json(args.directory / "runtime/state.json") or read_json(
            args.directory / "plan.json"
        )
        for job in state.get("jobs", []):
            print(job["id"], job["status"], job.get("wait_reason") or job.get("error", ""))
    else:
        controller = read_json(args.directory / "runtime/controller.json")
        if (
            controller
            and controller["identity"] is not None
            and process_identity(controller["pid"]) == controller["identity"]
        ):
            os.kill(controller["pid"], signal.SIGTERM)
            print("Stop requested; the controller will terminate only its own worker groups")
        else:
            print("No live controller for this queue")


if __name__ == "__main__":
    main()
