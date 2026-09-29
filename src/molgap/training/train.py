"""Run a configured Chemprop baseline without modifying Chemprop."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from molgap.config import load_config, resolve_path, run_directory
from molgap.training.commands import build_train_command
from molgap.utils.files import write_command_record, write_run_metadata


def train(config_path: str | Path, dry_run: bool = False, overwrite: bool = False) -> Any:
    """Validate inputs, record the run, and execute Chemprop."""
    config_path = Path(config_path).resolve()
    config, root = load_config(config_path)
    phase = int(config["experiment"]["phase"])
    if phase == 3:
        from molgap.training.phase3 import train_phase3

        return train_phase3(config_path, dry_run=dry_run, overwrite=overwrite)

    command = build_train_command(config, root)
    if dry_run:
        print(json.dumps(command, indent=2))
        return command

    processed_dir = resolve_path(root, config["data"]["processed_dir"])
    missing = [
        processed_dir / f"{name}.csv"
        for name in ("train", "val", "test")
        if not (processed_dir / f"{name}.csv").is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "Processed splits are missing. Run molgap-prepare first. Missing: "
            + ", ".join(str(path) for path in missing)
        )

    run_dir = run_directory(config, root)
    artifact_paths = (run_dir / "chemprop", run_dir / "train.log", run_dir / "command.json")
    if any(path.exists() for path in artifact_paths) and not overwrite:
        raise FileExistsError(
            f"Training artifacts already exist in {run_dir}. Use a new experiment name/seed "
            "or pass --overwrite."
        )
    if overwrite and (run_dir / "chemprop").exists():
        shutil.rmtree(run_dir / "chemprop")
    run_dir.mkdir(parents=True, exist_ok=True)
    write_run_metadata(run_dir, config, config_path, root)
    write_command_record(run_dir / "command.json", command, {"kind": "train"})
    started = time.perf_counter()
    subprocess.run(command, cwd=root, check=True)
    timing = {"fit_pipeline_seconds": time.perf_counter() - started}
    with (run_dir / "training_timing.json").open("w", encoding="utf-8") as handle:
        json.dump(timing, handle, indent=2)
        handle.write("\n")
    return command


def build_parser(default_config: str = "configs/baseline.yaml") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=default_config, help="Experiment YAML file")
    parser.add_argument("--dry-run", action="store_true", help="Print argv after validation")
    parser.add_argument("--overwrite", action="store_true", help="Replace this run's model outputs")
    return parser


def main(default_config: str = "configs/baseline.yaml") -> None:
    args = build_parser(default_config).parse_args()
    train(args.config, args.dry_run, args.overwrite)


if __name__ == "__main__":
    main()
