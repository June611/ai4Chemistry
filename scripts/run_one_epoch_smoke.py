#!/usr/bin/env python3
"""Run the current QM9 models sequentially through isolated one-epoch pipelines."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import shlex
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import yaml


def now() -> str:
    return datetime.now(UTC).isoformat()


def write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def models(root: Path) -> list[tuple[str, str, Path]]:
    result = [
        (name, "qm9", root / "configs/qm9_models" / f"{name}.yaml")
        for name in (
            "egnn",
            "schnet",
            "painn",
            "dimenet_pp",
            "esa",
            "torchmd_net",
            "cormorant",
            "lie_transformer",
            "alignn",
        )
    ]
    result.extend(
        [
            ("3dinfomax_pretrain", "pretrain", root / "configs/qm9_models/3dinfomax_pretrain.yaml"),
            ("3dinfomax", "qm9", root / "configs/qm9_models/3dinfomax.yaml"),
        ]
    )
    for matrix in ("multiseed.yaml", "spe_affinity_ablation_multiseed.yaml"):
        for item in yaml.safe_load((root / "configs" / matrix).read_text())["models"]:
            result.append((item["name"], item["trainer"], root / item["config"]))
    result.extend(
        (name, "ml", root / "configs" / f"{name}.yaml")
        for name in ("training_mean", "xgboost_rdkit29", "random_forest_rdkit29")
    )
    if len(result) != 23 or len({item[0] for item in result}) != 23:
        raise ValueError(
            "Expected 19 downstream neural models, 3 traditional models and pretraining"
        )
    return result


def prepare(
    root: Path, destination: Path, name: str, kind: str, source: Path
) -> tuple[Path, Path, dict]:
    config = copy.deepcopy(yaml.safe_load(source.read_text()))
    if int(config["data"]["split"]["seed"]) != 3407:
        raise ValueError(f"Unexpected split in {source}")
    output = destination / "runs"
    if kind == "pretrain":
        config["pretraining"].update(seed=3407, epochs=1, batch_size=256)
        config["output"]["root"] = str(output / name)
        run_dir = output / name / "seed3407"
    else:
        config["experiment"]["name"] = name
        # Some existing validators require this protocol ID; diagnostics are
        # explicitly distinguished by their separate root and smoke_suite field.
        config["experiment"]["protocol_id"] = "paired_split_gap_mae_v2"
        config["training"].update(
            seed=3407, split_seed=3407, seed_pairing="matched", run_mode="paired_split_seed"
        )
        if kind != "ml":
            config["training"].update(
                epochs=1, batch_size=256, accelerator="gpu" if kind != "qm9" else "cuda"
            )
        if kind == "chemprop" and config["model"]["backend"] == "chemprop_cli":
            # The official CLI requires epochs > warmup_epochs. The formal
            # two-epoch warmup cannot fit a diagnostic run of one epoch.
            config["training"]["warmup_epochs"] = 0
        config["output"]["root"] = str(output)
        run_dir = output / name / "seed3407"
        if name == "3dinfomax":
            config["model"]["params"]["pretrain_checkpoint"] = str(
                output / "3dinfomax_pretrain/seed3407/best_checkpoint.pt"
            )
    config["smoke_suite"] = {
        "purpose": "diagnostic_only",
        "full_fixed_split": True,
        "split_seed": 3407,
        "training_seed": 3407,
        "epochs": None if kind == "ml" else 1,
        "batch_size": None if kind == "ml" else 256,
        "source_config": str(source.relative_to(root)),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    }
    config_path = destination / "configs" / f"{name}.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    return config_path, run_dir, config


def commands(kind: str, config: Path) -> list[list[str]]:
    module = {
        "qm9": "molgap.qm9_models.train",
        "pretrain": "molgap.qm9_models.pretrain_infomax",
        "chemprop": "molgap.training.train",
        "transformer": "molgap.training.transformer",
        "spe_multitask": "molgap.training.spe_multitask",
        "ml": "molgap.training.ml",
    }[kind]
    result = [[sys.executable, "-m", module, "--config", str(config)]]
    if kind == "chemprop":
        result.extend(
            [
                [sys.executable, "-m", "molgap.training.predict", "--config", str(config)],
                [sys.executable, "-m", "molgap.training.evaluate", "--config", str(config)],
            ]
        )
    return result


def verify(root: Path, run_dir: Path, kind: str, config: dict) -> dict:
    if kind == "pretrain":
        import torch

        checkpoint = run_dir / "best_checkpoint.pt"
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        history = json.loads((run_dir / "history.json").read_text())
        if payload["epoch"] != 1 or len(history) != 1 or payload["training_seed"] != 3407:
            raise ValueError("Pretraining did not complete exactly one epoch with seed3407")
        if not math.isfinite(payload["best_validation_loss"]):
            raise ValueError("Nonfinite contrastive validation loss")
        return {
            "epochs_completed": 1,
            "validation_loss": payload["best_validation_loss"],
            "checkpoint": str(checkpoint),
        }
    from molgap.training.evaluate import evaluate_files

    truth = root / config["data"]["processed_dir"] / "test.csv"
    prediction = run_dir / "predictions/test_predictions.csv"
    metrics = evaluate_files(truth, prediction, config["data"]["targets"], "smiles", "sample_id")
    if metrics["rows"] != 13380:
        raise ValueError(f"Expected 13380 test predictions, received {metrics['rows']}")
    if not (run_dir / "metrics.json").is_file():
        raise FileNotFoundError("Pipeline did not produce metrics.json")
    checkpoints = [
        path
        for pattern in ("*.pt", "*.ckpt", "*.joblib", "model.json")
        for path in run_dir.rglob(pattern)
    ]
    if not checkpoints:
        raise FileNotFoundError("Pipeline did not save a model")
    timing = json.loads((run_dir / "training_timing.json").read_text())
    epoch_count = timing.get("epochs_completed")
    if kind != "ml" and epoch_count is not None and epoch_count != 1:
        raise ValueError(f"Expected exactly one epoch, received {epoch_count}")
    if kind == "chemprop" and epoch_count is None:
        import torch

        lightning = list(run_dir.rglob("*.ckpt"))
        if not lightning or any(
            torch.load(path, map_location="cpu", weights_only=False)["epoch"] != 0
            for path in lightning
        ):
            raise ValueError("Chemprop checkpoint does not establish a completed first epoch")
        epoch_count = 1
    return {
        "epochs_completed": epoch_count,
        "test_rows": metrics["rows"],
        "gap_mae_hartree": metrics["targets"]["delta_e"]["mae"],
    }


def publish(destination: Path, report: dict) -> None:
    write_json(destination / "status.json", report)
    rows = [
        "# QM9 one-epoch smoke results",
        "",
        "Seed/split 3407; neural batch 256, 1 epoch; full fixed split. "
        "Traditional models: one fit.",
        "Diagnostic results only; these do not replace the three-seed, 100-epoch experiments.",
        "",
        "| Model | Status | Seconds | Detail |",
        "|---|---|---:|---|",
    ]
    for item in report["models"]:
        detail = item.get("error", "")
        if item.get("checks", {}).get("gap_mae_hartree") is not None:
            detail = f"gap MAE {item['checks']['gap_mae_hartree']:.6g} Hartree; 13380 predictions"
        if item["kind"] == "ml":
            detail += "; epochs N/A"
        clean_detail = detail.replace("|", "/").replace("\n", " ")
        rows.append(
            f"| {item['model']} | {item['status']} | "
            f"{item.get('seconds', 0):.1f} | {clean_detail} |"
        )
    (destination / "results.md").write_text("\n".join(rows) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument(
        "--models", nargs="+", help="Run only these models in a new output directory"
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    destination = args.output.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("DGLBACKEND", "pytorch")
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("MKL_NUM_THREADS", "2")
    os.environ["PYTHONUNBUFFERED"] = "1"
    definitions = models(root)
    if args.models:
        selected = set(args.models)
        if "3dinfomax" in selected:
            selected.add("3dinfomax_pretrain")
        unknown = selected - {item[0] for item in definitions}
        if unknown:
            raise ValueError(f"Unknown model selection: {sorted(unknown)}")
        definitions = [item for item in definitions if item[0] in selected]
    report_path = destination / "status.json"
    if report_path.exists():
        report = json.loads(report_path.read_text())
    else:
        report = {
            "started": now(),
            "code_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=root, text=True
            ).strip(),
            "python": sys.executable,
            "purpose": "diagnostic_only",
            "seed": 3407,
            "epochs": 1,
            "batch_size": 256,
            "models": [],
        }
        for name, kind, source in definitions:
            config_path, run_dir, config = prepare(root, destination, name, kind, source)
            report["models"].append(
                {
                    "model": name,
                    "kind": kind,
                    "source": str(source.relative_to(root)),
                    "config": str(config_path),
                    "run_dir": str(run_dir),
                    "status": "pending",
                    "source_sha256": config["smoke_suite"]["source_sha256"],
                    "commands": commands(kind, config_path),
                }
            )
        publish(destination, report)
    if args.prepare_only:
        print(f"Prepared {len(report['models'])} sequential pipelines at {destination}")
        return
    if not report.get("finished"):
        monitor_log = (destination / "resource-monitor.log").open("a")
        monitor = subprocess.Popen(
            [
                sys.executable,
                str(root / "scripts/monitor_smoke_resources.py"),
                "--output",
                str(destination),
                "--runner-pid",
                str(os.getpid()),
            ],
            stdout=monitor_log,
            stderr=subprocess.STDOUT,
        )
        report["resource_monitor_pid"] = monitor.pid
        publish(destination, report)
        # Let the monitor observe all models as pending before training starts.
        ready_deadline = time.monotonic() + 10
        while not (destination / "resources.csv").exists():
            if monitor.poll() is not None or time.monotonic() > ready_deadline:
                raise RuntimeError("Resource monitor did not become ready; see its log")
            time.sleep(0.05)
    for entry in report["models"]:
        if entry["status"] in {"success", "failed", "blocked"}:
            continue
        if entry["status"] == "running":
            raise RuntimeError(
                f"Interrupted run needs inspection before resuming: {entry['model']}"
            )
        if entry["model"] == "3dinfomax":
            pretrain = next(item for item in report["models"] if item["kind"] == "pretrain")
            if pretrain["status"] != "success":
                entry.update(status="blocked", error="QM9 pretraining prerequisite failed")
                publish(destination, report)
                continue
        entry.update(status="running", started=now())
        log_path = destination / "logs" / f"{entry['model']}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        entry["log"] = str(log_path)
        publish(destination, report)
        print(f"[{now()}] START {entry['model']}", flush=True)
        started = time.perf_counter()
        try:
            with log_path.open("w", buffering=1) as log:
                for command in entry["commands"]:
                    log.write(f"\nCOMMAND: {shlex.join(command)}\n")
                    result = subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT)
                    entry["returncode"] = result.returncode
                    if result.returncode:
                        raise RuntimeError(f"Pipeline command exited with code {result.returncode}")
            config = yaml.safe_load(Path(entry["config"]).read_text())
            entry["checks"] = verify(root, Path(entry["run_dir"]), entry["kind"], config)
            entry["status"] = "success"
        except Exception as error:
            entry.update(status="failed", error=str(error))
            lines = log_path.read_text(errors="replace").splitlines()
            entry["log_tail"] = lines[-35:]
            useful = [
                line
                for line in lines
                if any(
                    token in line
                    for token in ("Error:", "Exception:", "out of memory", "Nonfinite")
                )
            ]
            if useful:
                entry["error"] += "; " + useful[-1][-900:]
        entry.update(seconds=time.perf_counter() - started, finished=now())
        publish(destination, report)
        print(
            f"[{now()}] {entry['status'].upper()} {entry['model']} ({entry['seconds']:.1f}s)",
            flush=True,
        )
    report["finished"] = now()
    publish(destination, report)
    failed = [item["model"] for item in report["models"] if item["status"] != "success"]
    print(json.dumps({"success": len(report["models"]) - len(failed), "failed_or_blocked": failed}))


if __name__ == "__main__":
    main()
