"""Check configured inputs, source dependencies and model construction without training."""

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main():
    from molgap.modular.config import load
    from molgap.modular.data import load_labels
    from molgap.modular.train import train

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=int, choices=[1, 2], default=1)
    parser.add_argument("--model", action="append")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--data-only", action="store_true")
    parser.add_argument(
        "--report", type=Path, default=Path("artifacts/qm9_epoch200/preflight.json")
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.config:
        paths = [args.config]
    else:
        rows = list(
            csv.DictReader(
                (root / "docs/features/qm9-epoch200-modular-study/training_queue.csv").open()
            )
        )
        paths = [
            root / r["config"]
            for r in rows
            if int(r["stage"]) == args.stage and (not args.model or r["model"] in args.model)
        ]
    results = []
    for path in paths:
        try:
            if args.data_only:
                cfg, project = load(path)
                _, _, identity = load_labels(cfg, project)
                result = dict(data_identity=identity)
            else:
                result = train(path, dry_run=True)
            results.append(dict(config=str(path), ok=True, result=result))
        except Exception as exc:
            results.append(
                dict(config=str(path), ok=False, error_type=type(exc).__name__, error=str(exc))
            )
        print(json.dumps({k: v for k, v in results[-1].items() if k != "result"}), flush=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(results, indent=2) + "\n")
    if not paths:
        raise ValueError("No matching jobs")
    if any(not r["ok"] for r in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
