"""Exercise resolved three-seed plans on local full data, without formal training."""

from __future__ import annotations

import argparse
import contextlib
import gc
import io
import json
from pathlib import Path

from molgap.experiments.multiseed import expand_matrix
from molgap.qm9_models.train import run as qm9_run
from molgap.training.ml import train_ml
from molgap.training.spe_multitask import train_spe_multitask
from molgap.training.train import train
from molgap.training.transformer import train_transformer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--infomax-checkpoint-template")
    parser.add_argument("--output", type=Path, default=Path("reports/gap_mae_runtime_audit.json"))
    args = parser.parse_args()
    records = []

    def record(model, seed, function):
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                plan = function()
            if isinstance(plan, list):
                assert plan[plan.index("--tracking-metric") + 1] == "gap_mae"
                assert plan[plan.index("--min-delta") + 1] == "1e-05"
                assert plan[plan.index("--devices") + 1] == "1"
                assert plan[plan.index("--batch-size") + 1] == "256"
                assert plan[plan.index("--epochs") + 1] == "100"
                assert plan[plan.index("--patience") + 1] == "20"
                assert plan[plan.index("--pytorch-seed") + 1] == str(seed)
            elif "selection" in plan:
                selection = plan["selection"]
                assert (selection["target"], selection["metric"], selection["unit"]) == (
                    "delta_e",
                    "mae",
                    "hartree",
                )
                assert selection["patience"] == 20 and selection["min_delta"] == 1e-5
                training = plan.get("training", {})
                budget = plan.get("budget", training)
                assert budget["batch_size"] == 256 and budget["epochs"] == 100
            else:
                assert plan["split_seed"] == plan["training_seed"] == seed
            records.append(
                {
                    "model": model,
                    "seed": seed,
                    "status": "pass",
                    "plan": plan,
                    "data_scope": (
                        "command/plan only; input data covered by split audit"
                        if model.startswith("chemprop_")
                        else "full-data loader and runtime plan"
                    ),
                }
            )
        except Exception as error:
            records.append(
                {
                    "model": model,
                    "seed": seed,
                    "status": "fail",
                    "error": f"{type(error).__name__}: {error}",
                }
            )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"records": records}, indent=2) + "\n")
        print(f"{len(records):02d} {model} seed{seed}: {records[-1]['status']}", flush=True)
        gc.collect()

    for seed in (3407, 42, 2026):
        suffix = "" if seed == 3407 else f"_seed{seed}"
        for stem in ("multiseed", "spe_affinity_ablation_multiseed"):
            matrix = Path(f"configs/{stem}{suffix}.yaml")
            _, _, _, _, specs = expand_matrix(matrix)
            for spec in specs:
                runners = {
                    "chemprop": train,
                    "transformer": train_transformer,
                    "spe_multitask": train_spe_multitask,
                }
                record(
                    spec.model,
                    seed,
                    lambda s=spec, fn=runners[spec.trainer]: fn(s.config_path, dry_run=True),
                )
        for model in (
            "egnn",
            "schnet",
            "painn",
            "dimenet_pp",
            "esa",
            "torchmd_net",
            "cormorant",
            "lie_transformer",
            "alignn",
            "3dinfomax",
        ):
            checkpoint = None
            if model == "3dinfomax":
                if not args.infomax_checkpoint_template:
                    records.append(
                        {
                            "model": model,
                            "seed": seed,
                            "status": "pending",
                            "reason": "Split-matched pretraining checkpoint required",
                        }
                    )
                    continue
                checkpoint = Path(
                    args.infomax_checkpoint_template.format(split_seed=seed, seed=seed)
                )
            record(
                model,
                seed,
                lambda m=model, cp=checkpoint, s=seed: qm9_run(
                    Path(f"configs/qm9_models/{m}.yaml"),
                    dry_run=True,
                    seed_override=s,
                    split_seed_override=s,
                    pretrain_checkpoint=cp,
                ),
            )
        for model in ("xgboost_rdkit29", "random_forest_rdkit29", "training_mean"):
            record(
                model,
                seed,
                lambda m=model, end=suffix: train_ml(Path(f"configs/{m}{end}.yaml"), dry_run=True),
            )
    counts = {
        status: sum(r["status"] == status for r in records)
        for status in ("pass", "pending", "fail")
    }
    payload = {
        "protocol_id": "paired_split_gap_mae_v2",
        "scope": "Resolved plans; Chemprop command-only, other methods full-data dry-run",
        "counts": counts,
        "records": records,
        "status": "fail" if counts["fail"] else "pending" if counts["pending"] else "pass",
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(counts), flush=True)
    if counts["fail"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
