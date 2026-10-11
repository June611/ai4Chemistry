#!/usr/bin/env python3
"""Audit the approved three-run paired split/training-seed protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

PROTOCOL_ID = "paired_split_gap_mae_v2"
SEEDS = (3407, 42, 2026)

CHEMPROP = ("baseline.yaml", "multitask.yaml", "consistency.yaml")
SEQUENCE = (
    "transformer.yaml",
    "spe_transformer.yaml",
    "spe_multitask.yaml",
    "spe_multitask_affinity.yaml",
    "spe_multitask_consistency.yaml",
    "spe_multitask_affinity_consistency.yaml",
)
QM9 = (
    "egnn.yaml",
    "schnet.yaml",
    "painn.yaml",
    "dimenet_pp.yaml",
    "esa.yaml",
    "torchmd_net.yaml",
    "cormorant.yaml",
    "lie_transformer.yaml",
    "alignn.yaml",
    "3dinfomax.yaml",
)


def _yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"YAML root is not a mapping: {path}")
    return payload


def _record(checks: dict[str, bool], **details: Any) -> dict[str, Any]:
    return {"status": "pass" if all(checks.values()) else "fail", "checks": checks, **details}


def audit(root: Path) -> dict[str, Any]:
    neural: dict[str, Any] = {}
    for name in CHEMPROP:
        config = _yaml(root / "configs" / name)
        training = config["training"]
        neural[name] = _record(
            {
                "batch_size_256": int(training["batch_size"]) == 256,
                "epochs_100": int(training["epochs"]) == 100,
                "patience_20": int(training["patience"]) == 20,
                "gap_mae_selection": training.get("selection_metric") == "gap_mae",
                "min_delta_hartree": training.get("early_stopping_min_delta") == 1e-5,
                "single_device": training["devices"] == 1,
            }
        )
    for name in SEQUENCE:
        config = _yaml(root / "configs" / name)
        training = config["training"]
        neural[name] = _record(
            {
                "batch_size_256": int(training["batch_size"]) == 256,
                "epochs_100": int(training["epochs"]) == 100,
                "patience_20": int(training["early_stopping_patience"]) == 20,
                "min_delta_hartree": training["early_stopping_min_delta"] == 1e-5,
                "single_device": training["devices"] == 1,
            }
        )
    for name in QM9:
        config = _yaml(root / "configs/qm9_models" / name)
        training = config["training"]
        neural[f"qm9_models/{name}"] = _record(
            {
                "batch_size_256": int(training["batch_size"]) == 256,
                "epochs_100": int(training["epochs"]) == 100,
                "patience_20": int(training["early_stopping"]["patience"]) == 20,
                "gap_mae_selection": training["early_stopping"]["monitor"] == "val_mae",
                "min_delta_hartree": training["early_stopping"]["min_delta"] == 1e-5,
                "preserved_loss": training["loss"] == "mae",
                "preserved_scaler": training["target_scaling"] == "mad",
                "isolated_output_root": config["output"]["root"] == "outputs/paired_gap_mae_v2",
            }
        )
    pretrain = _yaml(root / "configs/qm9_models/3dinfomax_pretrain.yaml")
    settings = pretrain["pretraining"]
    pretrain_result = _record(
        {
            "batch_size_256": int(settings["batch_size"]) == 256,
            "epochs_100": int(settings["epochs"]) == 100,
            "patience_20": int(settings["patience"]) == 20,
            "lr_decay_before_early_stop": settings["lr_patience"] < settings["patience"],
            "isolated_output_root": pretrain["output"]["root"]
            == "outputs/paired_gap_mae_v2/3dinfomax_pretrain",
        }
    )

    matrix_paths = {
        3407: (
            root / "configs/multiseed.yaml",
            root / "configs/spe_affinity_ablation_multiseed.yaml",
        ),
        42: (
            root / "configs/multiseed_seed42.yaml",
            root / "configs/spe_affinity_ablation_multiseed_seed42.yaml",
        ),
        2026: (
            root / "configs/multiseed_seed2026.yaml",
            root / "configs/spe_affinity_ablation_multiseed_seed2026.yaml",
        ),
    }
    matrices: dict[str, Any] = {}
    planned_neural_runs = 0
    for seed, paths in matrix_paths.items():
        for path in paths:
            config = _yaml(path)
            key = path.relative_to(root).as_posix()
            model_count = len(config["models"])
            planned_neural_runs += model_count
            matrices[key] = _record(
                {
                    "protocol_id": config.get("protocol_id") == PROTOCOL_ID,
                    "matched_pairing": config.get("seed_pairing") == "matched",
                    "split_seed": int(config["split_seed"]) == seed,
                    "one_matching_training_seed": list(map(int, config["training_seeds"]))
                    == [seed],
                    "no_legacy_seeds": not ({7, 123} & set(map(int, config["training_seeds"]))),
                },
                models=model_count,
            )

    traditional: dict[str, Any] = {}
    for seed in SEEDS:
        suffix = "" if seed == 3407 else f"_seed{seed}"
        for model in ("xgboost_rdkit29", "random_forest_rdkit29", "training_mean"):
            path = root / "configs" / f"{model}{suffix}.yaml"
            config = _yaml(path)
            training = config["training"]
            split_seed = int(config["data"]["split"]["seed"])
            traditional[path.name] = _record(
                {
                    "protocol_id": config["experiment"].get("protocol_id") == PROTOCOL_ID,
                    "split_seed": split_seed == seed,
                    "training_seed": int(training["seed"]) == seed,
                    "matched_pairing": training.get("seed_pairing") == "matched",
                    "run_mode": training.get("run_mode") == "paired_split_seed",
                    "processed_dir": str(config["data"]["processed_dir"]).endswith(f"/seed{seed}"),
                    "isolated_output_root": config["output"]["root"] == "outputs/paired_gap_mae_v2",
                },
                budget={
                    "batch_size": "N/A",
                    "epochs": "N/A",
                    "early_stopping_patience": "N/A",
                },
            )

    excluded = (
        "configs/experiments/exp003_consistency_0.yaml",
        "configs/experiments/exp004_consistency_01.yaml",
        "configs/qm9_models/egnn_test.yaml",
        "configs/spe_transformer_10epoch.yaml",
    )
    report = {
        "protocol_id": PROTOCOL_ID,
        "paired_seeds": list(SEEDS),
        "budget": {"batch_size": 256, "epochs": 100, "early_stopping_patience": 20},
        "selection": {
            "target": "delta_e",
            "metric": "mae",
            "unit": "hartree",
            "mode": "min",
            "min_delta": 1e-5,
            "checkpoint": "strict minimum",
            "early_stop": "min_delta improvement",
            "loss_scaler_policy": "preserve method recipes",
            "exceptions": ["contrastive pretraining", "non-iterative traditional baselines"],
        },
        "neural_configs": neural,
        "3dinfomax_pretraining": pretrain_result,
        "matrices": matrices,
        "traditional_configs": traditional,
        "excluded_configs": list(excluded),
        "planned_runs": {
            "chemprop_transformer_spe": planned_neural_runs,
            "qm9_downstream": len(QM9) * len(SEEDS),
            "traditional": 3 * len(SEEDS),
            "evaluated_total": planned_neural_runs + len(QM9) * len(SEEDS) + 3 * len(SEEDS),
            "3dinfomax_pretraining_auxiliary": len(SEEDS),
        },
    }
    sections = (neural, matrices, traditional)
    report["status"] = (
        "pass"
        if all(item["status"] == "pass" for section in sections for item in section.values())
        and pretrain_result["status"] == "pass"
        and report["planned_runs"]["evaluated_total"] == 66
        else "fail"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/paired_seed_100epoch_protocol_audit.json"),
    )
    args = parser.parse_args()
    result = audit(args.root.resolve())
    payload = json.dumps(result, indent=2) + "\n"
    output = args.output if args.output.is_absolute() else args.root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    if result["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
