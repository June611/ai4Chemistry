"""Verify that active training configs use only the project's fixed QM9 splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml
from audit_paired_seed_protocol import CHEMPROP, QM9, SEEDS, SEQUENCE


def audit(root: Path) -> dict:
    records = []
    configs = [root / "configs" / name for name in (*CHEMPROP, *SEQUENCE)]
    configs += [root / "configs/qm9_models" / name for name in (*QM9, "3dinfomax_pretrain.yaml")]
    for seed in SEEDS:
        suffix = "" if seed == 3407 else f"_seed{seed}"
        configs += [
            root / f"configs/{name}{suffix}.yaml"
            for name in ("training_mean", "random_forest_rdkit29", "xgboost_rdkit29")
        ]
    for path in configs:
        config = yaml.safe_load(path.read_text())
        data = config["data"]
        seed = int(data["split"]["seed"])
        checks = {
            "allowed_seed": seed in SEEDS,
            "qm9_split": data["processed_dir"] == f"data/processed/qm9_full/random/seed{seed}",
            "hartree": data["label_unit"] == "hartree",
            "gap_target": data["targets"][-1] == "delta_e",
            "qm9_geometry": "geometry" not in data
            or data["geometry"] == f"data/geometry/qm9_full_seed{seed}.npz",
        }
        if path.name == "3dinfomax.yaml":
            params = config["model"]["params"]
            checks["requires_qm9_pretraining"] = params["require_pretrained"]
            checks["no_external_weights_configured"] = params["pretrain_checkpoint"] is None
        if "pretraining" in config:
            checks["train_subset_within_qm9"] = 0 < config["pretraining"]["num_train"] <= 107038
            checks["paired_pretraining_seed"] = config["pretraining"]["seed"] == seed
        records.append(
            {"config": path.relative_to(root).as_posix(), "checks": checks, "split_seed": seed}
        )
    matrices = []
    for seed in SEEDS:
        suffix = "" if seed == 3407 else f"_seed{seed}"
        data_path = f"configs/data/qm9_full{suffix}.yaml"
        data_config = yaml.safe_load((root / data_path).read_text())
        source_checks = {
            "qm9_source": data_config["dataset"]["source"]["path"] == "data/raw/qm9_chn.csv",
            "qm9_name": data_config["dataset"]["name"] == "qm9_full",
            "paired_seed": data_config["split"]["seed"] == seed,
        }
        for stem in ("multiseed", "spe_affinity_ablation_multiseed"):
            path = root / f"configs/{stem}{suffix}.yaml"
            matrix = yaml.safe_load(path.read_text())
            matrices.append(
                {
                    "matrix": path.relative_to(root).as_posix(),
                    "checks": {
                        **source_checks,
                        "qm9_data_config": matrix["data_config"] == data_path,
                        "paired_seeds": matrix["split_seed"] == seed
                        and matrix["training_seeds"] == [seed],
                        "audited_models_only": all(
                            root / model["config"] in configs for model in matrix["models"]
                        ),
                    },
                }
            )
    passed = all(all(r["checks"].values()) for r in [*records, *matrices])
    return {
        "status": "pass" if passed else "fail",
        "scope": "active config/source routing; split contents verified by separate data audit",
        "dataset": "QM9 only",
        "configs": records,
        "matrices": matrices,
        "pretraining_policy": "QM9 train partition only; no QMugs/QMv4/external checkpoint",
        "historical_data": "QM9 CN/test subsets in the package are not active training inputs",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output", type=Path, default=Path("reports/qm9_training_scope_audit.json")
    )
    args = parser.parse_args()
    report = audit(args.root.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "dataset": report["dataset"]}))
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
