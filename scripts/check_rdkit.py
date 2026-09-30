"""Check the project environment against the fixed RDKit-29 descriptor contract."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from importlib.metadata import version
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors

# Reference notebook selected_columns order, excluding BCUT2D_LOGPHI.
FEATURES = (
    "SMR_VSA7",
    "FractionCSP3",
    "SMR_VSA10",
    "BertzCT",
    "SlogP_VSA6",
    "BCUT2D_MRHI",
    "HallKierAlpha",
    "MolLogP",
    "BCUT2D_MWHI",
    "SMR_VSA5",
    "BalabanJ",
    "PEOE_VSA11",
    "BCUT2D_CHGHI",
    "SMR_VSA9",
    "PEOE_VSA2",
    "SlogP_VSA2",
    "BCUT2D_LOGPLOW",
    "VSA_EState2",
    "BCUT2D_MWLOW",
    "VSA_EState4",
    "VSA_EState5",
    "BCUT2D_MRLOW",
    "Kappa3",
    "fr_piperzine",
    "fr_piperdine",
    "fr_Ar_NH",
    "fr_aniline",
    "fr_imidazole",
    "fr_pyridine",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data/processed/qm9_full/random/seed3407",
    )
    parser.add_argument("--rows-per-split", type=int, default=100)
    parser.add_argument(
        "--check-models",
        action="store_true",
        help="Also smoke-test CPU XGBoost and Random Forest on the sampled descriptors",
    )
    args = parser.parse_args()
    if args.rows_per_split < 1:
        parser.error("--rows-per-split must be positive")
    missing = [name for name in FEATURES if not callable(getattr(Descriptors, name, None))]
    if missing:
        raise RuntimeError(f"Missing RDKit descriptors: {missing}")
    calculator = MoleculeDescriptors.MolecularDescriptorCalculator(FEATURES)
    checked = {}
    sampled = {}
    for split in ("train", "val", "test"):
        count = 0
        features, targets = [], []
        with (args.data_dir / f"{split}.csv").open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                mol = Chem.MolFromSmiles(row["smiles"])
                if mol is None:
                    raise ValueError(f"Invalid SMILES: {split}/{row['sample_id']}")
                # Direct calls propagate descriptor exceptions that the calculator may hide.
                direct = np.asarray([getattr(Descriptors, name)(mol) for name in FEATURES])
                values = np.asarray(calculator.CalcDescriptors(mol))
                if not np.isfinite(values).all() or not np.isfinite(direct).all():
                    raise ValueError(f"Nonfinite descriptors: {split}/{row['sample_id']}")
                np.testing.assert_allclose(values, direct, rtol=1e-10, atol=1e-12)
                features.append(values)
                targets.append(float(row["delta_e"]))
                count += 1
                if count >= args.rows_per_split:
                    break
        if count == 0:
            raise ValueError(f"Empty split: {split}")
        checked[split] = count
        sampled[split] = (np.asarray(features), np.asarray(targets))
    model_checks = {}
    packages = ["rdkit", "numpy", "pandas"]
    if args.check_models:
        from sklearn.ensemble import RandomForestRegressor
        from xgboost import XGBRegressor

        packages.extend(["scikit-learn", "xgboost"])
        for name, model in (
            (
                "xgboost",
                XGBRegressor(
                    n_estimators=3,
                    max_depth=2,
                    tree_method="hist",
                    device="cpu",
                    n_jobs=2,
                    random_state=3407,
                ),
            ),
            (
                "random_forest",
                RandomForestRegressor(
                    n_estimators=3,
                    max_depth=2,
                    n_jobs=2,
                    random_state=3407,
                ),
            ),
        ):
            model.fit(*sampled["train"])
            prediction = model.predict(sampled["val"][0])
            if prediction.shape != sampled["val"][1].shape or not np.isfinite(prediction).all():
                raise ValueError(f"Invalid predictions from {name}")
            model_checks[name] = "fit/predict passed (smoke test only)"
    print(
        json.dumps(
            {
                "python": sys.executable,
                "versions": {name: version(name) for name in packages},
                "data_dir": str(args.data_dir.resolve()),
                "features": list(FEATURES),
                "checked_rows": checked,
                "model_checks": model_checks,
                "status": "passed",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
