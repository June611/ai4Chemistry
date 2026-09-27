from pathlib import Path

import pandas as pd
import pytest

from molgap.training.evaluate import evaluate_files


def test_multitask_evaluation_reports_physical_consistency(tmp_path: Path) -> None:
    truth_path = tmp_path / "truth.csv"
    prediction_path = tmp_path / "predictions.csv"
    pd.DataFrame(
        {
            "smiles": ["CC", "CO"],
            "homo": [-6.0, -5.0],
            "lumo": [-1.0, -1.0],
            "delta_e": [5.0, 4.0],
        }
    ).to_csv(truth_path, index=False)
    pd.DataFrame(
        {
            "smiles": ["CC", "CO"],
            "homo": [-6.0, -5.1],
            "lumo": [-1.0, -1.0],
            "delta_e": [5.0, 4.0],
        }
    ).to_csv(prediction_path, index=False)

    metrics = evaluate_files(
        truth_path,
        prediction_path,
        ["homo", "lumo", "delta_e"],
        "smiles",
    )

    assert metrics["targets"]["delta_e"]["mae"] == 0.0
    assert metrics["consistency"]["mae"] == pytest.approx(0.05)


def test_evaluation_aligns_rows_by_sample_id(tmp_path: Path) -> None:
    truth_path = tmp_path / "truth.csv"
    prediction_path = tmp_path / "predictions.csv"
    pd.DataFrame({"sample_id": ["a", "b"], "smiles": ["CC", "CO"], "delta_e": [1.0, 2.0]}).to_csv(
        truth_path, index=False
    )
    pd.DataFrame({"sample_id": ["b", "a"], "smiles": ["CO", "CC"], "delta_e": [2.0, 1.0]}).to_csv(
        prediction_path, index=False
    )

    metrics = evaluate_files(truth_path, prediction_path, ["delta_e"], "smiles")

    assert metrics["targets"]["delta_e"]["mae"] == 0.0
