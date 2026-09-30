import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from rdkit import Chem
from rdkit.Chem import Descriptors

from molgap.data.rdkit_features import FEATURES, descriptor_matrix
from molgap.experiments.summary import find_metric_files, load_records, validate_expected_seeds
from molgap.training.evaluate import evaluate_files
from molgap.training.ml import load_estimator, load_ml_config, train_ml
from molgap.utils.files import sha256


@pytest.fixture
def project(tmp_path: Path) -> Path:
    (tmp_path / "pyproject.toml").write_text("[project]\nname='tiny'\nversion='0.0.0'\n")
    data = tmp_path / "data"
    data.mkdir()
    molecules = {
        "train": ["C", "CC", "CCC", "CO", "CN", "CCO"],
        "val": ["CCN", "CCCO"],
        "test": ["CCCN", "CCCC"],
    }
    for split, smiles in molecules.items():
        pd.DataFrame(
            {
                "sample_id": [f"{split}_{i}" for i in range(len(smiles))],
                "smiles": smiles,
                "delta_e": np.linspace(0.12, 0.3, len(smiles)),
                "homo": np.nan,
                "lumo": np.inf,
            }
        ).to_csv(data / f"{split}.csv", index=False)
    return tmp_path


def make_config(project: Path, backend: str) -> Path:
    filename = "training_mean.yaml" if backend == "training_mean" else f"{backend}_rdkit29.yaml"
    reference = Path(__file__).resolve().parents[1] / "configs" / filename
    config = yaml.safe_load(reference.read_text())
    config["data"]["processed_dir"] = "data"
    if backend != "training_mean":
        config["model"]["params"].update(n_estimators=3, max_depth=2)
    config["training"]["n_jobs"] = 1
    path = project / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


@pytest.mark.parametrize("backend", ["xgboost", "random_forest"])
def test_ml_single_fit_artifacts_and_reload(project: Path, backend: str, monkeypatch) -> None:
    from molgap.training import ml

    config_path = make_config(project, backend)
    hashes = {s: sha256(project / "data" / f"{s}.csv") for s in ("train", "val", "test")}
    estimator = ml.XGBRegressor if backend == "xgboost" else ml.RandomForestRegressor
    original_fit = estimator.fit
    fits = []

    def track_fit(self, x, y, *args, **kwargs):
        fits.append((x.copy(), y.copy()))
        return original_fit(self, x, y, *args, **kwargs)

    monkeypatch.setattr(estimator, "fit", track_fit)
    result = train_ml(config_path)
    run_dir = Path(result["run_dir"])
    assert run_dir == project / "outputs" / f"{backend}_rdkit29" / "seed3407"
    assert len(fits) == 1
    train = pd.read_csv(project / "data/train.csv", float_precision="round_trip")
    expected_x, _ = descriptor_matrix(train.smiles.tolist(), train.sample_id.tolist())
    np.testing.assert_array_equal(fits[0][0], expected_x)
    np.testing.assert_array_equal(fits[0][1], train.delta_e.to_numpy())
    assert fits[0][0].shape == (6, 29)
    assert np.isfinite(fits[0][0]).all()  # Extra homo/lumo columns are never inputs.
    for split, digest in hashes.items():
        assert sha256(project / "data" / f"{split}.csv") == digest
    for name in (
        "config.yaml",
        "environment.json",
        "feature_names.json",
        "normalization.json",
        "training_timing.json",
        "metrics.json",
        "validation_metrics.json",
        "predictions/test_predictions.csv",
        "model.json" if backend == "xgboost" else "model.joblib",
    ):
        assert (run_dir / name).is_file()
    assert json.loads((run_dir / "feature_names.json").read_text()) == list(FEATURES)
    assert json.loads((run_dir / "normalization.json").read_text()) == {
        "features": "none",
        "target": "none",
        "label_unit": "hartree",
        "scaler": None,
    }
    environment = json.loads((run_dir / "environment.json").read_text())
    assert environment["inputs"]["test.csv"]["sha256"] == hashes["test"]
    assert "rdkit" in environment["packages"]
    predictions = pd.read_csv(run_dir / "predictions/test_predictions.csv")
    assert list(predictions) == ["sample_id", "smiles", "delta_e", "delta_e_pred"]
    x, _ = descriptor_matrix(predictions.smiles.tolist(), predictions.sample_id.tolist())
    np.testing.assert_allclose(load_estimator(run_dir).predict(x), predictions.delta_e_pred)
    evaluated = evaluate_files(
        project / "data/test.csv",
        run_dir / "predictions/test_predictions.csv",
        ["delta_e"],
        "smiles",
    )
    assert evaluated["targets"]["delta_e"] == pytest.approx(result["metrics"]["targets"]["delta_e"])
    assert len(list((run_dir / "predictions").glob("*.csv"))) == 1
    records = load_records(find_metric_files([run_dir]))
    assert records[0]["run_mode"] == "single"
    validate_expected_seeds(records, [3407, 42, 2026, 7, 123])
    with pytest.raises(FileExistsError):
        train_ml(config_path)
    assert len(fits) == 1


def test_dry_run_does_not_create_outputs(project: Path) -> None:
    result = train_ml(make_config(project, "xgboost"), dry_run=True)
    assert result["split_rows"] == {"train": 6, "val": 2, "test": 2}
    assert not (project / "outputs").exists()


@pytest.mark.parametrize("constant_train", [False, True])
def test_training_mean_uses_train_only_without_descriptors(
    project: Path, monkeypatch, constant_train: bool
) -> None:
    from molgap.training import ml

    path = make_config(project, "training_mean")
    for split in ("train", "val", "test"):
        csv_path = project / "data" / f"{split}.csv"
        frame = pd.read_csv(csv_path)
        frame["delta_e"] = (
            ([3.5] * len(frame) if constant_train else np.arange(1, len(frame) + 1))
            if split == "train"
            else np.arange(100, 100 + len(frame))
        )
        frame.to_csv(csv_path, index=False)

    def forbid_features(*args, **kwargs):
        raise AssertionError("Mean baseline must not calculate molecular features")

    monkeypatch.setattr(ml, "descriptor_matrix", forbid_features)
    result = train_ml(path)
    run = Path(result["run_dir"])
    predictions_path = run / "predictions/test_predictions.csv"
    predictions = pd.read_csv(predictions_path)
    assert predictions.delta_e_pred.tolist() == [3.5, 3.5]
    assert predictions.delta_e.tolist() == [100, 101]
    assert predictions.sample_id.tolist() == ["test_0", "test_1"]
    assert result["metrics"]["targets"]["delta_e"]["mae"] == pytest.approx(97)
    assert json.loads((run / "feature_names.json").read_text()) == []
    mean = json.loads((run / "mean_baseline.json").read_text())
    assert mean["fit_split"] == "train" and mean["train_rows"] == 6
    assert mean["mean_hartree"] == 3.5
    assert mean["normalization_check"]["passed"]
    assert mean["normalization_check"]["inverse_mean_hartree"] == pytest.approx(3.5)
    np.testing.assert_array_equal(load_estimator(run).predict(np.zeros((2, 1))), [3.5, 3.5])
    metrics = evaluate_files(project / "data/test.csv", predictions_path, ["delta_e"], "smiles")
    assert metrics["targets"] == result["metrics"]["targets"]
    validate_expected_seeds(load_records(find_metric_files([run])), [3407, 42, 2026, 7, 123])
    with pytest.raises(FileExistsError):
        train_ml(path)


@pytest.mark.parametrize(
    "problem",
    ["duplicate", "overlap", "nonfinite", "missing", "invalid_smiles", "canonical_overlap"],
)
def test_ml_rejects_bad_data_before_saving(project: Path, problem: str) -> None:
    config = make_config(project, "random_forest")
    path = project / "data/test.csv"
    frame = pd.read_csv(path)
    if problem == "duplicate":
        frame.loc[1, "sample_id"] = frame.loc[0, "sample_id"]
    elif problem == "overlap":
        frame.loc[0, "sample_id"] = "train_0"
    elif problem == "nonfinite":
        frame.loc[0, "delta_e"] = np.inf
    elif problem == "missing":
        frame.loc[0, "sample_id"] = None
    elif problem == "invalid_smiles":
        frame.loc[0, "smiles"] = "not-a-molecule"
    else:
        frame.loc[0, "smiles"] = "OCC"  # Same molecule as train's CCO.
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError):
        train_ml(config)
    assert not (project / "outputs").exists()


def test_descriptor_order_and_nonfinite_guard(monkeypatch) -> None:
    x, _ = descriptor_matrix(["CCO"], ["one"])
    expected = [getattr(Descriptors, name)(Chem.MolFromSmiles("CCO")) for name in FEATURES]
    np.testing.assert_array_equal(x[0], expected)
    monkeypatch.setattr(Descriptors, "SMR_VSA7", lambda mol: np.nan)
    with pytest.raises(ValueError, match="Nonfinite"):
        descriptor_matrix(["CCO"], ["one"])


@pytest.mark.parametrize(
    "section,key,value",
    [
        ("normalization", "target", "standard"),
        ("normalization", "features", "standard"),
        ("training", "seed", 42),
        ("training", "run_mode", "repeated"),
        ("training", "n_jobs", 0),
    ],
)
def test_ml_rejects_contract_changes(project: Path, section: str, key: str, value) -> None:
    path = make_config(project, "xgboost")
    config = yaml.safe_load(path.read_text())
    config[section][key] = value
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError):
        load_ml_config(path)
