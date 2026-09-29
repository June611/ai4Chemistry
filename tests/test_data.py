import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from molgap.data.dataset import DataValidationError
from molgap.data.preprocess import ConfirmationRequired, prepare_dataset
from molgap.data.split import (
    regression_stratified_random_split,
    scaffold_key,
    split_with_manifest,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_fixture_project(tmp_path: Path, conflicting_duplicate: bool = False) -> Path:
    project = tmp_path / "project"
    project.mkdir()
    (project / "pyproject.toml").write_text(
        "[project]\nname='test'\nversion='0'\n", encoding="utf-8"
    )
    raw = project / "data" / "raw"
    raw.mkdir(parents=True)
    duplicate_offset = 5e-4 if conflicting_duplicate else 5e-5
    frame = pd.DataFrame(
        {
            "id": list(range(1, 9)),
            "smiles_raw": ["CCO", "OCC", "CC", "CCC", "CCCC", "N", "CN", "CO"],
            "h": [-0.3, -0.3, -0.25, -0.26, -0.27, -0.4, -0.35, -0.32],
            "l": [0.1, 0.1 + duplicate_offset, 0.05, 0.06, 0.07, 0.1, 0.08, 0.09],
            "gap": [0.4, 0.4 + duplicate_offset, 0.3, 0.32, 0.34, 0.5, 0.43, 0.41],
        }
    )
    source = raw / "sample.csv"
    frame.to_csv(source, index=False)
    config = {
        "schema_version": 1,
        "dataset": {
            "name": "sample",
            "artifact_version": "1.0.0",
            "expected_rows": 8,
            "source": {
                "path": "data/raw/sample.csv",
                "encoding": "utf-8",
                "id_column": "id",
                "columns": {
                    "smiles": "smiles_raw",
                    "homo": "h",
                    "lumo": "l",
                    "delta_e": "gap",
                },
            },
        },
        "labels": {
            "unit": "hartree",
            "consistency_tolerance": 1e-3,
            "derive_ev": False,
        },
        "processing": {
            "canonicalize_smiles": True,
            "preserve_stereo": True,
            "duplicate_tolerance": 1.1e-4,
            "duplicate_policy": "mean_within_tolerance",
            "max_rejected_fraction": 0.05,
        },
        "split": {"method": "random", "sizes": [0.6, 0.2, 0.2], "seed": 7},
        "chemprop": {"target_columns": ["delta_e"]},
        "output": {"root": "data/processed"},
    }
    config_path = project / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return config_path


def test_preprocess_is_auditable_and_deterministic(tmp_path: Path) -> None:
    config_path = _write_fixture_project(tmp_path)
    source_path = config_path.parent / "data" / "raw" / "sample.csv"
    source_hash = _sha256(source_path)

    report = prepare_dataset(config_path)

    output = config_path.parent / "data" / "processed" / "sample" / "random" / "seed7"
    assert report["status"] == "complete"
    assert report["final_unique_rows"] == 7
    assert report["checks"]["no_smiles_leakage"]
    assert report["checks"]["adapted_contract_rollback_matches_source"]
    assert _sha256(source_path) == source_hash
    assert list(pd.read_csv(output / "train.csv").columns) == [
        "sample_id",
        "smiles",
        "delta_e",
    ]
    assert len(pd.read_csv(output / "duplicates.csv")) == 2
    assert json.loads((output / "pending_confirmations.json").read_text())["items"] == []
    for name in (
        "dataset.csv",
        "train.csv",
        "val.csv",
        "test.csv",
        "split_manifest.csv",
        "preprocess_report.json",
        "target_statistics.json",
        "duplicates.csv",
        "rejected_rows.csv",
        "processing_log.jsonl",
        "change_map.csv",
        "data_lineage.json",
        "artifact_manifest.json",
        "lineage.sha256",
    ):
        assert (output / name).is_file()

    first_manifest = (output / "split_manifest.csv").read_bytes()
    prepare_dataset(config_path)
    assert (output / "split_manifest.csv").read_bytes() == first_manifest


def test_conflicting_duplicates_stop_for_confirmation(tmp_path: Path) -> None:
    config_path = _write_fixture_project(tmp_path, conflicting_duplicate=True)

    with pytest.raises(ConfirmationRequired):
        prepare_dataset(config_path)

    output = config_path.parent / "data" / "processed" / "sample" / "random" / "seed7"
    pending = json.loads((output / "pending_confirmations.json").read_text())["items"]
    assert any(item["type"] == "conflicting_duplicate_labels" for item in pending)
    assert set(pd.read_csv(output / "duplicates.csv")["status"]) == {"conflict"}


def test_scaffold_split_has_no_scaffold_leakage() -> None:
    smiles = [
        "c1ccccc1",
        "c1ccncc1",
        "C1CCCCC1",
        "C1CCCC1",
        "c1ccoc1",
        "c1ccsc1",
        "c1ccc2ccccc2c1",
        "c1ncc[nH]1",
    ]
    frame = pd.DataFrame(
        {
            "sample_id": [f"molecule:{index}" for index in range(len(smiles))],
            "smiles": smiles,
        }
    )

    splits, _ = split_with_manifest(frame, "smiles", "scaffold_balanced", [0.6, 0.2, 0.2], 7)

    scaffold_sets = [
        {scaffold_key(smiles_value) for smiles_value in splits[name]["smiles"]}
        for name in ("train", "val", "test")
    ]
    assert not scaffold_sets[0] & scaffold_sets[1]
    assert not scaffold_sets[0] & scaffold_sets[2]
    assert not scaffold_sets[1] & scaffold_sets[2]


def test_regression_stratified_split_is_balanced_exact_and_deterministic() -> None:
    frame = pd.DataFrame(
        {
            "sample_id": [f"sample:{index}" for index in range(500)],
            "delta_e": [*range(450), *range(1000, 1050)],
        }
    )

    first, bins_by_sample, effective_bins = regression_stratified_random_split(
        frame, "delta_e", [0.8, 0.1, 0.1], seed=3407, bins=10
    )
    second, second_bins, second_effective = regression_stratified_random_split(
        frame, "delta_e", [0.8, 0.1, 0.1], seed=3407, bins=10
    )

    assert [len(first[name]) for name in ("train", "val", "test")] == [400, 50, 50]
    assert effective_bins == second_effective == 10
    assert bins_by_sample == second_bins
    for name in ("train", "val", "test"):
        assert first[name]["sample_id"].tolist() == second[name]["sample_id"].tolist()
    bin_counts = {
        name: first[name]["sample_id"].map(bins_by_sample).value_counts().sort_index().tolist()
        for name in ("train", "val", "test")
    }
    assert bin_counts["train"] == [40] * 10
    assert bin_counts["val"] == [5] * 10
    assert bin_counts["test"] == [5] * 10


def test_regression_stratification_rejects_constant_target() -> None:
    frame = pd.DataFrame(
        {"sample_id": [f"sample:{index}" for index in range(30)], "delta_e": [0.2] * 30}
    )

    with pytest.raises(DataValidationError, match="at least two unique"):
        regression_stratified_random_split(frame, "delta_e", [0.8, 0.1, 0.1], seed=7, bins=5)


def test_stratification_metadata_is_written_to_manifest() -> None:
    frame = pd.DataFrame(
        {
            "sample_id": [f"sample:{index}" for index in range(60)],
            "smiles": [f"[{index + 1}CH4]" for index in range(60)],
            "delta_e": [index / 100 for index in range(60)],
        }
    )

    _, manifest = split_with_manifest(
        frame,
        "smiles",
        "random",
        [0.8, 0.1, 0.1],
        7,
        stratify_column="delta_e",
        stratify_bins=5,
    )

    assert set(manifest["split_strategy"]) == {"quantile_stratified"}
    assert set(manifest["stratify_column"]) == {"delta_e"}
    assert set(manifest["stratify_bins_requested"]) == {5}
    assert set(manifest["stratify_bins_effective"]) == {5}
    assert set(manifest["stratify_bin"]) == set(range(5))
