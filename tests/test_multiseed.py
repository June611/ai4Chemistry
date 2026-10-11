from pathlib import Path

import pytest
import yaml

from molgap.config import ConfigError
from molgap.experiments.multiseed import expand_matrix


def _write_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def test_matrix_reuses_one_split_across_training_seeds(tmp_path: Path) -> None:
    root = tmp_path / "project"
    (root / "pyproject.toml").parent.mkdir(parents=True)
    (root / "pyproject.toml").write_text(
        "[project]\nname='matrix-test'\nversion='0.0.0'\n", encoding="utf-8"
    )
    split = {
        "method": "random",
        "sizes": [0.8, 0.1, 0.1],
        "seed": 3407,
        "stratify": {"strategy": "quantile", "column": "delta_e", "bins": 10},
    }
    data_config = {
        "dataset": {"name": "tiny"},
        "split": split,
        "output": {"root": "data/processed"},
    }
    chemprop = {
        "experiment": {"name": "base", "phase": 1},
        "data": {"processed_dir": "unused", "split": split},
        "training": {"seed": 3407},
    }
    transformer = {
        "experiment": {"name": "sequence", "type": "smiles_transformer"},
        "data": {
            "processed_dir": "unused",
            "split": split,
            "chemprop_reference_config": "configs/chemprop.yaml",
        },
        "training": {"seed": 3407},
    }
    _write_yaml(root / "configs" / "data.yaml", data_config)
    _write_yaml(root / "configs" / "chemprop.yaml", chemprop)
    _write_yaml(root / "configs" / "transformer.yaml", transformer)
    matrix = {
        "name": "five_seed_test",
        "split_seed": 3407,
        "training_seeds": [1, 2, 3, 4, 5],
        "parallel_jobs": 2,
        "gpu": "0",
        "data_config": "configs/data.yaml",
        "models": [
            {"name": "graph", "trainer": "chemprop", "config": "configs/chemprop.yaml"},
            {
                "name": "sequence",
                "trainer": "transformer",
                "config": "configs/transformer.yaml",
            },
        ],
        "output": {"root": "outputs/batches"},
    }
    matrix_path = root / "configs" / "multiseed.yaml"
    _write_yaml(matrix_path, matrix)

    _, _, _, preparation_paths, specs = expand_matrix(matrix_path, overwrite=True)

    assert len(preparation_paths) == 1
    assert preparation_paths[0].name == "split_seed3407.yaml"
    assert len(specs) == 10
    assert {spec.split_seed for spec in specs} == {3407}
    assert {spec.training_seed for spec in specs} == {1, 2, 3, 4, 5}
    graph_seed2 = next(spec for spec in specs if spec.model == "graph" and spec.training_seed == 2)
    sequence_seed2 = next(
        spec for spec in specs if spec.model == "sequence" and spec.training_seed == 2
    )
    graph_config = yaml.safe_load(graph_seed2.config_path.read_text(encoding="utf-8"))
    sequence_config = yaml.safe_load(sequence_seed2.config_path.read_text(encoding="utf-8"))
    assert graph_config["data"]["processed_dir"] == sequence_config["data"]["processed_dir"]
    assert graph_config["data"]["processed_dir"].endswith("tiny/random/seed3407")
    assert graph_config["data"]["split"]["seed"] == 3407
    assert sequence_config["data"]["split"]["seed"] == 3407
    assert graph_config["training"]["seed"] == sequence_config["training"]["seed"] == 2
    assert len(graph_seed2.commands) == 3
    assert len(sequence_seed2.commands) == 1
    assert "--overwrite" in graph_seed2.commands[0]
    assert "--overwrite" in sequence_seed2.commands[0]

    reference_path = root / sequence_config["data"]["chemprop_reference_config"]
    reference = yaml.safe_load(reference_path.read_text(encoding="utf-8"))
    assert reference["data"]["processed_dir"] == graph_config["data"]["processed_dir"]
    assert reference["data"]["split"] == sequence_config["data"]["split"]
    assert {
        yaml.safe_load(spec.config_path.read_text(encoding="utf-8"))["data"]["processed_dir"]
        for spec in specs
    } == {"data/processed/tiny/random/seed3407"}


def test_matched_seed_matrix_generates_one_isolated_protocol_run(tmp_path: Path) -> None:
    root = tmp_path / "project"
    (root / "pyproject.toml").parent.mkdir(parents=True)
    (root / "pyproject.toml").write_text(
        "[project]\nname='matrix-test'\nversion='0.0.0'\n", encoding="utf-8"
    )
    split = {
        "method": "random",
        "sizes": [0.8, 0.1, 0.1],
        "seed": 42,
        "stratify": {"strategy": "quantile", "column": "delta_e", "bins": 10},
    }
    _write_yaml(
        root / "configs/data.yaml",
        {
            "dataset": {"name": "tiny"},
            "split": split,
            "output": {"root": "data/processed"},
        },
    )
    _write_yaml(
        root / "configs/model.yaml",
        {
            "experiment": {"name": "base", "phase": 1},
            "data": {"processed_dir": "unused", "split": split},
            "training": {"seed": 3407},
            "output": {"root": "outputs"},
        },
    )
    matrix = {
        "name": "paired",
        "protocol_id": "paired_split_gap_mae_v2",
        "seed_pairing": "matched",
        "split_seed": 42,
        "training_seeds": [42],
        "parallel_jobs": 1,
        "gpu": "0",
        "data_config": "configs/data.yaml",
        "models": [{"name": "graph", "trainer": "chemprop", "config": "configs/model.yaml"}],
        "output": {"root": "outputs/batches"},
    }
    path = root / "configs/matrix.yaml"
    _write_yaml(path, matrix)
    _, _, _, _, specs = expand_matrix(path)
    assert len(specs) == 1
    generated = yaml.safe_load(specs[0].config_path.read_text())
    assert generated["data"]["split"]["seed"] == 42
    assert generated["training"] == {
        "seed": 42,
        "run_mode": "paired_split_seed",
        "split_seed": 42,
        "seed_pairing": "matched",
    }
    assert generated["experiment"]["protocol_id"] == "paired_split_gap_mae_v2"
    assert generated["output"]["root"] == "outputs/paired_gap_mae_v2"

    matrix["training_seeds"] = [3407]
    _write_yaml(path, matrix)
    with pytest.raises(ConfigError, match="equal to split_seed"):
        expand_matrix(path)
