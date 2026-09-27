from pathlib import Path

import pandas as pd
import yaml

from molgap.training.phase3 import train_phase3
from molgap.training.predict import predict


def test_phase3_tiny_cpu_training(tmp_path: Path) -> None:
    project = tmp_path / "project"
    processed = project / "data" / "processed" / "tiny" / "random" / "seed7"
    processed.mkdir(parents=True)
    (project / "pyproject.toml").write_text(
        "[project]\nname='tiny-molgap'\nversion='0.0.0'\n", encoding="utf-8"
    )
    molecules = [
        ("a", "C", -0.30, 0.10),
        ("b", "CC", -0.31, 0.08),
        ("c", "CCC", -0.32, 0.07),
        ("d", "CO", -0.35, 0.05),
        ("e", "CN", -0.34, 0.06),
        ("f", "O", -0.40, 0.03),
        ("g", "N", -0.38, 0.04),
        ("h", "CCl", -0.33, 0.09),
        ("i", "CBr", -0.36, 0.02),
        ("j", "CF", -0.37, 0.01),
    ]
    frame = pd.DataFrame(molecules, columns=["sample_id", "smiles", "homo", "lumo"])
    frame["delta_e"] = frame["lumo"] - frame["homo"]
    frame.iloc[:6].to_csv(processed / "train.csv", index=False)
    frame.iloc[6:8].to_csv(processed / "val.csv", index=False)
    frame.iloc[8:].to_csv(processed / "test.csv", index=False)
    config = {
        "experiment": {"name": "tiny_phase3", "phase": 3},
        "data": {
            "processed_dir": "data/processed/tiny/random/seed7",
            "smiles_column": "smiles",
            "id_column": "sample_id",
            "targets": ["homo", "lumo", "delta_e"],
            "label_unit": "hartree",
            "split": {"method": "random", "sizes": [0.6, 0.2, 0.2], "seed": 7},
        },
        "model": {
            "backend": "chemprop_python",
            "message_hidden_dim": 16,
            "depth": 2,
            "dropout": 0.0,
            "activation": "relu",
            "aggregation": "mean",
            "batch_norm": False,
            "ffn_hidden_dim": 16,
            "ffn_num_layers": 1,
        },
        "training": {
            "batch_size": 2,
            "epochs": 1,
            "warmup_epochs": 1,
            "init_lr": 1e-4,
            "max_lr": 1e-3,
            "final_lr": 1e-4,
            "seed": 7,
            "num_workers": 0,
            "accelerator": "cpu",
            "devices": 1,
            "patience": 1,
        },
        "loss": {
            "function": "mse",
            "weights": {"homo": 1.0, "lumo": 1.0, "gap": 1.0, "consistency": 0.1},
        },
        "evaluation": {"metrics": ["mae", "rmse", "r2"]},
        "output": {"root": "outputs"},
    }
    config_path = project / "phase3.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    result = train_phase3(config_path)
    run_dir = project / "outputs" / "tiny_phase3" / "seed7"

    assert result["metrics"]["rows"] == 2
    assert (run_dir / "chemprop" / "model.pt").is_file()
    assert (run_dir / "predictions" / "test_predictions.csv").is_file()
    assert (run_dir / "environment.json").is_file()

    repeated_path = run_dir / "predictions" / "cli_predictions.csv"
    assert predict(config_path, output_override=repeated_path) == repeated_path
    assert list(pd.read_csv(repeated_path)["sample_id"]) == ["i", "j"]
