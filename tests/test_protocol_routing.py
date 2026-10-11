from pathlib import Path

import pandas as pd
import pytest
import torch

from molgap.config import run_directory
from molgap.qm9_models import train


@pytest.mark.parametrize("seed", [3407, 42, 2026])
def test_qm9_seed_overrides_preserve_model_group(monkeypatch, seed: int) -> None:
    root = Path(__file__).resolve().parents[1]
    captured = {}

    def fake_load(config, project_root, smoke):
        captured.update(config)
        frames = {split: pd.DataFrame({"delta_e": [0.2]}) for split in ("train", "val", "test")}
        return frames, {}, {}, {"geometry_sha256": "test-only"}

    monkeypatch.setattr(train, "load_data", fake_load)
    monkeypatch.setattr(train, "build_model", lambda *args: torch.nn.Linear(1, 1))
    plan = train.run(
        root / "configs/qm9_models/egnn.yaml",
        split_seed_override=seed,
        seed_override=seed,
        dry_run=True,
    )
    assert plan["training_seed"] == plan["split_seed"] == seed
    assert plan["budget"] == {"batch_size": 256, "epochs": 100, "early_stopping_patience": 20}
    assert captured["experiment"]["name"] == "egnn_qm9"
    assert captured["experiment"]["model_id"] == "egnn"
    assert captured["data"]["processed_dir"].endswith(f"/seed{seed}")
    assert captured["data"]["geometry"].endswith(f"_seed{seed}.npz")
    assert run_directory(captured, root) == (
        root / "outputs/paired_gap_mae_v2/egnn_qm9" / f"seed{seed}"
    )


def test_qm9_mismatched_seed_rejected_before_data_loading(monkeypatch) -> None:
    root = Path(__file__).resolve().parents[1]

    def unexpected_load(*args):
        pytest.fail("Mismatched seeds must be rejected before data loading")

    monkeypatch.setattr(train, "load_data", unexpected_load)
    with pytest.raises(ValueError, match="training seed to equal split seed"):
        train.run(root / "configs/qm9_models/egnn.yaml", split_seed_override=42, dry_run=True)
