"""A complete CPU checkpoint/predict/evaluate path on an isolated synthetic fixture."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from molgap.data.qm9_properties import digest
from molgap.multitask7.config import TASK_NAMES, TASK_UNITS
from molgap.multitask7.evaluation import verify_run
from molgap.training.multitask7 import MultiTask7LightningModule, train


def fixture_project(root):
    (root / "pyproject.toml").write_text('[project]\nname="mtl7-fixture"\nversion="0.0.0"\n')
    data_root = root / "data/processed/qm9_single_properties_v1"
    split_dir = data_root / "random/seed42"
    split_dir.mkdir(parents=True)
    manifest, extended = [], {}
    cursor = 0
    for split, count in [("train", 12), ("val", 4), ("test", 4)]:
        values = np.arange(cursor * 7, (cursor + count) * 7, dtype=float).reshape(count, 7) / 100
        frame = pd.DataFrame(values, columns=TASK_NAMES)
        frame["sample_id"] = [f"id{i}" for i in range(cursor, cursor + count)]
        frame["smiles"] = (["C", "CC", "CCC", "CO"] * count)[:count]
        path = split_dir / f"{split}.csv"
        frame.to_csv(path, index=False)
        extended[f"random/seed42/{split}.csv"] = digest(path)
        manifest.extend(
            dict(sample_id=s, split=split, split_position=i) for i, s in enumerate(frame.sample_id)
        )
        cursor += count
    pd.DataFrame(manifest).to_csv(split_dir / "split_manifest.csv", index=False)
    identity = dict(
        targets=TASK_UNITS,
        preserved_original_membership_order_and_energy_labels=True,
        extended_files=extended,
        source_files={
            "data/processed/qm9_full/random/seed42/split_manifest.csv": digest(
                split_dir / "split_manifest.csv"
            )
        },
    )
    (data_root / "data_identity.json").write_text(json.dumps(identity))
    return root


@pytest.mark.parametrize("epochs", [100, 200])
def test_checkpoint_roundtrip_and_seven_saved_metrics(tmp_path, epochs):
    root = fixture_project(tmp_path)
    source = (
        Path(__file__).resolve().parents[1]
        / "configs/qm9_multitask7/models/chemprop_mtl7_affinity_uncertainty.yaml"
    )
    cfg = yaml.safe_load(source.read_text())
    if epochs == 200:
        from molgap.multitask7.config import PROTOCOL_ID_200

        cfg["experiment"]["protocol_id"] = PROTOCOL_ID_200
        cfg["training"]["epochs"] = 200
    cfg["training"].update(accelerator="cpu", num_workers=0, pin_memory=False)
    path = root / "config.yaml"
    path.write_text(yaml.safe_dump(cfg))
    run = root / "smoke"
    result = train(path, max_epochs=2, output_dir=run)
    assert result["rows"] == 4 and result["diagnostic"]
    assert verify_run(run, root, allow_diagnostic=True)["all_seven_targets_verified"]
    checkpoint = torch.load(run / "best_checkpoint.pt", weights_only=False)
    first = MultiTask7LightningModule(checkpoint["config"], checkpoint["scalers"])
    first.load_state_dict(checkpoint["state_dict"])
    second = MultiTask7LightningModule(checkpoint["config"], checkpoint["scalers"])
    second.load_state_dict(checkpoint["state_dict"])
    first.eval()
    second.eval()
    x = torch.randn(3, 300)
    torch.testing.assert_close(first.network.head(x), second.network.head(x))
    assert len(pd.read_csv(run / "training_log.csv")) == 2
    assert (run / "affinity_matrix.csv").exists() and (run / "uncertainty_params.json").exists()
    best = torch.load(run / "checkpoints/best.ckpt", weights_only=False)
    callback_states = best["callbacks"]
    assert any("ModelCheckpoint" in name and "val_delta_e_mae" in name for name in callback_states)
    assert any("EarlyStopping" in name and "val_delta_e_mae" in name for name in callback_states)
    if epochs == 200:
        terminal = torch.load(run / "checkpoints/terminal_fit.ckpt", weights_only=False)
        assert terminal["state_dict"] and terminal["optimizer_states"] and terminal["lr_schedulers"]
        assert terminal["loops"]["fit_loop"]["epoch_progress"]["total"]["completed"] == 2
        assert terminal["global_step"] == 2
        assert terminal["lr_schedulers"][0]["last_epoch"] == terminal["global_step"]
        assert set(terminal["rng_snapshot"]) == {"python", "numpy", "torch_cpu", "torch_cuda"}
        assert any("log_vars" in name for name in terminal["state_dict"])
        assert len(json.loads((run / "scalers.json").read_text())) == 7
        assert (run / "config.yaml").exists() and (run / "split_identity.json").exists()
        # This export is an inference bundle, not an optimizer-resume checkpoint.
        assert "optimizer_states" not in checkpoint and "lr_schedulers" not in checkpoint
