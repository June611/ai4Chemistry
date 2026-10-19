"""Exercise target-scale isolation and original-unit validation selection."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from molgap.multitask7.config import TASK_NAMES
from molgap.multitask7.data import MultiTask7Dataset
from molgap.multitask7.losses import UncertaintyWeightedMSE
from molgap.multitask7.metrics import MultiTask7Metrics
from molgap.training.multitask7 import MultiTask7LightningModule

ROOT = Path(__file__).resolve().parents[1]


def test_independent_train_scalers_remove_large_target_units_from_loss(tmp_path):
    base = np.arange(1, 43, dtype=float).reshape(6, 7)
    units = np.array([1e-6, 1e-3, 0.1, 1e3, 1e8, 1e4, 1e-8])
    values = base * units + 10 * units
    frame = pd.DataFrame(values, columns=TASK_NAMES)
    frame["sample_id"], frame["smiles"] = list("abcdef"), ["C"] * 6
    path = tmp_path / "train.csv"
    frame.to_csv(path, index=False)
    train = MultiTask7Dataset(path, split="train")
    scalers = train.fit_scalers()
    normalized = train.transform(values)
    np.testing.assert_allclose(normalized.mean(0), 0, atol=1e-12)
    np.testing.assert_allclose(normalized.std(0), 1, atol=1e-12)
    np.testing.assert_allclose(train.inverse_transform(normalized), values)
    assert all(s["train_rows"] == 6 and s["fit_split"] == "train" for s in scalers.values())
    for split in ("val", "test"):
        other = MultiTask7Dataset(path, split=split, scalers=scalers)
        with pytest.raises(ValueError, match="Only the training split"):
            other.fit_scalers()
        assert other.scalers is scalers

    truth = torch.tensor(normalized, dtype=torch.float64)
    prediction = (truth + 0.25).detach().requires_grad_()
    objective = UncertaintyWeightedMSE().double()
    loss = objective(prediction, truth)
    assert loss.item() == pytest.approx(0.25**2)
    loss.backward()
    # Equal standardized errors have equal gradient weight, despite 1e16 unit ratio.
    torch.testing.assert_close(prediction.grad, prediction.grad[:, :1].expand_as(prediction))
    torch.testing.assert_close(
        objective.log_vars.grad, objective.log_vars.grad[:1].expand_as(objective.log_vars)
    )
    metrics = MultiTask7Metrics(scalers).compute(prediction, truth)
    for task in TASK_NAMES:
        assert metrics[task]["mae"] == pytest.approx(0.25 * scalers[task]["scale"])


def test_validation_gap_mae_uses_original_units_and_sample_weighting(monkeypatch):
    from molgap.modular.config import load

    cfg, _ = load(ROOT / "configs/qm9_epoch200/fresh/chemprop_variants/D1_seed42.yaml")
    cfg = cfg["source"]
    scales = [1e-6, 1e-3, 0.1, 1e3, 1e8, 1e4, 1e-8]
    scalers = {task: {"scale": scale} for task, scale in zip(TASK_NAMES, scales, strict=True)}
    model = MultiTask7LightningModule(cfg, scalers)
    monkeypatch.setattr(model, "forward", lambda batch: batch.predictions)
    recorded = {}
    monkeypatch.setattr(
        model, "log", lambda name, value, **kwargs: recorded.update({name: float(value)})
    )
    model.on_validation_epoch_start()
    for n, error in [(3, 1.0), (1, 3.0)]:
        batch = SimpleNamespace(Y=torch.zeros(n, 7), predictions=torch.full((n, 7), error))
        model.validation_step(batch, 0)
    model.on_validation_epoch_end()
    for task, scale in zip(TASK_NAMES, scales, strict=True):
        assert recorded[f"val_{task}_mae"] == pytest.approx(1.5 * scale)
        assert recorded[f"val_{task}_rmse"] == pytest.approx(np.sqrt(3) * scale)


@pytest.mark.parametrize("selection", ["loss", "val_loss", "rmse", "mean_label_mae"])
def test_epoch200_refuses_loss_or_mixed_unit_selection(selection):
    from molgap.modular.config import load, validate

    cfg, _ = load(ROOT / "configs/qm9_epoch200/fresh/chemprop_variants/D2_seed42.yaml")
    cfg["training"]["selection_metric"] = selection
    with pytest.raises(ValueError, match="original-unit gap validation MAE"):
        validate(cfg)
