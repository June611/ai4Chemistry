"""Mathematical and lineage invariants for the independent Phase 1 protocol."""

import copy
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import torch
from chemprop.nn.ffn import MLP

from molgap.config import ConfigError
from molgap.multitask7.config import (
    TASK_NAMES,
    VARIANTS,
    load_multitask7_config,
    validate_multitask7_config,
)
from molgap.multitask7.data import MultiTask7Dataset
from molgap.multitask7.losses import UncertaintyWeightedMSE
from molgap.multitask7.metrics import MultiTask7Metrics
from molgap.multitask7.models import (
    ChempropMultiTask7,
    ChempropMultiTask7Affinity,
    ChempropMultiTask7AffinityUncertainty,
    ChempropMultiTask7Uncertainty,
)

ROOT = Path(__file__).resolve().parents[1]


def config():
    return load_multitask7_config(ROOT / "configs/qm9_multitask7/models/chemprop_mtl7.yaml")[0]


def test_configs_and_invalid_protocols():
    for model in VARIANTS:
        load_multitask7_config(ROOT / f"configs/qm9_multitask7/models/{model}.yaml")
    for section, key, value in [
        ("data", "targets", list(reversed(TASK_NAMES))),
        ("training", "seed", 2026),
        ("training", "epochs", 10),
        ("training", "selection_metric", "zpve_mae"),
    ]:
        cfg = copy.deepcopy(config())
        cfg[section][key] = value
        with pytest.raises(ConfigError):
            validate_multitask7_config(cfg)


def test_scalers_train_only_and_original_unit_metrics(tmp_path):
    values = np.arange(1, 36, dtype=float).reshape(5, 7)
    frame = pd.DataFrame(values, columns=TASK_NAMES)
    frame["sample_id"], frame["smiles"] = list("abcde"), ["C"] * 5
    path = tmp_path / "train.csv"
    frame.to_csv(path, index=False)
    ds = MultiTask7Dataset(path)
    scalers = ds.fit_scalers()
    np.testing.assert_allclose(ds.transform(values).mean(0), 0, atol=1e-14)
    np.testing.assert_allclose(ds.transform(values).std(0), 1)
    np.testing.assert_allclose(ds.inverse_transform(ds.transform(values)), values)
    assert all(s["fit_split"] == "train" and s["ddof"] == 0 for s in scalers.values())
    with pytest.raises(ValueError):
        MultiTask7Dataset(path, split="val").fit_scalers()
    pred = values.copy()
    pred[:, 2] += 0.125
    pred[:, -1] += 10
    metrics = MultiTask7Metrics(scalers).compute(pred, values, normalized=False)
    assert metrics["delta_e"]["mae"] == pytest.approx(0.125)
    assert metrics["zpve"]["mae"] == pytest.approx(10)
    assert MultiTask7Metrics.get_selection_metric(metrics) == pytest.approx(0.125)
    frame.loc[0, "mu"] = np.nan
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError):
        MultiTask7Dataset(path)


def test_four_variants_gradients_parameter_deltas_and_exact_identity():
    torch.manual_seed(42)
    base = ChempropMultiTask7(input_dim=8, hidden_dim=12, dropout=0)
    torch.manual_seed(42)
    affinity = ChempropMultiTask7Affinity(input_dim=8, hidden_dim=12, dropout=0)
    torch.manual_seed(42)
    dynamic = ChempropMultiTask7Uncertainty(input_dim=8, hidden_dim=12, dropout=0)
    torch.manual_seed(42)
    both = ChempropMultiTask7AffinityUncertainty(input_dim=8, hidden_dim=12, dropout=0)

    def count(model):
        return sum(p.numel() for p in model.parameters())

    assert count(affinity) - count(base) == 49
    assert count(dynamic) - count(base) == 7
    assert count(both) - count(base) == 56
    x, y = torch.randn(20, 8), torch.randn(20, 7)
    for model in (base, affinity, dynamic, both):
        prediction = model(x)
        assert prediction.shape == y.shape
        loss = model.loss_fn(prediction, y)
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    with (
        patch.object(affinity.affinity, "forward", return_value=torch.eye(7)),
        patch.object(both.affinity, "forward", return_value=torch.eye(7)),
    ):
        torch.testing.assert_close(affinity(x), base(x))
        torch.testing.assert_close(both(x), base(x))
    torch.testing.assert_close(dynamic.loss_fn(dynamic(x), y), base.loss_fn(base(x), y))
    matrix = affinity.affinity()
    torch.testing.assert_close(matrix.sum(-1), torch.ones(7))
    assert not torch.allclose(matrix, torch.eye(7))
    native = MLP.build(8, 1, 12, 1, 0, "relu")
    native[0][0].load_state_dict(base.task_projectors[0][0][0].state_dict())
    native[-1][-1].load_state_dict(base.task_heads[0].state_dict())
    torch.testing.assert_close(native(x).squeeze(-1), base(x)[:, 0])


def test_uncertainty_formula_and_weight_direction():
    criterion = UncertaintyWeightedMSE()
    prediction = torch.zeros(4, 7)
    truth = torch.ones_like(prediction)
    truth[:, 0] = 2
    truth[:, 1] = 0.5
    loss = criterion(prediction, truth)
    torch.testing.assert_close(loss, (prediction - truth).square().mean())
    loss.backward()
    # Gradient descent increases log variance for the larger-MSE task, reducing precision.
    assert criterion.log_vars.grad[0] < 0 < criterion.log_vars.grad[1]
    optimizer = torch.optim.SGD(criterion.parameters(), lr=0.1)
    optimizer.step()
    assert criterion.get_task_weights()[0] < 1 < criterion.get_task_weights()[1]


def test_literal_extra_projection_layout():
    base = ChempropMultiTask7()
    affinity = ChempropMultiTask7Affinity(head_layout="extra_projection")
    both = ChempropMultiTask7AffinityUncertainty(head_layout="extra_projection")

    def count(model):
        return sum(p.numel() for p in model.parameters())

    assert count(affinity) - count(base) == 7 * (300 * 300 + 300) + 49
    assert count(both) - count(affinity) == 7
    x = torch.randn(5, 300)
    affinity.eval()
    with patch.object(affinity.affinity, "forward", return_value=torch.eye(7)):
        expected = torch.cat(
            [
                head(affinity.task_projectors[i](affinity.extra_projectors[i](x)))
                for i, head in enumerate(affinity.task_heads)
            ],
            dim=1,
        )
        torch.testing.assert_close(affinity(x), expected)


@pytest.mark.parametrize(
    "constructor",
    [
        ChempropMultiTask7,
        ChempropMultiTask7Affinity,
        ChempropMultiTask7Uncertainty,
        ChempropMultiTask7AffinityUncertainty,
    ],
)
def test_small_sample_overfit(constructor):
    torch.manual_seed(7)
    x = torch.randn(100, 8)
    y = x @ torch.randn(8, 7)
    y = (y - y.mean(0)) / y.std(0)
    model = constructor(8, 32, dropout=0, head_layout="extra_projection")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
    for _ in range(250):
        optimizer.zero_grad()
        loss = model.loss_fn(model(x), y)
        loss.backward()
        optimizer.step()
    assert (model(x) - y).square().mean().item() < 0.01
