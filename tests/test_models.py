from pathlib import Path

import numpy as np
import pytest
import torch
from chemprop.models.utils import load_model, save_model
from sklearn.preprocessing import StandardScaler

from molgap.config import load_config
from molgap.models import PhysicsConsistencyMSE, QuantumMultiTaskHead, build_phase3_model


def test_physics_consistency_loss_value_and_gradient() -> None:
    criterion = PhysicsConsistencyMSE(
        task_weights=[1.0, 1.0, 1.0],
        consistency_weight=0.1,
        target_means=[0.0, 0.0, 0.0],
        target_scales=[1.0, 1.0, 1.0],
    )
    criterion.eval()
    predictions = torch.tensor([[0.0, 1.0, 0.0]], requires_grad=True)
    targets = torch.zeros_like(predictions)

    loss = criterion(predictions, targets)
    loss.backward()

    assert loss.item() == pytest.approx(1.1 / 3)
    assert predictions.grad is not None
    assert torch.isfinite(predictions.grad).all()


def test_phase3_model_round_trip(tmp_path: Path) -> None:
    config, _ = load_config(Path(__file__).parents[1] / "configs" / "consistency.yaml")
    scaler = StandardScaler().fit(
        np.array([[-0.3, 0.1, 0.4], [-0.4, 0.2, 0.6], [-0.2, 0.05, 0.25]])
    )
    model = build_phase3_model(config, scaler)
    path = tmp_path / "model.pt"

    save_model(path, model, output_columns=config["data"]["targets"])
    restored = load_model(path)

    assert isinstance(restored.predictor, QuantumMultiTaskHead)
    assert restored.predictor.output_dim == 3
