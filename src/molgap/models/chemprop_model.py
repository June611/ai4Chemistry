"""Construct the project-local Phase 3 model through Chemprop's Python API."""

from __future__ import annotations

from typing import Any

from chemprop.models import MPNN
from chemprop.nn import (
    MAE,
    RMSE,
    BondMessagePassing,
    MeanAggregation,
    NormAggregation,
    R2Score,
    SumAggregation,
)
from chemprop.nn.transforms import UnscaleTransform
from sklearn.preprocessing import StandardScaler

from molgap.config import task_weights
from molgap.models.losses import PhysicsConsistencyMSE
from molgap.models.multitask_head import QuantumMultiTaskHead


def build_phase3_model(config: dict[str, Any], scaler: StandardScaler) -> MPNN:
    """Build a D-MPNN with independent orbital heads and consistency-aware MSE."""
    model_config = config["model"]
    training = config["training"]
    weights = task_weights(config)
    criterion = PhysicsConsistencyMSE(
        task_weights=weights,
        consistency_weight=float(config["loss"]["weights"]["consistency"]),
        target_means=scaler.mean_,
        target_scales=scaler.scale_,
    )
    message_passing = BondMessagePassing(
        d_h=int(model_config["message_hidden_dim"]),
        depth=int(model_config["depth"]),
        dropout=float(model_config["dropout"]),
        activation=str(model_config["activation"]),
    )
    aggregations = {"mean": MeanAggregation, "sum": SumAggregation, "norm": NormAggregation}
    aggregation = aggregations[model_config["aggregation"]]()
    predictor = QuantumMultiTaskHead(
        input_dim=message_passing.output_dim,
        hidden_dim=int(model_config["ffn_hidden_dim"]),
        n_layers=int(model_config["ffn_num_layers"]),
        dropout=float(model_config["dropout"]),
        activation=str(model_config["activation"]),
        criterion=criterion,
        output_transform=UnscaleTransform.from_standard_scaler(scaler),
    )
    metric_types = {"mae": MAE, "rmse": RMSE, "r2": R2Score}
    metrics = [metric_types[name](task_weights=weights) for name in config["evaluation"]["metrics"]]
    return MPNN(
        message_passing,
        aggregation,
        predictor,
        batch_norm=bool(model_config["batch_norm"]),
        metrics=metrics,
        warmup_epochs=int(training["warmup_epochs"]),
        init_lr=float(training["init_lr"]),
        max_lr=float(training["max_lr"]),
        final_lr=float(training["final_lr"]),
    )
