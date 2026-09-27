"""Loss functions for physically constrained orbital-energy prediction."""

from __future__ import annotations

import torch
from chemprop.nn.metrics import ChempropMetric
from torch import Tensor
from torch.nn import functional as F


def gap_consistency_residual(predictions: Tensor) -> Tensor:
    """Return gap - (LUMO - HOMO) for columns [HOMO, LUMO, gap]."""
    if predictions.ndim != 2 or predictions.shape[1] != 3:
        raise ValueError("Predictions must have shape [batch, 3] ordered homo, lumo, delta_e.")
    return predictions[:, 2] - (predictions[:, 1] - predictions[:, 0])


class PhysicsConsistencyMSE(ChempropMetric):
    """Task-weighted MSE plus an orbital-gap consistency penalty in Hartree."""

    alias = "physics_consistency_mse"

    def __init__(
        self,
        task_weights: Tensor | list[float],
        consistency_weight: float,
        target_means: Tensor | list[float] | None = None,
        target_scales: Tensor | list[float] | None = None,
        mean_values: Tensor | list[float] | None = None,
        scale_values: Tensor | list[float] | None = None,
    ) -> None:
        super().__init__(task_weights=task_weights)
        if self.task_weights.numel() != 3:
            raise ValueError("PhysicsConsistencyMSE requires exactly three task weights.")
        if consistency_weight < 0:
            raise ValueError("consistency_weight must be non-negative.")
        target_means = target_means if target_means is not None else mean_values
        target_scales = target_scales if target_scales is not None else scale_values
        if target_means is None or target_scales is None:
            raise ValueError("target means and scales are required.")
        means = torch.as_tensor(target_means, dtype=torch.float).view(1, -1)
        scales = torch.as_tensor(target_scales, dtype=torch.float).view(1, -1)
        if means.shape[1] != 3 or scales.shape[1] != 3 or torch.any(scales <= 0):
            raise ValueError(
                "target_means/scales must contain three values and scales must be > 0."
            )
        self.consistency_weight = float(consistency_weight)
        # Plain attributes allow Chemprop's cross-device metric reconstruction helper
        # to recover constructor arguments from ``metric.__dict__``.
        self.mean_values = means.detach().cpu().flatten().tolist()
        self.scale_values = scales.detach().cpu().flatten().tolist()
        self.register_buffer("target_means", means)
        self.register_buffer("target_scales", scales)

    def _calc_unreduced_loss(
        self,
        preds: Tensor,
        targets: Tensor,
        *_: Tensor,
    ) -> Tensor:
        supervised = F.mse_loss(preds, targets, reduction="none")
        physical = preds * self.target_scales + self.target_means if self.training else preds
        consistency = gap_consistency_residual(physical).square().unsqueeze(1)
        weight_sum = self.task_weights.sum().clamp_min(torch.finfo(preds.dtype).eps)
        return supervised + self.consistency_weight * consistency / weight_sum
