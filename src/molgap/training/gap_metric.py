"""Gap-only validation MAE for Chemprop's normalized validation datasets."""

from __future__ import annotations

import torch
from chemprop.nn.metrics import MAE, MetricRegistry
from lightning.pytorch.callbacks import Callback

GAP_MONITOR = "val/gap_mae"


@MetricRegistry.register("gap_mae")
class GapMAE(MAE):
    """Measure only the last target (delta_e), independently of auxiliary tasks."""

    def __init__(self, task_weights=1.0):
        # Selection must not inherit training-task weights.
        super().__init__(task_weights=1.0)
        self.register_buffer("gap_scale", torch.tensor(1.0))

    def update(self, preds, targets, mask=None, weights=None, lt_mask=None, gt_mask=None):
        if preds.ndim != 2 or preds.shape != targets.shape:
            raise ValueError("GapMAE requires matching (batch, tasks) regression tensors")
        selected_mask = None if mask is None else mask[:, -1:]
        super().update(
            preds[:, -1:] * self.gap_scale,
            targets[:, -1:] * self.gap_scale,
            selected_mask,
            # Validation selection is an unweighted per-molecule MAE.
            torch.ones(targets.shape[0], device=targets.device),
        )


class GapMetricScale(Callback):
    """Convert validation errors to Hartree; test targets already use Hartree."""

    def on_validation_epoch_start(self, trainer, pl_module):
        from molgap.models.losses import PhysicsConsistencyMSE

        transform = pl_module.predictor.output_transform
        if not transform.training or not hasattr(transform, "scale"):
            raise ValueError("Expected Chemprop's train-normalized validation output transform")
        scale = transform.scale.reshape(-1)[-1]
        if not torch.isfinite(scale) or scale <= 0:
            raise ValueError("Training gap scale must be finite and positive")
        for metric in pl_module.metrics:
            if isinstance(metric, GapMAE):
                metric.gap_scale.copy_(scale)
            elif isinstance(metric, PhysicsConsistencyMSE):
                # Chemprop validates on normalized tensors. Its cloned criterion
                # has no stochastic layers; training mode restores the existing
                # Hartree conversion for the diagnostic physical residual.
                metric.train()

    def on_test_epoch_start(self, trainer, pl_module):
        from molgap.models.losses import PhysicsConsistencyMSE

        for metric in pl_module.metrics:
            if isinstance(metric, GapMAE):
                metric.gap_scale.fill_(1.0)
            elif isinstance(metric, PhysicsConsistencyMSE):
                metric.eval()
