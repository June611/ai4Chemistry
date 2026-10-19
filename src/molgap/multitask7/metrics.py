"""Per-property metrics in their own physical units."""

import numpy as np
import torch
from sklearn.metrics import r2_score

from molgap.multitask7.config import TASK_NAMES, TASK_UNITS


class MultiTask7Metrics:
    def __init__(self, scalers):
        self.scalers = scalers
        self.task_names = TASK_NAMES

    def inverse_transform(self, normalized):
        means = [self.scalers[t]["mean"] for t in TASK_NAMES]
        scales = [self.scalers[t]["scale"] for t in TASK_NAMES]
        if isinstance(normalized, torch.Tensor):
            normalized = normalized.to(torch.float64)
            return normalized * normalized.new_tensor(scales) + normalized.new_tensor(means)
        return np.asarray(normalized, dtype=np.float64) * scales + means

    def compute(self, prediction, truth, *, normalized=True):
        if isinstance(prediction, torch.Tensor):
            prediction = prediction.detach().cpu().numpy()
        if isinstance(truth, torch.Tensor):
            truth = truth.detach().cpu().numpy()
        prediction, truth = np.asarray(prediction), np.asarray(truth)
        if normalized:
            prediction, truth = self.inverse_transform(prediction), self.inverse_transform(truth)
        if prediction.shape != truth.shape or prediction.ndim != 2 or prediction.shape[1] != 7:
            raise ValueError("Expected aligned [samples, 7] predictions and targets")
        if len(prediction) < 2 or not np.isfinite(prediction).all() or not np.isfinite(truth).all():
            raise ValueError("Metrics require at least two finite, aligned predictions")
        result = {}
        for i, task in enumerate(TASK_NAMES):
            errors = prediction[:, i] - truth[:, i]
            mse = float(np.mean(errors**2))
            result[task] = dict(
                mae=float(np.mean(abs(errors))),
                mse=mse,
                rmse=float(np.sqrt(mse)),
                r2=float(r2_score(truth[:, i], prediction[:, i])),
                unit=TASK_UNITS[task],
            )
        return result

    @staticmethod
    def get_selection_metric(metrics):
        return metrics["delta_e"]["mae"]
