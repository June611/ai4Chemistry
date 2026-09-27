"""Numerically explicit regression metrics."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike


def regression_metrics(y_true: ArrayLike, y_pred: ArrayLike) -> dict[str, float | int | None]:
    """Compute MAE, RMSE, and R² after pairwise finite-value masking."""
    truth = np.asarray(y_true, dtype=float).reshape(-1)
    prediction = np.asarray(y_pred, dtype=float).reshape(-1)
    if truth.shape != prediction.shape:
        raise ValueError(f"Shape mismatch: truth {truth.shape}, prediction {prediction.shape}")
    mask = np.isfinite(truth) & np.isfinite(prediction)
    truth = truth[mask]
    prediction = prediction[mask]
    if truth.size == 0:
        raise ValueError("No finite truth/prediction pairs are available.")

    residual = prediction - truth
    mae = float(np.mean(np.abs(residual)))
    rmse = float(math.sqrt(float(np.mean(residual**2))))
    denominator = float(np.sum((truth - np.mean(truth)) ** 2))
    r2 = (
        None
        if truth.size < 2 or denominator == 0.0
        else 1.0 - float(np.sum(residual**2)) / denominator
    )
    return {"n": int(truth.size), "mae": mae, "rmse": rmse, "r2": r2}
