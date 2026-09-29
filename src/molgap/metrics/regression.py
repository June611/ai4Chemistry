"""Numerically explicit regression metrics."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike


def regression_metrics(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    mape_zero_tolerance: float = 1e-12,
) -> dict[str, float | int | None]:
    """Compute common metrics after pairwise finite-value masking.

    MAPE is reported as a percentage. Targets whose absolute value is at or below
    ``mape_zero_tolerance`` are excluded from MAPE only and counted explicitly.
    """
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
    mape_mask = np.abs(truth) > float(mape_zero_tolerance)
    mape_n = int(mape_mask.sum())
    mape = (
        float(np.mean(np.abs(residual[mape_mask] / truth[mape_mask])) * 100.0) if mape_n else None
    )
    denominator = float(np.sum((truth - np.mean(truth)) ** 2))
    r2 = (
        None
        if truth.size < 2 or denominator == 0.0
        else 1.0 - float(np.sum(residual**2)) / denominator
    )
    return {
        "n": int(truth.size),
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "mape_n": mape_n,
        "mape_excluded_zero_targets": int(truth.size) - mape_n,
        "r2": r2,
    }
