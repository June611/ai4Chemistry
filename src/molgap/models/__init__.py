"""Project-owned model extensions built around the external Chemprop library."""

from molgap.models.chemprop_model import build_phase3_model
from molgap.models.losses import PhysicsConsistencyMSE, gap_consistency_residual
from molgap.models.multitask_head import QuantumMultiTaskHead

__all__ = [
    "PhysicsConsistencyMSE",
    "QuantumMultiTaskHead",
    "build_phase3_model",
    "gap_consistency_residual",
]
