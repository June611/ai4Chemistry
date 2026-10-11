"""Pinned official TorchMD-Net equivariant-transformer adapter."""

import os
import tempfile
from pathlib import Path

import torch
from torch import nn

from molgap.qm9_models.batch import flat_atoms
from molgap.qm9_models.sources import activate_source


class TorchMDNetAdapter(nn.Module):
    def __init__(self, root: Path, config: dict) -> None:
        super().__init__()
        os.environ.setdefault(
            "WARP_CACHE_PATH", str(Path(tempfile.gettempdir()) / "molgap-warp-cache")
        )
        activate_source(root, "torchmd-net")
        from torchmdnet.models.model import create_model

        self.cutoff = float(config.get("cutoff", 5.0))

        # Create TorchMD-Net model
        # Using ET (Equivariant Transformer) architecture
        model_config = {
            "model": config.get("architecture", "equivariant-transformer"),
            "embedding_dimension": config.get("embedding_dimension", 128),
            "num_layers": config.get("num_layers", 6),
            "num_rbf": config.get("num_rbf", 50),
            "rbf_type": config.get("rbf_type", "expnorm"),
            "trainable_rbf": config.get("trainable_rbf", False),
            "activation": config.get("activation", "silu"),
            "cutoff_lower": 0.0,
            "cutoff_upper": self.cutoff,
            "max_z": 100,
            "max_num_neighbors": 32,
            "equivariance_invariance_group": "O(3)",
            "num_heads": config.get("num_heads", 8),
            "distance_influence": config.get("distance_influence", "both"),
            "attn_activation": "silu",
            "num_layers_phi": 2,
            "derivative": False,
            "output_model": "Scalar",
            "reduce_op": "sum",
            "precision": 32,
            "atom_filter": -1,
            "prior_model": None,
            "neighbor_embedding": False,
            "aggr": "add",
            "vector_cutoff": False,
            "output_mlp_num_layers": 0,
        }

        self.network = create_model(model_config)

    def forward(self, batch: dict) -> torch.Tensor:
        z, pos, groups = flat_atoms(batch)
        out, _ = self.network(z=z, pos=pos, batch=groups, num_systems=len(batch["smiles"]))
        return out.reshape(-1)
