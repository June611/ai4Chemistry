"""Thin adapter for the official CormorantQM9 network."""

from pathlib import Path

import torch
from torch import nn

from molgap.qm9_models.sources import activate_source


class CormorantAdapter(nn.Module):
    def __init__(self, root: Path, config: dict, device: torch.device) -> None:
        super().__init__()
        activate_source(root, "cormorant", "src")
        from cormorant.models import CormorantQM9

        self.network = CormorantQM9(
            maxl=config["maxl"],
            max_sh=config["max_sh"],
            num_cg_levels=config["num_cg_levels"],
            num_channels=config["num_channels"],
            num_species=5,
            cutoff_type=["learn"],
            hard_cut_rad=1.73,
            soft_cut_rad=1.73,
            soft_cut_width=0.2,
            weight_init="rand",
            level_gain=10.0,
            charge_power=2,
            basis_set=[3, 3],
            charge_scale=9.0,
            gaussian_mask=False,
            top="linear",
            input="linear",
            num_mpnn_layers=1,
            device=device,
            dtype=torch.float32,
        )

    def forward(self, batch: dict) -> torch.Tensor:
        return self.network(
            {k: batch[k] for k in ("positions", "charges", "one_hot", "atom_mask", "edge_mask")}
        ).reshape(-1)
