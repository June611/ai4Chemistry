"""Adapter for the authors' QM9 EGNN, including charge features and edge masks."""

import sys
from pathlib import Path

import torch
from torch import nn

from molgap.qm9_models.sources import activate_source


class EGNNAdapter(nn.Module):
    def __init__(self, root: Path, config: dict, device: torch.device) -> None:
        super().__init__()
        activate_source(root, "egnn")
        # EGNN and 3D Infomax both expose an upstream top-level package named
        # ``models``. Clear the other checkout's cached namespace before EGNN's
        # ``qm9.models`` imports ``models.gcl``.
        for module_name in list(sys.modules):
            if module_name == "models" or module_name.startswith("models."):
                del sys.modules[module_name]
        from qm9.models import EGNN

        self.network = EGNN(
            in_node_nf=15,
            in_edge_nf=0,
            hidden_nf=int(config["hidden_nf"]),
            n_layers=int(config["n_layers"]),
            attention=bool(config["attention"]),
            node_attr=int(config["node_attr"]),
            device=device,
        )

    def forward(self, batch: dict) -> torch.Tensor:
        charges = batch["charges"]
        size, width = charges.shape
        powers = (charges.float()[:, :, None] / 9).pow(torch.arange(3, device=charges.device))
        features = (batch["one_hot"][:, :, :, None] * powers[:, :, None, :]).reshape(-1, 15)
        src = torch.arange(width, device=charges.device).repeat_interleave(width)
        dst = torch.arange(width, device=charges.device).repeat(width)
        offsets = torch.arange(size, device=charges.device)[:, None] * width
        edges = [(src[None] + offsets).reshape(-1), (dst[None] + offsets).reshape(-1)]
        return self.network(
            h0=features,
            x=batch["positions"].reshape(-1, 3),
            edges=edges,
            edge_attr=None,
            node_mask=batch["atom_mask"].reshape(-1, 1).float(),
            edge_mask=batch["edge_mask"].reshape(-1, 1).float(),
            n_nodes=width,
        ).reshape(-1)
