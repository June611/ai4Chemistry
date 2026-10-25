"""Official SchNet/PaiNN algorithm subset with a molecular gap readout."""

import torch
from torch import nn

from molgap._vendor.schnetpack import properties
from molgap._vendor.schnetpack.atomistic.atomwise import Atomwise
from molgap._vendor.schnetpack.nn import CosineCutoff, GaussianRBF
from molgap._vendor.schnetpack.representation.painn import PaiNN
from molgap._vendor.schnetpack.representation.schnet import SchNet
from molgap.qm9_models.batch import flat_atoms, padded_radius_edges


class SchNetPackAdapter(nn.Module):
    def __init__(self, name: str, config: dict) -> None:
        super().__init__()
        self.cutoff = float(config["cutoff"])
        representation = {"schnet": SchNet, "painn": PaiNN}[name]
        width = int(config["n_atom_basis"])
        self.representation = representation(
            n_atom_basis=width,
            n_interactions=int(config["n_interactions"]),
            radial_basis=GaussianRBF(n_rbf=int(config["n_rbf"]), cutoff=self.cutoff),
            cutoff_fn=CosineCutoff(self.cutoff),
        )
        self.readout = Atomwise(n_in=width, n_out=1, aggregation_mode="sum", output_key="gap")

    def forward(self, batch: dict) -> torch.Tensor:
        z, pos, groups = flat_atoms(batch)
        i, j = padded_radius_edges(batch, self.cutoff)
        inputs = {
            properties.Z: z,
            properties.R: pos,
            properties.Rij: pos[j] - pos[i],
            properties.idx_i: i,
            properties.idx_j: j,
            properties.idx_m: groups,
            properties.n_atoms: batch["atom_mask"].sum(dim=1),
        }
        return self.readout(self.representation(inputs))["gap"].reshape(-1)
