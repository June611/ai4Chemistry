"""Official LieTransformer attention on the authors' molecular charge features."""

from pathlib import Path

import torch
from torch import nn

from molgap.qm9_models.sources import activate_source


class LieTransformerAdapter(nn.Module):
    def __init__(self, root: Path, config: dict) -> None:
        super().__init__()
        activate_source(root, "lieconv")
        activate_source(root, "lie-transformer")
        from eqv_transformer.eqv_attention import EquivariantTransformer
        from lie_conv.lieGroups import SE3

        self.network = EquivariantTransformer(
            dim_input=15,
            dim_output=1,
            group=SE3(0.2),
            global_pool=True,
            global_pool_mean=True,
            **config,
        )

    def forward(self, batch: dict) -> torch.Tensor:
        charges = batch["charges"].float() / 9
        powers = torch.stack((torch.ones_like(charges), charges, charges.square()), dim=-1)
        features = (batch["one_hot"][:, :, :, None] * powers[:, :, None, :]).flatten(2)
        # Remove global translation before the stochastic group lift.
        mask = batch["atom_mask"]
        center = (batch["positions"] * mask[:, :, None]).sum(1) / mask.sum(1)[:, None]
        positions = batch["positions"] - center[:, None]
        return self.network((positions, features, mask)).reshape(-1)
