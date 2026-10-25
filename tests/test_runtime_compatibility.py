import pytest
import torch
from torch.utils.data import DataLoader

from molgap.qm9_models.batch import flat_atoms, padded_radius_edges, radius_edges
from molgap.qm9_models.runtime import dataloader_kwargs


@pytest.mark.parametrize("workers", [0, 1, 2])
def test_three_epoch_shuffle_matches_legacy(workers):
    def orders(options):
        loader = DataLoader(
            list(range(32)), batch_size=8, shuffle=True,
            generator=torch.Generator().manual_seed(42), **options,
        )
        return [torch.cat(list(loader)).tolist() for _ in range(3)]

    assert orders({"num_workers": workers}) == orders(
        dataloader_kwargs({"num_workers": workers}, device=torch.device("cpu"))
    )


@pytest.mark.parametrize("width", [24, 25, 26, 29])
def test_mixed_padding_preserves_reference_graph(width):
    torch.manual_seed(42)
    positions = torch.randn(2, width, 3)
    positions[0, :2] = torch.tensor([
        [-1.74247229, -.44673184, 3.43472290],
        [-6.65604877, .44034275, 3.17040777],
    ])
    mask = torch.ones(2, width, dtype=torch.bool)
    mask[0, 2:] = False
    batch = {"positions": positions, "atom_mask": mask, "charges": mask.long()}
    _, flat, groups = flat_atoms(batch)
    assert torch.equal(padded_radius_edges(batch, 5.), radius_edges(flat, groups, 5.))
