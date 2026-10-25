from pathlib import Path

import pytest
import torch

from molgap.qm9_models.batch import (
    Molecule,
    collate_molecules,
    flat_atoms,
    padded_radius_edges,
    radius_edges,
)


def molecular_fixture():
    return [
        Molecule(
            "water",
            "O",
            torch.tensor([8, 1, 1]),
            torch.tensor([[0.0, 0.0, 0.0], [0.95, 0.0, 0.0], [-0.24, 0.92, 0.0]]),
            0.3,
        ),
        Molecule(
            "methane",
            "C",
            torch.tensor([6, 1, 1, 1, 1]),
            torch.tensor(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 1.0, 1.0],
                    [-1.0, -1.0, 1.0],
                    [-1.0, 1.0, -1.0],
                    [1.0, -1.0, -1.0],
                ]
            ),
            0.5,
        ),
    ]


def test_no_cross_molecule_or_padding_edges():
    batch = collate_molecules(molecular_fixture())
    assert batch["atom_mask"].sum().item() == 8
    assert not batch["edge_mask"][:, torch.arange(5), torch.arange(5)].any()
    z, pos, groups = flat_atoms(batch)
    assert len(z) == 8
    edges = radius_edges(pos, groups, 5.0)
    vectorized = padded_radius_edges(batch, 5.0)
    assert torch.equal(vectorized, edges)
    assert torch.equal(groups[edges[0]], groups[edges[1]])
    assert (edges[0] != edges[1]).all()


def test_official_egnn_invariance_padding_gradients_and_reload():
    root = Path(__file__).parents[1]
    if not (root / ".external/qm9_models/egnn").exists():
        pytest.skip("Pinned EGNN source required for upstream integration test")
    from molgap.qm9_models.egnn import EGNNAdapter

    torch.manual_seed(3)
    torch.set_num_threads(2)
    config = {"hidden_nf": 16, "n_layers": 2, "attention": True, "node_attr": 0}
    model = EGNNAdapter(root, config, torch.device("cpu"))
    batch = collate_molecules(molecular_fixture())
    pred = model(batch)
    assert pred.shape == (2,) and torch.isfinite(pred).all()
    loss = (pred - batch["target"]).square().mean()
    loss.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    assert sum(p.grad.abs().sum().item() for p in model.parameters() if p.grad is not None) > 0
    model.eval()
    with torch.no_grad():
        reference = model(batch)
        rotation, _ = torch.linalg.qr(torch.randn(3, 3))
        changed = dict(batch, positions=batch["positions"] @ rotation + 4)
        torch.testing.assert_close(model(changed), reference, atol=1e-5, rtol=1e-5)
        single = collate_molecules(molecular_fixture()[:1])
        torch.testing.assert_close(model(single), reference[:1], atol=1e-5, rtol=1e-5)
        clone = EGNNAdapter(root, config, torch.device("cpu"))
        clone.load_state_dict(model.state_dict())
        torch.testing.assert_close(clone(batch), reference)


def test_original_xyz_id_gap_and_geometry_contract():
    from molgap.qm9_models.geometry import read_xyz, validate_identity

    xyz = "\n".join(
        [
            "3",
            "gdb 3 1 2 3 4 5 -0.3 0.1 0.4 19 0.02 -1 -1 -1 -1 6",
            "O 0 0 0 0",
            "H 9.5*^-1 0 0 0",
            "H -0.24 0.92 0 0",
            "100 200 300",
            "O O",
            "InChI=1S/H2O/h1H2",
        ]
    )
    record = read_xyz(xyz)
    assert record.gap_hartree == 0.4  # Must not accidentally read r2 (19) as gap.
    validate_identity(record, "gdb_3", "O", 0.4)
    with pytest.raises(ValueError, match="ID mismatch"):
        validate_identity(record, "gdb_2", "O", 0.4)
    with pytest.raises(ValueError, match="gap mismatch"):
        validate_identity(record, "gdb_3", "O", 10.8845)  # eV is not Hartree.
    with pytest.raises(ValueError, match="SMILES identity"):
        validate_identity(record, "gdb_3", "N", 0.4)


@pytest.mark.parametrize("name", ["schnet", "painn"])
def test_schnetpack_invariance_gradient_and_batch_padding(name):
    from molgap.qm9_models.schnetpack import SchNetPackAdapter

    torch.manual_seed(3)
    torch.set_num_threads(2)
    model = SchNetPackAdapter(
        name, {"cutoff": 5.0, "n_atom_basis": 16, "n_interactions": 2, "n_rbf": 10}
    )
    batch = collate_molecules(molecular_fixture())
    reference = model(batch)
    assert reference.shape == (2,) and torch.isfinite(reference).all()
    reference.sum().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    with torch.no_grad():
        rotation, _ = torch.linalg.qr(torch.randn(3, 3))
        rotated = dict(batch, positions=batch["positions"] @ rotation + 2)
        torch.testing.assert_close(model(rotated), reference, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(
            model(collate_molecules(molecular_fixture()[:1])), reference[:1], atol=1e-5, rtol=1e-5
        )
