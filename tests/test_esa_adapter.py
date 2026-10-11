"""Test ESA adapter with synthetic data."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from molgap.qm9_models.batch import Molecule, collate_molecules
from molgap.qm9_models.esa import ATOM_FEATURE_DIM, BOND_FEATURE_DIM, ESAAdapter, chemical_graph


def test_esa():
    """Test ESA with synthetic batch."""
    print("Testing ESA adapter...")

    device = torch.device("cpu")
    config = {"hidden_dim": 64, "num_heads": 4, "layer_types": ["M", "S", "P"], "max_edges": 128}

    model = ESAAdapter(config).to(device)
    print(f"✓ Model created: {sum(p.numel() for p in model.parameters())} parameters")

    # ESA uses SMILES, not 3D coordinates
    mol1 = Molecule(
        sample_id="test_1",
        smiles="C",
        atomic_numbers=torch.tensor([6, 1, 1, 1, 1], dtype=torch.long),
        positions=torch.randn(5, 3),  # Not used by ESA
        target=0.5,
    )
    mol2 = Molecule(
        sample_id="test_2",
        smiles="O",
        atomic_numbers=torch.tensor([8, 1, 1], dtype=torch.long),
        positions=torch.randn(3, 3),  # Not used by ESA
        target=0.3,
    )

    batch = collate_molecules([mol1, mol2])

    # Forward pass
    output = model(batch)
    assert output.shape == (2,)
    assert torch.isfinite(output).all()
    print(f"✓ Forward pass: output shape {output.shape}, values {output.tolist()}")

    # Backward pass
    loss = output.sum()
    loss.backward()
    print("✓ Backward pass successful")

    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    assert has_grad
    print(f"✓ Gradients computed: {has_grad}")

    print("\n✅ ESA adapter test passed!\n")


def test_esa_official_qm9_feature_dimensions():
    nodes, edges, attributes = chemical_graph("C[C@H](F)O")
    assert nodes.shape[1] == ATOM_FEATURE_DIM == 35
    assert attributes.shape[1] == BOND_FEATURE_DIM == 13
    assert edges.shape[1] == attributes.shape[0]


if __name__ == "__main__":
    test_esa()
