"""Test ALIGNN adapter with synthetic data."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from molgap.qm9_models.alignn import ALIGNNAdapter
from molgap.qm9_models.batch import Molecule, collate_molecules


def test_alignn():
    """Test ALIGNN with synthetic batch."""
    print("Testing ALIGNN adapter...")

    device = torch.device("cpu")
    root = Path(__file__).parent.parent
    config = {
        "cutoff": 5.0,
        "alignn_layers": 2,
        "gcn_layers": 2,
        "atom_input_features": 92,
        "edge_input_features": 80,
        "triplet_input_features": 40,
        "embedding_features": 32,
        "hidden_features": 64,
        "output_features": 1,
    }

    model = ALIGNNAdapter(root, config).to(device)
    print(f"✓ Model created: {sum(p.numel() for p in model.parameters())} parameters")

    # Create synthetic molecules with positions spread enough to form edges
    mol1 = Molecule(
        sample_id="test_1",
        smiles="C",
        atomic_numbers=torch.tensor([6, 1, 1, 1, 1], dtype=torch.long),
        positions=torch.randn(5, 3) * 2.0,
        target=0.5,
    )
    mol2 = Molecule(
        sample_id="test_2",
        smiles="O",
        atomic_numbers=torch.tensor([8, 1, 1], dtype=torch.long),
        positions=torch.randn(3, 3) * 2.0,
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

    print("\n✅ ALIGNN adapter test passed!\n")


if __name__ == "__main__":
    test_alignn()
