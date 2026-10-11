"""Test DimeNet++ adapter with synthetic data."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from molgap.qm9_models.batch import Molecule, collate_molecules
from molgap.qm9_models.dimenet import DimeNetPPAdapter


def test_dimenet_pp():
    """Test DimeNet++ with synthetic batch."""
    print("Testing DimeNet++ adapter...")

    device = torch.device("cpu")
    config = {
        "hidden_channels": 32,
        "num_blocks": 3,
        "int_emb_size": 16,
        "basis_emb_size": 8,
        "out_emb_channels": 16,
        "num_spherical": 4,
        "num_radial": 6,
        "cutoff": 5.0,
        "envelope_exponent": 5,
        "num_before_skip": 1,
        "num_after_skip": 1,
        "num_output_layers": 2,
    }

    model = DimeNetPPAdapter(config).to(device)
    print(f"✓ Model created: {sum(p.numel() for p in model.parameters())} parameters")

    # Create synthetic molecules
    mol1 = Molecule(
        sample_id="test_1",
        smiles="C",
        atomic_numbers=torch.tensor([6, 1, 1, 1, 1], dtype=torch.long),
        positions=torch.randn(5, 3) * 2.0,  # Larger spread for edge formation
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

    print("\n✅ DimeNet++ adapter test passed!\n")


if __name__ == "__main__":
    test_dimenet_pp()
