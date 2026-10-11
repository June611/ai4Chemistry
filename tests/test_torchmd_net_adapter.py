"""Test TorchMD-Net adapter with synthetic data."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from molgap.qm9_models.batch import Molecule, collate_molecules
from molgap.qm9_models.torchmd_net import TorchMDNetAdapter


def test_torchmd_net():
    """Test TorchMD-Net with synthetic batch."""
    print("Testing TorchMD-Net adapter...")

    device = torch.device("cpu")
    config = {
        "architecture": "equivariant-transformer",
        "embedding_dimension": 64,
        "num_layers": 3,
        "num_rbf": 20,
        "rbf_type": "expnorm",
        "trainable_rbf": False,
        "activation": "silu",
        "cutoff": 5.0,
        "num_heads": 4,
        "distance_influence": "both",
    }

    root = Path(__file__).parent.parent
    model = TorchMDNetAdapter(root, config).to(device)
    print(f"✓ Model created: {sum(p.numel() for p in model.parameters())} parameters")

    # Create synthetic molecules
    mol1 = Molecule(
        sample_id="test_1",
        smiles="C",
        atomic_numbers=torch.tensor([6, 1, 1, 1, 1], dtype=torch.long),
        positions=torch.randn(5, 3) * 1.0,
        target=0.5,
    )
    mol2 = Molecule(
        sample_id="test_2",
        smiles="O",
        atomic_numbers=torch.tensor([8, 1, 1], dtype=torch.long),
        positions=torch.randn(3, 3) * 1.0,
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

    print("\n✅ TorchMD-Net adapter test passed!\n")


if __name__ == "__main__":
    test_torchmd_net()
