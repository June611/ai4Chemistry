"""Test EGNN adapter with small synthetic data to verify forward/backward pass."""

import sys
from pathlib import Path

import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from molgap.qm9_models.egnn import EGNNAdapter


def test_egnn_adapter():
    """Test EGNN with synthetic batch."""
    print("Testing EGNN adapter...")

    device = torch.device("cpu")
    root = Path(__file__).parent.parent
    config = {"hidden_nf": 32, "n_layers": 3, "attention": True, "node_attr": 1}

    # Create model
    model = EGNNAdapter(root, config, device)
    print(f"✓ Model created: {sum(p.numel() for p in model.parameters())} parameters")

    # Create synthetic batch (2 molecules: CH4 and H2O)
    batch = {
        "charges": torch.tensor(
            [[6, 1, 1, 1, 1], [8, 1, 1, 0, 0]], dtype=torch.long
        ),  # C with 4 H, O with 2 H
        "one_hot": torch.zeros(2, 5, 5),
        "positions": torch.randn(2, 5, 3) * 0.1,
        "atom_mask": torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]], dtype=torch.bool),
        "edge_mask": torch.ones(2, 5, 5, dtype=torch.bool),
    }

    # Set one-hot encoding
    batch["one_hot"][0, 0, 4] = 1  # C at position 0
    batch["one_hot"][0, 1:5, 0] = 1  # H at positions 1-4
    batch["one_hot"][1, 0, 3] = 1  # O at position 0
    batch["one_hot"][1, 1:3, 0] = 1  # H at positions 1-2

    # Set edge mask (no self-loops for padded atoms)
    for i in range(2):
        mask = batch["atom_mask"][i]
        batch["edge_mask"][i] = mask.unsqueeze(0) & mask.unsqueeze(1)

    # Forward pass
    output = model(batch)
    assert output.shape == (2,)
    assert torch.isfinite(output).all()
    print(f"✓ Forward pass: output shape {output.shape}, values {output.tolist()}")

    # Backward pass
    loss = output.sum()
    loss.backward()
    print("✓ Backward pass successful")

    # Check gradients
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    assert has_grad
    print(f"✓ Gradients computed: {has_grad}")

    print("\n✅ EGNN adapter test passed!\n")


if __name__ == "__main__":
    test_egnn_adapter()
