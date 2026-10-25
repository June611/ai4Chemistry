"""
Test script for MPNN-Transformer hybrid models.

Tests:
1. Work 1 (2D only)
2. Work 2 with λ=0 (should match Work 1 dimensions)
3. Work 2 with λ=1.0 and 3D coords
"""

import torch
from chemprop.data import MoleculeDatapoint, MoleculeDataset, build_dataloader
from chemprop.featurizers import SimpleMoleculeMolGraphFeaturizer

from molgap.models.mpnn_transformer import MPNNTransformerHybrid
from molgap.models.mpnn_transformer_3d import MPNNTransformer3D


def test_work1():
    """Test Work 1: 2D only"""
    print("=" * 60)
    print("Testing Work 1: MPNN-Transformer (2D only)")
    print("=" * 60)

    # Create test molecules
    smiles = ["CCO", "c1ccccc1", "CC(C)C"]
    datapoints = [MoleculeDatapoint.from_smi(smi) for smi in smiles]
    dataset = MoleculeDataset(datapoints, SimpleMoleculeMolGraphFeaturizer())
    loader = build_dataloader(dataset, batch_size=3, shuffle=False)

    # Get one batch
    batch = next(iter(loader))
    bmg = batch.bmg

    print(f"Batch size: {len(batch)}")
    print(f"Total atoms: {len(bmg.batch)}")
    print(f"bmg.batch: {bmg.batch}")

    # Create model
    model = MPNNTransformerHybrid(
        d_h=300,
        mpnn_depth=3,
        d_model=256,  # Must be divisible by nhead
        nhead=8,
        num_layers=4,
        n_tasks=1,  # single task
    )

    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Forward pass
    model.eval()
    with torch.no_grad():
        y_pred = model(bmg)

    print(f"Output shape: {y_pred.shape}")
    print(f"Predictions: {y_pred.squeeze().tolist()}")
    print("✅ Work 1 test passed!\n")


def test_work2_2d():
    """Test Work 2 with λ=0 (pure 2D, should match Work 1 architecture)"""
    print("=" * 60)
    print("Testing Work 2 with λ=0 (degraded to 2D)")
    print("=" * 60)

    smiles = ["CCO", "c1ccccc1", "CC(C)C"]
    datapoints = [MoleculeDatapoint.from_smi(smi) for smi in smiles]
    dataset = MoleculeDataset(datapoints, SimpleMoleculeMolGraphFeaturizer())
    loader = build_dataloader(dataset, batch_size=3, shuffle=False)
    batch = next(iter(loader))
    bmg = batch.bmg

    # Create model with λ=0
    model = MPNNTransformer3D(
        d_h=300,
        mpnn_depth=3,
        d_model=256,
        nhead=8,
        num_layers=4,
        pe_type="mlp",
        lambda_geometry=0.0,  # Pure 2D
        n_tasks=7,  # multi-task
    )

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Forward without coords
    model.eval()
    with torch.no_grad():
        y_pred = model(bmg, coords=None)

    print(f"Output shape: {y_pred.shape}")
    print(f"First molecule predictions (7 tasks): {y_pred[0].tolist()}")
    print("✅ Work 2 (λ=0) test passed!\n")


def test_work2_3d():
    """Test Work 2 with 3D coords and λ=1.0"""
    print("=" * 60)
    print("Testing Work 2 with 3D coordinates (λ=1.0)")
    print("=" * 60)

    smiles = ["CCO", "c1ccccc1"]
    datapoints = [MoleculeDatapoint.from_smi(smi) for smi in smiles]
    dataset = MoleculeDataset(datapoints, SimpleMoleculeMolGraphFeaturizer())
    loader = build_dataloader(dataset, batch_size=2, shuffle=False)
    batch = next(iter(loader))
    bmg = batch.bmg

    # Generate fake 3D coords (in real training, these come from QM9 SDF)
    n_atoms = len(bmg.batch)
    coords = torch.randn(n_atoms, 3) * 2.0  # fake coordinates

    print(f"Total atoms: {n_atoms}")
    print(f"Coords shape: {coords.shape}")

    # Test MLP PE
    print("\n--- Testing MLP PE ---")
    model_mlp = MPNNTransformer3D(
        d_h=300,
        mpnn_depth=3,
        d_model=256,
        nhead=8,
        num_layers=4,
        pe_type="mlp",
        lambda_geometry=1.0,
        n_tasks=7,
    )

    model_mlp.eval()
    with torch.no_grad():
        y_pred_mlp = model_mlp(bmg, coords=coords)

    print(f"Output shape: {y_pred_mlp.shape}")
    print(f"First molecule predictions: {y_pred_mlp[0].tolist()}")

    # Test Fourier PE
    print("\n--- Testing Fourier PE ---")
    model_fourier = MPNNTransformer3D(
        d_h=300,
        mpnn_depth=3,
        d_model=256,
        nhead=8,
        num_layers=4,
        pe_type="fourier",
        num_frequencies=50,
        lambda_geometry=1.0,
        n_tasks=7,
    )

    model_fourier.eval()
    with torch.no_grad():
        y_pred_fourier = model_fourier(bmg, coords=coords)

    print(f"Output shape: {y_pred_fourier.shape}")
    print(f"First molecule predictions: {y_pred_fourier[0].tolist()}")

    print("✅ Work 2 (3D) test passed!\n")


def test_learnable_lambda():
    """Test learnable λ"""
    print("=" * 60)
    print("Testing learnable λ")
    print("=" * 60)

    smiles = ["CCO"]
    datapoints = [MoleculeDatapoint.from_smi(smi) for smi in smiles]
    dataset = MoleculeDataset(datapoints, SimpleMoleculeMolGraphFeaturizer())
    loader = build_dataloader(dataset, batch_size=1, shuffle=False)
    batch = next(iter(loader))
    bmg = batch.bmg

    coords = torch.randn(len(bmg.batch), 3)

    model = MPNNTransformer3D(lambda_geometry=0.5, learnable_lambda=True, n_tasks=1)

    print(f"Initial λ: {model.lambda_geometry.item():.4f}")
    print(f"λ requires_grad: {model.lambda_geometry.requires_grad}")

    # Forward + backward
    y_pred = model(bmg, coords=coords)
    loss = y_pred.sum()
    loss.backward()

    print(f"λ gradient: {model.lambda_geometry.grad.item():.4f}")
    print("✅ Learnable λ test passed!\n")


if __name__ == "__main__":
    test_work1()
    test_work2_2d()
    test_work2_3d()
    test_learnable_lambda()

    print("=" * 60)
    print("🎉 All tests passed!")
    print("=" * 60)
