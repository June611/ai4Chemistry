"""Test 3D Infomax adapter with synthetic data."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from molgap.qm9_models.batch import Molecule, collate_molecules
from molgap.qm9_models.infomax import ThreeDInfomaxAdapter
from molgap.qm9_models.train import _validate_infomax_checkpoint


def test_3dinfomax(tmp_path):
    """Test 3D Infomax with synthetic batch (without pretrained weights)."""
    print("Testing 3D Infomax adapter...")

    device = torch.device("cpu")
    root = Path(__file__).parent.parent
    config = {
        "hidden_dim": 32,
        "propagation_depth": 2,
        "aggregators": ["mean", "max"],
        "scalers": ["identity"],
        "readout_aggregators": ["mean", "sum"],
        "mid_batch_norm": False,
        "last_batch_norm": False,
        "readout_batchnorm": False,
        "pretrain_checkpoint": None,
        "require_pretrained": False,
    }

    model = ThreeDInfomaxAdapter(root, config).to(device)
    print(f"✓ Model created: {sum(p.numel() for p in model.parameters())} parameters")
    print("  Note: Training from scratch (no pretrained weights)")

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

    encoder_state = {
        key: value.detach().clone()
        for key, value in model.network.state_dict().items()
        if key.startswith("node_gnn.") and "batch_norm" not in key
    }
    checkpoint = tmp_path / "qm9_infomax.pt"
    torch.save({"model_state_dict": encoder_state}, checkpoint)
    transfer_config = dict(
        config,
        pretrain_checkpoint=str(checkpoint),
        require_pretrained=True,
    )
    transferred = ThreeDInfomaxAdapter(root, transfer_config)
    assert transferred.transferred_parameter_count == len(encoder_state)

    print("\n✅ 3D Infomax adapter test passed!\n")
    print("⚠️  WARNING: For proper results, must use QM9-pretrained weights")


def test_3dinfomax_requires_pretraining_checkpoint():
    root = Path(__file__).parent.parent
    with pytest.raises(ValueError, match="requires a QM9-pretrained checkpoint"):
        ThreeDInfomaxAdapter(root, {"hidden_dim": 16, "propagation_depth": 1})


def test_3dinfomax_rejects_checkpoint_from_another_split(tmp_path):
    checkpoint = tmp_path / "other_split.pt"
    torch.save(
        {
            "model_state_dict": {},
            "split_manifest_sha256": "other-manifest",
            "geometry_sha256": "geometry-a",
        },
        checkpoint,
    )
    config = {
        "model": {
            "name": "3dinfomax",
            "params": {"pretrain_checkpoint": str(checkpoint)},
        }
    }
    with pytest.raises(ValueError, match="not pretrained on this split"):
        _validate_infomax_checkpoint(
            config,
            Path(__file__).parent.parent,
            {"manifest": {"sha256": "selected-manifest"}},
            {"geometry_sha256": "geometry-a"},
        )


def test_3dinfomax_rejects_unpaired_pretraining_seed(tmp_path):
    checkpoint = tmp_path / "wrong_seed.pt"
    torch.save(
        {
            "model_state_dict": {},
            "split_manifest_sha256": "selected-manifest",
            "geometry_sha256": "geometry-a",
            "training_seed": 3407,
        },
        checkpoint,
    )
    config = {
        "data": {"split": {"seed": 42}},
        "model": {
            "name": "3dinfomax",
            "params": {"pretrain_checkpoint": str(checkpoint)},
        },
    }
    with pytest.raises(ValueError, match="pretraining seed differs"):
        _validate_infomax_checkpoint(
            config,
            Path(__file__).parent.parent,
            {"manifest": {"sha256": "selected-manifest"}},
            {"geometry_sha256": "geometry-a"},
        )


def test_3dinfomax_cross_modal_pretraining_step():
    from molgap.qm9_models.pretrain_infomax import _activate, _forward

    root = Path(__file__).parent.parent
    PNA, Net3D, NTXent = _activate(root)
    model2d = PNA(
        hidden_dim=16,
        target_dim=8,
        aggregators=["mean", "max"],
        scalers=["identity"],
        readout_aggregators=["mean"],
        readout_batchnorm=False,
        mid_batch_norm=False,
        last_batch_norm=False,
        propagation_depth=1,
    )
    model3d = Net3D(
        node_dim=0,
        edge_dim=0,
        hidden_dim=8,
        target_dim=8,
        readout_aggregators=["mean"],
        readout_batchnorm=False,
        batch_norm=False,
        propagation_depth=1,
        node_wise_output_layers=0,
        message_net_layers=1,
        update_net_layers=1,
    )
    molecules = [
        Molecule("methane", "C", torch.tensor([6, 1, 1, 1, 1]), torch.randn(5, 3), 0.5),
        Molecule("water", "O", torch.tensor([8, 1, 1]), torch.randn(3, 3), 0.3),
    ]
    z2d, z3d = _forward(model2d, model3d, collate_molecules(molecules), torch.device("cpu"))
    assert z2d.shape == z3d.shape == (2, 8)
    loss = NTXent(tau=0.1)(z2d, z3d)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(parameter.grad is not None for parameter in model2d.parameters())
    assert any(parameter.grad is not None for parameter in model3d.parameters())


if __name__ == "__main__":
    test_3dinfomax()
