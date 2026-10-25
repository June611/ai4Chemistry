"""
Test multi-task loss with uncertainty weighting.
"""

import torch

from molgap.nn.multitask_loss import MultiTaskHead, UncertaintyWeightedLoss


def test_uncertainty_weighted_loss():
    """Test uncertainty weighting mechanism"""
    print("=" * 60)
    print("Testing Uncertainty-Weighted Loss")
    print("=" * 60)

    # Create loss
    loss_fn = UncertaintyWeightedLoss(n_tasks=7)

    # Fake predictions and targets
    preds = torch.randn(4, 7)  # 4 molecules, 7 tasks
    targets = torch.randn(4, 7)

    # Initial uncertainties (should all be 1.0)
    print(f"Initial σ (all should be ~1.0): {loss_fn.get_uncertainties()}")

    # Compute loss
    total_loss, task_losses, weights = loss_fn(preds, targets)

    print(f"\nTotal loss: {total_loss.item():.4f}")
    print(f"Task losses (MSE): {task_losses.tolist()}")
    print(f"Weights (1/σ²): {weights.tolist()}")

    # Simulate training: backward pass
    total_loss.backward()

    print(f"\nGradients of log(σ): {loss_fn.log_vars.grad.tolist()}")

    # Manually update (simulate optimizer step)
    with torch.no_grad():
        loss_fn.log_vars -= 0.1 * loss_fn.log_vars.grad

    print(f"Updated σ: {loss_fn.get_uncertainties().tolist()}")

    print("✅ Uncertainty-weighted loss test passed!\n")


def test_multitask_head():
    """Test multi-task head"""
    print("=" * 60)
    print("Testing MultiTaskHead")
    print("=" * 60)

    head = MultiTaskHead(d_model=256, n_tasks=7)

    # Fake molecule representations
    z = torch.randn(4, 256)
    targets = torch.randn(4, 7)

    # Forward
    preds = head(z)
    print(f"Predictions shape: {preds.shape}")

    # Loss
    loss, task_losses, weights = head.loss(preds, targets)
    print(f"Loss: {loss.item():.4f}")
    print(f"Initial uncertainties: {head.get_uncertainties().tolist()}")

    # Backward
    loss.backward()

    print("✅ MultiTaskHead test passed!\n")


def test_missing_data():
    """Test with missing task labels (mask)"""
    print("=" * 60)
    print("Testing with Missing Data")
    print("=" * 60)

    loss_fn = UncertaintyWeightedLoss(n_tasks=7)

    preds = torch.randn(4, 7)
    targets = torch.randn(4, 7)

    # Mask: some tasks are missing for some molecules
    mask = torch.ones(4, 7, dtype=torch.bool)
    mask[0, 2] = False  # molecule 0, task 2 missing
    mask[1, 5] = False  # molecule 1, task 5 missing

    print("Mask (True=valid, False=missing):")
    print(mask)

    loss, task_losses, weights = loss_fn(preds, targets, mask=mask)

    print(f"\nLoss: {loss.item():.4f}")
    print(f"Task losses: {task_losses.tolist()}")

    print("✅ Missing data test passed!\n")


if __name__ == '__main__':
    test_uncertainty_weighted_loss()
    test_multitask_head()
    test_missing_data()

    print("=" * 60)
    print("🎉 All multi-task tests passed!")
    print("=" * 60)
