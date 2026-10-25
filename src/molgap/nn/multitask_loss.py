"""Uncertainty-weighted multi-task loss (Kendall et al. 2018)."""

import torch
import torch.nn as nn


class UncertaintyWeightedLoss(nn.Module):
    """
    Multi-task loss with learnable uncertainty weighting.

    L = Σ_t [ (1/2σ²_t) · MSE_t + log σ_t ]

    Reference:
        Kendall, A., Gal, Y., & Cipolla, R. (2018).
        Multi-task learning using uncertainty to weigh losses for scene geometry and semantics.
        CVPR 2018.

    Each task t has a learnable uncertainty parameter σ_t:
    - Large σ_t → task t is uncertain → small weight (1/σ²_t)
    - Small σ_t → task t is certain → large weight
    - log σ_t term prevents σ_t → ∞
    """

    def __init__(self, n_tasks: int = 7):
        """
        Args:
            n_tasks: number of tasks
        """
        super().__init__()
        # Store log(σ) instead of σ to ensure σ > 0
        # Initialize to 0 → σ = exp(0) = 1
        self.log_vars = nn.Parameter(torch.zeros(n_tasks))

    def forward(self, preds, targets, mask=None):
        """
        Args:
            preds: [B, n_tasks] predictions
            targets: [B, n_tasks] ground truth
            mask: [B, n_tasks] optional mask (True = valid, False = missing)
                  If None, assumes all tasks are valid

        Returns:
            loss: scalar weighted loss
            task_losses: [n_tasks] individual task losses (for logging)
            weights: [n_tasks] current weights (1/σ²) (for logging)
        """
        # Compute per-task MSE: [n_tasks]
        squared_errors = (preds - targets).pow(2)  # [B, n_tasks]

        if mask is not None:
            # Only compute loss on valid entries
            squared_errors = squared_errors * mask.float()
            n_valid = mask.float().sum(dim=0).clamp(min=1)  # [n_tasks]
            mse_per_task = squared_errors.sum(dim=0) / n_valid
        else:
            mse_per_task = squared_errors.mean(dim=0)  # [n_tasks]

        # Precision = 1/σ² = exp(-2 log σ)
        precision = torch.exp(-2 * self.log_vars)  # [n_tasks]

        # Weighted loss = (1/2σ²) MSE + log σ
        weighted_loss = 0.5 * precision * mse_per_task + self.log_vars

        # Sum over tasks
        total_loss = weighted_loss.sum()

        # For logging
        task_losses = mse_per_task.detach()
        weights = precision.detach()

        return total_loss, task_losses, weights

    def get_uncertainties(self):
        """Return current uncertainty σ for each task."""
        return torch.exp(self.log_vars).detach()


class MultiTaskHead(nn.Module):
    """
    Multi-task prediction head with uncertainty-weighted loss.

    Usage in training:
        head = MultiTaskHead(d_model=256, n_tasks=7)
        preds = head(z)  # z: [B, d_model]
        loss, task_losses, weights = head.loss(preds, targets)
    """

    def __init__(
        self,
        d_model: int,
        n_tasks: int = 7,
        hidden_dim: int = None,
        dropout: float = 0.1,
    ):
        super().__init__()

        if hidden_dim is None:
            hidden_dim = d_model // 2

        self.predictor = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_tasks),
        )

        self.loss_fn = UncertaintyWeightedLoss(n_tasks)

    def forward(self, z):
        """
        Args:
            z: [B, d_model] molecule representations

        Returns:
            preds: [B, n_tasks] predictions
        """
        return self.predictor(z)

    def loss(self, preds, targets, mask=None):
        """Compute uncertainty-weighted loss."""
        return self.loss_fn(preds, targets, mask)

    def get_uncertainties(self):
        """Return current σ for each task."""
        return self.loss_fn.get_uncertainties()
