"""Attention-based pooling for molecule-level representation."""

import torch
import torch.nn as nn
from torch import Tensor


class AttentionPooling(nn.Module):
    """
    Learnable attention pooling: aggregate atom representations into molecule representation.

    α_i = softmax(w^T h_i)
    z = Σ α_i h_i
    """

    def __init__(self, d_model: int):
        """
        Args:
            d_model: dimension of atom representations
        """
        super().__init__()
        self.attention_weight = nn.Linear(d_model, 1, bias=False)

    def forward(self, h: Tensor, mask: Tensor) -> Tensor:
        """
        Args:
            h: [B, N_max, d] batched atom representations
            mask: [B, N_max] padding mask (True = padding)

        Returns:
            z: [B, d] molecule representations
        """
        # Compute attention scores: [B, N_max, 1]
        scores = self.attention_weight(h)  # [B, N_max, 1]

        # Mask padding atoms
        scores = scores.masked_fill(mask.unsqueeze(-1), float("-inf"))

        # Softmax over atoms
        alpha = torch.softmax(scores, dim=1)  # [B, N_max, 1]

        # Weighted sum
        z = (alpha * h).sum(dim=1)  # [B, d]

        return z
