"""3D positional encoding for coordinates."""

import math

import torch
import torch.nn as nn
from torch import Tensor


class MLPPositionalEncoding(nn.Module):
    """
    Learnable MLP positional encoding for 3D coordinates.

    PE(r) = W_2 · ReLU(W_1 r + b_1) + b_2

    Note: This does NOT guarantee rotation/translation invariance.
    Acceptable because QM9 coordinates are pre-aligned.
    """

    def __init__(self, d_model: int = 300, hidden_dim: int = 128):
        """
        Args:
            d_model: output dimension (must match atom representations)
            hidden_dim: hidden layer dimension
        """
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, d_model)
        )

    def forward(self, coords: Tensor) -> Tensor:
        """
        Args:
            coords: [N, 3] 3D coordinates (flat, not batched)

        Returns:
            pe: [N, d_model] positional encodings
        """
        return self.mlp(coords)


class FourierPositionalEncoding(nn.Module):
    """
    Fourier Features positional encoding for 3D coordinates.

    PE(r) = [sin(2π B r), cos(2π B r)] → Linear projection

    Note: This does NOT guarantee rotation/translation invariance.
    """

    def __init__(
        self,
        d_model: int = 300,
        num_frequencies: int = 50,
        learnable: bool = False
    ):
        """
        Args:
            d_model: output dimension
            num_frequencies: number of frequency bands
            learnable: if True, frequency matrix B is learnable
        """
        super().__init__()
        self.num_frequencies = num_frequencies

        if learnable:
            # Learnable frequency matrix
            self.B = nn.Parameter(torch.randn(num_frequencies, 3) * 0.1)
        else:
            # Fixed frequency matrix (exponentially spaced)
            freq = torch.exp(torch.linspace(0, math.log(10), num_frequencies))
            B = freq[:, None] * torch.randn(num_frequencies, 3)
            self.register_buffer('B', B)

        # Project Fourier features to d_model
        self.proj = nn.Linear(2 * num_frequencies, d_model)

    def forward(self, coords: Tensor) -> Tensor:
        """
        Args:
            coords: [N, 3] 3D coordinates

        Returns:
            pe: [N, d_model] positional encodings
        """
        # Compute 2π B r: [N, 3] @ [3, L] = [N, L]
        x = 2 * math.pi * coords @ self.B.T

        # Fourier features: [N, 2L]
        fourier = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)

        # Project to d_model
        return self.proj(fourier)
