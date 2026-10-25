"""
Work 2: MPNN-Transformer with 3D Geometry Injection and Multi-Task Learning.

Extends Work 1 with:
- 3D positional encoding (MLP or Fourier)
- λ gating for geometry injection strength
- Multi-task prediction (7 quantum properties)
"""

import torch
import torch.nn as nn
from chemprop.data import BatchMolGraph
from chemprop.nn import BondMessagePassing

from molgap.nn.batch_utils import batch_atoms
from molgap.nn.pooling import AttentionPooling
from molgap.nn.positional_encoding import FourierPositionalEncoding, MLPPositionalEncoding


class MPNNTransformer3D(nn.Module):
    """
    Work 2: Unified 2D/3D framework with λ gating.

    When λ=0 or coords=None, degrades to Work 1 (pure 2D).
    """

    def __init__(
        self,
        # MPNN parameters
        d_v: int = 72,
        d_e: int = 14,
        d_h: int = 300,
        mpnn_depth: int = 3,
        # Transformer parameters
        d_model: int = 256,  # Must be divisible by nhead
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        # 3D encoding
        pe_type: str = 'mlp',  # 'mlp' or 'fourier'
        num_frequencies: int = 50,  # for Fourier PE
        lambda_geometry: float = 1.0,  # geometry injection strength
        learnable_lambda: bool = False,  # if True, λ is learnable
        # Multi-task
        n_tasks: int = 7,  # 7 quantum properties
    ):
        super().__init__()

        # 1. Local encoding: BondMessagePassing
        self.mpnn = BondMessagePassing(
            d_v=d_v,
            d_e=d_e,
            d_h=d_h,
            depth=mpnn_depth,
            dropout=dropout,
        )

        mpnn_output_dim = d_h

        if mpnn_output_dim != d_model:
            self.projection = nn.Linear(mpnn_output_dim, d_model)
        else:
            self.projection = nn.Identity()

        # 2. 3D Positional Encoding
        if pe_type == 'mlp':
            self.pos_encoder = MLPPositionalEncoding(d_model=d_model)
        elif pe_type == 'fourier':
            self.pos_encoder = FourierPositionalEncoding(
                d_model=d_model,
                num_frequencies=num_frequencies,
                learnable=False
            )
        else:
            raise ValueError(f"Unknown pe_type: {pe_type}")

        # λ gating parameter
        if learnable_lambda:
            self.lambda_geometry = nn.Parameter(torch.tensor(lambda_geometry))
        else:
            self.register_buffer('lambda_geometry', torch.tensor(lambda_geometry))

        # 3. Global encoding: Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 4. Pooling
        self.pooling = AttentionPooling(d_model)

        # 5. Multi-task prediction head
        self.predictor = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_tasks)
        )

    def forward(self, bmg: BatchMolGraph, coords=None, V_d=None):
        """
        Args:
            bmg: BatchMolGraph from Chemprop
            coords: [N_total, 3] 3D coordinates (optional)
            V_d: optional additional atom descriptors

        Returns:
            y_pred: [B, n_tasks] predictions
        """
        # Stage 1: MPNN encodes local environment
        h_local = self.mpnn(bmg, V_d)  # [N_total, d_h + d_v]
        h_local = self.projection(h_local)  # [N_total, d_model]

        # Stage 2: Add 3D positional encoding if coordinates provided
        if coords is not None:
            pe = self.pos_encoder(coords)  # [N_total, d_model]
            h_local = h_local + self.lambda_geometry * pe

        # Stage 3: Batch atoms
        h_batched, mask = batch_atoms(h_local, bmg.batch)  # [B, N_max, d_model]

        # Stage 4: Transformer
        h_global = self.transformer(
            src=h_batched,
            src_key_padding_mask=mask
        )  # [B, N_max, d_model]

        # Stage 5: Pooling
        z = self.pooling(h_global, mask)  # [B, d_model]

        # Stage 6: Predict
        y_pred = self.predictor(z)  # [B, n_tasks]

        return y_pred
