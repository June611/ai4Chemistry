"""
Work 1: MPNN-Transformer Hybrid Architecture (2D only).

This model combines:
- Chemprop's BondMessagePassing for local chemical environment encoding
- Transformer encoder for global interaction modeling
"""

import torch
import torch.nn as nn
from chemprop.nn import BondMessagePassing
from chemprop.data import BatchMolGraph

from molgap.nn.batch_utils import batch_atoms
from molgap.nn.pooling import AttentionPooling


class MPNNTransformerHybrid(nn.Module):
    """
    Hybrid architecture: BondMessagePassing (local) + Transformer (global).

    This is Work 1 (2D only, no 3D coordinates).
    """

    def __init__(
        self,
        # MPNN parameters
        d_v: int = 72,   # atom feature dim (SimpleMoleculeMolGraphFeaturizer default)
        d_e: int = 14,   # bond feature dim (Chemprop default)
        d_h: int = 300,  # MPNN hidden dim
        mpnn_depth: int = 3,
        # Transformer parameters
        d_model: int = 256,  # Must be divisible by nhead
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        # Task
        n_tasks: int = 1,  # single task (gap) for work 1
    ):
        super().__init__()

        # 1. Local encoding: BondMessagePassing (Chemprop 2.x)
        self.mpnn = BondMessagePassing(
            d_v=d_v,
            d_e=d_e,
            d_h=d_h,
            depth=mpnn_depth,
            dropout=dropout,
        )

        # MPNN output is [m_i ⊕ v_i] with shape [N, d_h + d_v]
        mpnn_output_dim = d_h

        # Project MPNN output to d_model if needed
        if mpnn_output_dim != d_model:
            self.projection = nn.Linear(mpnn_output_dim, d_model)
        else:
            self.projection = nn.Identity()

        # 2. Global encoding: Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,  # Important: input is [B, N, d]
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 3. Pooling: attention-based aggregation
        self.pooling = AttentionPooling(d_model)

        # 4. Prediction head
        self.predictor = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_tasks)
        )

    def forward(self, bmg: BatchMolGraph, V_d=None):
        """
        Args:
            bmg: BatchMolGraph from Chemprop
            V_d: optional additional atom descriptors (not used in work 1)

        Returns:
            y_pred: [B, n_tasks] predictions
        """
        # Stage 1: MPNN encodes local chemical environment
        h_local = self.mpnn(bmg, V_d)  # [N_total, d_h + d_v]

        # Project to d_model
        h_local = self.projection(h_local)  # [N_total, d_model]

        # Stage 2: Batch atoms for Transformer
        h_batched, mask = batch_atoms(h_local, bmg.batch)  # [B, N_max, d_model], [B, N_max]

        # Stage 3: Transformer encodes global interactions
        h_global = self.transformer(
            src=h_batched,
            src_key_padding_mask=mask  # True = padding
        )  # [B, N_max, d_model]

        # Stage 4: Attention pooling
        z = self.pooling(h_global, mask)  # [B, d_model]

        # Stage 5: Predict
        y_pred = self.predictor(z)  # [B, n_tasks]

        return y_pred
