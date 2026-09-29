"""Compact Transformer encoder for tokenized canonical SMILES."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


class SmilesTransformerRegressor(nn.Module):
    """Token and position embeddings, Transformer encoder, and a two-layer MLP head."""

    def __init__(self, vocab_size: int, max_length: int, config: dict[str, Any]) -> None:
        super().__init__()
        d_model = int(config["d_model"])
        dropout = float(config["dropout"])
        self.pooling = str(config["pooling"])
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.position_embedding = nn.Embedding(max_length, d_model)
        self.embedding_dropout = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=int(config["num_heads"]),
            dim_feedforward=int(config["dim_feedforward"]),
            dropout=dropout,
            activation=str(config["activation"]),
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=int(config["num_layers"]),
            norm=nn.LayerNorm(d_model),
            enable_nested_tensor=False,
        )
        hidden = int(config["regression_hidden_dim"])
        self.regression_head = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        self._reset_embeddings()

    def _reset_embeddings(self) -> None:
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.token_embedding.weight[0].zero_()

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Predict one normalized delta_e value per sequence."""
        positions = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
        hidden = self.token_embedding(input_ids) + self.position_embedding(positions)
        hidden = self.embedding_dropout(hidden)
        hidden = self.encoder(hidden, src_key_padding_mask=~attention_mask.bool())
        if self.pooling == "cls":
            pooled = hidden[:, 0]
        else:
            mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return self.regression_head(pooled).squeeze(-1)


def count_trainable_parameters(model: nn.Module) -> int:
    """Count trainable scalar parameters."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
