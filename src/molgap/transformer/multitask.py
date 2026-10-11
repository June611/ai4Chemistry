"""SPE Transformer with three task representations and affinity fusion."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


class SpeAffinityMultiTaskRegressor(nn.Module):
    """Predict normalized HOMO, LUMO, and gap values from shared SPE features."""

    task_names = ("homo", "lumo", "delta_e")

    def __init__(self, vocab_size: int, max_length: int, config: dict[str, Any]) -> None:
        super().__init__()
        d_model = int(config["d_model"])
        hidden = int(config["regression_hidden_dim"])
        dropout = float(config["dropout"])
        self.pooling = str(config["pooling"])
        self.use_affinity = bool(config["use_affinity"])
        self.affinity_temperature = float(config.get("affinity_temperature", 1.0))

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
        self.task_projectors = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d_model, hidden),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
                for _ in self.task_names
            ]
        )
        self.task_heads = nn.ModuleList([nn.Linear(hidden, 1) for _ in self.task_names])
        if self.use_affinity:
            identity_bias = float(config.get("affinity_identity_bias", 2.0))
            self.affinity_logits = nn.Parameter(torch.eye(3) * identity_bias)
        else:
            self.register_buffer("fixed_affinity", torch.eye(3), persistent=True)
        self._reset_embeddings()

    def _reset_embeddings(self) -> None:
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.token_embedding.weight[0].zero_()

    def affinity_matrix(self) -> torch.Tensor:
        """Return the row-normalized task mixing matrix in canonical task order."""
        if self.use_affinity:
            return torch.softmax(self.affinity_logits / self.affinity_temperature, dim=-1)
        return self.fixed_affinity

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
        hidden = self.token_embedding(input_ids) + self.position_embedding(positions)
        hidden = self.embedding_dropout(hidden)
        hidden = self.encoder(hidden, src_key_padding_mask=~attention_mask.bool())
        if self.pooling == "cls":
            return hidden[:, 0]
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Return normalized predictions ordered as HOMO, LUMO, delta_e."""
        pooled = self.encode(input_ids, attention_mask)
        task_features = torch.stack(
            [projector(pooled) for projector in self.task_projectors], dim=1
        )
        fused = torch.einsum("ij,bjh->bih", self.affinity_matrix(), task_features)
        return torch.cat(
            [head(fused[:, index]) for index, head in enumerate(self.task_heads)], dim=1
        )
