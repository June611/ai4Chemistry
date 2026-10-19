"""Backbone-independent heads. All outputs are in per-task Z-score space."""

import torch
from torch import nn


class IndependentTaskHead(nn.Module):
    def __init__(self, input_dim, n_tasks, hidden_dim=128, dropout=0.1):
        super().__init__()
        self.projectors = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout))
                for _ in range(n_tasks)
            ]
        )
        self.outputs = nn.ModuleList([nn.Linear(hidden_dim, 1) for _ in range(n_tasks)])

    def features(self, h):
        return torch.stack([p(h) for p in self.projectors], dim=1)

    def predict(self, tokens):
        return torch.cat([head(tokens[:, j]) for j, head in enumerate(self.outputs)], dim=-1)

    def forward(self, h):
        return self.predict(self.features(h))


class TaskAttentionHead(IndependentTaskHead):
    def __init__(self, input_dim, n_tasks=7, hidden_dim=128, num_heads=4, dropout=0.1):
        super().__init__(input_dim, n_tasks, hidden_dim, dropout)
        if hidden_dim % num_heads:
            raise ValueError("Task width must be divisible by the attention head count")
        self.norm = nn.LayerNorm(hidden_dim)
        self.attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.gate = nn.Parameter(torch.zeros(()))
        self.last_attention = None

    def forward(self, h):
        tokens = self.features(h)
        normalized = self.norm(tokens)
        update, weights = self.attention(
            normalized, normalized, normalized, need_weights=True, average_attn_weights=False
        )
        self.last_attention = weights.detach()
        return self.predict(tokens + self.gate * update)
