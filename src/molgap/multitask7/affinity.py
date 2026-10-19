"""Global directed task-feature mixing, independent of labels at inference."""

import torch
from torch import nn


class TaskAffinityModule(nn.Module):
    def __init__(self, n_tasks=7, temperature=1.0, identity_bias=2.0):
        super().__init__()
        if temperature <= 0:
            raise ValueError("Temperature must be positive")
        self.temperature, self.identity_bias = temperature, identity_bias
        self.affinity_logits = nn.Parameter(torch.eye(n_tasks) * identity_bias)

    def forward(self):
        return torch.softmax(self.affinity_logits / self.temperature, dim=-1)

    def fuse(self, task_features):
        return torch.einsum("ij,bjh->bih", self(), task_features)

    def get_matrix_dict(self, task_names):
        return dict(
            matrix=self().detach().cpu().tolist(),
            task_names=list(task_names),
            row_interpretation="receiver task",
            col_interpretation="provider task",
            temperature=self.temperature,
            identity_bias=self.identity_bias,
        )
