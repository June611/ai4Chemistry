"""The specified uncertainty objective, with an equal-weight starting point."""

import torch
from torch import nn


class UncertaintyWeightedMSE(nn.Module):
    def __init__(self, n_tasks=7):
        super().__init__()
        self.log_vars = nn.Parameter(torch.zeros(n_tasks))

    def forward(self, prediction, truth):
        mse = (prediction - truth).square().mean(dim=0)
        return (self.get_task_weights() * mse + self.log_vars).mean()

    def get_task_weights(self):
        return torch.exp(-self.log_vars)
