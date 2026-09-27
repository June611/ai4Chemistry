"""Independent HOMO, LUMO, and gap heads on a shared Chemprop fingerprint."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from chemprop.nn.ffn import MLP
from chemprop.nn.metrics import MSE, ChempropMetric
from chemprop.nn.predictors import Predictor
from chemprop.nn.transforms import UnscaleTransform
from lightning.pytorch.core.mixins import HyperparametersMixin
from torch import Tensor, nn


class QuantumMultiTaskHead(Predictor, HyperparametersMixin):
    """Three independent MLP heads with a shared molecular representation."""

    n_targets = 1
    _T_default_metric = MSE

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int | Sequence[int] = 300,
        n_layers: int = 1,
        dropout: float = 0.0,
        activation: str | nn.Module = "relu",
        criterion: ChempropMetric | None = None,
        output_transform: UnscaleTransform | None = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["criterion", "output_transform", "activation"])
        self.hparams["criterion"] = criterion
        self.hparams["output_transform"] = output_transform
        self.hparams["activation"] = activation
        self.hparams["cls"] = self.__class__
        self.heads = nn.ModuleList(
            [MLP.build(input_dim, 1, hidden_dim, n_layers, dropout, activation) for _ in range(3)]
        )
        self.criterion = criterion or MSE(task_weights=torch.ones(3))
        self.output_transform = output_transform if output_transform is not None else nn.Identity()

    @property
    def input_dim(self) -> int:
        return self.heads[0].input_dim

    @property
    def output_dim(self) -> int:
        return 3

    @property
    def n_tasks(self) -> int:
        return 3

    def forward(self, fingerprints: Tensor) -> Tensor:
        predictions = torch.cat([head(fingerprints) for head in self.heads], dim=1)
        return self.output_transform(predictions)

    train_step = forward

    def encode(self, fingerprints: Tensor, i: int) -> Tensor:
        if i == 0:
            return fingerprints
        return torch.cat([head[:i](fingerprints) for head in self.heads], dim=1)
