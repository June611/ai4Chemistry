"""New Chemprop heads and molecular network; no legacy model is modified."""

import torch
from chemprop.nn import BondMessagePassing, MeanAggregation, NormAggregation, SumAggregation
from chemprop.nn.ffn import MLP
from torch import nn

from molgap.multitask7.affinity import TaskAffinityModule
from molgap.multitask7.config import TASK_NAMES, VARIANTS
from molgap.multitask7.losses import UncertaintyWeightedMSE


class ChempropMultiTask7(nn.Module):
    """Seven native Chemprop MLPs, split before their final output linear layer."""

    def __init__(
        self,
        input_dim=300,
        hidden_dim=300,
        n_layers=1,
        dropout=0.1,
        activation="relu",
        *,
        use_affinity=False,
        use_uncertainty=False,
        temperature=1.0,
        identity_bias=2.0,
        head_layout="matched",
    ):
        super().__init__()
        self.extra_projectors = None
        if use_affinity and head_layout == "extra_projection":
            # User-selected literal specification: project -> mix -> a full native MLP.
            self.extra_projectors = nn.ModuleList(
                [
                    nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout))
                    for _ in TASK_NAMES
                ]
            )
        head_input = hidden_dim if self.extra_projectors is not None else input_dim
        self.task_projectors, self.task_heads = nn.ModuleList(), nn.ModuleList()
        for _ in TASK_NAMES:
            native = MLP.build(head_input, 1, hidden_dim, n_layers, dropout, activation)
            # Match native Linear -> activation -> dropout -> Linear exactly.
            blocks = list(native.children())
            final = list(blocks[-1].children())
            self.task_projectors.append(nn.Sequential(*blocks[:-1], *final[:-1]))
            self.task_heads.append(final[-1])
        self.affinity = TaskAffinityModule(7, temperature, identity_bias) if use_affinity else None
        self.criterion = UncertaintyWeightedMSE() if use_uncertainty else None

    def forward(self, fingerprints):
        if self.extra_projectors is not None:
            features = torch.stack(
                [projector(fingerprints) for projector in self.extra_projectors], dim=1
            )
            features = self.affinity.fuse(features)
            return torch.cat(
                [
                    head(self.task_projectors[i](features[:, i]))
                    for i, head in enumerate(self.task_heads)
                ],
                dim=1,
            )
        features = torch.stack(
            [projector(fingerprints) for projector in self.task_projectors], dim=1
        )
        if self.affinity is not None:
            features = self.affinity.fuse(features)
        return torch.cat([head(features[:, i]) for i, head in enumerate(self.task_heads)], dim=1)

    def loss_fn(self, prediction, truth):
        if self.criterion is not None:
            return self.criterion(prediction, truth)
        return (prediction - truth).square().mean()

    def get_uncertainty_params(self):
        if self.criterion is None:
            return None
        return dict(
            task_names=list(TASK_NAMES),
            log_vars=self.criterion.log_vars.detach().cpu().tolist(),
            precision_weights=self.criterion.get_task_weights().detach().cpu().tolist(),
        )

    def get_affinity_matrix(self):
        return None if self.affinity is None else self.affinity.get_matrix_dict(TASK_NAMES)


class ChempropMultiTask7Affinity(ChempropMultiTask7):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, use_affinity=True)


class ChempropMultiTask7Uncertainty(ChempropMultiTask7):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, use_uncertainty=True)


class ChempropMultiTask7AffinityUncertainty(ChempropMultiTask7):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, use_affinity=True, use_uncertainty=True)


class ChempropSevenTaskNetwork(nn.Module):
    def __init__(self, config):
        super().__init__()
        m = config["model"]
        # Instantiate the encoder first in every variant to pair its initialization.
        self.message_passing = BondMessagePassing(
            d_h=m["message_hidden_dim"],
            depth=m["depth"],
            dropout=m["dropout"],
            activation=m["activation"],
        )
        self.aggregation = {
            "mean": MeanAggregation,
            "sum": SumAggregation,
            "norm": NormAggregation,
        }[m["aggregation"]]()
        self.batch_norm = (
            nn.BatchNorm1d(m["message_hidden_dim"]) if m["batch_norm"] else nn.Identity()
        )
        constructors = {
            "B": ChempropMultiTask7,
            "C": ChempropMultiTask7Affinity,
            "D1": ChempropMultiTask7Uncertainty,
            "D2": ChempropMultiTask7AffinityUncertainty,
        }
        variant = VARIANTS[config["experiment"]["model_id"]][0]
        self.head = constructors[variant](
            self.message_passing.output_dim,
            m["ffn_hidden_dim"],
            m["ffn_num_layers"],
            m["dropout"],
            m["activation"],
            temperature=m["affinity_temperature"],
            identity_bias=m["affinity_identity_bias"],
            head_layout=m["head_layout"],
        )

    def forward(self, bmg, atom_descriptors=None):
        atoms = self.message_passing(bmg, atom_descriptors)
        fingerprints = self.batch_norm(self.aggregation(atoms, bmg.batch))
        return self.head(fingerprints)
