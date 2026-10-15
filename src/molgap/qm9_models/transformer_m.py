"""Pinned official Transformer-M QM9 encoder, trained from scratch.

The authors' QM9 default activates the 3D channel and disables 2D biases.
No Fairseq installation or external pretrained checkpoint is required.
"""

import torch
from torch import nn

from molgap._vendor.transformer_m.compat import get_activation_fn
from molgap._vendor.transformer_m.transformer_m_encoder import TransformerMEncoderQM9


class TransformerMAdapter(nn.Module):
    def __init__(self, config):
        super().__init__()
        p = dict(config)
        checkpointing = p.pop("activation_checkpointing", True)
        dim = p.pop("embed_dim", 768)
        activation = p.pop("activation_fn", "relu")
        if not p.pop("no_2d", True):
            raise ValueError("This adapter follows the official QM9 no_2d=True configuration")
        self.encoder = TransformerMEncoderQM9(
            num_atoms=512 * 9,
            num_in_degree=512,
            num_out_degree=512,
            num_edges=512 * 3,
            num_spatial=512,
            num_edge_dis=128,
            edge_type="multi_hop",
            multi_hop_max_dist=5,
            num_encoder_layers=p.pop("num_layers", 12),
            embedding_dim=dim,
            ffn_embedding_dim=p.pop("ffn_dim", 768),
            num_attention_heads=p.pop("num_heads", 32),
            dropout=p.pop("dropout", 0.0),
            attention_dropout=p.pop("attention_dropout", 0.1),
            activation_dropout=p.pop("activation_dropout", 0.1),
            activation_fn=activation,
            add_3d=True,
            num_3d_bias_kernel=p.pop("num_kernel", 128),
            no_2d=True,
        )
        if p:
            raise ValueError(f"Unknown Transformer-M parameters: {sorted(p)}")
        self.encoder.activation_checkpointing = checkpointing
        self.head = nn.Sequential(nn.Linear(dim, dim), _Activation(activation), nn.Linear(dim, 1))

    @staticmethod
    def inputs(batch):
        mask = batch["atom_mask"]
        b, n = mask.shape
        features = batch["node_features"]
        offsets = torch.arange(9, device=mask.device) * 512 + 2
        x = (features + offsets).masked_fill(~mask[:, :, None], 0)
        tokens = torch.cat([torch.ones(b, 1, dtype=torch.bool, device=mask.device), mask], 1)
        bias = batch["positions"].new_zeros(b, n + 1, n + 1)
        bias = bias.masked_fill(~tokens[:, None, :], float("-inf"))
        atom_type = x[:, :, 0]
        pair = torch.stack(
            (
                atom_type[:, :, None].expand(-1, -1, n) + 1,
                atom_type[:, None, :].expand(-1, n, -1) + 513,
            ),
            -1,
        )
        pair_mask = mask[:, :, None] & mask[:, None, :]
        pair[..., 0].masked_fill_(~pair_mask, 1)
        pair[..., 1].masked_fill_(~pair_mask, 513)
        # These 2D tensors are read by the upstream API but not used when no_2d.
        zeros = torch.zeros(b, n, dtype=torch.long, device=mask.device)
        return {
            "x": x,
            "pos": batch["positions"],
            "node_type_edge": pair,
            "attn_bias": bias,
            "in_degree": zeros,
            "out_degree": zeros,
            "spatial_pos": batch["spatial"],
            "edge_input": zeros[:, :, None, None, None],
            "attn_edge_type": zeros[:, :, None, None],
        }

    def forward(self, batch):
        states, _ = self.encoder(self.inputs(batch), last_state_only=True)
        return self.head(states[-1][0]).squeeze(-1)


class _Activation(nn.Module):
    def __init__(self, name):
        super().__init__()
        self.function = get_activation_fn(name)

    def forward(self, x):
        return self.function(x)
