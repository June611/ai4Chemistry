"""Explicit invariant molecular feature adapters; never flatten vector xyz channels."""

import torch
from torch import nn


class MolecularEncoder(nn.Module):
    def __init__(self, config, root, device, vocab_size=None):
        super().__init__()
        self.name = config["backbone"]["model"]
        source = config["source"]
        m = source["model"]
        p = m.get("params", {})
        if self.name == "chemprop":
            from chemprop.nn import BondMessagePassing, MeanAggregation

            self.core = BondMessagePassing(
                d_h=m["message_hidden_dim"],
                depth=m["depth"],
                dropout=m["dropout"],
                activation=m["activation"],
            )
            self.pool = MeanAggregation()
            if m["aggregation"] != "mean" or m.get("batch_norm"):
                raise ValueError("This adapter preserves the reviewed mean/no-BN Chemprop")
            self.output_dim = self.core.output_dim
        elif self.name in {"smiles_transformer", "spe_transformer"}:
            from molgap.transformer.model import SmilesTransformerRegressor

            self.core = SmilesTransformerRegressor(vocab_size, source["tokenizer"]["max_length"], m)
            self.core.regression_head = nn.Identity()
            self.output_dim = m["d_model"]
        else:
            from molgap.qm9_models.factory import build_model

            self.core = build_model(self.name, p, root, device)
            if self.name in {"schnet", "painn"}:
                self.output_dim = p["n_atom_basis"]
                self.core.readout.outnet = nn.Identity()
            elif self.name == "egnn":
                self.output_dim = p["hidden_nf"]
                self.core.network.graph_dec = nn.Identity()
            elif self.name == "esa":
                self.output_dim = p["hidden_dim"]
                self.core.readout = nn.Identity()
            elif self.name == "demol":
                self.output_dim = 2 * p["embed_dim"]
                self.core.readout = nn.Identity()
                if self.core.bond_regularization:
                    raise ValueError("Use the reviewed bond_regularization=false protocol")
                # Unused original auxiliary head is not an optimized model parameter.
                self.core.bond_head = nn.Identity()
            elif self.name == "transformer_m":
                self.output_dim = p["embed_dim"]
                self.core.head = nn.Identity()
            elif self.name == "3dinfomax":
                self.output_dim = p["hidden_dim"] * len(p["readout_aggregators"])
                self.core.network.output = nn.Identity()
            elif self.name == "torchmd_net":
                self.output_dim = p["embedding_dimension"]
                self.core.network.output_model = nn.Identity()
            elif self.name == "dimenet_pp":
                self.output_dim = p["out_emb_channels"]
                for output in self.core.network.output_blocks:
                    output.lin = nn.Identity()
            else:
                raise ValueError(f"No reviewed feature adapter: {self.name}")
        self.to(device)

    def forward(self, inputs):
        if self.name == "chemprop":
            return self.pool(self.core(inputs["graph"], inputs.get("V_d")), inputs["graph"].batch)
        if self.name in {"smiles_transformer", "spe_transformer"}:
            return self.core.encode(inputs["input_ids"], inputs["attention_mask"])
        if self.name == "torchmd_net":
            from molgap.qm9_models.batch import flat_atoms

            z, pos, groups = flat_atoms(inputs)
            scalar, _, _, _, groups = self.core.network.representation_model(z, pos, groups)
            out = scalar.new_zeros(len(inputs["smiles"]), self.output_dim)
            return out.index_add(0, groups, scalar)
        if self.name == "dimenet_pp":
            from molgap.qm9_models.batch import flat_atoms, radius_edges
            from molgap.qm9_models.dimenet import triplets

            z, pos, groups = flat_atoms(inputs)
            i, j, ii, jj, kk, kj, ji = triplets(radius_edges(pos, groups, self.core.cutoff), len(z))
            distance = (pos[i] - pos[j]).norm(dim=-1)
            a, b = pos[jj] - pos[kk], pos[ii] - pos[jj]
            angle = torch.atan2(torch.cross(b, a, dim=-1).norm(dim=-1), (a * b).sum(-1))
            net = self.core.network
            rbf, sbf = net.rbf(distance), net.sbf(distance, angle, kj)
            x = net.emb(z, rbf, i, j)
            atoms = net.output_blocks[0](x, rbf, i, num_nodes=len(z))
            for interaction, output in zip(
                net.interaction_blocks, net.output_blocks[1:], strict=True
            ):
                x = interaction(x, rbf, sbf, kj, ji)
                atoms = atoms + output(x, rbf, i, num_nodes=len(z))
            return atoms.new_zeros(len(inputs["smiles"]), self.output_dim).index_add(
                0, groups, atoms
            )
        value = self.core(inputs)
        return value.reshape(len(inputs["smiles"]), self.output_dim)


def move_inputs(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {k: move_inputs(v, device) for k, v in value.items()}
    if hasattr(value, "to"):
        # Chemprop BatchMolGraph.to mutates and returns None.
        result = value.to(device)
        return value if result is None else result
    return value
