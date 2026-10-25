"""DimeNet++ via the PyG implementation linked by the authors.

The PyG neural blocks are unchanged. Pure-Torch neighbor/triplet enumeration avoids
compiled torch-cluster/torch-sparse dependencies for small nonperiodic QM9 molecules.
"""

import math

import torch
from torch import nn

from molgap.qm9_models.batch import flat_atoms, padded_radius_edges


def triplets(edge_index: torch.Tensor, num_nodes: int) -> tuple:
    row, col = edge_index
    # Build incoming edge lists per node, then enumerate k->j->i without i==k.
    counts = torch.bincount(col, minlength=num_nodes)
    width = int(counts.max()) if counts.numel() else 0
    incoming = torch.full((num_nodes, width), -1, dtype=torch.long, device=row.device)
    order = torch.argsort(col, stable=True)
    starts = counts.cumsum(0) - counts
    ranks = torch.arange(len(row), device=row.device) - starts[col[order]]
    incoming[col[order], ranks] = order
    candidates = incoming[row]
    ji = torch.arange(len(row), device=row.device)[:, None].expand_as(candidates)
    valid = candidates >= 0
    kj, ji = candidates[valid], ji[valid]
    valid = row[kj] != col[ji]
    kj, ji = kj[valid], ji[valid]
    return col, row, col[ji], row[ji], row[kj], kj, ji


class DimeNetPPAdapter(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        from torch_geometric.nn.models import DimeNetPlusPlus, dimenet_utils

        # PyG 2.6.1 uses the removed NumPy 2 np.math alias; same scalar formula.
        def prefactor(k: int, m: int) -> float:
            return math.sqrt(
                (2 * k + 1)
                * math.factorial(k - abs(m))
                / (4 * math.pi * math.factorial(k + abs(m)))
            )

        dimenet_utils.sph_harm_prefactor = prefactor

        self.cutoff = float(config.get("cutoff", 5.0))
        self.network = DimeNetPlusPlus(out_channels=1, **config)

    def forward(self, batch: dict) -> torch.Tensor:
        z, pos, groups = flat_atoms(batch)
        edges = padded_radius_edges(batch, self.cutoff)
        i, j, idx_i, idx_j, idx_k, idx_kj, idx_ji = triplets(edges, len(z))
        dist = (pos[i] - pos[j]).square().sum(-1).sqrt()
        pos_jk, pos_ij = pos[idx_j] - pos[idx_k], pos[idx_i] - pos[idx_j]
        angle = torch.atan2(
            torch.cross(pos_ij, pos_jk, dim=1).norm(dim=-1), (pos_ij * pos_jk).sum(-1)
        )
        network = self.network
        rbf = network.rbf(dist)
        sbf = network.sbf(dist, angle, idx_kj)
        x = network.emb(z, rbf, i, j)
        prediction = network.output_blocks[0](x, rbf, i, num_nodes=len(z))
        for interaction, output in zip(
            network.interaction_blocks, network.output_blocks[1:], strict=True
        ):
            x = interaction(x, rbf, sbf, idx_kj, idx_ji)
            prediction = prediction + output(x, rbf, i, num_nodes=len(z))
        result = prediction.new_zeros((batch["charges"].shape[0], 1))
        return result.index_add(0, groups, prediction).reshape(-1)
