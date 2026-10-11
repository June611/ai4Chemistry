"""Official ALIGNN on nonperiodic molecular radius graphs and line graphs."""

from pathlib import Path

import torch
from torch import nn

from molgap.qm9_models.sources import activate_source


class ALIGNNAdapter(nn.Module):
    def __init__(self, root: Path, config: dict) -> None:
        super().__init__()
        activate_source(root, "alignn")
        from alignn.models.alignn import ALIGNN, ALIGNNConfig
        from jarvis.core.specie import get_node_attributes
        from rdkit import Chem

        params = dict(config)
        self.cutoff = params.pop("cutoff")
        self.network = ALIGNN(ALIGNNConfig(name="alignn", **params))
        features = torch.zeros(10, 92)
        for z in (1, 6, 7, 8, 9):
            symbol = Chem.GetPeriodicTable().GetElementSymbol(z)
            features[z] = torch.tensor(get_node_attributes(symbol, atom_features="cgcnn"))
        self.register_buffer("element_features", features)

    def forward(self, batch: dict) -> torch.Tensor:
        import dgl

        graphs, lines = [], []
        for z, pos, mask in zip(
            batch["charges"], batch["positions"], batch["atom_mask"], strict=True
        ):
            z, pos = z[mask], pos[mask]
            distance = torch.cdist(pos, pos)
            edges = (distance < self.cutoff) & ~torch.eye(len(z), device=z.device, dtype=torch.bool)
            src, dst = torch.where(edges)
            graph = dgl.graph((src, dst), num_nodes=len(z), device=z.device)
            graph.ndata["atom_features"] = self.element_features[z]
            graph.edata["r"] = pos[dst] - pos[src]
            line = graph.line_graph(backtracking=False, shared=True)
            a, b = line.edges()
            r1, r2 = -graph.edata["r"][a], graph.edata["r"][b]
            line.edata["h"] = torch.nn.functional.cosine_similarity(r1, r2, dim=-1).clamp(-1, 1)
            graphs.append(graph)
            lines.append(line)
        return self.network((dgl.batch(graphs), dgl.batch(lines), None)).reshape(-1)
