"""Official ESA core with the authors' QM9 atom/bond feature schema."""

import torch
from rdkit import Chem
from torch import nn

ATOM_FEATURE_DIM = 35
BOND_FEATURE_DIM = 13


def _one_hot_unknown(value, choices):
    """Match the official ESA Chemprop-style encoding, including its unknown bin."""
    encoded = [0] * len(choices)
    encoded[choices.index(value) if value in choices else -1] = 1
    return encoded


def chemical_graph(smiles: str):
    """Construct a 2D graph from SMILES; no target or coordinates enter ESA."""
    molecule = Chem.AddHs(Chem.MolFromSmiles(smiles))
    features = []
    for atom in molecule.GetAtoms():
        features.append(
            _one_hot_unknown(atom.GetAtomicNum(), list(range(9)))
            + _one_hot_unknown(atom.GetTotalDegree(), [0, 1, 2, 3, 4, 5])
            + _one_hot_unknown(atom.GetFormalCharge(), [-1, -2, 1, 2, 0])
            + _one_hot_unknown(int(atom.GetChiralTag()), [0, 1, 2, 3])
            + _one_hot_unknown(int(atom.GetTotalNumHs()), [0, 1, 2, 3, 4])
            + _one_hot_unknown(
                int(atom.GetHybridization()),
                [
                    int(Chem.HybridizationType.SP),
                    int(Chem.HybridizationType.SP2),
                    int(Chem.HybridizationType.SP3),
                    int(Chem.HybridizationType.SP3D),
                    int(Chem.HybridizationType.SP3D2),
                ],
            )
            + [int(atom.GetIsAromatic())]
        )
    edges, attrs = [], []
    for bond in molecule.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bond_type = bond.GetBondType()
        attr = [
            0,
            int(bond_type == Chem.BondType.SINGLE),
            int(bond_type == Chem.BondType.DOUBLE),
            int(bond_type == Chem.BondType.TRIPLE),
            int(bond_type == Chem.BondType.AROMATIC),
            int(bond.GetIsConjugated()),
            int(bond.IsInRing()),
            *_one_hot_unknown(int(bond.GetStereo()), list(range(6))),
        ]
        edges.extend([(i, j), (j, i)])
        attrs.extend([attr, attr])
    nodes = torch.tensor(features).float()
    attributes = torch.tensor(attrs).float()
    if nodes.shape[1] != ATOM_FEATURE_DIM or attributes.shape[1] != BOND_FEATURE_DIM:
        raise ValueError("ESA feature dimensions differ from the official QM9 schema")
    return nodes, torch.tensor(edges).long().T, attributes


class ESAAdapter(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        from molgap._vendor.esa.masked_layers import ESA
        from molgap._vendor.esa.mlp_utils import SmallMLP

        width = config["hidden_dim"]
        self.max_edges = config["max_edges"]
        self.embed = SmallMLP(
            in_dim=2 * ATOM_FEATURE_DIM + BOND_FEATURE_DIM,
            inter_dim=int(config.get("mlp_hidden_size", 256)),
            out_dim=width,
            use_ln=False,
            dropout_p=0,
            num_layers=int(config.get("num_mlp_layers", 3)),
        )
        layers = config["layer_types"]
        if not layers or "P" not in layers:
            raise ValueError("ESA graph regression requires a PMA ('P') pooling layer")
        if any(layer not in {"M", "S", "P"} for layer in layers):
            raise ValueError("ESA layer_types may contain only 'M', 'S', and 'P'")
        self.network = ESA(
            num_outputs=int(config.get("num_outputs", 32)),
            dim_output=width,
            dim_hidden=[width] * len(layers),
            num_heads=[config["num_heads"]] * len(layers),
            layer_types=layers,
            node_or_edge="edge",
            xformers_or_torch_attn="torch",
            pre_or_post=config.get("pre_or_post", "post"),
            norm_type=config.get("norm_type", "BN"),
            use_mlps=bool(config.get("use_mlps", True)),
            mlp_hidden_size=int(config.get("mlp_hidden_size", 256)),
            num_mlp_layers=int(config.get("num_mlp_layers", 3)),
            mlp_type="standard",
            use_bfloat16=False,
            set_max_items=self.max_edges,
        )
        self.readout = nn.Linear(width, 1)

    def forward(self, batch: dict) -> torch.Tensor:
        from torch_geometric.utils import to_dense_batch

        device = next(self.parameters()).device
        nodes, edges, attributes, groups = [], [], [], []
        offset = 0
        for i, smiles in enumerate(batch["smiles"]):
            x, edge, attr = chemical_graph(smiles)
            if edge.shape[1] > self.max_edges:
                raise ValueError("ESA edge sequence exceeds configured capacity")
            nodes.append(x)
            edges.append(edge + offset)
            attributes.append(attr)
            groups.append(torch.full((len(x),), i, dtype=torch.long))
            offset += len(x)
        x = torch.cat(nodes).to(device)
        edge = torch.cat(edges, dim=1).to(device)
        attr = torch.cat(attributes).to(device)
        groups = torch.cat(groups).to(device)
        h = self.embed(torch.cat((x[edge[0]], x[edge[1]], attr), dim=-1))
        h, _ = to_dense_batch(h, groups[edge[0]], max_num_nodes=self.max_edges)
        h = self.network(h, edge, groups, num_max_items=self.max_edges)
        return self.readout(h).reshape(-1)
