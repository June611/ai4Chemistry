"""3D Infomax fine-tuning adapter using the authors' 2D PNA encoder."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import torch
from rdkit import Chem
from torch import nn

from molgap.qm9_models.sources import activate_source


def _molecular_graph(smiles: str, device: torch.device):
    import dgl
    from ogb.utils.features import atom_to_feature_vector, bond_to_feature_vector

    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"Invalid SMILES for 3D Infomax: {smiles}")
    molecule = Chem.AddHs(molecule)
    nodes = [atom_to_feature_vector(atom) for atom in molecule.GetAtoms()]
    edges: list[tuple[int, int]] = []
    edge_features = []
    for bond in molecule.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        feature = bond_to_feature_vector(bond)
        edges.extend(((i, j), (j, i)))
        edge_features.extend((feature, feature))
    if not edges:
        raise ValueError(f"3D Infomax PNA requires at least one bond: {smiles}")
    src, dst = zip(*edges, strict=True)
    graph = dgl.graph((src, dst), num_nodes=len(nodes), device=device)
    graph.ndata["feat"] = torch.tensor(nodes, dtype=torch.long, device=device)
    graph.edata["feat"] = torch.tensor(edge_features, dtype=torch.long, device=device)
    return graph


class ThreeDInfomaxAdapter(nn.Module):
    """Official PNA downstream model initialized from a QM9 3D-Infomax checkpoint."""

    def __init__(self, root: Path, config: dict) -> None:
        super().__init__()
        activate_source(root, "3dinfomax")
        # Several research repositories use top-level packages named ``models``
        # and ``commons``. Remove an earlier adapter's modules so Python resolves
        # these names from the just-activated, pinned 3D Infomax checkout.
        for module_name in list(sys.modules):
            if module_name in {"models", "commons"} or module_name.startswith(
                ("models.", "commons.")
            ):
                del sys.modules[module_name]
        from models.pna import PNA

        hidden = int(config.get("hidden_dim", 200))
        self.network = PNA(
            hidden_dim=hidden,
            target_dim=1,
            aggregators=config.get("aggregators", ["mean", "max", "min", "std"]),
            scalers=config.get("scalers", ["identity", "amplification", "attenuation"]),
            readout_aggregators=config.get("readout_aggregators", ["min", "max", "mean", "sum"]),
            readout_batchnorm=bool(config.get("readout_batchnorm", True)),
            readout_hidden_dim=int(config.get("readout_hidden_dim", hidden)),
            readout_layers=int(config.get("readout_layers", 2)),
            residual=bool(config.get("residual", True)),
            mid_batch_norm=bool(config.get("mid_batch_norm", True)),
            last_batch_norm=bool(config.get("last_batch_norm", True)),
            propagation_depth=int(config.get("propagation_depth", 7)),
            dropout=float(config.get("dropout", 0.0)),
            posttrans_layers=int(config.get("posttrans_layers", 1)),
            pretrans_layers=int(config.get("pretrans_layers", 2)),
            batch_norm_momentum=float(config.get("batch_norm_momentum", 0.1)),
        )
        checkpoint = config.get("pretrain_checkpoint")
        if checkpoint:
            path = Path(checkpoint)
            if not path.is_absolute():
                path = root / path
            self._load_pretrained_weights(path)
        elif config.get("require_pretrained", True):
            raise ValueError(
                "3D Infomax requires a QM9-pretrained checkpoint; set model.params."
                "pretrain_checkpoint after the official pretraining stage"
            )

    def _load_pretrained_weights(self, path: Path) -> None:
        if not path.is_file():
            raise FileNotFoundError(f"3D Infomax checkpoint does not exist: {path}")
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        source = checkpoint.get("model_state_dict", checkpoint)
        transferable = {}
        destination = self.network.state_dict()
        for key, value in source.items():
            if "teacher" in key or "batch_norm" in key:
                continue
            mapped = re.sub(r"^gnn\.|^gnn2\.", "node_gnn.", key.replace("student.", ""))
            if (
                mapped.startswith("node_gnn.")
                and mapped in destination
                and destination[mapped].shape == value.shape
            ):
                transferable[mapped] = value
        if not transferable:
            raise ValueError(f"No compatible 2D PNA encoder weights found in {path}")
        destination.update(transferable)
        self.network.load_state_dict(destination)
        self.transferred_parameter_count = len(transferable)

    def forward(self, batch: dict) -> torch.Tensor:
        import dgl

        device = next(self.parameters()).device
        graph = dgl.batch([_molecular_graph(smiles, device) for smiles in batch["smiles"]])
        return self.network(graph).reshape(-1)
