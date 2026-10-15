"""Model selection without importing unused optional frameworks."""

from pathlib import Path

import torch


def build_model(name: str, config: dict, root: Path, device: torch.device):
    if name == "egnn":
        from molgap.qm9_models.egnn import EGNNAdapter

        model = EGNNAdapter(root, config, device)
    elif name == "demol":
        from molgap.qm9_models.demol import DeMolAdapter

        model = DeMolAdapter(config)
    elif name == "transformer_m":
        from molgap.qm9_models.transformer_m import TransformerMAdapter

        model = TransformerMAdapter(config)
    elif name in {"schnet", "painn"}:
        from molgap.qm9_models.schnetpack import SchNetPackAdapter

        model = SchNetPackAdapter(name, config)
    elif name == "cormorant":
        from molgap.qm9_models.cormorant import CormorantAdapter

        model = CormorantAdapter(root, config, device)
    elif name == "dimenet_pp":
        from molgap.qm9_models.dimenet import DimeNetPPAdapter

        model = DimeNetPPAdapter(config)
    elif name == "lie_transformer":
        from molgap.qm9_models.lie_transformer import LieTransformerAdapter

        model = LieTransformerAdapter(root, config)
    elif name == "esa":
        from molgap.qm9_models.esa import ESAAdapter

        model = ESAAdapter(config)
    elif name == "alignn":
        from molgap.qm9_models.alignn import ALIGNNAdapter

        model = ALIGNNAdapter(root, config)
    elif name == "torchmd_net":
        from molgap.qm9_models.torchmd_net import TorchMDNetAdapter

        model = TorchMDNetAdapter(root, config)
    elif name == "3dinfomax":
        from molgap.qm9_models.infomax import ThreeDInfomaxAdapter

        model = ThreeDInfomaxAdapter(root, config)
    else:
        raise ValueError(f"Adapter is not implemented/registered: {name}")
    return model.to(device)
